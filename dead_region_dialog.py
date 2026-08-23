"""
Phase 6 — Dead Region Dialog
PySide6 dialog that lets the user interactively define dead (excluded) regions
for a parameter and previews the resulting allowed sub-ranges.
"""
from __future__ import annotations

import copy
from typing import List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from parameter_config import AllowedSubRange, ParameterConfig, ParameterType
from sampler_utils import compute_allowed_subranges, validate_subranges


class DeadRegionDialog(QDialog):
    """
    Modal dialog for configuring dead (excluded) regions of a parameter.

    For INT / FLOAT parameters
    --------------------------
    Shows a table of (Dead Low, Dead High) intervals that the user can add or
    remove.  A live preview shows the resulting allowed sub-ranges.

    For CATEGORICAL parameters
    --------------------------
    Shows a checklist: checked = allowed, unchecked = dead.

    For BOOL parameters
    -------------------
    Not applicable (use Fix = True / False on the card instead).
    """

    def __init__(self, param_config: ParameterConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = copy.deepcopy(param_config)
        self.setWindowTitle(f"Dead Regions — {param_config.name}")
        self.setMinimumWidth(520)
        self._ok_button: QPushButton | None = None
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        if self._config.ptype in (ParameterType.INT, ParameterType.FLOAT):
            self._build_numeric_ui(layout)
        elif self._config.ptype == ParameterType.CATEGORICAL:
            self._build_categorical_ui(layout)
        else:  # BOOL
            layout.addWidget(
                QLabel("""
                    "Bool parameters use "Fix = True" or "Fix = False" on the\n" 
                    "parameter card instead of dead regions." """
                )
            )

        # OK / Cancel
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        self._ok_button = btn_box.button(QDialogButtonBox.Ok)
        layout.addWidget(btn_box)

    # ── Numeric UI (INT / FLOAT) ───────────────────────────────────────────

    def _build_numeric_ui(self, layout: QVBoxLayout) -> None:
        self._is_int = self._config.ptype == ParameterType.INT
        self._full_min = self._config.full_min
        self._full_max = self._config.full_max

        layout.addWidget(
            QLabel(f"Full range: [{self._full_min}  –  {self._full_max}]")
        )
        layout.addWidget(QLabel("Define dead (excluded) intervals:"))

        # Table
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Dead Low", "Dead High", ""])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setFixedHeight(180)
        layout.addWidget(self._table)

        # Populate from existing dead regions (complement of allowed_subranges)
        for lo, hi in self._infer_dead_regions():
            self._add_row(lo, hi)

        add_btn = QPushButton("+ Add Dead Region")
        add_btn.clicked.connect(lambda: self._add_row(self._full_min, self._full_max))
        layout.addWidget(add_btn)

        # Allowed sub-ranges preview
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)
        layout.addWidget(QLabel("Resulting allowed sub-ranges:"))
        self._preview_label = QLabel()
        self._preview_label.setWordWrap(True)
        bold = QFont()
        bold.setBold(True)
        self._preview_label.setFont(bold)
        layout.addWidget(self._preview_label)

        self._update_preview()

    def _infer_dead_regions(self) -> List[Tuple[float, float]]:
        """Compute dead regions as the complement of the current allowed_subranges."""
        subranges = sorted(self._config.allowed_subranges, key=lambda r: r.low)
        dead: List[Tuple[float, float]] = []
        prev = self._full_min
        for r in subranges:
            if prev < r.low:
                dead.append((prev, r.low))
            prev = r.high
        if prev < self._full_max:
            dead.append((prev, self._full_max))
        return dead

    def _add_row(self, lo: float = 0.0, hi: float = 1.0) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        if self._is_int:
            lo_spin: QSpinBox | QDoubleSpinBox = QSpinBox()
            hi_spin: QSpinBox | QDoubleSpinBox = QSpinBox()
            for spin in (lo_spin, hi_spin):
                spin.setRange(int(self._full_min), int(self._full_max))
            lo_spin.setValue(int(lo))
            hi_spin.setValue(int(hi))
        else:
            lo_spin = QDoubleSpinBox()
            hi_spin = QDoubleSpinBox()
            for spin in (lo_spin, hi_spin):
                spin.setRange(self._full_min, self._full_max)
                spin.setDecimals(6)
                spin.setSingleStep((self._full_max - self._full_min) / 100)
            lo_spin.setValue(float(lo))
            hi_spin.setValue(float(hi))

        lo_spin.valueChanged.connect(self._update_preview)
        hi_spin.valueChanged.connect(self._update_preview)

        remove_btn = QPushButton("✕")
        remove_btn.setFixedWidth(32)
        remove_btn.clicked.connect(lambda _checked, btn=remove_btn: self._remove_row(btn))

        self._table.setCellWidget(row, 0, lo_spin)
        self._table.setCellWidget(row, 1, hi_spin)
        self._table.setCellWidget(row, 2, remove_btn)

        self._update_preview()

    def _remove_row(self, btn: QPushButton) -> None:
        for row in range(self._table.rowCount()):
            if self._table.cellWidget(row, 2) is btn:
                self._table.removeRow(row)
                break
        self._update_preview()

    def _get_dead_regions(self) -> List[Tuple[float, float]]:
        dead: List[Tuple[float, float]] = []
        for row in range(self._table.rowCount()):
            lo_w = self._table.cellWidget(row, 0)
            hi_w = self._table.cellWidget(row, 1)
            if lo_w and hi_w:
                dead.append((float(lo_w.value()), float(hi_w.value())))
        return dead

    def _update_preview(self) -> None:
        dead = self._get_dead_regions()
        try:
            allowed = compute_allowed_subranges(self._full_min, self._full_max, dead)
            if self._is_int:
                parts = [f"[{int(r.low)} – {int(r.high)}]" for r in allowed]
            else:
                parts = [f"[{r.low:.5g} – {r.high:.5g}]" for r in allowed]
            self._preview_label.setText("  ".join(parts))
            self._preview_label.setStyleSheet("color: #1a7a1a;")
            if self._ok_button:
                self._ok_button.setEnabled(True)
        except ValueError as exc:
            self._preview_label.setText(f"⚠  {exc}")
            self._preview_label.setStyleSheet("color: #cc0000;")
            if self._ok_button:
                self._ok_button.setEnabled(False)

    # ── Categorical UI ─────────────────────────────────────────────────────

    def _build_categorical_ui(self, layout: QVBoxLayout) -> None:
        layout.addWidget(
            QLabel("Check choices to ALLOW (uncheck to exclude / mark as dead):")
        )

        self._list_widget = QListWidget()
        for choice in self._config.all_choices:
            item = QListWidgetItem(str(choice))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            state = (
                Qt.Checked
                if str(choice) in self._config.allowed_choices
                else Qt.Unchecked
            )
            item.setCheckState(state)
            self._list_widget.addItem(item)
        self._list_widget.itemChanged.connect(self._validate_categorical)
        layout.addWidget(self._list_widget)

        btn_row = QHBoxLayout()
        sel_all = QPushButton("Select All")
        sel_all.clicked.connect(self._select_all)
        desel_all = QPushButton("Deselect All")
        desel_all.clicked.connect(self._deselect_all)
        btn_row.addWidget(sel_all)
        btn_row.addWidget(desel_all)
        layout.addLayout(btn_row)

        self._cat_warning = QLabel("")
        self._cat_warning.setStyleSheet("color: #cc0000;")
        layout.addWidget(self._cat_warning)

    def _select_all(self) -> None:
        for i in range(self._list_widget.count()):
            self._list_widget.item(i).setCheckState(Qt.Checked)

    def _deselect_all(self) -> None:
        for i in range(self._list_widget.count()):
            self._list_widget.item(i).setCheckState(Qt.Unchecked)

    def _validate_categorical(self) -> None:
        checked = [
            self._list_widget.item(i).text()
            for i in range(self._list_widget.count())
            if self._list_widget.item(i).checkState() == Qt.Checked
        ]
        if not checked:
            self._cat_warning.setText("⚠  At least one choice must be allowed.")
            if self._ok_button:
                self._ok_button.setEnabled(False)
        else:
            self._cat_warning.setText("")
            if self._ok_button:
                self._ok_button.setEnabled(True)

    # ── Result ─────────────────────────────────────────────────────────────

    def get_updated_config(self) -> ParameterConfig:
        """
        Return a copy of the original ParameterConfig updated with the user's
        dead-region choices.  Call only after the dialog has been accepted.
        """
        updated = copy.deepcopy(self._config)

        if self._config.ptype in (ParameterType.INT, ParameterType.FLOAT):
            dead = self._get_dead_regions()
            try:
                updated.allowed_subranges = compute_allowed_subranges(
                    self._full_min, self._full_max, dead
                )
            except ValueError:
                pass  # keep original if somehow invalid at accept time

        elif self._config.ptype == ParameterType.CATEGORICAL:
            updated.allowed_choices = [
                self._list_widget.item(i).text()
                for i in range(self._list_widget.count())
                if self._list_widget.item(i).checkState() == Qt.Checked
            ]

        return updated
