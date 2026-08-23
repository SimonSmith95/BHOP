"""
Phase 7 — Parameter Card Widget
One collapsible card per parameter displayed in the main window's scroll area.
"""
from __future__ import annotations

import copy
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dead_region_dialog import DeadRegionDialog
from parameter_config import AllowedSubRange, ParameterConfig, ParameterType


class ParamCardWidget(QFrame):
    """
    A framed card representing one experiment parameter.

    Layout
    ------
    Row 1 (always visible):
        [ ☑ Enabled ]  [ Name (bold) ]  [ Type ▼ ]
    Row 2 (type-dependent sub-widget):
        INT / FLOAT  : Min / Max spinboxes  +  "Dead Regions…" button
                       + allowed-ranges preview label
        CATEGORICAL  : compact scrollable checklist
        BOOL         : radio group  (Optimize / Fix=True / Fix=False)

    Signals
    -------
    config_changed()
        Emitted after any meaningful state change (type switch, range edit,
        dead-region update, enable/disable).
    """

    config_changed = Signal()

    def __init__(self, param_config: ParameterConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = copy.deepcopy(param_config)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self._build_ui()

    # ── Top-level construction ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(8, 6, 8, 6)
        self._outer.setSpacing(4)

        # ── Row 1: checkbox + name + type selector ─────────────────────────
        row1 = QHBoxLayout()
        self._enabled_cb = QCheckBox()
        self._enabled_cb.setChecked(self._config.enabled)
        self._enabled_cb.toggled.connect(self._on_enabled_changed)
        row1.addWidget(self._enabled_cb)

        name_lbl = QLabel(self._config.name)
        bold = QFont()
        bold.setBold(True)
        name_lbl.setFont(bold)
        row1.addWidget(name_lbl)
        row1.addStretch()

        self._type_combo = QComboBox()
        for pt in ParameterType:
            self._type_combo.addItem(pt.value, pt)
        self._type_combo.setCurrentText(self._config.ptype.value)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        row1.addWidget(self._type_combo)
        self._outer.addLayout(row1)

        # ── Row 2: type-specific sub-widget ───────────────────────────────
        self._sub_container = QWidget()
        self._sub_layout = QVBoxLayout(self._sub_container)
        self._sub_layout.setContentsMargins(20, 0, 0, 0)
        self._sub_layout.setSpacing(4)
        self._outer.addWidget(self._sub_container)

        self._build_sub_widget()
        self._update_enabled_state()

    # ── Sub-widget factory ─────────────────────────────────────────────────

    def _clear_sub_widget(self) -> None:
        while self._sub_layout.count():
            item = self._sub_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _build_sub_widget(self) -> None:
        self._clear_sub_widget()
        ptype = self._config.ptype
        if ptype in (ParameterType.INT, ParameterType.FLOAT):
            self._build_numeric_sub()
        elif ptype == ParameterType.CATEGORICAL:
            self._build_categorical_sub()
        elif ptype == ParameterType.BOOL:
            self._build_bool_sub()

    # ── INT / FLOAT sub-widget ─────────────────────────────────────────────

    def _build_numeric_sub(self) -> None:
        is_int = self._config.ptype == ParameterType.INT
        row = QHBoxLayout()

        row.addWidget(QLabel("Min:"))
        if is_int:
            self._min_spin: QSpinBox | QDoubleSpinBox = QSpinBox()
            self._min_spin.setRange(-999_999_999, 999_999_999)
            self._min_spin.setValue(int(self._config.full_min))
        else:
            self._min_spin = QDoubleSpinBox()
            self._min_spin.setRange(-1e12, 1e12)
            self._min_spin.setDecimals(6)
            self._min_spin.setValue(self._config.full_min)
        self._min_spin.valueChanged.connect(self._on_range_changed)
        row.addWidget(self._min_spin)

        row.addWidget(QLabel("Max:"))
        if is_int:
            self._max_spin: QSpinBox | QDoubleSpinBox = QSpinBox()
            self._max_spin.setRange(-999_999_999, 999_999_999)
            self._max_spin.setValue(int(self._config.full_max))
        else:
            self._max_spin = QDoubleSpinBox()
            self._max_spin.setRange(-1e12, 1e12)
            self._max_spin.setDecimals(6)
            self._max_spin.setValue(self._config.full_max)
        self._max_spin.valueChanged.connect(self._on_range_changed)
        row.addWidget(self._max_spin)

        dead_btn = QPushButton("Dead Regions…")
        dead_btn.setFixedWidth(130)
        dead_btn.clicked.connect(self._open_dead_region_dialog)
        row.addWidget(dead_btn)
        self._sub_layout.addLayout(row)

        self._range_preview = QLabel()
        self._range_preview.setStyleSheet("font-size: 11px;")
        self._sub_layout.addWidget(self._range_preview)
        self._refresh_range_preview()

    # ── CATEGORICAL sub-widget ─────────────────────────────────────────────

    def _build_categorical_sub(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(120)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(2)

        self._cat_checkboxes: List[QCheckBox] = []
        for choice in self._config.all_choices:
            cb = QCheckBox(str(choice))
            cb.setChecked(str(choice) in self._config.allowed_choices)
            cb.toggled.connect(self._on_cat_changed)
            inner_layout.addWidget(cb)
            self._cat_checkboxes.append(cb)

        scroll.setWidget(inner)
        self._sub_layout.addWidget(scroll)

    # ── BOOL sub-widget ────────────────────────────────────────────────────

    def _build_bool_sub(self) -> None:
        self._bool_group = QButtonGroup(self)
        rb_opt   = QRadioButton("Optimize (both)")
        rb_true  = QRadioButton("Fix = True")
        rb_false = QRadioButton("Fix = False")

        if self._config.fixed_value is None:
            rb_opt.setChecked(True)
        elif self._config.fixed_value:
            rb_true.setChecked(True)
        else:
            rb_false.setChecked(True)

        self._bool_group.addButton(rb_opt,   0)
        self._bool_group.addButton(rb_true,  1)
        self._bool_group.addButton(rb_false, 2)
        self._bool_group.idClicked.connect(self._on_bool_changed)

        row = QHBoxLayout()
        row.addWidget(rb_opt)
        row.addWidget(rb_true)
        row.addWidget(rb_false)
        row.addStretch()
        self._sub_layout.addLayout(row)

    # ── Slot handlers ──────────────────────────────────────────────────────

    def _on_enabled_changed(self, checked: bool) -> None:
        self._config.enabled = checked
        self._update_enabled_state()
        self.config_changed.emit()

    def _update_enabled_state(self) -> None:
        self._sub_container.setEnabled(self._config.enabled)

    def _on_type_changed(self) -> None:
        new_type: ParameterType = self._type_combo.currentData()
        self._config.ptype = new_type
        # Ensure sub-range list is consistent
        if new_type in (ParameterType.INT, ParameterType.FLOAT):
            if not self._config.allowed_subranges:
                self._config.allowed_subranges = [
                    AllowedSubRange(self._config.full_min, self._config.full_max)
                ]
        self._build_sub_widget()
        self.config_changed.emit()

    def _on_range_changed(self) -> None:
        self._config.full_min = float(self._min_spin.value())
        self._config.full_max = float(self._max_spin.value())
        # Reset sub-ranges to the new full range
        self._config.allowed_subranges = [
            AllowedSubRange(self._config.full_min, self._config.full_max)
        ]
        self._refresh_range_preview()
        self.config_changed.emit()

    def _on_cat_changed(self) -> None:
        self._config.allowed_choices = [
            cb.text() for cb in self._cat_checkboxes if cb.isChecked()
        ]
        self.config_changed.emit()

    def _on_bool_changed(self, btn_id: int) -> None:
        self._config.fixed_value = {0: None, 1: True, 2: False}[btn_id]
        self.config_changed.emit()

    def _open_dead_region_dialog(self) -> None:
        dlg = DeadRegionDialog(self._config, self)
        if dlg.exec() == QDialog.Accepted:
            self._config = dlg.get_updated_config()
            self._refresh_range_preview()
            self.config_changed.emit()

    def _refresh_range_preview(self) -> None:
        if not hasattr(self, "_range_preview"):
            return
        subranges = self._config.allowed_subranges
        is_int = self._config.ptype == ParameterType.INT
        if subranges:
            if is_int:
                parts = [f"[{int(r.low)}–{int(r.high)}]" for r in subranges]
            else:
                parts = [f"[{r.low:.5g}–{r.high:.5g}]" for r in subranges]
            self._range_preview.setText("Allowed: " + "  ".join(parts))
            self._range_preview.setStyleSheet("color: #1a7a1a; font-size: 11px;")
        else:
            self._range_preview.setText("⚠ No allowed ranges!")
            self._range_preview.setStyleSheet("color: #cc0000; font-size: 11px;")

    # ── Public API ─────────────────────────────────────────────────────────

    def get_config(self) -> ParameterConfig:
        """Return a deep copy of the current parameter configuration."""
        return copy.deepcopy(self._config)

    def set_config(self, param_config: ParameterConfig) -> None:
        """Repopulate the card from an external ParameterConfig (e.g. on session load)."""
        self._config = copy.deepcopy(param_config)
        # Block signals while rebuilding to avoid spurious config_changed emissions
        self._type_combo.blockSignals(True)
        self._type_combo.setCurrentText(self._config.ptype.value)
        self._type_combo.blockSignals(False)
        self._enabled_cb.blockSignals(True)
        self._enabled_cb.setChecked(self._config.enabled)
        self._enabled_cb.blockSignals(False)
        self._build_sub_widget()
        self._update_enabled_state()


# ── Local import guard (avoids circular import at module level) ────────────────
from PySide6.QtWidgets import QDialog  # noqa: E402 — must come after class body
