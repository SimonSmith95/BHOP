"""
Constraint Dialog
=================
Lets the user define, edit and delete parameter constraints.

Each constraint is a named rule with four parts:

    expression   — arithmetic formula over parameter names,
                   e.g. "CsPbI + FAPbI + MAPbI"  or  "temp * time"
    operator     — one of  =  <=  >=
    target       — numeric right-hand-side value
    residual     — (equality only) the parameter that will be auto-computed
                   as  target − (expression evaluated without it).
                   This parameter is removed from Optuna's search space.

Allowed expression operators: +  −  *  /  **  ( )  and parameter names.
No arbitrary Python — expressions are parsed via ``ast`` and evaluated safely.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from parameter_config import ParameterConstraint


# ── Catppuccin Mocha colours reused from main palette ─────────────────────────
_BG    = "#1e1e2e"
_FG    = "#cdd6f4"
_GRID  = "#313244"
_GREEN = "#a6e3a1"
_RED   = "#f38ba8"
_AMBER = "#f9e2af"


# ══════════════════════════════════════════════════════════════════════════════
# Single-constraint edit form (reused for Add and Edit)
# ══════════════════════════════════════════════════════════════════════════════

class _ConstraintEditDialog(QDialog):
    """
    Small modal dialog to create or edit a single :class:`ParameterConstraint`.
    """

    def __init__(
        self,
        param_names: List[str],
        constraint: Optional[ParameterConstraint] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._param_names = param_names
        self.setWindowTitle("Edit Constraint" if constraint else "Add Constraint")
        self.setMinimumWidth(460)
        self._build_ui(constraint)

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self, c: Optional[ParameterConstraint]) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(8)

        # Name
        self._name_edit = QLineEdit(c.name if c else "")
        self._name_edit.setPlaceholderText('e.g. "Composition"')
        form.addRow("Name:", self._name_edit)

        # Expression
        self._expr_edit = QLineEdit(c.expression if c else "")
        self._expr_edit.setPlaceholderText(
            "e.g.  CsPbI + FAPbI + MAPbI   or   temp * time"
        )
        form.addRow("Expression:", self._expr_edit)

        # Operator
        self._op_combo = QComboBox()
        self._op_combo.addItems(["=", "<=", ">="])
        if c:
            self._op_combo.setCurrentText(c.operator)
        self._op_combo.currentTextChanged.connect(self._on_op_changed)
        form.addRow("Operator:", self._op_combo)

        # Target
        self._target_spin = QDoubleSpinBox()
        self._target_spin.setRange(-1e12, 1e12)
        self._target_spin.setDecimals(6)
        self._target_spin.setSingleStep(0.1)
        self._target_spin.setValue(c.target if c else 1.0)
        form.addRow("Target value:", self._target_spin)

        # Residual parameter (only relevant for "=")
        self._residual_combo = QComboBox()
        self._residual_combo.addItem("(none — all params are free)")
        for pn in self._param_names:
            self._residual_combo.addItem(pn)
        if c and c.residual_param:
            idx = self._residual_combo.findText(c.residual_param)
            if idx >= 0:
                self._residual_combo.setCurrentIndex(idx)
        self._residual_label = QLabel("Residual param:")
        form.addRow(self._residual_label, self._residual_combo)

        layout.addLayout(form)

        # Help text
        self._help = QLabel()
        self._help.setWordWrap(True)
        self._help.setStyleSheet(f"color: {_AMBER}; font-size: 11px;")
        layout.addWidget(self._help)

        # Known parameters hint
        param_hint = QLabel(
            "Known parameters: " + (", ".join(self._param_names) or "(none yet)")
        )
        param_hint.setWordWrap(True)
        param_hint.setStyleSheet(f"color: {_FG}; font-size: 10px; font-style: italic;")
        layout.addWidget(param_hint)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._on_op_changed(self._op_combo.currentText())

    # ── Slots ───────────────────────────────────────────────────────────────

    def _on_op_changed(self, op: str) -> None:
        is_eq = (op == "=")
        self._residual_label.setVisible(is_eq)
        self._residual_combo.setVisible(is_eq)
        if is_eq:
            self._help.setText(
                "Equality (=): one parameter will be <b>auto-computed</b> so the "
                "expression always equals the target.  Select it as the "
                "<i>Residual param</i>.  It will be removed from Optuna's search "
                "space — the surrogate works in the reduced (N-1) dimension."
            )
        else:
            self._help.setText(
                "Inequality (≤/≥): all parameters remain free.  Optuna's surrogate "
                "learns to avoid infeasible regions via its constraints_func hook.  "
                "Any violated suggestion is projected to the constraint boundary "
                "before being shown to the user."
            )

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        expr = self._expr_edit.text().strip()
        op   = self._op_combo.currentText()
        tgt  = self._target_spin.value()
        res  = self._residual_combo.currentText()
        if res.startswith("(none"):
            res = ""

        if not name:
            QMessageBox.warning(self, "Validation", "Please enter a constraint name.")
            return

        c = ParameterConstraint(
            name=name,
            expression=expr,
            operator=op,
            target=tgt,
            residual_param=res,
        )
        errs = c.validate_expression(self._param_names)
        if errs:
            QMessageBox.warning(
                self, "Expression Error",
                "The constraint has the following issues:\n\n• " + "\n• ".join(errs),
            )
            return

        self._result = c
        self.accept()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def result(self) -> Optional[ParameterConstraint]:
        return getattr(self, "_result", None)


# ══════════════════════════════════════════════════════════════════════════════
# Main Constraint Manager Dialog
# ══════════════════════════════════════════════════════════════════════════════

class ConstraintDialog(QDialog):
    """
    Non-modal* dialog that lists all :class:`ParameterConstraint` objects
    and provides Add / Edit / Remove controls.

    (*) Opened modally from main_window via exec() so that constraints are
    applied before the next "Ask Batch" click.

    Usage
    -----
    ::
        dlg = ConstraintDialog(param_names, existing_constraints, parent=self)
        if dlg.exec():
            config.constraints = dlg.constraints
    """

    def __init__(
        self,
        param_names: List[str],
        constraints: Optional[List[ParameterConstraint]] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._param_names = param_names
        self._constraints: List[ParameterConstraint] = list(constraints or [])

        self.setWindowTitle("📐  Parameter Constraints")
        self.setMinimumWidth(680)
        self.setMinimumHeight(360)
        self._build_ui()
        self._refresh_table()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Intro text
        intro = QLabel(
            "<b>Define rules that constrain how parameter values relate to each other.</b><br>"
            "Examples: fractions that must sum to 1 (equality), or a processing "
            "budget that must not be exceeded (inequality)."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_FG}; font-size: 12px;")
        layout.addWidget(intro)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {_GRID};")
        layout.addWidget(sep)

        # ── Constraint table ──────────────────────────────────────────────
        cols = ["Name", "Expression", "Op", "Target", "Residual param (= only)"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.doubleClicked.connect(self._on_edit)
        layout.addWidget(self._table, stretch=1)

        # ── Toolbar ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        add_btn = QPushButton("＋  Add Constraint")
        add_btn.clicked.connect(self._on_add)
        btn_row.addWidget(add_btn)

        edit_btn = QPushButton("✏  Edit Selected")
        edit_btn.clicked.connect(self._on_edit)
        btn_row.addWidget(edit_btn)

        del_btn = QPushButton("🗑  Remove Selected")
        del_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(del_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Status / hint ─────────────────────────────────────────────────
        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {_AMBER}; font-size: 11px;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # ── Dialog buttons ────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ── Table helpers ───────────────────────────────────────────────────────

    def _refresh_table(self) -> None:
        self._table.setRowCount(0)
        for c in self._constraints:
            row = self._table.rowCount()
            self._table.insertRow(row)
            items = [
                c.name,
                c.expression,
                c.operator,
                str(c.target),
                c.residual_param or "—",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                # Colour the operator cell
                if col == 2:
                    if c.operator == "=":
                        item.setForeground(QColor(_GREEN))
                    else:
                        item.setForeground(QColor(_AMBER))
                self._table.setItem(row, col, item)

        self._update_status()

    def _selected_row(self) -> int:
        rows = self._table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _update_status(self) -> None:
        issues: List[str] = []
        for c in self._constraints:
            errs = c.validate_expression(self._param_names)
            if errs:
                issues.append(f"• {c.name}: " + "; ".join(errs))
        if issues:
            self._status.setText("⚠ Issues detected:\n" + "\n".join(issues))
        else:
            self._status.setText(
                f"{len(self._constraints)} constraint(s) defined." if self._constraints
                else "No constraints defined — parameters are optimized freely."
            )

    # ── Slot handlers ───────────────────────────────────────────────────────

    def _on_add(self) -> None:
        dlg = _ConstraintEditDialog(self._param_names, parent=self)
        if dlg.exec() and dlg.result:
            self._constraints.append(dlg.result)
            self._refresh_table()

    def _on_edit(self) -> None:
        row = self._selected_row()
        if row < 0:
            QMessageBox.information(self, "Edit", "Select a constraint row first.")
            return
        dlg = _ConstraintEditDialog(
            self._param_names, self._constraints[row], parent=self
        )
        if dlg.exec() and dlg.result:
            self._constraints[row] = dlg.result
            self._refresh_table()

    def _on_delete(self) -> None:
        row = self._selected_row()
        if row < 0:
            return
        name = self._constraints[row].name
        reply = QMessageBox.question(
            self, "Remove",
            f"Remove constraint '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            del self._constraints[row]
            self._refresh_table()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def constraints(self) -> List[ParameterConstraint]:
        """The current list of constraints (after the dialog is accepted)."""
        return list(self._constraints)
