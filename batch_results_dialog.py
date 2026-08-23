"""
Phase 8 — Batch Results Dialog
After a batch is suggested, lets the user enter or import measured lab results
before they are told back to the Optuna study.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from parameter_config import ObjectiveConfig
from session_manager import SessionManager, SessionState


class BatchResultsDialog(QDialog):
    """
    Modal dialog displayed after ask_batch() completes.

    Shows the suggested experiment parameters (read-only) and collects the
    measured objective values from the user.  Results can be imported
    automatically from an updated CSV or entered manually.

    Signals
    -------
    results_submitted(list)
        Emitted when the user clicks "Submit All".
        Payload: list of {"trial_number": int, "values": list[float]}
    """

    results_submitted = Signal(list)

    def __init__(
        self,
        pending_batch: List[dict],          # [{"trial_number": int, "params": dict}, ...]
        objectives: List[ObjectiveConfig],
        session_state: SessionState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pending_batch = pending_batch
        self._objectives = objectives
        self._session_state = session_state

        self.setWindowTitle("Enter Batch Results")
        self.setMinimumWidth(740)
        self.setMinimumHeight(420)
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        n = len(self._pending_batch)
        layout.addWidget(
            QLabel(f"<b>Enter measured results for {n} suggested experiment(s).</b>")
        )
        layout.addWidget(
            QLabel(
                "Parameter columns show the <b>suggested</b> target values.  "
                "You can <b>edit them</b> to enter the <i>actual</i> composition "
                "you achieved in the lab before submitting."
            )
        )

        # ── Toolbar buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        export_btn = QPushButton("⬆  Export Suggested Params as CSV")
        export_btn.setToolTip(
            "Save the suggested parameter values to a CSV file for use in the lab."
        )
        export_btn.clicked.connect(self._export_params_csv)
        btn_row.addWidget(export_btn)

        import_btn = QPushButton("⬇  Import Results from CSV")
        import_btn.setToolTip(
            "Load an updated CSV and auto-fill matched objective cells."
        )
        import_btn.clicked.connect(self._import_results_csv)
        btn_row.addWidget(import_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Results table ──────────────────────────────────────────────────
        param_names: List[str] = (
            list(self._pending_batch[0]["params"].keys()) if self._pending_batch else []
        )
        obj_names = [o.column_name for o in self._objectives]
        all_cols = ["Trial #"] + param_names + obj_names

        self._table = QTableWidget(len(self._pending_batch), len(all_cols))
        self._table.setHorizontalHeaderLabels(all_cols)

        hdr = self._table.horizontalHeader()
        for i in range(len(all_cols)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hdr.setStretchLastSection(True)
        # Allow double-click / key editing — param cells are editable;
        # trial # and objective spinboxes handle their own interaction.
        self._table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.AnyKeyPressed
        )

        # Column-header tooltips
        self._table.horizontalHeaderItem(0)  # set after items exist; done below
        _PARAM_TIP = (
            "Suggested target value — double-click to edit with the actual "
            "composition you achieved in the lab."
        )

        # Store spinboxes: self._obj_spins[row][obj_idx]
        self._obj_spins: List[List[QDoubleSpinBox]] = []
        self._param_names = param_names
        self._obj_names = obj_names

        for row_idx, trial_data in enumerate(self._pending_batch):
            col = 0

            # Trial number (always read-only)
            self._set_readonly_cell(row_idx, col, str(trial_data["trial_number"]))
            col += 1

            # Param values — editable, amber tint to indicate they can be changed
            for pname in param_names:
                raw_val = trial_data["params"].get(pname, "")
                # Round floats to 6 significant figures so they fit in the cell
                try:
                    display_val = f"{float(raw_val):.6g}"
                except (ValueError, TypeError):
                    display_val = str(raw_val)
                item = QTableWidgetItem(display_val)
                item.setBackground(QColor("#f59e0b"))   # vivid amber — "suggested, may edit"
                item.setForeground(QColor("#1a1a2e"))   # near-black text for contrast
                item.setToolTip(_PARAM_TIP)
                self._table.setItem(row_idx, col, item)
                col += 1

            # Objective inputs — styled to stand out clearly as user-entry fields
            _SPIN_STYLE = (
                "QDoubleSpinBox {"
                "  background-color: #ffffff;"
                "  color: #1a1a2e;"
                "  border: 2px solid #89b4fa;"
                "  border-radius: 4px;"
                "  padding: 3px 6px;"
                "  font-size: 13px;"
                "}"
                "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0; }"
            )
            row_spins: List[QDoubleSpinBox] = []
            for _obj in obj_names:
                spin = QDoubleSpinBox()
                spin.setRange(-1e12, 1e12)
                spin.setDecimals(6)
                spin.setSingleStep(0.001)
                spin.setStyleSheet(_SPIN_STYLE)
                # Use a sentinel: minimum value = "not entered"
                spin.setSpecialValueText(" ")
                spin.setValue(spin.minimum())
                self._table.setCellWidget(row_idx, col, spin)
                row_spins.append(spin)
                col += 1
            self._obj_spins.append(row_spins)

        layout.addWidget(self._table)

        # ── Dialog buttons ─────────────────────────────────────────────────
        btn_box = QDialogButtonBox()
        submit_btn = btn_box.addButton("Submit All", QDialogButtonBox.AcceptRole)
        submit_btn.clicked.connect(self._on_submit)
        cancel_btn = btn_box.addButton("Cancel", QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(btn_box)

    def _set_readonly_cell(self, row: int, col: int, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, col, item)
        return item

    # ── Slot handlers ──────────────────────────────────────────────────────

    def _on_submit(self) -> None:
        results: List[dict] = []

        for row_idx, row_spins in enumerate(self._obj_spins):
            # ── Objective values ───────────────────────────────────────────
            values: List[float] = []
            for obj_idx, spin in enumerate(row_spins):
                if spin.value() == spin.minimum():
                    QMessageBox.warning(
                        self,
                        "Missing Values",
                        f"Row {row_idx + 1}, objective '{self._obj_names[obj_idx]}' "
                        "has no value.  Please fill in all cells before submitting.",
                    )
                    return
                values.append(spin.value())

            # ── Read back actual param values (user may have edited them) ──
            actual_params: dict = {}
            nominal_params: dict = self._pending_batch[row_idx]["params"]
            for pi, pname in enumerate(self._param_names):
                col_idx = 1 + pi       # col 0 = Trial #
                item = self._table.item(row_idx, col_idx)
                raw = item.text().strip() if item else ""
                if raw == "":
                    actual_params[pname] = nominal_params.get(pname)
                else:
                    try:
                        # Preserve int if the nominal value was an int
                        nom = nominal_params.get(pname)
                        if isinstance(nom, int):
                            actual_params[pname] = int(round(float(raw)))
                        else:
                            actual_params[pname] = float(raw)
                    except ValueError:
                        actual_params[pname] = raw   # keep as string for categorical

            results.append({
                "trial_number":  self._pending_batch[row_idx]["trial_number"],
                "values":        values,
                "actual_params": actual_params,
                "nominal_params": nominal_params,
            })

        self.results_submitted.emit(results)
        self.accept()

    def _on_cancel(self) -> None:
        reply = QMessageBox.question(
            self,
            "Cancel",
            "Cancel result entry?\n\n"
            "The pending batch is still saved in the session file.\n"
            "Use the '📋 Enter Pending Results…' toolbar button to reopen\n"
            "this dialog at any time — no need to reload the app.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.reject()

    # ── Export / Import ────────────────────────────────────────────────────

    def _export_params_csv(self) -> None:
        if not self._pending_batch:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Suggested Parameters", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        rows = [
            {"trial_number": td["trial_number"], **td["params"]}
            for td in self._pending_batch
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        QMessageBox.information(self, "Exported", f"Suggested parameters saved to:\n{path}")

    def _import_results_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Results CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "CSV Error", f"Cannot read file:\n{exc}")
            return

        matches = SessionManager.match_pending_to_csv(self._session_state, df)
        if not matches:
            QMessageBox.information(
                self,
                "No Matches Found",
                "No rows in the CSV matched the pending trial parameters.\n\n"
                "Make sure the parameter column values are identical to those "
                "that were suggested.",
            )
            return

        self.prefill_results(matches)
        filled = len(matches)
        total = len(self._pending_batch)
        QMessageBox.information(
            self,
            "Import Complete",
            f"Pre-filled {filled} of {total} trial(s).\n"
            + (
                f"{total - filled} trial(s) were not matched and must be entered manually."
                if filled < total
                else "All trials matched!"
            ),
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def prefill_results(self, matches: Dict[int, List[float]]) -> None:
        """
        Pre-fill objective spinboxes from a {trial_number: [values]} dict.
        Matched cells are highlighted green.
        """
        n_params = len(self._param_names)
        for row_idx, trial_data in enumerate(self._pending_batch):
            trial_num = trial_data["trial_number"]
            if trial_num not in matches:
                continue
            values = matches[trial_num]
            for obj_idx, val in enumerate(values):
                if obj_idx < len(self._obj_spins[row_idx]):
                    spin = self._obj_spins[row_idx][obj_idx]
                    spin.setValue(val)
                    # Visual highlight
                    spin.setStyleSheet("background-color: #d4edda;")  # light green
