"""
Phases 9 & 10 — Main Window + Warning System
Central PySide6 QMainWindow that ties all components together.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import optuna
import pandas as pd
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from batch_results_dialog import BatchResultsDialog
from design_space_widget import DesignSpaceDialog
from csv_loader import extract_param_defaults, load_csv, load_trials_from_csv
from optuna_builder import (
    build_study,
    get_pareto_front,
    load_historical_trials,
    tell_batch,
)
from param_card_widget import ParamCardWidget
from constraint_dialog import ConstraintDialog
from parameter_config import ObjectiveConfig, ParameterConfig, ParameterConstraint, StudyConfig
from session_manager import SessionManager, SessionState
from worker import OptimizationWorker

# ── Warning thresholds (Phase 10) ─────────────────────────────────────────────
TPE_PARAM_WARNING_THRESHOLD = 8
MIN_TRIALS_PER_PARAM_RATIO = 2

# ── Dark theme stylesheet ──────────────────────────────────────────────────────
DARK_STYLESHEET = """
/* ── Base ── */
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
/* ── Dock widgets ── */
QDockWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
QDockWidget::title {
    background-color: #2a2a3e;
    padding: 4px 8px;
    font-weight: bold;
}
/* ── Group boxes ── */
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 5px;
    margin-top: 10px;
    color: #bac2de;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #89b4fa;
}
/* ── Buttons ── */
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 5px 12px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #1e1e2e;
}
QPushButton:disabled {
    color: #585b70;
    background-color: #1e1e2e;
    border-color: #313244;
}
QPushButton:checked {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-color: #89b4fa;
}
/* ── Combo boxes ── */
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px 8px;
    selection-background-color: #89b4fa;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #2a2a3e;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
    border: 1px solid #45475a;
}
/* ── Spin boxes ── */
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: #89b4fa;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #585b70;
    border: 1px solid #6c6f85;
    border-radius: 2px;
    width: 16px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QSpinBox::up-button:pressed, QSpinBox::down-button:pressed,
QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {
    background-color: #7aa2f7;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0;
    height: 0;
    border-left:  5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 6px solid #cdd6f4;
}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {
    border-bottom-color: #1e1e2e;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-left:  5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #cdd6f4;
}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {
    border-top-color: #1e1e2e;
}
/* ── Labels ── */
QLabel {
    color: #cdd6f4;
}
/* ── Checkboxes & radio buttons ── */
QCheckBox, QRadioButton {
    color: #cdd6f4;
    spacing: 6px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #45475a;
    border-radius: 3px;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QRadioButton::indicator {
    border-radius: 7px;
}
QRadioButton::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
/* ── Tables ── */
QTableWidget, QTableView {
    background-color: #1e1e2e;
    color: #cdd6f4;
    gridline-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    alternate-background-color: #24243e;
}
QTableWidget::item, QTableView::item {
    padding: 4px;
}
QTableWidget::item:selected, QTableView::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QHeaderView::section {
    background-color: #2a2a3e;
    color: #89b4fa;
    border: 1px solid #45475a;
    padding: 5px;
    font-weight: bold;
}
/* ── Scroll bars ── */
QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background-color: #1e1e2e;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #45475a;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #585b70;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
/* ── Tabs ── */
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 4px;
    background-color: #1e1e2e;
}
QTabBar::tab {
    background-color: #2a2a3e;
    color: #bac2de;
    padding: 6px 14px;
    border: 1px solid #45475a;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
}
QTabBar::tab:selected {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border-bottom: 2px solid #89b4fa;
}
QTabBar::tab:hover:!selected {
    background-color: #313244;
}
/* ── Menu bar ── */
QMenuBar {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}
QMenuBar::item:selected {
    background-color: #313244;
    border-radius: 3px;
}
QMenu {
    background-color: #2a2a3e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-radius: 3px;
}
QMenu::separator {
    height: 1px;
    background-color: #45475a;
    margin: 4px 8px;
}
/* ── Toolbar ── */
QToolBar {
    background-color: #2a2a3e;
    border-bottom: 1px solid #313244;
    spacing: 4px;
    padding: 3px;
}
QToolBar QToolButton {
    background-color: transparent;
    color: #cdd6f4;
    border-radius: 4px;
    padding: 4px 8px;
}
QToolBar QToolButton:hover {
    background-color: #313244;
}
/* ── Status bar ── */
QStatusBar {
    background-color: #181825;
    color: #bac2de;
    border-top: 1px solid #313244;
}
/* ── Scroll area ── */
QScrollArea {
    border: none;
    background-color: #1e1e2e;
}
/* ── Text browser (docs viewer) ── */
QTextBrowser {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    font-size: 13px;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}
/* ── Dialog ── */
QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
/* ── Frame separators ── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: #45475a;
}
"""

# Warning / banner colours per theme
_WARN_STYLE_DARK  = (
    "background:#3d2e00; color:#ffc107; padding:6px; border-radius:4px; border:1px solid #ffc107;"
)
_WARN_STYLE_LIGHT = (
    "background:#fff3cd; color:#856404; padding:6px; border-radius:4px; border:1px solid #ffc107;"
)
_BANNER_STYLE_DARK  = (
    "background:#3d2e00; border:1px solid #ffc107; padding:8px; border-radius:4px;"
)
_BANNER_STYLE_LIGHT = (
    "background:#fff3cd; border:1px solid #ffc107; padding:8px; border-radius:4px;"
)


# ── Simple Markdown → HTML converter (for the docs viewer) ─────────────────────
def _markdown_to_html(text: str) -> str:
    """
    Convert a subset of Markdown to HTML suitable for QTextBrowser.
    Handles: headings (##-####), bold (**), inline code (`),
    fenced code blocks (```), bullet lists (- ), blockquotes (>), tables (|).
    """
    import re
    lines = text.split("\n")
    html_lines: list[str] = []
    in_code_block = False
    in_list = False
    in_table = False

    def close_list():
        nonlocal in_list
        if in_list:
            html_lines.append("</ul>")
            in_list = False

    def close_table():
        nonlocal in_table
        if in_table:
            html_lines.append("</table>")
            in_table = False

    def inline(s: str) -> str:
        """Apply inline formatting: bold, inline code."""
        # Inline code first (to avoid double-processing)
        s = re.sub(r"`([^`]+)`", r"<code style='background:#2a2a3e;padding:1px 4px;"
                   r"border-radius:3px;font-family:monospace'>\1</code>", s)
        # Bold
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return s

    for line in lines:
        # Fenced code blocks
        if line.startswith("```"):
            if in_code_block:
                html_lines.append("</pre>")
                in_code_block = False
            else:
                close_list()
                close_table()
                html_lines.append(
                    "<pre style='background:#181825;color:#a6e3a1;"
                    "padding:10px;border-radius:6px;"
                    "border:1px solid #45475a;overflow-x:auto'>"
                )
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Headings
        if line.startswith("#### "):
            close_list(); close_table()
            html_lines.append(f"<h4 style='color:#cba6f7'>{inline(line[5:])}</h4>")
        elif line.startswith("### "):
            close_list(); close_table()
            html_lines.append(f"<h3 style='color:#89b4fa'>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            close_list(); close_table()
            html_lines.append(f"<h2 style='color:#89dceb;border-bottom:1px solid #45475a;"
                              f"padding-bottom:4px'>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            close_list(); close_table()
            html_lines.append(f"<h1 style='color:#89dceb'>{inline(line[2:])}</h1>")

        # Blockquotes
        elif line.startswith("> "):
            close_list(); close_table()
            html_lines.append(
                f"<blockquote style='border-left:3px solid #89b4fa;"
                f"margin:4px 0;padding:4px 12px;color:#bac2de'>"
                f"{inline(line[2:])}</blockquote>"
            )

        # Table rows
        elif line.startswith("|") and line.endswith("|"):
            stripped = line.strip("|")
            if re.match(r"^[\s\-|:]+$", stripped):
                # Separator row — skip
                continue
            cells = [c.strip() for c in stripped.split("|")]
            if not in_table:
                html_lines.append(
                    "<table style='border-collapse:collapse;width:100%;"
                    "margin:8px 0'>"
                )
                in_table = True
                tag = "th"
                style = ("background:#2a2a3e;color:#89b4fa;font-weight:bold;"
                         "padding:6px 10px;border:1px solid #45475a")
            else:
                tag = "td"
                style = "padding:5px 10px;border:1px solid #313244"
            row = "".join(f"<{tag} style='{style}'>{inline(c)}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")

        # Bullet list items
        elif re.match(r"^[-*] ", line):
            close_table()
            if not in_list:
                html_lines.append("<ul style='margin:4px 0 4px 20px;padding:0'>")
                in_list = True
            html_lines.append(f"<li style='margin:2px 0'>{inline(line[2:])}</li>")

        # Horizontal rules
        elif re.match(r"^---+$", line.strip()):
            close_list(); close_table()
            html_lines.append("<hr style='border:none;border-top:1px solid #45475a;margin:8px 0'>")

        # Blank lines
        elif line.strip() == "":
            close_list(); close_table()
            html_lines.append("<p></p>")

        # Normal paragraph text
        else:
            close_table()
            html_lines.append(f"<p style='margin:2px 0'>{inline(line)}</p>")

    close_list()
    close_table()

    body = "\n".join(html_lines)
    return (
        "<html><head><style>"
        "body{font-family:sans-serif;font-size:13px;margin:12px;color:#cdd6f4;"
        "background:#1e1e2e}"
        "code{font-family:monospace}"
        "</style></head>"
        f"<body>{body}</body></html>"
    )


class MainWindow(QMainWindow):
    """
    Main application window for BHOP.

    Layout
    ------
    Menu bar   — File (session/config management) + Help
    Toolbar    — Load CSV / Reload CSV / path label
    Left dock  — Study settings (objectives, sampler, batch, warnings, Ask button)
    Centre     — Resume banner (if pending batch) + scrollable param cards
    Bottom dock— Results tabs (All Trials / Best-Pareto) + export button
    Status bar — trial count, best value, status text
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BHOP — Lab Optimizer")
        self.resize(1280, 820)

        # ── Application state ──────────────────────────────────────────────
        self._study: Optional[optuna.Study] = None
        self._session_state: Optional[SessionState] = None
        self._df: Optional[pd.DataFrame] = None
        self._csv_path: Optional[str] = None
        self._param_cards: List[ParamCardWidget] = []
        self._worker: Optional[OptimizationWorker] = None
        self._batches_done: int = 0
        self._design_space_dlg: Optional[DesignSpaceDialog] = None

        # Objective-row widgets [(col_name_label, direction_combo), ...]
        self._obj_row_widgets: List[tuple] = []
        # Per-column checkboxes in the objectives selector
        self._obj_col_rows: List[tuple] = []  # (col, QCheckBox, QComboBox)
        # User-defined parameter constraints (equality / inequality rules)
        self._constraints: List[ParameterConstraint] = []

        self._build_menu()
        self._build_toolbar()
        self._build_left_dock()
        self._build_centre()
        self._build_status_bar()

        # ── Theme (dark by default, persisted via QSettings) ──────────────
        settings = QSettings()
        dark = settings.value("dark_mode", True, type=bool)
        self._dark_mode_action.setChecked(dark)
        self._apply_theme(dark)

    # ══════════════════════════════════════════════════════════════════════
    # Menu bar
    # ══════════════════════════════════════════════════════════════════════

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # ── File ──────────────────────────────────────────────────────────
        file_menu = mb.addMenu("File")

        new_act = QAction("New Session…", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self._action_new_session)
        file_menu.addAction(new_act)

        load_sess_act = QAction("Load Session…", self)
        load_sess_act.setShortcut("Ctrl+O")
        load_sess_act.triggered.connect(self._action_load_session)
        file_menu.addAction(load_sess_act)

        self._recent_menu = file_menu.addMenu("Recent Sessions")
        self._refresh_recent_menu()

        file_menu.addSeparator()

        save_cfg_act = QAction("Save Config…", self)
        save_cfg_act.triggered.connect(self._action_save_config)
        file_menu.addAction(save_cfg_act)

        load_cfg_act = QAction("Load Config…", self)
        load_cfg_act.triggered.connect(self._action_load_config)
        file_menu.addAction(load_cfg_act)

        file_menu.addSeparator()
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # ── View ──────────────────────────────────────────────────────────
        view_menu = mb.addMenu("View")
        self._dark_mode_action = QAction("Dark Mode", self)
        self._dark_mode_action.setCheckable(True)
        self._dark_mode_action.setChecked(True)   # default; overridden in __init__
        self._dark_mode_action.setShortcut("Ctrl+Shift+D")
        self._dark_mode_action.setToolTip(
            "Toggle between dark (Catppuccin Mocha) and light (system default) theme."
        )
        self._dark_mode_action.triggered.connect(self._action_toggle_theme)
        view_menu.addAction(self._dark_mode_action)

        # ── Help ──────────────────────────────────────────────────────────
        help_menu = mb.addMenu("Help")

        how_to_act = QAction("How to Use…", self)
        how_to_act.setShortcut("F1")
        how_to_act.setToolTip("Open the How to Use section of the documentation.")
        how_to_act.triggered.connect(self._action_show_how_to_use)
        help_menu.addAction(how_to_act)

        help_menu.addSeparator()

        about_act = QAction("About", self)
        about_act.triggered.connect(self._action_about)
        help_menu.addAction(about_act)

    def _refresh_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = SessionManager.list_recent_sessions(5)
        for entry in recent:
            act = QAction(entry.get("study_name", "Unknown"), self)
            path = entry.get("session_path", "")
            act.triggered.connect(
                lambda _checked, p=path: self._load_session_from_path(p)
            )
            self._recent_menu.addAction(act)
        if not recent:
            dummy = QAction("(no recent sessions)", self)
            dummy.setEnabled(False)
            self._recent_menu.addAction(dummy)

    # ══════════════════════════════════════════════════════════════════════
    # Toolbar
    # ══════════════════════════════════════════════════════════════════════

    def _build_toolbar(self) -> None:
        tb = QToolBar("CSV", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        load_csv_act = QAction("Load CSV…", self)
        load_csv_act.setShortcut("Ctrl+L")
        load_csv_act.triggered.connect(self._action_load_csv)
        tb.addAction(load_csv_act)

        reload_csv_act = QAction("Reload CSV", self)
        reload_csv_act.setShortcut("Ctrl+R")
        reload_csv_act.triggered.connect(self._action_reload_csv)
        tb.addAction(reload_csv_act)

        self._csv_path_label = QLabel("  No CSV loaded")
        self._csv_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(self._csv_path_label)

        tb.addSeparator()

        self._pending_results_btn = QPushButton("📋  Enter Pending Results…")
        self._pending_results_btn.setToolTip(
            "Reopen the results dialog for the current pending batch.\n"
            "Use this when you come back after running experiments —\n"
            "you can correct the actual compositions and enter your measurements."
        )
        self._pending_results_btn.setEnabled(False)
        self._pending_results_btn.clicked.connect(self._action_enter_pending_results)
        tb.addWidget(self._pending_results_btn)

    # ══════════════════════════════════════════════════════════════════════
    # Left dock — Study settings
    # ══════════════════════════════════════════════════════════════════════

    def _build_left_dock(self) -> None:
        self._left_dock = QDockWidget("BHOP Controls", self)
        self._left_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self._left_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        # Outer container holds a QTabWidget: ⚙ Settings | 📊 Results
        outer = QWidget()
        outer.setMinimumWidth(310)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self._left_tabs = QTabWidget()
        outer_layout.addWidget(self._left_tabs)

        # ══════════════════════════════════════════════════════════════
        # Tab 1 — ⚙ Settings
        # ══════════════════════════════════════════════════════════════
        settings_widget = QWidget()
        vbox = QVBoxLayout(settings_widget)
        vbox.setSpacing(8)
        vbox.setContentsMargins(6, 6, 6, 6)

        # ── Objectives ────────────────────────────────────────────────
        self._obj_group = QGroupBox("Objectives")
        self._obj_inner_layout = QVBoxLayout(self._obj_group)
        self._obj_inner_layout.addWidget(
            QLabel("Load a CSV to configure objectives.")
        )
        vbox.addWidget(self._obj_group)

        # ── Sampler ───────────────────────────────────────────────────
        sampler_group = QGroupBox("Sampler")
        sf = QFormLayout(sampler_group)
        self._sampler_combo = QComboBox()
        self._sampler_combo.addItems(["TPE", "NSGAII", "Random"])
        self._sampler_combo.setToolTip(
            "TPE: good for single-objective, ≤8 parameters.\n"
            "NSGAII: recommended for multi-objective.\n"
            "Random: baseline / exploration only."
        )
        self._sampler_combo.currentIndexChanged.connect(self._update_warnings)
        sf.addRow("Sampler:", self._sampler_combo)
        vbox.addWidget(sampler_group)

        # ── Batch settings ────────────────────────────────────────────
        batch_group = QGroupBox("Batch Settings")
        bf = QFormLayout(batch_group)

        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(1, 500)
        self._batch_size_spin.setValue(1)
        self._batch_size_spin.setToolTip(
            "How many experiments to suggest per batch (run in parallel in the lab)."
        )
        self._batch_size_spin.valueChanged.connect(self._update_warnings)
        bf.addRow("Batch size:", self._batch_size_spin)

        self._n_batches_spin = QSpinBox()
        self._n_batches_spin.setRange(1, 100_000)
        self._n_batches_spin.setValue(10)
        bf.addRow("Total batches:", self._n_batches_spin)

        self._batches_done_label = QLabel("0 / 10")
        bf.addRow("Batches done:", self._batches_done_label)
        vbox.addWidget(batch_group)

        # ── Warning label ──────────────────────────────────────────────
        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.hide()
        vbox.addWidget(self._warning_label)

        # ── Action buttons ─────────────────────────────────────────────
        self._ask_btn = QPushButton("Ask Next Batch")
        self._ask_btn.setEnabled(False)
        bold_big = QFont()
        bold_big.setBold(True)
        bold_big.setPointSize(11)
        self._ask_btn.setFont(bold_big)
        self._ask_btn.setMinimumHeight(42)
        self._ask_btn.setToolTip("Suggest the next batch of parameters from Optuna.")
        self._ask_btn.clicked.connect(self._action_ask_next_batch)
        vbox.addWidget(self._ask_btn)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setEnabled(False)
        self._pause_btn.setCheckable(True)
        self._pause_btn.clicked.connect(self._action_toggle_pause)
        vbox.addWidget(self._pause_btn)

        vbox.addStretch()
        self._left_tabs.addTab(settings_widget, "⚙  Settings")

        # ══════════════════════════════════════════════════════════════
        # Tab 2 — 📊 Results
        # ══════════════════════════════════════════════════════════════
        results_widget = QWidget()
        rvbox = QVBoxLayout(results_widget)
        rvbox.setContentsMargins(4, 4, 4, 4)
        rvbox.setSpacing(4)

        rbtn_row = QHBoxLayout()
        design_space_btn = QPushButton("📊  Design Space…")
        design_space_btn.setToolTip(
            "Open the Design Space + Correlation Matrix window.\n"
            "Shows historical data in parameter space and how variables\n"
            "correlate with each other and the objective."
        )
        design_space_btn.clicked.connect(self._action_open_design_space)
        rbtn_row.addWidget(design_space_btn)
        rbtn_row.addStretch()
        export_btn = QPushButton("Export CSV")
        export_btn.setToolTip("Export all completed trial results to a CSV file.")
        export_btn.clicked.connect(self._action_export_results)
        rbtn_row.addWidget(export_btn)
        rvbox.addLayout(rbtn_row)

        self._results_tabs = QTabWidget()

        self._trials_table = QTableWidget()
        self._trials_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._trials_table.horizontalHeader().setStretchLastSection(True)
        self._trials_table.setAlternatingRowColors(True)
        self._results_tabs.addTab(self._trials_table, "All Trials")

        self._pareto_table = QTableWidget()
        self._pareto_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._pareto_table.horizontalHeader().setStretchLastSection(True)
        self._pareto_table.setAlternatingRowColors(True)
        self._results_tabs.addTab(self._pareto_table, "Best / Pareto")

        rvbox.addWidget(self._results_tabs, stretch=1)
        self._left_tabs.addTab(results_widget, "📊  Results")

        self._left_dock.setWidget(outer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._left_dock)
        # Give the dock a generous initial width so the results table is readable
        self.resizeDocks([self._left_dock], [480], Qt.Horizontal)

    # ══════════════════════════════════════════════════════════════════════
    # Centre — resume banner + param card scroll area
    # ══════════════════════════════════════════════════════════════════════

    def _build_centre(self) -> None:
        centre = QWidget()
        vbox = QVBoxLayout(centre)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(6)

        # Resume banner (hidden by default; style applied by _apply_theme())
        self._resume_banner = QFrame()
        self._resume_banner.setFrameShape(QFrame.StyledPanel)
        banner_h = QHBoxLayout(self._resume_banner)
        self._resume_label = QLabel()
        self._resume_label.setWordWrap(True)
        banner_h.addWidget(self._resume_label, stretch=1)
        load_updated_btn = QPushButton("Load Updated CSV…")
        load_updated_btn.clicked.connect(self._action_load_updated_csv)
        banner_h.addWidget(load_updated_btn)
        self._resume_banner.hide()
        vbox.addWidget(self._resume_banner)

        # Param card scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setAlignment(Qt.AlignTop)
        self._cards_layout.setSpacing(6)

        self._placeholder_lbl = QLabel(
            "No CSV loaded.\n\nUse  File → New Session…  or  Load CSV…  to begin."
        )
        self._placeholder_lbl.setAlignment(Qt.AlignCenter)
        italic = QFont()
        italic.setItalic(True)
        self._placeholder_lbl.setFont(italic)
        self._cards_layout.addWidget(self._placeholder_lbl)

        self._scroll.setWidget(self._cards_container)
        vbox.addWidget(self._scroll)

        self.setCentralWidget(centre)

    # ══════════════════════════════════════════════════════════════════════
    # Status bar
    # ══════════════════════════════════════════════════════════════════════

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        self._status_trials = QLabel("Trials: 0")
        self._status_best   = QLabel("Best: —")
        self._status_text   = QLabel("Status: Ready")
        for lbl in (self._status_trials, self._status_best, self._status_text):
            sb.addWidget(lbl)
            sb.addWidget(self._make_separator())

    @staticmethod
    def _make_separator() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        return sep

    # ══════════════════════════════════════════════════════════════════════
    # CSV loading
    # ══════════════════════════════════════════════════════════════════════

    def _action_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Experiment CSV", "", "CSV Files (*.csv)"
        )
        if path:
            self._do_load_csv(path)

    def _action_reload_csv(self) -> None:
        if self._csv_path:
            self._do_load_csv(self._csv_path)
        elif self._session_state and self._session_state.csv_path:
            self._do_load_csv(self._session_state.csv_path)

    def _do_load_csv(self, path: str) -> None:
        try:
            df = load_csv(path)
        except ValueError as exc:
            QMessageBox.critical(self, "CSV Error", str(exc))
            return
        self._df = df
        self._csv_path = path
        if self._session_state:
            self._session_state.csv_path = path
        self._csv_path_label.setText(f"  {os.path.basename(path)}")
        self._populate_objectives_selector(df)
        self._set_status("CSV loaded")

    def _populate_objectives_selector(self, df: pd.DataFrame) -> None:
        # Clear existing objective widgets
        while self._obj_inner_layout.count():
            item = self._obj_inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._obj_col_rows = []

        self._obj_inner_layout.addWidget(QLabel("Select result column(s):"))

        for col in df.columns:
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)

            cb = QCheckBox(col)
            dir_combo = QComboBox()
            dir_combo.addItems(["minimize", "maximize"])
            dir_combo.setFixedWidth(96)
            dir_combo.hide()
            cb.toggled.connect(lambda checked, c=dir_combo: c.setVisible(checked))

            row_l.addWidget(cb)
            row_l.addWidget(dir_combo)
            self._obj_inner_layout.addWidget(row_w)
            self._obj_col_rows.append((col, cb, dir_combo))

        apply_btn = QPushButton("Apply Objectives →")
        apply_btn.clicked.connect(self._action_apply_objectives)
        self._obj_inner_layout.addWidget(apply_btn)

        constraint_btn = QPushButton("📐  Constraints…")
        constraint_btn.setToolTip(
            "Define rules that constrain how parameter values relate to each other.\n"
            "Examples:\n"
            "  • CsPbI + FAPbI + MAPbI = 1  (compositional fractions)\n"
            "  • temp * time ≤ 50000  (processing budget)"
        )
        constraint_btn.clicked.connect(self._action_open_constraints)
        self._obj_inner_layout.addWidget(constraint_btn)

    # ══════════════════════════════════════════════════════════════════════
    # Apply objectives → build param cards + study
    # ══════════════════════════════════════════════════════════════════════

    def _action_apply_objectives(self) -> None:
        if self._df is None:
            QMessageBox.warning(self, "No CSV", "Load a CSV file first.")
            return

        selected = [
            (col, combo.currentText())
            for col, cb, combo in self._obj_col_rows
            if cb.isChecked()
        ]
        if not selected:
            QMessageBox.warning(
                self, "No Objectives", "Select at least one result column."
            )
            return

        objectives = [ObjectiveConfig(col, direction) for col, direction in selected]
        result_cols = [o.column_name for o in objectives]
        params = extract_param_defaults(self._df, result_cols)

        config = StudyConfig(
            parameters=params,
            objectives=objectives,
            constraints=list(self._constraints),
            batch_size=self._batch_size_spin.value(),
            n_batches=self._n_batches_spin.value(),
            sampler_name=self._sampler_combo.currentText(),
        )

        # Create a new session if one doesn't exist yet
        if self._session_state is None:
            session_dir = (
                os.path.dirname(self._csv_path)
                if self._csv_path
                else os.path.join(os.path.expanduser("~"), "bhop_sessions")
            )
            self._session_state = SessionManager.create_new_session(
                config, self._csv_path or "", session_dir
            )

        self._session_state.study_config = config
        # Persist the updated config (including constraints) immediately so it
        # survives a restart even if the user never runs a batch.
        SessionManager.save(self._session_state)

        # Build / reload the Optuna study
        self._study = build_study(
            config,
            self._session_state.storage_path,
            self._session_state.study_name,
        )

        # Seed historical trials from CSV
        trial_dicts = load_trials_from_csv(self._df, config)
        added, skipped = load_historical_trials(self._study, trial_dicts, config)

        # Build parameter cards
        self._build_param_cards(params)
        self._ask_btn.setEnabled(True)
        self._update_warnings()
        self._refresh_results_tables()
        self._update_status_bar()
        self._refresh_recent_menu()
        self._refresh_design_space()   # show historical data, no suggestions yet
        self._set_status(f"Ready — {added} historical trials loaded, {skipped} skipped.")

    # ══════════════════════════════════════════════════════════════════════
    # Constraint editor
    # ══════════════════════════════════════════════════════════════════════

    def _action_open_constraints(self) -> None:
        """Open the ConstraintDialog so the user can add / edit / remove rules."""
        # Build the list of available parameter names from the current CSV
        # (all non-objective columns).
        param_names: List[str] = []
        if self._df is not None and self._session_state is not None:
            result_cols = {o.column_name for o in self._session_state.study_config.objectives}
            param_names = [c for c in self._df.columns if c not in result_cols]
        elif self._df is not None:
            param_names = list(self._df.columns)

        dlg = ConstraintDialog(
            constraints=list(self._constraints),
            param_names=param_names,
            parent=self,
        )
        if dlg.exec() == QDialog.Accepted:
            self._constraints = dlg.constraints
            # Persist immediately so constraints survive a restart without needing
            # the user to click "Apply Objectives" again.
            if self._session_state:
                self._session_state.study_config.constraints = list(self._constraints)
                SessionManager.save(self._session_state)
            n = len(self._constraints)
            self._set_status(
                f"{n} constraint(s) defined. "
                "Click 'Apply Objectives' to rebuild the study with these constraints."
            )

    # ══════════════════════════════════════════════════════════════════════
    # Parameter cards
    # ══════════════════════════════════════════════════════════════════════

    def _build_param_cards(self, params: List[ParameterConfig]) -> None:
        self._placeholder_lbl.hide()

        for card in self._param_cards:
            card.deleteLater()
        self._param_cards.clear()

        for param in params:
            card = ParamCardWidget(param, self._cards_container)
            card.config_changed.connect(self._update_warnings)
            self._cards_layout.addWidget(card)
            self._param_cards.append(card)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 10 — Warning system
    # ══════════════════════════════════════════════════════════════════════

    def _update_warnings(self) -> None:
        n_enabled = sum(1 for c in self._param_cards if c.get_config().enabled)
        n_trials   = len(self._study.trials) if self._study else 0
        sampler    = self._sampler_combo.currentText()
        batch_size = self._batch_size_spin.value()

        warnings: List[str] = []

        if sampler == "TPE" and n_enabled > TPE_PARAM_WARNING_THRESHOLD:
            warnings.append(
                f"⚠ TPE may struggle with {n_enabled} parameters "
                f"(recommended ≤ {TPE_PARAM_WARNING_THRESHOLD}). "
                "Consider switching to NSGAII or reducing parameters."
            )
        if n_enabled > 0 and n_trials < MIN_TRIALS_PER_PARAM_RATIO * n_enabled:
            warnings.append(
                f"⚠ Only {n_trials} historical trials for {n_enabled} parameters. "
                "Initial suggestions may be poor — add more data first."
            )
        if sampler == "TPE" and n_trials > 0 and batch_size > n_trials:
            warnings.append(
                f"⚠ Batch size ({batch_size}) exceeds historical trial count ({n_trials}). "
                "TPE batch quality degrades without sufficient prior data."
            )

        if warnings:
            self._warning_label.setText("\n\n".join(warnings))
            self._warning_label.show()
        else:
            self._warning_label.hide()

        self._batches_done_label.setText(
            f"{self._batches_done} / {self._n_batches_spin.value()}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Ask next batch
    # ══════════════════════════════════════════════════════════════════════

    def _action_ask_next_batch(self) -> None:
        if self._study is None or self._session_state is None:
            QMessageBox.warning(
                self, "No Session", "Load a CSV and apply objectives first."
            )
            return

        # Warn confirmation
        if self._warning_label.isVisible():
            reply = QMessageBox.question(
                self,
                "Warnings Active",
                f"There are active warnings:\n\n{self._warning_label.text()}\n\n"
                "Proceed anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._sync_config_from_ui()

        config = self._session_state.study_config
        self._worker = OptimizationWorker(
            self._study, config, SessionManager, self._session_state, parent=self
        )
        self._worker.batch_ready.connect(self._on_batch_ready)
        self._worker.batch_complete.connect(self._on_batch_complete)
        self._worker.optimization_done.connect(self._on_optimization_done)
        self._worker.error.connect(self._on_worker_error)

        self._ask_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._set_status("Asking batch…")
        self._worker.start()

    def _sync_config_from_ui(self) -> None:
        if not self._session_state:
            return
        cfg = self._session_state.study_config
        cfg.batch_size   = self._batch_size_spin.value()
        cfg.n_batches    = self._n_batches_spin.value()
        cfg.sampler_name = self._sampler_combo.currentText()
        cfg.parameters   = [c.get_config() for c in self._param_cards]

    # ── Worker callbacks ───────────────────────────────────────────────────

    def _on_batch_ready(self, param_dicts: list) -> None:
        if not self._session_state or not self._session_state.pending_batch:
            return

        # Overlay the suggested points on the design space BEFORE opening the dialog
        # so the user can switch to the Design Space tab to review placement
        suggestions = [item["params"] for item in self._session_state.pending_batch]
        self._refresh_design_space(suggestions=suggestions)

        dlg = BatchResultsDialog(
            self._session_state.pending_batch,
            self._session_state.study_config.objectives,
            self._session_state,
            parent=self,
        )
        dlg.results_submitted.connect(self._on_results_submitted)
        if dlg.exec() != dlg.accepted:
            # User cancelled — stop the worker, session is already saved
            if self._worker:
                self._worker.cancel_results()
            self._ask_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._set_status("Awaiting results — session saved. Reload to resume.")

    def _on_results_submitted(self, results: list) -> None:
        # Append actual compositions + results to the CSV before telling Optuna
        self._append_results_to_csv(results)
        if self._worker:
            self._worker.submit_results(results)
        self._update_pending_btn_state()
        self._set_status("Submitting results…")

    def _on_batch_complete(self, done: int, total: int) -> None:
        self._batches_done = done
        self._batches_done_label.setText(f"{done} / {total}")
        self._refresh_results_tables()
        self._update_status_bar()
        # Auto-switch to Results tab so the user sees the new data immediately
        self._left_tabs.setCurrentIndex(1)
        self._set_status(f"Batch {done}/{total} complete.")

    def _on_optimization_done(self) -> None:
        self._ask_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._refresh_results_tables()
        self._set_status("Optimization complete.")
        QMessageBox.information(self, "Done", "All batches complete!")

    def _on_worker_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Worker Error", msg)
        self._ask_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._set_status("Error — see dialog.")

    def _action_toggle_pause(self, checked: bool) -> None:
        if self._worker:
            if checked:
                self._worker.pause()
                self._pause_btn.setText("Resume")
                self._set_status("Paused.")
            else:
                self._worker.resume()
                self._pause_btn.setText("Pause")
                self._set_status("Running…")

    # ══════════════════════════════════════════════════════════════════════
    # Resume path (pending batch on session load)
    # ══════════════════════════════════════════════════════════════════════

    def _action_load_updated_csv(self) -> None:
        """Resume path: user loads CSV with completed results."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Updated CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            df = load_csv(path)
        except ValueError as exc:
            QMessageBox.critical(self, "CSV Error", str(exc))
            return
        self._df = df
        self._csv_path = path

        if not self._session_state or not self._session_state.pending_batch:
            return

        matches = SessionManager.match_pending_to_csv(self._session_state, df)
        dlg = BatchResultsDialog(
            self._session_state.pending_batch,
            self._session_state.study_config.objectives,
            self._session_state,
            parent=self,
        )
        if matches:
            dlg.prefill_results(matches)
        dlg.results_submitted.connect(self._on_resume_results_submitted)
        dlg.exec()

    def _on_resume_results_submitted(self, results: list) -> None:
        if not self._session_state or not self._study:
            return
        trial_numbers = [r["trial_number"] for r in results]
        values_list   = [r["values"]        for r in results]
        self._append_results_to_csv(results)
        # Pass constrained params so the surrogate trains on feasible values.
        pending_by_num = {
            p["trial_number"]: p["params"]
            for p in (self._session_state.pending_batch or [])
        }
        constrained = [pending_by_num.get(n, {}) for n in trial_numbers]
        tell_batch(self._study, trial_numbers, values_list,
                   constrained, self._session_state.study_config)
        SessionManager.clear_pending_batch(self._session_state)
        self._resume_banner.hide()
        self._refresh_results_tables()
        self._update_status_bar()
        self._update_pending_btn_state()
        self._ask_btn.setEnabled(True)
        self._set_status("Results submitted. Ready for next batch.")
        QMessageBox.information(
            self, "Resumed", "Results submitted. Ready to ask the next batch."
        )

    # ══════════════════════════════════════════════════════════════════════
    # Session management
    # ══════════════════════════════════════════════════════════════════════

    def _action_new_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV for New Session", "", "CSV Files (*.csv)"
        )
        if path:
            self._session_state = None  # will be created on Apply Objectives
            self._do_load_csv(path)
            self._set_status("New session: configure objectives, then click Apply Objectives.")

    def _action_load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "Session Files (*_session.json)"
        )
        if path:
            self._load_session_from_path(path)

    def _load_session_from_path(self, path: str) -> None:
        try:
            state = SessionManager.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._session_state = state
        cfg = state.study_config

        try:
            self._study = build_study(cfg, state.storage_path, state.study_name)
        except Exception as exc:
            QMessageBox.critical(self, "Study Load Error", str(exc))
            return

        # Restore UI from config (including constraints)
        self._constraints = list(cfg.constraints)
        self._sampler_combo.setCurrentText(cfg.sampler_name)
        self._batch_size_spin.setValue(cfg.batch_size)
        self._n_batches_spin.setValue(cfg.n_batches)
        self._build_param_cards(cfg.parameters)
        self._refresh_results_tables()
        self._update_warnings()
        self._update_status_bar()
        self._ask_btn.setEnabled(True)
        self._refresh_recent_menu()

        # Load CSV if path is still valid
        if state.csv_path and os.path.exists(state.csv_path):
            try:
                df = load_csv(state.csv_path)
                self._df = df
                self._csv_path = state.csv_path
                self._csv_path_label.setText(f"  {os.path.basename(state.csv_path)}")
                self._populate_objectives_selector(df)
            except Exception:
                pass

        # Check for pending batch — update toolbar button and optional banner
        self._update_pending_btn_state()
        if state.pending_batch:
            n = len(state.pending_batch)
            self._resume_label.setText(
                f"⚠  Session has <b>{n}</b> pending trial(s) awaiting lab results.\n"
                "Click <b>📋 Enter Pending Results…</b> in the toolbar to enter your "
                "measurements, or use 'Load Updated CSV…' to auto-fill from a file."
            )
            self._resume_banner.show()
        else:
            self._resume_banner.hide()

        self._set_status(f"Session '{state.study_name}' loaded.")

    # ══════════════════════════════════════════════════════════════════════
    # Config save / load
    # ══════════════════════════════════════════════════════════════════════

    def _action_save_config(self) -> None:
        if not self._session_state:
            QMessageBox.warning(self, "No Config", "No active session to save from.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "", "JSON Files (*.json)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._session_state.study_config.to_dict(), fh, indent=2)

    def _action_load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cfg = StudyConfig.from_dict(json.load(fh))
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", str(exc))
            return

        if self._session_state:
            self._session_state.study_config = cfg
        # Restore constraints to UI state so they are included in the next Apply
        self._constraints = list(cfg.constraints)
        self._sampler_combo.setCurrentText(cfg.sampler_name)
        self._batch_size_spin.setValue(cfg.batch_size)
        self._n_batches_spin.setValue(cfg.n_batches)
        self._build_param_cards(cfg.parameters)
        self._update_warnings()

    # ══════════════════════════════════════════════════════════════════════
    # Results tables
    # ══════════════════════════════════════════════════════════════════════

    def _refresh_results_tables(self) -> None:
        if not self._study:
            return
        from optuna.trial import TrialState

        completed = [t for t in self._study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            return

        obj_cols = (
            [o.column_name for o in self._session_state.study_config.objectives]
            if self._session_state
            else ["value"]
        )
        param_cols = list(completed[0].params.keys())
        all_cols   = ["Trial #"] + param_cols + obj_cols

        def _fill_table(table: QTableWidget, trials: list) -> None:
            table.setRowCount(len(trials))
            table.setColumnCount(len(all_cols))
            table.setHorizontalHeaderLabels(all_cols)
            for r, trial in enumerate(sorted(trials, key=lambda t: t.number)):
                table.setItem(r, 0, QTableWidgetItem(str(trial.number)))
                for ci, pname in enumerate(param_cols, start=1):
                    table.setItem(r, ci, QTableWidgetItem(str(trial.params.get(pname, ""))))
                for oi in range(len(obj_cols)):
                    if trial.values and oi < len(trial.values):
                        val = trial.values[oi]
                    elif trial.value is not None and oi == 0:
                        val = trial.value
                    else:
                        val = ""
                    table.setItem(r, len(param_cols) + 1 + oi, QTableWidgetItem(str(val)))
            table.resizeColumnsToContents()

        _fill_table(self._trials_table, completed)
        _fill_table(self._pareto_table, get_pareto_front(self._study))

    def _update_status_bar(self) -> None:
        if not self._study:
            return
        from optuna.trial import TrialState

        done = [t for t in self._study.trials if t.state == TrialState.COMPLETE]
        self._status_trials.setText(f"Trials: {len(done)}")
        try:
            pareto = get_pareto_front(self._study)
            if pareto:
                t = pareto[0]
                if t.value is not None:
                    self._status_best.setText(f"Best: {t.value:.4g}")
                elif t.values:
                    vals = ", ".join(f"{v:.4g}" for v in t.values)
                    self._status_best.setText(f"Best: [{vals}]")
        except Exception:
            pass

    def _action_export_results(self) -> None:
        if not self._study:
            QMessageBox.warning(self, "No Study", "No study loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        from optuna.trial import TrialState

        rows = []
        obj_cols = (
            [o.column_name for o in self._session_state.study_config.objectives]
            if self._session_state
            else []
        )
        for t in self._study.trials:
            if t.state != TrialState.COMPLETE:
                continue
            row: dict = {"trial_number": t.number}
            row.update(t.params)
            if t.values:
                for i, v in enumerate(t.values):
                    col = obj_cols[i] if i < len(obj_cols) else f"objective_{i}"
                    row[col] = v
            elif t.value is not None:
                col = obj_cols[0] if obj_cols else "value"
                row[col] = t.value
            rows.append(row)

        pd.DataFrame(rows).to_csv(path, index=False)
        QMessageBox.information(self, "Exported", f"Results exported to:\n{path}")

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _set_status(self, msg: str) -> None:
        self._status_text.setText(f"Status: {msg}")

    def _update_pending_btn_state(self) -> None:
        """Enable the 'Enter Pending Results…' button iff there is a pending batch."""
        has_pending = bool(
            self._session_state and self._session_state.pending_batch
        )
        self._pending_results_btn.setEnabled(has_pending)

    def _action_enter_pending_results(self) -> None:
        """
        Open the BatchResultsDialog for the current pending batch.

        This is the primary way to re-enter results after closing the dialog
        (e.g. because experiments take 1–2 days).  The user can correct the
        actual composition values before submitting.
        """
        if not self._session_state or not self._session_state.pending_batch:
            QMessageBox.information(
                self,
                "No Pending Batch",
                "There is no pending batch to enter results for.\n"
                "Use 'Ask Next Batch' to generate a new suggestion.",
            )
            return
        if not self._study:
            QMessageBox.warning(
                self, "No Study",
                "Load a session first (File → Load Session…).",
            )
            return

        dlg = BatchResultsDialog(
            self._session_state.pending_batch,
            self._session_state.study_config.objectives,
            self._session_state,
            parent=self,
        )
        dlg.results_submitted.connect(self._on_pending_results_submitted)
        dlg.exec()

    def _on_pending_results_submitted(self, results: list) -> None:
        """
        Handle results entered via the toolbar '📋 Enter Pending Results…' button.

        Tells Optuna, appends to CSV, clears the pending batch, and re-enables
        the 'Ask Next Batch' button.
        """
        if not self._session_state or not self._study:
            return
        trial_numbers = [r["trial_number"] for r in results]
        values_list   = [r["values"]        for r in results]
        # Append actual compositions + results to the experiment CSV
        self._append_results_to_csv(results)
        # Pass constrained params so the surrogate trains on feasible values.
        pending_by_num = {
            p["trial_number"]: p["params"]
            for p in (self._session_state.pending_batch or [])
        }
        constrained = [pending_by_num.get(n, {}) for n in trial_numbers]
        tell_batch(self._study, trial_numbers, values_list,
                   constrained, self._session_state.study_config)
        SessionManager.clear_pending_batch(self._session_state)
        self._resume_banner.hide()
        self._refresh_results_tables()
        self._update_status_bar()
        self._update_pending_btn_state()
        self._ask_btn.setEnabled(True)
        self._set_status("Results submitted. Ready for next batch.")
        QMessageBox.information(
            self, "Results Submitted",
            "Results recorded successfully. Ready to ask the next batch."
        )

    def _append_results_to_csv(self, results: list) -> None:
        """
        Append actual compositions + measured objective values to the experiment CSV.

        Parameters
        ----------
        results : list of dicts, each containing
            ``actual_params`` (edited compositions),
            ``nominal_params`` (original suggestion),
            ``values``         (measured objective values).

        The CSV is updated in-place.  Failures are non-fatal — a warning is
        shown but the submission continues.
        """
        if not self._csv_path or not os.path.exists(self._csv_path):
            return   # no CSV to append to
        if not self._session_state:
            return
        cfg = self._session_state.study_config
        obj_cols = [o.column_name for o in cfg.objectives]

        try:
            rows = []
            for r in results:
                # Prefer actual (lab-measured) compositions; fall back to nominal
                params = r.get("actual_params") or r.get("nominal_params") or {}
                row: dict = dict(params)
                for i, v in enumerate(r.get("values", [])):
                    col = obj_cols[i] if i < len(obj_cols) else f"objective_{i}"
                    row[col] = v
                rows.append(row)

            if not rows:
                return

            df_existing = pd.read_csv(self._csv_path)
            df_new      = pd.DataFrame(rows)
            # Align columns — add missing columns as NaN
            for c in df_existing.columns:
                if c not in df_new.columns:
                    df_new[c] = float("nan")
            df_new = df_new[df_existing.columns]   # same column order
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(self._csv_path, index=False)
            # Refresh our in-memory copy
            self._df = df_combined
        except Exception as exc:
            QMessageBox.warning(
                self,
                "CSV Update Warning",
                f"Results were submitted to Optuna, but could not be appended "
                f"to the CSV:\n{exc}\n\nYou can add them manually if needed.",
            )

    def _refresh_design_space(
        self, suggestions: Optional[List[dict]] = None
    ) -> None:
        """
        Push fresh data to the Design Space dialog (if it is open).

        Parameters
        ----------
        suggestions : optional list of param-value dicts for the latest
                      suggested batch; if None the plot shows historical
                      data only (suggestions are cleared).
        """
        if self._df is None or not self._session_state:
            return
        if self._design_space_dlg is None:
            return  # dialog not opened yet — nothing to update
        cfg = self._session_state.study_config
        self._design_space_dlg.refresh(
            df=self._df,
            params=cfg.parameters,
            objectives=cfg.objectives,
            suggestions=suggestions,
        )

    def _action_open_design_space(self) -> None:
        """
        Open (or raise) the Design Space visualisation window.

        Creates the dialog the first time; thereafter reuses the same
        window so the user can keep it open alongside BHOP.
        """
        if self._df is None or not self._session_state:
            QMessageBox.information(
                self,
                "No Data",
                "Load a CSV and apply objectives first, then open the Design Space.",
            )
            return

        # Create once, reuse thereafter
        if self._design_space_dlg is None:
            self._design_space_dlg = DesignSpaceDialog(parent=self)

        cfg = self._session_state.study_config
        self._design_space_dlg.refresh(
            df=self._df,
            params=cfg.parameters,
            objectives=cfg.objectives,
            suggestions=None,   # suggestions already set by _refresh_design_space if pending
        )
        self._design_space_dlg.show()
        self._design_space_dlg.raise_()
        self._design_space_dlg.activateWindow()

    def _action_about(self) -> None:
        QMessageBox.information(
            self,
            "About BHOP",
            "BHOP — Bayesian Hyperparameter Optimization for the Lab\n\n"
            "Built with Optuna + PySide6\n\n"
            "Supports:\n"
            "  • CSV-seeded historical trials\n"
            "  • Dead regions (excluded parameter zones)\n"
            "  • Multi-objective optimization (Pareto front)\n"
            "  • Batched experiment suggestions\n"
            "  • Full session persistence (close & resume)\n"
            "  • TPE / NSGAII / Random samplers",
        )

    # ══════════════════════════════════════════════════════════════════════
    # Theme management
    # ══════════════════════════════════════════════════════════════════════

    def _apply_theme(self, dark: bool) -> None:
        """Apply dark or light theme to the whole application and update themed widgets."""
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(DARK_STYLESHEET if dark else "")

        # Warning label (theme-specific amber style)
        self._warning_label.setStyleSheet(
            _WARN_STYLE_DARK if dark else _WARN_STYLE_LIGHT
        )
        # Resume banner
        self._resume_banner.setStyleSheet(
            _BANNER_STYLE_DARK if dark else _BANNER_STYLE_LIGHT
        )

        # Persist choice
        QSettings().setValue("dark_mode", dark)

    def _action_toggle_theme(self) -> None:
        """Called when the user clicks View → Dark Mode (checkable action)."""
        self._apply_theme(self._dark_mode_action.isChecked())

    # ══════════════════════════════════════════════════════════════════════
    # In-app documentation viewer
    # ══════════════════════════════════════════════════════════════════════

    def _action_show_how_to_use(self) -> None:
        """Help → How to Use… — shows §3 of DOCUMENTATION.md in a QTextBrowser."""
        doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DOCUMENTATION.md")
        if not os.path.exists(doc_path):
            QMessageBox.warning(
                self, "Documentation Not Found",
                f"Could not find DOCUMENTATION.md at:\n{doc_path}"
            )
            return

        with open(doc_path, "r", encoding="utf-8") as fh:
            content = fh.read()

        # Extract §3 (How to Use) through end of §3 (before §4)
        start = content.find("## 3. How to Use")
        end   = content.find("\n## 4.", start) if start != -1 else -1
        if start == -1:
            section_text = content          # fallback: show everything
        elif end == -1:
            section_text = content[start:]
        else:
            section_text = content[start:end]

        html = _markdown_to_html(section_text)

        dlg = QDialog(self)
        dlg.setWindowTitle("How to Use BHOP")
        dlg.resize(820, 640)

        vbox = QVBoxLayout(dlg)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(html)
        vbox.addWidget(browser, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        vbox.addLayout(btn_row)

        dlg.exec()

    # ══════════════════════════════════════════════════════════════════════
    # Close event — save session before exit
    # ══════════════════════════════════════════════════════════════════════

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._session_state:
            if self._session_state.pending_batch:
                n = len(self._session_state.pending_batch)
                reply = QMessageBox.question(
                    self,
                    "Pending Trials",
                    f"You have {n} suggested trial(s) awaiting lab results.\n\n"
                    "The session will be saved automatically.\n"
                    "When your experiments are done, reopen the app, reload\n"
                    "this session, then click  📋 Enter Pending Results…  in\n"
                    "the toolbar to submit your measurements.\n\n"
                    "Close anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    event.ignore()
                    return
            # Sync UI state into config then save
            self._sync_config_from_ui()
            SessionManager.save(self._session_state)

        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)

        event.accept()
