"""
Design Space Visualisation Widget
==================================
Embeds a Matplotlib figure that shows where the historical data sits in
parameter space and how the most-recently suggested batch relates to it.

Plot type is chosen automatically from the number of enabled numeric params:

  ≤ 5  params → pairwise scatter grid (lower-triangle pairplot)
  6–12 params → parallel coordinates
  > 12 params → 1-D marginal strip (histogram + suggestion markers)

All plots use the Catppuccin Mocha dark palette to match the rest of the app.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from PySide6.QtCore import Qt
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizeGrip,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from parameter_config import ObjectiveConfig, ParameterConfig, ParameterType

# ── Catppuccin Mocha palette ──────────────────────────────────────────────────
_BG      = "#1e1e2e"   # base background
_AX_BG   = "#181825"   # axis background (slightly darker)
_GRID    = "#313244"   # grid / spine colour
_FG      = "#cdd6f4"   # foreground text
_HIST    = "#89b4fa"   # historical data points (blue)
_PARETO  = "#a6e3a1"   # pareto-optimal points (green)
_SUGGEST = "#f38ba8"   # suggested batch points (red/pink)
_SUGGEST2 = "#fab387"  # second suggestion colour if needed (peach)


class DesignSpaceWidget(QWidget):
    """
    Embedded Matplotlib canvas that visualises the parameter design space.

    Public API
    ----------
    refresh(df, params, objectives, suggestions=None)
        Rebuild the plot with new data / suggestions.
    clear()
        Hide the canvas and show the placeholder text.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._df: Optional[pd.DataFrame] = None
        self._params: List[ParameterConfig] = []
        self._objectives: List[ObjectiveConfig] = []
        # suggestions: list of param-value dicts [{"Temperature": 200, ...}, ...]
        self._suggestions: Optional[List[dict]] = None

        # ── Pairplot page state ────────────────────────────────────────────
        self._page: int = 0        # current page index (0-based)
        _PAGE_SIZE = 5             # max params per pairplot page (class-level below)

        self._fig = Figure(facecolor=_BG)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Colour by:"))
        self._obj_combo = QComboBox()
        self._obj_combo.setFixedWidth(180)
        self._obj_combo.currentIndexChanged.connect(self._redraw)
        toolbar.addWidget(self._obj_combo)

        toolbar.addWidget(QLabel("  View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["Auto", "Pairplot", "Parallel Coords", "Marginals"])
        self._view_combo.setFixedWidth(130)
        self._view_combo.setToolTip(
            "Auto: choose plot type based on parameter count.\n"
            "Pairplot: scatter grid (paged, 5 params per page).\n"
            "Parallel Coords: all params on parallel axes.\n"
            "Marginals: stacked 1-D histograms."
        )
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        toolbar.addWidget(self._view_combo)

        toolbar.addStretch()

        # Page navigation — hidden when paging is not applicable
        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(32)
        self._prev_btn.setToolTip("Previous page of parameters")
        self._prev_btn.clicked.connect(self._action_prev_page)
        toolbar.addWidget(self._prev_btn)

        self._page_label = QLabel("Page 1 / 1")
        self._page_label.setFixedWidth(90)
        self._page_label.setAlignment(Qt.AlignCenter)
        self._page_label.setStyleSheet(f"color: {_FG}; font-size: 11px;")
        toolbar.addWidget(self._page_label)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(32)
        self._next_btn.setToolTip("Next page of parameters")
        self._next_btn.clicked.connect(self._action_next_page)
        toolbar.addWidget(self._next_btn)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet(f"color: {_FG}; font-size: 11px;")
        toolbar.addWidget(self._info_label)

        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(36)
        refresh_btn.setToolTip("Redraw the design space plot.")
        refresh_btn.clicked.connect(self._redraw)
        toolbar.addWidget(refresh_btn)
        layout.addLayout(toolbar)

        # Start with page nav hidden
        self._set_page_nav_visible(False)

        # ── Canvas ────────────────────────────────────────────────────────
        layout.addWidget(self._canvas, stretch=1)

        # ── Placeholder ───────────────────────────────────────────────────
        self._placeholder = QLabel(
            "Load a CSV and apply objectives to see the design space."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_FG}; font-size: 13px; font-style: italic;"
        )
        layout.addWidget(self._placeholder)

        self._canvas.hide()

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def refresh(
        self,
        df: pd.DataFrame,
        params: List[ParameterConfig],
        objectives: List[ObjectiveConfig],
        suggestions: Optional[List[dict]] = None,
    ) -> None:
        """
        Update data and redraw.

        Parameters
        ----------
        df          : Full historical DataFrame (all rows, all columns).
        params      : Parameter configurations (all types; non-numeric are skipped).
        objectives  : Objective configurations used to colour points.
        suggestions : Optional list of param-value dicts for the latest suggested
                      batch.  Each dict maps param name → value.
        """
        self._df          = df
        self._params      = params
        self._objectives  = objectives
        self._suggestions = suggestions

        # Sync objective combobox
        prev = self._obj_combo.currentText()
        self._obj_combo.blockSignals(True)
        self._obj_combo.clear()
        for o in objectives:
            self._obj_combo.addItem(o.column_name)
        if prev in [o.column_name for o in objectives]:
            self._obj_combo.setCurrentText(prev)
        self._obj_combo.blockSignals(False)

        self._redraw()

    def clear(self) -> None:
        """Remove data and show the placeholder message."""
        self._df = None
        self._params = []
        self._objectives = []
        self._suggestions = None
        self._fig.clear()
        self._canvas.draw_idle()
        self._canvas.hide()
        self._placeholder.show()

    # ══════════════════════════════════════════════════════════════════════
    # Internal: choose & dispatch to the right plot type
    # ══════════════════════════════════════════════════════════════════════

    # ── Pairplot page navigation ───────────────────────────────────────────

    _PAGE_SIZE = 5   # max number of parameters shown per pairplot page

    def _set_page_nav_visible(self, visible: bool) -> None:
        for w in (self._prev_btn, self._page_label, self._next_btn):
            w.setVisible(visible)

    def _update_page_nav(self, n_total: int) -> None:
        n_pages = max(1, math.ceil(n_total / self._PAGE_SIZE))
        self._page = max(0, min(self._page, n_pages - 1))
        self._page_label.setText(f"Page {self._page + 1} / {n_pages}")
        self._prev_btn.setEnabled(self._page > 0)
        self._next_btn.setEnabled(self._page < n_pages - 1)

    def _on_view_changed(self) -> None:
        self._page = 0
        self._redraw()

    def _action_prev_page(self) -> None:
        self._page = max(0, self._page - 1)
        self._redraw()

    def _action_next_page(self) -> None:
        self._page += 1
        self._redraw()

    # ── Main dispatch ──────────────────────────────────────────────────────

    def _redraw(self) -> None:
        if self._df is None or not self._params:
            return

        numeric = [
            p for p in self._params
            if p.enabled
            and p.ptype in (ParameterType.FLOAT, ParameterType.INT)
            and p.name in self._df.columns
        ]

        n_hist = len(self._df)
        n_sug  = len(self._suggestions) if self._suggestions else 0
        self._info_label.setText(
            f"{n_hist} historical pts" + (f"  |  {n_sug} suggested" if n_sug else "")
        )

        self._fig.clear()

        view = self._view_combo.currentText()   # "Auto" | "Pairplot" | ...
        n    = len(numeric)

        if n == 0:
            self._draw_no_numeric()
            self._set_page_nav_visible(False)
        else:
            # Determine whether to use the pairplot
            use_pairplot = (
                view == "Pairplot"
                or (view == "Auto" and n <= self._PAGE_SIZE)
            )
            use_parallel = (
                view == "Parallel Coords"
                or (view == "Auto" and not use_pairplot and n <= 12)
            )

            if use_pairplot:
                self._draw_pairplot(numeric)
                n_pages = max(1, math.ceil(n / self._PAGE_SIZE))
                self._set_page_nav_visible(n_pages > 1)
                self._update_page_nav(n)
            elif use_parallel:
                self._draw_parallel(numeric)
                self._set_page_nav_visible(False)
            else:
                self._draw_marginals(numeric)
                self._set_page_nav_visible(False)

        try:
            self._fig.tight_layout()
        except Exception:
            pass

        self._canvas.draw_idle()
        self._placeholder.hide()
        self._canvas.show()

    # ── Colour-mapping helper ──────────────────────────────────────────────

    def _get_colormap(self) -> tuple:
        """
        Returns (ScalarMappable | None, values_array | None).

        The colormap direction: minimize → RdYlGn_r (low=green, high=red)
                                maximize → RdYlGn   (low=red, high=green)
        """
        col = self._obj_combo.currentText()
        obj = next((o for o in self._objectives if o.column_name == col), None)
        if col not in (self._df.columns if self._df is not None else []):
            return None, None

        vals = pd.to_numeric(self._df[col], errors="coerce").values
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return None, None

        vmin, vmax = finite.min(), finite.max()
        if vmin == vmax:
            vmin -= 1; vmax += 1

        cmap_name = "RdYlGn_r" if (obj and obj.direction == "minimize") else "RdYlGn"
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm   = cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array(vals)
        return sm, vals

    def _style_ax(self, ax) -> None:
        """Apply dark theme styling to a single Axes."""
        ax.set_facecolor(_AX_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_FG, labelsize=7)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)

    def _add_colorbar(self, sm, ax_or_axes, label: str) -> None:
        cbar = self._fig.colorbar(sm, ax=ax_or_axes, shrink=0.6, pad=0.04, aspect=28)
        cbar.ax.yaxis.set_tick_params(color=_FG, labelsize=7)
        cbar.outline.set_edgecolor(_GRID)

        # Replace Matplotlib's floating "1e6"-style offset text with human-readable
        # suffixes (k / M / G) baked into each tick label.  This avoids the offset
        # text overlapping the rotated axis label on large-valued objectives.
        from matplotlib.ticker import FuncFormatter

        def _human(x, _):
            for mag, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
                if abs(x) >= mag:
                    return f"{x / mag:.3g}{suf}"
            return f"{x:.4g}"

        cbar.ax.yaxis.set_major_formatter(FuncFormatter(_human))

        import matplotlib.pyplot as _plt
        _plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_FG)
        cbar.set_label(label, color=_FG, fontsize=8, labelpad=12)

    # ══════════════════════════════════════════════════════════════════════
    # Plot type 1: Pairwise scatter grid (≤ 5 numeric params)
    # ══════════════════════════════════════════════════════════════════════

    def _draw_pairplot(self, numeric_all: list[ParameterConfig]) -> None:
        from matplotlib.gridspec import GridSpec
        from matplotlib.ticker import FuncFormatter
        import matplotlib.pyplot as _plt

        # ── Apply page slicing ─────────────────────────────────────────────
        n_total = len(numeric_all)
        n_pages = max(1, math.ceil(n_total / self._PAGE_SIZE))
        self._page = max(0, min(self._page, n_pages - 1))
        start   = self._page * self._PAGE_SIZE
        numeric = numeric_all[start : start + self._PAGE_SIZE]
        n       = len(numeric)

        sm, vals = self._get_colormap()
        has_color = sm is not None and vals is not None

        # ── GridSpec: N plot columns + 1 narrow colorbar column ───────────
        # The colorbar gets its own column so it NEVER steals space from the
        # scatter plots.  `tight_layout()` then works correctly.
        if has_color:
            gs = GridSpec(
                n, n + 1, figure=self._fig,
                width_ratios=[1.0] * n + [0.05],
                hspace=0.08, wspace=0.08,
            )
            cax = self._fig.add_subplot(gs[:, n])   # rightmost column = colorbar
        else:
            gs  = GridSpec(n, n, figure=self._fig, hspace=0.08, wspace=0.08)
            cax = None

        # Build 2-D list of Axes
        axes = [[self._fig.add_subplot(gs[i, j]) for j in range(n)]
                for i in range(n)]

        for i, pi in enumerate(numeric):
            xi = pd.to_numeric(self._df[pi.name], errors="coerce").values

            for j, pj in enumerate(numeric):
                ax = axes[i][j]
                self._style_ax(ax)
                yj = pd.to_numeric(self._df[pj.name], errors="coerce").values

                if i == j:
                    # Diagonal — histogram
                    mask = np.isfinite(xi)
                    ax.hist(xi[mask], bins=min(20, max(5, mask.sum() // 3)),
                            color=_HIST, alpha=0.75,
                            edgecolor=_BG, linewidth=0.4)
                    if i == n - 1:
                        ax.set_xlabel(pi.name, fontsize=7)
                    if j == 0:
                        ax.set_ylabel(pi.name, fontsize=7)

                elif i < j:
                    # Upper triangle — hide
                    ax.set_visible(False)

                else:
                    # Lower triangle — scatter
                    mask = np.isfinite(xi) & np.isfinite(yj)
                    if has_color:
                        c_vals = np.where(
                            np.isfinite(vals),
                            vals,
                            np.nanmean(vals[np.isfinite(vals)]),
                        )
                        ax.scatter(
                            yj[mask], xi[mask],
                            c=c_vals[mask],
                            cmap=sm.cmap, norm=sm.norm,
                            s=20, alpha=0.70, linewidths=0,
                        )
                    else:
                        ax.scatter(yj[mask], xi[mask],
                                   c=_HIST, s=20, alpha=0.70, linewidths=0)

                    # Overlay suggestions
                    if self._suggestions:
                        sxi = [s.get(pi.name) for s in self._suggestions]
                        syj = [s.get(pj.name) for s in self._suggestions]
                        valid = [
                            (x, y) for x, y in zip(sxi, syj)
                            if x is not None and y is not None
                        ]
                        if valid:
                            vx, vy = zip(*valid)
                            ax.scatter(
                                vy, vx,
                                marker="*", s=180, c=_SUGGEST,
                                edgecolors="white", linewidths=0.6,
                                zorder=5,
                            )

                    if i == n - 1:
                        ax.set_xlabel(pj.name, fontsize=7)
                    if j == 0:
                        ax.set_ylabel(pi.name, fontsize=7)

        # ── Colorbar in its dedicated GridSpec column ──────────────────────
        if has_color and cax is not None:
            cbar = self._fig.colorbar(sm, cax=cax)
            cbar.ax.yaxis.set_tick_params(color=_FG, labelsize=7)
            cbar.outline.set_edgecolor(_GRID)

            def _human(x, _):
                for mag, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
                    if abs(x) >= mag:
                        return f"{x / mag:.3g}{suf}"
                return f"{x:.4g}"

            cbar.ax.yaxis.set_major_formatter(FuncFormatter(_human))
            _plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_FG)
            cbar.set_label(self._obj_combo.currentText(),
                           color=_FG, fontsize=8, labelpad=10)

        # ── Legend (top-right, inside figure space) ────────────────────────
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_HIST, markersize=7,
                   linestyle="None", label="Historical data"),
        ]
        if self._suggestions:
            legend_handles.append(
                Line2D([0], [0], marker="*", color="w",
                       markerfacecolor=_SUGGEST, markersize=12,
                       linestyle="None", label="Suggested")
            )
        # Place legend anchored to the figure's top-right; `bbox_transform` is
        # figure coordinates so it never collides with any subplot.
        self._fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.88 if has_color else 1.0, 1.0),
            bbox_transform=self._fig.transFigure,
            facecolor=_BG, edgecolor=_GRID, labelcolor=_FG, fontsize=8,
        )

        # ── Suptitle with page info ────────────────────────────────────────
        if n_pages > 1:
            title = (
                f"Pairplot  (page {self._page + 1} / {n_pages}"
                f"  ·  params {start + 1}–{start + n} of {n_total})"
            )
        else:
            title = "Pairplot — Design Space"
        self._fig.suptitle(title, color=_FG, fontsize=9, y=1.01)
        self._fig.patch.set_facecolor(_BG)

    # ══════════════════════════════════════════════════════════════════════
    # Plot type 2: Parallel coordinates (6–12 numeric params)
    # ══════════════════════════════════════════════════════════════════════

    def _draw_parallel(self, numeric: list[ParameterConfig]) -> None:
        ax = self._fig.add_subplot(111)
        self._style_ax(ax)

        n     = len(numeric)
        xs    = list(range(n))
        names = [p.name for p in numeric]

        # Build raw matrix & normalise each column to [0, 1]
        raw = np.column_stack([
            pd.to_numeric(self._df[p.name], errors="coerce").values
            for p in numeric
        ])  # shape: (n_rows, n_params)

        col_min = np.nanmin(raw, axis=0)
        col_max = np.nanmax(raw, axis=0)
        col_rng = np.where(col_max > col_min, col_max - col_min, 1.0)
        norm_raw = (raw - col_min) / col_rng   # [0, 1]

        sm, vals = self._get_colormap()
        has_color = sm is not None and vals is not None

        # Historical lines
        for row_i in range(len(self._df)):
            y = norm_raw[row_i]
            if np.any(np.isnan(y)):
                continue
            if has_color and np.isfinite(vals[row_i]):
                rgba = sm.to_rgba(vals[row_i], alpha=0.35)
            else:
                rgba = (*mcolors.to_rgb(_HIST), 0.25)
            ax.plot(xs, y, c=rgba, linewidth=0.9)

        # Axis spine lines
        for xi in xs:
            ax.axvline(xi, color=_GRID, linewidth=0.8, zorder=2)

        # Suggested lines
        if self._suggestions:
            suggest_colours = [_SUGGEST, _SUGGEST2, "#cba6f7", "#94e2d5"]
            for s_idx, s in enumerate(self._suggestions):
                sy = []
                for p_idx, p in enumerate(numeric):
                    v = s.get(p.name)
                    if v is not None:
                        sy.append((float(v) - col_min[p_idx]) / col_rng[p_idx])
                    else:
                        sy.append(np.nan)
                col = suggest_colours[s_idx % len(suggest_colours)]
                ax.plot(xs, sy, c=col, linewidth=2.8, alpha=0.95, zorder=5,
                        label=f"Suggestion #{s_idx + 1}")
                for xi, yi in zip(xs, sy):
                    if np.isfinite(yi):
                        ax.scatter(xi, yi, c=col, s=70, zorder=6,
                                   edgecolors="white", linewidths=0.5)

        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=25, ha="right", color=_FG, fontsize=8)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["min", "25%", "50%", "75%", "max"],
                           color=_FG, fontsize=7)
        ax.set_xlim(-0.3, n - 0.7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Parallel Coordinates — Parameter Space",
                     color=_FG, fontsize=9, pad=6)

        if has_color:
            self._add_colorbar(sm, ax, self._obj_combo.currentText())

        if self._suggestions:
            ax.legend(loc="upper left", facecolor=_BG, edgecolor=_GRID,
                      labelcolor=_FG, fontsize=7)

        self._fig.patch.set_facecolor(_BG)

    # ══════════════════════════════════════════════════════════════════════
    # Plot type 3: 1-D marginal strip (> 12 numeric params)
    # ══════════════════════════════════════════════════════════════════════

    def _draw_marginals(self, numeric: list[ParameterConfig]) -> None:
        n    = len(numeric)
        axes = self._fig.subplots(n, 1, squeeze=False)
        suggest_colours = [_SUGGEST, _SUGGEST2, "#cba6f7", "#94e2d5"]

        for i, p in enumerate(numeric):
            ax = axes[i][0]
            self._style_ax(ax)

            col_vals = pd.to_numeric(self._df[p.name], errors="coerce").dropna().values
            if len(col_vals) > 1:
                bins = min(25, max(5, len(col_vals) // 5))
                ax.hist(col_vals, bins=bins, color=_HIST, alpha=0.65,
                        edgecolor=_BG, linewidth=0.3)

            # Suggestion vertical lines
            if self._suggestions:
                for s_idx, s in enumerate(self._suggestions):
                    sv = s.get(p.name)
                    if sv is not None:
                        col = suggest_colours[s_idx % len(suggest_colours)]
                        ax.axvline(
                            float(sv), color=col, linewidth=2.0,
                            linestyle="--", alpha=0.9,
                            label=(f"Sug #{s_idx + 1}" if i == 0 else "_nolegend_"),
                        )

            ax.set_ylabel(p.name, fontsize=7, color=_FG,
                          rotation=0, labelpad=65, va="center")
            ax.set_yticks([])

        if self._suggestions:
            axes[0][0].legend(
                loc="upper right", facecolor=_BG,
                edgecolor=_GRID, labelcolor=_FG, fontsize=7,
            )

        axes[-1][0].set_xlabel("Parameter value", color=_FG, fontsize=8)
        self._fig.suptitle("Parameter Marginal Distributions",
                           color=_FG, fontsize=9)
        self._fig.patch.set_facecolor(_BG)

    # ══════════════════════════════════════════════════════════════════════
    # Fallback: no numeric parameters
    # ══════════════════════════════════════════════════════════════════════

    def _draw_no_numeric(self) -> None:
        ax = self._fig.add_subplot(111)
        self._style_ax(ax)
        ax.text(0.5, 0.5,
                "No enabled numeric parameters found.\n"
                "Enable at least one FLOAT or INT parameter to see the design space.",
                ha="center", va="center", color=_FG, fontsize=11,
                transform=ax.transAxes, wrap=True)
        ax.set_axis_off()
        self._fig.patch.set_facecolor(_BG)


# ══════════════════════════════════════════════════════════════════════════════
# Correlation Matrix Widget
# ══════════════════════════════════════════════════════════════════════════════

class CorrelationWidget(QWidget):
    """
    Annotated Pearson / Spearman correlation heatmap for all numeric parameters
    and objective columns.

    Each cell shows the correlation coefficient.  A dashed line visually
    separates the parameter block from the objective block so you can quickly
    see which inputs drive each output.

    Public API
    ----------
    refresh(df, params, objectives)
        Rebuild the heatmap.
    clear()
        Hide the canvas and show the placeholder.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._df: Optional[pd.DataFrame] = None
        self._params: List[ParameterConfig] = []
        self._objectives: List[ObjectiveConfig] = []

        self._fig = Figure(facecolor=_BG)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Correlation method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["Pearson", "Spearman"])
        self._method_combo.setFixedWidth(130)
        self._method_combo.setToolTip(
            "Pearson: linear correlation (sensitive to outliers).\n"
            "Spearman: rank-based (robust to outliers, catches monotonic non-linear)."
        )
        self._method_combo.currentIndexChanged.connect(self._redraw)
        toolbar.addWidget(self._method_combo)
        toolbar.addStretch()
        refresh_btn = QPushButton("⟳  Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self._redraw)
        toolbar.addWidget(refresh_btn)
        layout.addLayout(toolbar)

        layout.addWidget(self._canvas, stretch=1)

        self._placeholder = QLabel(
            "Load a CSV and apply objectives to see the correlation matrix."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_FG}; font-size: 13px; font-style: italic;"
        )
        layout.addWidget(self._placeholder)
        self._canvas.hide()

    # ── Public API ─────────────────────────────────────────────────────────

    def refresh(
        self,
        df: pd.DataFrame,
        params: List[ParameterConfig],
        objectives: List[ObjectiveConfig],
    ) -> None:
        self._df         = df
        self._params     = params
        self._objectives = objectives
        self._redraw()

    def clear(self) -> None:
        self._df = None
        self._fig.clear()
        self._canvas.draw_idle()
        self._canvas.hide()
        self._placeholder.show()

    # ── Internal ───────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        if self._df is None:
            return

        # Collect numeric parameter columns + objective columns
        param_cols = [
            p.name for p in self._params
            if p.ptype in (ParameterType.FLOAT, ParameterType.INT)
            and p.name in self._df.columns
        ]
        obj_cols = [
            o.column_name for o in self._objectives
            if o.column_name in self._df.columns
        ]
        # Avoid duplicates (an objective column could share a name with a param)
        all_cols = param_cols + [c for c in obj_cols if c not in param_cols]

        if len(all_cols) < 2:
            self._canvas.hide()
            self._placeholder.setText(
                "Need at least 2 numeric columns to show a correlation matrix."
            )
            self._placeholder.show()
            return

        # Build numeric sub-dataframe
        sub = pd.DataFrame({c: pd.to_numeric(self._df[c], errors="coerce")
                            for c in all_cols})

        method = self._method_combo.currentText().lower()
        corr = sub.corr(method=method)
        n = len(all_cols)

        # ── Draw ───────────────────────────────────────────────────────────
        self._fig.clear()
        # Leave extra right margin for the colorbar
        ax = self._fig.add_axes([0.18, 0.18, 0.62, 0.72])
        ax.set_facecolor(_AX_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)

        im = ax.imshow(
            corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
        )

        # Colorbar with fixed limits — no need for human formatter here
        cax = self._fig.add_axes([0.83, 0.18, 0.03, 0.72])
        cbar = self._fig.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_tick_params(color=_FG, labelsize=7)
        cbar.outline.set_edgecolor(_GRID)
        import matplotlib.pyplot as _plt
        _plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_FG)
        cbar.set_label(
            f"{self._method_combo.currentText()} r",
            color=_FG, fontsize=8, labelpad=10,
        )

        # Tick labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(all_cols, rotation=40, ha="right",
                           fontsize=min(9, max(6, 80 // n)), color=_FG)
        ax.set_yticklabels(all_cols,
                           fontsize=min(9, max(6, 80 // n)), color=_FG)
        ax.tick_params(colors=_FG)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                if np.isfinite(val):
                    text_col = "white" if abs(val) > 0.6 else _FG
                    fs = max(5, min(9, 72 // n))
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=fs, color=text_col, fontweight="bold",
                    )

        # Dashed separator: parameters vs. objectives
        n_params = len(param_cols)
        if 0 < n_params < n:
            sep = n_params - 0.5
            ax.axhline(sep, color=_FG, linewidth=1.2, linestyle="--", alpha=0.45)
            ax.axvline(sep, color=_FG, linewidth=1.2, linestyle="--", alpha=0.45)
            # Small annotations
            ax.text(-0.6, sep / 2, "params", ha="right", va="center",
                    fontsize=6, color=_FG, alpha=0.7)
            ax.text(-0.6, (sep + n) / 2, "objectives", ha="right", va="center",
                    fontsize=6, color=_FG, alpha=0.7)

        ax.set_title(
            f"{self._method_combo.currentText()} Correlation Matrix  "
            "(dashed line: params | objectives)",
            color=_FG, fontsize=9, pad=8,
        )
        self._fig.patch.set_facecolor(_BG)

        self._canvas.draw_idle()
        self._placeholder.hide()
        self._canvas.show()


# ══════════════════════════════════════════════════════════════════════════════
# Stand-alone dialog wrapper
# ══════════════════════════════════════════════════════════════════════════════

class DesignSpaceDialog(QDialog):
    """
    Non-modal window hosting a tabbed interface:

    * **📊 Design Space** — pairplot / parallel coordinates / marginals
      (the original :class:`DesignSpaceWidget`)
    * **🔗 Correlation Matrix** — annotated Pearson / Spearman heatmap
      (:class:`CorrelationWidget`)

    Usage
    -----
    ::

        # Create once, reuse thereafter
        dlg = DesignSpaceDialog(parent=main_window)
        dlg.refresh(df, params, objectives, suggestions=None)
        dlg.show()
        dlg.raise_()

    Calling ``refresh()`` while the dialog is already visible simply redraws
    both tabs with new data; no new window is opened.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Design Space — BHOP")

        # ── Size: fit within screen ────────────────────────────────────────
        screen = (
            QApplication.primaryScreen()
            if QApplication.instance() is not None
            else None
        )
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(avail.width()  - 80, 1200)
            h = min(avail.height() - 80, 820)
        else:
            w, h = 1000, 720
        self.resize(w, h)
        self.setMinimumSize(500, 380)

        # ── Layout ────────────────────────────────────────────────────────
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(4)

        # Two-tab interface
        self._tabs = QTabWidget()

        self._widget = DesignSpaceWidget(self)
        self._tabs.addTab(self._widget, "📊  Design Space")

        self._corr_widget = CorrelationWidget(self)
        self._tabs.addTab(self._corr_widget, "🔗  Correlation Matrix")

        vbox.addWidget(self._tabs, stretch=1)

        # ── Bottom row: size grip + close button ──────────────────────────
        bottom_row = QHBoxLayout()
        size_grip = QSizeGrip(self)
        bottom_row.addWidget(size_grip, alignment=Qt.AlignLeft | Qt.AlignBottom)
        bottom_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.hide)
        bottom_row.addWidget(close_btn)
        vbox.addLayout(bottom_row)

    # ── Public API ─────────────────────────────────────────────────────────

    def refresh(
        self,
        df: pd.DataFrame,
        params: List[ParameterConfig],
        objectives: List[ObjectiveConfig],
        suggestions: Optional[List[dict]] = None,
    ) -> None:
        """Refresh both the Design Space and Correlation Matrix tabs."""
        self._widget.refresh(df, params, objectives, suggestions)
        self._corr_widget.refresh(df, params, objectives)

    def clear(self) -> None:
        """Clear both tabs."""
        self._widget.clear()
        self._corr_widget.clear()
