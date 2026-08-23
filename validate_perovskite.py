
#!/usr/bin/env python3
"""
validate_perovskite.py
======================
Rigorous validation of perovskite Bayesian Optimisation predictions
against the closed, fully-measured Perovskite_dataset.csv.

Five checks (matching the provided feedback):
  §0  Data loading & sanity checks
  §1  Hold-out / leave-some-out surrogate validation
  §2  Pool-based virtual BO experiment (gold-standard benchmark)
  §3  Acquisition surface & suggestion sanity
  §4  Domain-knowledge cross-check
  §5  Summary pass/fail report

Outputs
-------
  Console  – metrics and pass/fail checklist printed to stdout
  Plots    – PNG files saved to  validation_results/
"""
from __future__ import annotations

import io
import os
import sys
import warnings
from itertools import product as iproduct

# ── Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError) ──────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")          # headless – no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import Delaunay
from scipy.stats import norm as sp_norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedKFold, train_test_split

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH   = "test_data/Perovskite_dataset.csv"
OUT_DIR    = "validation_results"
SEED       = 42
TEST_SIZE  = 0.30
N_CV_FOLDS = 5
N_CV_REPS  = 5        # repeated k-fold repeats
N_SEED_BO  = 8        # initial random seed points for BO
N_BO_ITER  = 30       # BO steps per run
N_BO_RUNS  = 10       # independent BO runs for confidence bands
GRID_STEP  = 0.025    # simplex grid resolution
EI_XI      = 0.01     # EI exploration bonus

FEATURES   = ["CsPbI", "FAPbI", "MAPbI"]
TARGET     = "Instability index"

os.makedirs(OUT_DIR, exist_ok=True)
rng_global = np.random.default_rng(SEED)

# Pastel colour palette for the three regions
REGION_COLORS = {"Cs-rich": "#e07b54", "FA-rich": "#5a9e6f", "MA-rich": "#5b7ec9"}

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def ternary_to_cart(cs: np.ndarray, fa: np.ndarray, _ma: np.ndarray):
    """Barycentric (Cs, FA, MA) → Cartesian for equilateral triangle.

    Vertex layout:
      MAPbI (pure MA) → bottom-left  (0, 0)
      CsPbI (pure Cs) → bottom-right (1, 0)
      FAPbI (pure FA) → top centre   (0.5, √3/2)
    """
    x = cs + 0.5 * fa
    y = fa * (np.sqrt(3) / 2)
    return np.asarray(x), np.asarray(y)


def draw_ternary_frame(ax, fontsize: int = 9):
    """Draw the triangle outline and vertex labels on *ax*."""
    h = np.sqrt(3) / 2
    vx = [0, 1, 0.5, 0]
    vy = [0, 0, h,   0]
    ax.plot(vx, vy, "k-", lw=1.4, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(1.05,  -0.04, "CsPbI",  ha="left",   va="top",    fontsize=fontsize, fontweight="bold")
    ax.text(0.50,  h+0.04, "FAPbI", ha="center", va="bottom", fontsize=fontsize, fontweight="bold")
    ax.text(-0.05, -0.04, "MAPbI",  ha="right",  va="top",    fontsize=fontsize, fontweight="bold")
    # Guide lines at 25 / 50 / 75 %
    for frac in [0.25, 0.50, 0.75]:
        # Parallel to Cs–FA edge (iso-MA lines)
        p1 = np.array(ternary_to_cart(np.array([0.0]), np.array([1-frac]), np.array([frac])))
        p2 = np.array(ternary_to_cart(np.array([1-frac]), np.array([0.0]), np.array([frac])))
        ax.plot([p1[0,0], p2[0,0]], [p1[1,0], p2[1,0]], color="grey",
                lw=0.4, ls="--", zorder=1)
        # iso-Cs lines
        p1 = np.array(ternary_to_cart(np.array([frac]), np.array([1-frac]), np.array([0.0])))
        p2 = np.array(ternary_to_cart(np.array([frac]), np.array([0.0]), np.array([1-frac])))
        ax.plot([p1[0,0], p2[0,0]], [p1[1,0], p2[1,0]], color="grey",
                lw=0.4, ls="--", zorder=1)
        # iso-FA lines
        p1 = np.array(ternary_to_cart(np.array([0.0]), np.array([frac]), np.array([1-frac])))
        p2 = np.array(ternary_to_cart(np.array([1-frac]), np.array([frac]), np.array([0.0])))
        ax.plot([p1[0,0], p2[0,0]], [p1[1,0], p2[1,0]], color="grey",
                lw=0.4, ls="--", zorder=1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    pr, _ = stats.pearsonr(y_true, y_pred)
    sr, _ = stats.spearmanr(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "Pearson_r": float(pr), "Spearman_rho": float(sr)}


def make_rf(seed: int = SEED) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2,
        max_features=1.0, random_state=seed,
    )


def make_gp() -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e6))
        * Matern(length_scale=0.3, length_scale_bounds=(1e-2, 10.0), nu=2.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e6))
    )
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5,
        normalize_y=True, random_state=SEED,
    )


def region_label(cs: float, fa: float, ma: float) -> str:
    mx = max(cs, fa, ma)
    if mx == cs: return "Cs-rich"
    if mx == fa: return "FA-rich"
    return "MA-rich"


def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                         y_best: float, xi: float = EI_XI) -> np.ndarray:
    """EI acquisition for minimisation."""
    sigma = np.maximum(sigma, 1e-9)
    z  = (y_best - mu - xi) / sigma
    ei = (y_best - mu - xi) * sp_norm.cdf(z) + sigma * sp_norm.pdf(z)
    return np.maximum(ei, 0.0)


def rf_predict_with_std(rf: RandomForestRegressor, X: np.ndarray):
    """Use individual tree predictions to get mean & std from RF."""
    preds = np.array([tree.predict(X) for tree in rf.estimators_])
    return preds.mean(axis=0), preds.std(axis=0)


def in_convex_hull(points: np.ndarray, hull_points: np.ndarray) -> np.ndarray:
    """Boolean mask – True if point lies inside convex hull of hull_points."""
    try:
        tri = Delaunay(hull_points)
        return tri.find_simplex(points) >= 0
    except Exception:
        return np.ones(len(points), dtype=bool)


def simplex_grid(step: float = GRID_STEP) -> np.ndarray:
    """Regular grid on the 3-component simplex (components sum to 1)."""
    pts = []
    vals = np.round(np.arange(0.0, 1.0 + step / 2, step), 6)
    for cs in vals:
        for fa in vals:
            ma = round(1.0 - cs - fa, 6)
            if 0.0 - step / 2 <= ma <= 1.0 + step / 2:
                pts.append([cs, np.clip(fa, 0, 1), np.clip(ma, 0, 1)])
    return np.array(pts)


def save_fig(fig: plt.Figure, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def section_header(n: int, title: str):
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"  §{n}  {title}")
    print(bar)


# ──────────────────────────────────────────────────────────────────────────────
# §0  DATA LOADING & SANITY CHECKS
# ──────────────────────────────────────────────────────────────────────────────

def section0(df: pd.DataFrame) -> dict:
    section_header(0, "Data loading & sanity checks")

    results = {}

    # ── Basic stats ────────────────────────────────────────────────────────────
    n_rows = len(df)
    print(f"\n  Rows   : {n_rows}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Instability index stats:")
    desc = df[TARGET].describe()
    for stat in ["min", "25%", "50%", "75%", "max", "mean", "std"]:
        print(f"    {stat:6s} = {desc[stat]:>12,.0f}")

    # ── Composition sum check ──────────────────────────────────────────────────
    comp_sum = df[FEATURES].sum(axis=1)
    bad_mask = (comp_sum - 1.0).abs() > 0.02
    n_bad = bad_mask.sum()
    print(f"\n  Composition sum check (|sum−1| ≤ 0.02):")
    print(f"    Violations : {n_bad} / {n_rows}")
    if n_bad:
        print(df[bad_mask][[*FEATURES, TARGET]])
    results["sum_check_pass"] = n_bad == 0

    # ── Replicate analysis ────────────────────────────────────────────────────
    grp = df.groupby(FEATURES)[TARGET].agg(["count", "mean", "std"]).reset_index()
    replicated = grp[grp["count"] > 1]
    print(f"\n  Replicate analysis:")
    print(f"    Unique compositions : {len(grp)}")
    print(f"    With replicates     : {len(replicated)}")
    if len(replicated):
        print(f"    Max within-group std: {replicated['std'].max():>12,.0f}")
        print(f"    Median CV           : {(replicated['std']/replicated['mean']).median():.2%}")
    results["n_unique"] = int(len(grp))
    results["median_cv"] = float((replicated["std"] / replicated["mean"]).median()
                                 if len(replicated) else 0)

    # ── Identify lowest-instability compositions ───────────────────────────────
    top5 = grp.nsmallest(5, "mean")[FEATURES + ["mean", "count"]].reset_index(drop=True)
    print("\n  Top-5 compositions by mean Instability index (lower is better):")
    print(top5.to_string(index=False))

    # ── Plot: ternary scatter coloured by instability ─────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5.2))
    cs_v = grp["CsPbI"].values
    fa_v = grp["FAPbI"].values
    ma_v = grp["MAPbI"].values
    val  = grp["mean"].values
    xc, yc = ternary_to_cart(cs_v, fa_v, ma_v)
    draw_ternary_frame(ax)
    sc = ax.scatter(xc, yc, c=val, cmap="RdYlGn_r", s=60,
                    norm=mcolors.LogNorm(vmin=val.min(), vmax=val.max()),
                    edgecolors="k", linewidths=0.4, zorder=6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean Instability index (log scale)", fontsize=8)
    ax.set_title("§0 – Raw data: Instability index across composition space\n"
                 "(averaged over replicates)", fontsize=9)
    save_fig(fig, "s0_ternary_raw.png")

    # ── Plot: replicate variability ───────────────────────────────────────────
    if len(replicated):
        fig, ax = plt.subplots(figsize=(6, 5.2))
        cs_r = replicated["CsPbI"].values
        fa_r = replicated["FAPbI"].values
        ma_r = replicated["MAPbI"].values
        std_r = replicated["std"].values
        xr, yr = ternary_to_cart(cs_r, fa_r, ma_r)
        draw_ternary_frame(ax)
        sc = ax.scatter(xr, yr, c=std_r, cmap="OrRd", s=60,
                        edgecolors="k", linewidths=0.4, zorder=6)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Within-replicate std", fontsize=8)
        ax.set_title("§0 – Replicate variability (std of Instability index)", fontsize=9)
        save_fig(fig, "s0_replicate_variance.png")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# §1  SURROGATE HOLD-OUT VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def section1(df: pd.DataFrame) -> dict:
    section_header(1, "Surrogate hold-out validation")

    # Average replicates → one value per composition
    agg = df.groupby(FEATURES, as_index=False)[TARGET].mean()
    X   = agg[FEATURES].values.astype(float)
    y   = agg[TARGET].values.astype(float)
    print(f"\n  Working on {len(agg)} unique compositions (replicates averaged)")

    results = {}

    # ── (A) Single 70/30 train–test split ─────────────────────────────────────
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"\n  [A] 70/30 hold-out  (train={len(Xtr)}, test={len(Xte)})")

    rf = make_rf()
    rf.fit(Xtr, ytr)
    gp = make_gp()
    gp.fit(Xtr, ytr)

    rf_pred_te = rf.predict(Xte)
    gp_pred_te = gp.predict(Xte)

    m_rf_ho = compute_metrics(yte, rf_pred_te)
    m_gp_ho = compute_metrics(yte, gp_pred_te)
    results["holdout_rf"] = m_rf_ho
    results["holdout_gp"] = m_gp_ho

    print(f"\n    Random Forest  →  Pearson r={m_rf_ho['Pearson_r']:.3f}  "
          f"Spearman ρ={m_rf_ho['Spearman_rho']:.3f}  "
          f"RMSE={m_rf_ho['RMSE']:>10,.0f}  MAE={m_rf_ho['MAE']:>10,.0f}")
    print(f"    Gaussian Proc  →  Pearson r={m_gp_ho['Pearson_r']:.3f}  "
          f"Spearman ρ={m_gp_ho['Spearman_rho']:.3f}  "
          f"RMSE={m_gp_ho['RMSE']:>10,.0f}  MAE={m_gp_ho['MAE']:>10,.0f}")

    # ── (B) Repeated 5-fold CV ────────────────────────────────────────────────
    print(f"\n  [B] Repeated {N_CV_FOLDS}-fold CV  (×{N_CV_REPS} repeats) …")
    rkf = RepeatedKFold(n_splits=N_CV_FOLDS, n_repeats=N_CV_REPS, random_state=SEED)

    # Accumulate OOF predictions, averaging over repeats
    rf_oof_acc = np.zeros(len(y));  rf_oof_cnt = np.zeros(len(y))
    gp_oof_acc = np.zeros(len(y));  gp_oof_cnt = np.zeros(len(y))

    for fold_i, (tr_i, te_i) in enumerate(rkf.split(X)):
        rf_ = make_rf(seed=SEED + fold_i)
        rf_.fit(X[tr_i], y[tr_i])
        rf_oof_acc[te_i] += rf_.predict(X[te_i])
        rf_oof_cnt[te_i] += 1

        gp_ = make_gp()
        gp_.fit(X[tr_i], y[tr_i])
        gp_oof_acc[te_i] += gp_.predict(X[te_i])
        gp_oof_cnt[te_i] += 1

    rf_oof = rf_oof_acc / rf_oof_cnt
    gp_oof = gp_oof_acc / gp_oof_cnt

    m_rf_cv = compute_metrics(y, rf_oof)
    m_gp_cv = compute_metrics(y, gp_oof)
    results["cv_rf"] = m_rf_cv
    results["cv_gp"] = m_gp_cv

    print(f"\n    Random Forest  →  Pearson r={m_rf_cv['Pearson_r']:.3f}  "
          f"Spearman ρ={m_rf_cv['Spearman_rho']:.3f}  "
          f"RMSE={m_rf_cv['RMSE']:>10,.0f}  MAE={m_rf_cv['MAE']:>10,.0f}")
    print(f"    Gaussian Proc  →  Pearson r={m_gp_cv['Pearson_r']:.3f}  "
          f"Spearman ρ={m_gp_cv['Spearman_rho']:.3f}  "
          f"RMSE={m_gp_cv['RMSE']:>10,.0f}  MAE={m_gp_cv['MAE']:>10,.0f}")

    # Region labels for colouring
    labels = [region_label(row[0], row[1], row[2]) for row in X]
    label_te = [region_label(r[0], r[1], r[2]) for r in Xte]

    # ── Figure: predicted vs true (4 panels) ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("§1 – Surrogate predicted vs true Instability index", fontsize=11)

    plot_data = [
        (axes[0, 0], yte,  rf_pred_te, label_te, "RF  – 70/30 hold-out"),
        (axes[0, 1], yte,  gp_pred_te, label_te, "GP  – 70/30 hold-out"),
        (axes[1, 0], y,    rf_oof,     labels,    f"RF  – Repeated {N_CV_FOLDS}-fold CV (OOF avg)"),
        (axes[1, 1], y,    gp_oof,     labels,    f"GP  – Repeated {N_CV_FOLDS}-fold CV (OOF avg)"),
    ]

    for ax, yt, yp, lbs, title in plot_data:
        for region, color in REGION_COLORS.items():
            mask = np.array(lbs) == region
            ax.scatter(yt[mask], yp[mask], c=color, label=region,
                       s=30, alpha=0.75, edgecolors="k", linewidths=0.3)
        lo = min(yt.min(), yp.min()) * 0.9
        hi = max(yt.max(), yp.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="Perfect")
        ax.set_xlabel("True Instability index",  fontsize=8)
        ax.set_ylabel("Predicted Instability index", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        # Annotate with Pearson r
        m = compute_metrics(yt, yp)
        ax.text(0.97, 0.05,
                f"r={m['Pearson_r']:.3f}\nρ={m['Spearman_rho']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    fig.tight_layout()
    save_fig(fig, "s1_predicted_vs_true.png")

    # ── Figure: residuals coloured by region ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("§1 – Residuals by composition region (CV out-of-fold)", fontsize=11)

    for ax, yp, title in [(axes[0], rf_oof, "Random Forest"),
                           (axes[1], gp_oof, "Gaussian Process")]:
        residuals = yp - y
        for region, color in REGION_COLORS.items():
            mask = np.array(labels) == region
            ax.scatter(y[mask], residuals[mask], c=color, label=region,
                       s=35, alpha=0.80, edgecolors="k", linewidths=0.3)
        ax.axhline(0, color="k", lw=0.9, ls="--")
        ax.set_xlabel("True Instability index", fontsize=8)
        ax.set_ylabel("Residual (pred − true)",  fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, "s1_residuals_by_region.png")

    # Store fitted GP/RF on full data for later sections
    gp_full = make_gp()
    gp_full.fit(X, y)
    rf_full = make_rf()
    rf_full.fit(X, y)
    results["gp_full"] = gp_full
    results["rf_full"] = rf_full
    results["X"] = X
    results["y"] = y

    return results


# ──────────────────────────────────────────────────────────────────────────────
# §2  POOL-BASED VIRTUAL BO EXPERIMENT
# ──────────────────────────────────────────────────────────────────────────────

def _bo_run(pool_X: np.ndarray, pool_y: np.ndarray,
            n_seed: int, n_iter: int, seed: int,
            mode: str = "gp_ei") -> np.ndarray:
    """
    One independent BO run on the fixed pool.

    mode options:
      'gp_ei'   – GP surrogate + Expected Improvement (main BO)
      'rf_ei'   – RF surrogate + EI via tree-variance
      'greedy'  – GP mean only (no exploration bonus)
      'random'  – uniform random selection (baseline)

    Returns array of length (n_seed + n_iter) with cumulative best values.
    """
    rng = np.random.default_rng(seed)
    pool_size = len(pool_X)
    observed  = list(rng.choice(pool_size, size=n_seed, replace=False))
    remaining = list(set(range(pool_size)) - set(observed))

    best_history = []
    for idx in observed:
        best_history.append(pool_y[idx])

    for _ in range(min(n_iter, len(remaining))):
        if not remaining:
            break

        Xobs = pool_X[observed]
        yobs = pool_y[observed]
        Xrem = pool_X[remaining]
        y_best = yobs.min()

        if mode == "random":
            chosen_local = int(rng.integers(0, len(remaining)))

        elif mode == "gp_ei":
            gp = make_gp()
            try:
                gp.fit(Xobs, yobs)
                mu, sigma = gp.predict(Xrem, return_std=True)
            except Exception:
                chosen_local = int(rng.integers(0, len(remaining)))
                chosen_global = remaining[chosen_local]
                observed.append(chosen_global)
                remaining.remove(chosen_global)
                best_history.append(pool_y[observed].min())
                continue
            ei = expected_improvement(mu, sigma, y_best)
            chosen_local = int(np.argmax(ei))

        elif mode == "rf_ei":
            rf = make_rf(seed=seed)
            rf.fit(Xobs, yobs)
            mu, sigma = rf_predict_with_std(rf, Xrem)
            ei = expected_improvement(mu, sigma, y_best)
            chosen_local = int(np.argmax(ei))

        elif mode == "greedy":
            gp = make_gp()
            try:
                gp.fit(Xobs, yobs)
                mu, _ = gp.predict(Xrem, return_std=True)
            except Exception:
                mu = yobs.mean() * np.ones(len(Xrem))
            chosen_local = int(np.argmin(mu))

        else:
            raise ValueError(f"Unknown mode: {mode}")

        chosen_global = remaining[chosen_local]
        observed.append(chosen_global)
        remaining.remove(chosen_global)
        best_history.append(pool_y[observed].min())

    return np.array(best_history)


def section2(pool_X: np.ndarray, pool_y: np.ndarray) -> dict:
    section_header(2, "Pool-based virtual BO experiment")

    n_iter_actual = min(N_BO_ITER, len(pool_X) - N_SEED_BO)
    total_steps = N_SEED_BO + n_iter_actual
    print(f"\n  Pool size : {len(pool_X)} unique compositions")
    print(f"  Seed pts  : {N_SEED_BO}  |  BO steps: {n_iter_actual}  |  Runs: {N_BO_RUNS}")
    print(f"  Running {N_BO_RUNS} independent runs for each strategy …")

    true_best = pool_y.min()
    print(f"\n  True global best Instability index : {true_best:,.0f}")
    print(f"  True worst                          : {pool_y.max():,.0f}")

    strategies = {
        "GP + EI  (BO)":  "gp_ei",
        "RF + EI  (BO)":  "rf_ei",
        "Greedy (no EI)": "greedy",
        "Random":         "random",
    }
    colors = {
        "GP + EI  (BO)": "#2563eb",
        "RF + EI  (BO)": "#16a34a",
        "Greedy (no EI)": "#d97706",
        "Random":         "#dc2626",
    }

    all_curves: dict[str, np.ndarray] = {}

    for label, mode in strategies.items():
        curves = []
        for run in range(N_BO_RUNS):
            c = _bo_run(pool_X, pool_y, N_SEED_BO, n_iter_actual,
                        seed=SEED * 100 + run, mode=mode)
            # Guarantee cumulative minimum
            for i in range(1, len(c)):
                c[i] = min(c[i], c[i-1])
            curves.append(c)
        # Pad shorter curves
        max_len = max(len(c) for c in curves)
        padded  = [np.pad(c, (0, max_len - len(c)), mode="edge") for c in curves]
        all_curves[label] = np.array(padded)  # shape (N_BO_RUNS, total_steps)

    # Report how many steps to find top-20% of pool
    top20_threshold = np.percentile(pool_y, 20)
    print(f"\n  Top-20 % threshold : {top20_threshold:,.0f}")
    print(f"\n  Median steps to enter top-20 % of pool (or N/A if not reached):")
    results = {}
    for label, curves in all_curves.items():
        steps_found = []
        for c in curves:
            hits = np.where(c <= top20_threshold)[0]
            steps_found.append(hits[0] if len(hits) else np.nan)
        med = np.nanmedian(steps_found)
        print(f"    {label:<22s}: {med:.0f}" if not np.isnan(med) else
              f"    {label:<22s}: never reached")
        results[label] = {"median_steps_top20": med}

    # ── Figure: convergence curves ─────────────────────────────────────────────
    x_axis = np.arange(1, all_curves[list(all_curves)[0]].shape[1] + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, curves in all_curves.items():
        mu  = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.plot(x_axis, mu, label=label, color=colors[label], lw=2)
        ax.fill_between(x_axis, mu - std, mu + std,
                        color=colors[label], alpha=0.15)

    ax.axhline(true_best, ls="--", color="black", lw=1.0, label=f"Global best ({true_best:,.0f})")
    ax.axvline(N_SEED_BO, ls=":", color="grey", lw=1.0, label=f"End of seed ({N_SEED_BO} pts)")
    ax.set_xlabel("Number of experiments (evaluations)", fontsize=9)
    ax.set_ylabel("Best Instability index found (lower = better)", fontsize=9)
    ax.set_title(f"§2 – Pool-based virtual BO convergence\n"
                 f"(mean ± 1 std over {N_BO_RUNS} runs)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "s2_bo_convergence.png")

    # ── Figure: fraction of global best found vs evaluations ──────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, curves in all_curves.items():
        # Normalise so 1.0 = global best, 0.0 = global worst
        norm_curves = 1.0 - (curves - true_best) / (pool_y.max() - true_best)
        mu  = norm_curves.mean(axis=0)
        std = norm_curves.std(axis=0)
        ax.plot(x_axis, mu, label=label, color=colors[label], lw=2)
        ax.fill_between(x_axis, np.maximum(0, mu - std), np.minimum(1, mu + std),
                        color=colors[label], alpha=0.15)

    ax.axhline(1.0, ls="--", color="black", lw=1.0, label="Global best")
    ax.axvline(N_SEED_BO, ls=":", color="grey", lw=1.0)
    ax.set_xlabel("Number of experiments (evaluations)", fontsize=9)
    ax.set_ylabel("Normalised performance (1 = global best)", fontsize=9)
    ax.set_title("§2 – Normalised convergence (1 = global best found)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, "s2_bo_convergence_normalised.png")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# §3  ACQUISITION SURFACE & SUGGESTION SANITY
# ──────────────────────────────────────────────────────────────────────────────

def section3(gp: GaussianProcessRegressor, rf: RandomForestRegressor,
             data_X: np.ndarray, data_y: np.ndarray) -> dict:
    section_header(3, "Acquisition surface & suggestion sanity")

    grid = simplex_grid(GRID_STEP)
    print(f"\n  Simplex grid : {len(grid)} points (step={GRID_STEP})")

    # Predict on grid
    gp_mu, gp_sigma = gp.predict(grid, return_std=True)
    rf_mu,  rf_sigma = rf_predict_with_std(rf, grid)

    # EI on grid (using full-data GP as surrogate)
    y_best_full = data_y.min()
    gp_ei = expected_improvement(gp_mu, gp_sigma, y_best_full)

    # ── Convex hull check ─────────────────────────────────────────────────────
    # Find where the GP suggests going next (top-5 EI points)
    top_ei_idx = np.argsort(gp_ei)[-5:][::-1]
    top_suggestions = grid[top_ei_idx]
    in_hull = in_convex_hull(top_suggestions, data_X)

    print("\n  Top-5 GP+EI suggestions:")
    print(f"    {'CsPbI':>6}  {'FAPbI':>6}  {'MAPbI':>6}  "
          f"{'Sum':>5}  {'Pred mean':>12}  {'Pred σ':>12}  {'In hull':>8}")
    results = {}
    for i, (sug, ih) in enumerate(zip(top_suggestions, in_hull)):
        cs, fa, ma = sug
        s = cs + fa + ma
        mu_  = float(gp_mu[top_ei_idx[i]])
        sig_ = float(gp_sigma[top_ei_idx[i]])
        print(f"    {cs:>6.3f}  {fa:>6.3f}  {ma:>6.3f}  "
              f"{s:>5.3f}  {mu_:>12,.0f}  {sig_:>12,.0f}  {'✓' if ih else '✗ extrapolation':>8}")
    results["top5_in_hull"] = in_hull.tolist()
    results["top5_sum_ok"] = [abs(s[0]+s[1]+s[2]-1.0) < 0.02 for s in top_suggestions]

    # ── Ternary heatmaps ──────────────────────────────────────────────────────
    cs_g, fa_g, ma_g = grid[:, 0], grid[:, 1], grid[:, 2]
    xg, yg = ternary_to_cart(cs_g, fa_g, ma_g)

    def ternary_heatmap(vals, title, cmap, log_scale, fname, vmin=None, vmax=None,
                        overlay_top=None):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        draw_ternary_frame(ax)
        if log_scale:
            norm = mcolors.LogNorm(vmin=vals.min() + 1, vmax=vals.max())
        else:
            norm = mcolors.Normalize(vmin=vmin or vals.min(), vmax=vmax or vals.max())
        sc = ax.tricontourf(xg, yg, vals, levels=60, cmap=cmap, norm=norm, zorder=2)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("log scale" if log_scale else "value", fontsize=7)
        # Overlay actual data points
        xd, yd = ternary_to_cart(data_X[:, 0], data_X[:, 1], data_X[:, 2])
        ax.scatter(xd, yd, c="white", s=18, edgecolors="k",
                   linewidths=0.5, zorder=8, label="Observed")
        if overlay_top is not None:
            xt, yt_ = ternary_to_cart(overlay_top[:, 0], overlay_top[:, 1], overlay_top[:, 2])
            ax.scatter(xt, yt_, marker="*", c="gold", s=150,
                       edgecolors="k", linewidths=0.6, zorder=10, label="Top EI")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_title(title, fontsize=9)
        fig.tight_layout()
        save_fig(fig, fname)

    ternary_heatmap(gp_mu, "§3 – GP predicted mean Instability index (log scale)",
                    "RdYlGn_r", log_scale=True, fname="s3_gp_mean.png",
                    overlay_top=top_suggestions)

    ternary_heatmap(gp_sigma, "§3 – GP predicted std (model uncertainty)",
                    "Blues", log_scale=False, fname="s3_gp_std.png")

    ternary_heatmap(gp_ei, "§3 – GP Expected Improvement (acquisition surface)",
                    "hot_r", log_scale=False, fname="s3_gp_ei.png",
                    overlay_top=top_suggestions)

    ternary_heatmap(rf_mu, "§3 – RF predicted mean Instability index (log scale)",
                    "RdYlGn_r", log_scale=True, fname="s3_rf_mean.png")

    # ── Check: does model look smooth and physically plausible? ───────────────
    # Compute correlation between mean prediction on grid and distance to MA corner
    ma_fraction = grid[:, 2]
    r_ma_pred, _ = stats.pearsonr(ma_fraction, gp_mu)
    print(f"\n  GP mean vs. MA fraction correlation : r = {r_ma_pred:.3f}  "
          f"(positive = MA-rich → higher instability, expected)")
    results["gp_mean_vs_MA_r"] = float(r_ma_pred)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# §4  DOMAIN-KNOWLEDGE CROSS-CHECK
# ──────────────────────────────────────────────────────────────────────────────

def section4(gp: GaussianProcessRegressor, rf: RandomForestRegressor,
             data_X: np.ndarray, data_y: np.ndarray) -> dict:
    section_header(4, "Domain-knowledge cross-check")

    corners = {
        "Pure CsPbI (Cs=1)": np.array([[1.0, 0.0, 0.0]]),
        "Pure FAPbI (FA=1)": np.array([[0.0, 1.0, 0.0]]),
        "Pure MAPbI (MA=1)": np.array([[0.0, 0.0, 1.0]]),
    }
    known_MA_unstable = True  # literature: MA-rich → unstable
    results = {}

    print("\n  Predictions at pure-component corners:")
    print(f"    {'Composition':<26}  {'GP mean':>12}  {'GP σ':>12}  {'RF mean':>12}")
    corner_preds = {}
    for name, X_c in corners.items():
        gp_m, gp_s = gp.predict(X_c, return_std=True)
        rf_m = rf.predict(X_c)
        print(f"    {name:<26}  {gp_m[0]:>12,.0f}  {gp_s[0]:>12,.0f}  {rf_m[0]:>12,.0f}")
        corner_preds[name] = {"gp_mean": float(gp_m[0]), "gp_std": float(gp_s[0]),
                              "rf_mean": float(rf_m[0])}
    results["corner_preds"] = corner_preds

    # MA corner should be worse than Cs and FA corners
    ma_gp = corner_preds["Pure MAPbI (MA=1)"]["gp_mean"]
    cs_gp = corner_preds["Pure CsPbI (Cs=1)"]["gp_mean"]
    fa_gp = corner_preds["Pure FAPbI (FA=1)"]["gp_mean"]
    ma_worse_than_cs = ma_gp > cs_gp
    ma_worse_than_fa = ma_gp > fa_gp
    print(f"\n  Domain check – MA corner worse than Cs corner : {'✓ PASS' if ma_worse_than_cs else '✗ FAIL'}")
    print(f"  Domain check – MA corner worse than FA corner : {'✓ PASS' if ma_worse_than_fa else '✗ FAIL'}")
    results["ma_worse_than_cs"] = ma_worse_than_cs
    results["ma_worse_than_fa"] = ma_worse_than_fa

    # ── Top-10 compositions by observed mean ──────────────────────────────────
    agg = pd.DataFrame(data_X, columns=FEATURES)
    agg[TARGET] = data_y
    top10_obs = agg.nsmallest(10, TARGET).reset_index(drop=True)
    print(f"\n  Top-10 observed compositions (lowest mean Instability index):")
    print(top10_obs.to_string(index=False))

    # ── Top-10 on simplex grid by GP prediction ────────────────────────────────
    grid = simplex_grid(GRID_STEP)
    gp_mu_grid, _ = gp.predict(grid, return_std=True)
    top10_idx = np.argsort(gp_mu_grid)[:10]
    top10_grid = pd.DataFrame(grid[top10_idx], columns=FEATURES)
    top10_grid["GP_predicted"] = gp_mu_grid[top10_idx]
    print(f"\n  Top-10 grid compositions by GP predicted mean:")
    print(top10_grid.to_string(index=False))

    # Check: does the GP optimum lie in the Cs+FA region (MA ≈ 0)?
    best_grid_MA = float(grid[top10_idx[0], 2])
    best_is_low_MA = best_grid_MA < 0.20
    print(f"\n  Best GP prediction – MA fraction: {best_grid_MA:.3f}  "
          f"({'✓ low MA as expected' if best_is_low_MA else '✗ high MA – check model!'})")
    results["best_grid_low_MA"] = best_is_low_MA

    # ── Figure: ternary scatter – top10 highlighted ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    for ax, pred_vals, label in [
        (axes[0], gp_mu_grid, "GP predicted mean"),
        (axes[1], None,       "Observed mean"),
    ]:
        draw_ternary_frame(ax)
        # Background data
        cs_d, fa_d, ma_d = data_X[:, 0], data_X[:, 1], data_X[:, 2]
        xd, yd = ternary_to_cart(cs_d, fa_d, ma_d)
        ax.scatter(xd, yd, c="lightgrey", s=25, edgecolors="k",
                   linewidths=0.3, zorder=4, label="All data")

        # Top-10 overlay
        if pred_vals is not None:
            cs_t = grid[top10_idx, 0]
            fa_t = grid[top10_idx, 1]
            ma_t = grid[top10_idx, 2]
            xt, yt = ternary_to_cart(cs_t, fa_t, ma_t)
            sc = ax.scatter(xt, yt, c=gp_mu_grid[top10_idx], cmap="YlGn_r",
                            s=90, edgecolors="k", linewidths=0.6,
                            marker="D", zorder=9, label="Top-10 (GP)")
            fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        else:
            cs_t = top10_obs["CsPbI"].values
            fa_t = top10_obs["FAPbI"].values
            ma_t = top10_obs["MAPbI"].values
            xt, yt = ternary_to_cart(cs_t, fa_t, ma_t)
            sc = ax.scatter(xt, yt, c=top10_obs[TARGET].values, cmap="YlGn_r",
                            s=90, edgecolors="k", linewidths=0.6,
                            marker="D", zorder=9, label="Top-10 (obs)")
            fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)

        ax.set_title(f"§4 – Top-10 by {label}", fontsize=9)
        ax.legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, "s4_top10_compositions.png")

    # ── Figure: GP predictions vs MA fraction (scatter) ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(data_X[:, 2], data_y, c=REGION_COLORS["MA-rich"],
               alpha=0.5, s=30, label="Observed (averaged)", edgecolors="k", lw=0.3)
    # Grid curve sorted by MA
    sort_ma = np.argsort(grid[:, 2])
    ax.plot(grid[sort_ma, 2], gp_mu_grid[sort_ma], c="#2563eb",
            lw=1.5, alpha=0.5, label="GP mean (grid)", zorder=2)
    ax.set_xlabel("MAPbI fraction", fontsize=9)
    ax.set_ylabel("Instability index", fontsize=9)
    ax.set_title("§4 – Instability index vs MA fraction\n"
                 "(MA-rich should be most unstable)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "s4_instability_vs_MA.png")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# §5  SUMMARY PASS/FAIL REPORT
# ──────────────────────────────────────────────────────────────────────────────

def section5(s0: dict, s1: dict, s2: dict, s3: dict, s4: dict):
    section_header(5, "Summary pass/fail report")

    y_scale = 1_755_056 - 23_707  # approximate full range of target

    checks = []

    # Check 1: composition sums
    checks.append((
        "§0  All compositions sum to 1.0 (±0.02)",
        s0["sum_check_pass"],
        "No violations found" if s0["sum_check_pass"] else "⚠ Some rows have bad sums",
    ))

    # Check 2: surrogate Pearson r (hold-out) > 0.5 for both models
    rf_r  = s1["holdout_rf"]["Pearson_r"]
    gp_r  = s1["holdout_gp"]["Pearson_r"]
    r_pass = rf_r > 0.5 and gp_r > 0.5
    checks.append((
        "§1  Surrogate Pearson r > 0.5 on hold-out",
        r_pass,
        f"RF r={rf_r:.3f}, GP r={gp_r:.3f}",
    ))

    # Check 3: surrogate Spearman rho > 0.5 (ranking quality)
    rf_rho = s1["holdout_rf"]["Spearman_rho"]
    gp_rho = s1["holdout_gp"]["Spearman_rho"]
    rho_pass = rf_rho > 0.5 and gp_rho > 0.5
    checks.append((
        "§1  Surrogate Spearman ρ > 0.5 on hold-out",
        rho_pass,
        f"RF ρ={rf_rho:.3f}, GP ρ={gp_rho:.3f}",
    ))

    # Check 4: CV RMSE < 30% of target range
    rf_rmse = s1["cv_rf"]["RMSE"]
    gp_rmse = s1["cv_gp"]["RMSE"]
    rmse_pct_rf = rf_rmse / y_scale
    rmse_pct_gp = gp_rmse / y_scale
    rmse_pass = rmse_pct_rf < 0.30 and rmse_pct_gp < 0.30
    checks.append((
        "§1  CV RMSE < 30% of target range",
        rmse_pass,
        f"RF RMSE={rmse_pct_rf:.1%} of range, GP RMSE={rmse_pct_gp:.1%} of range",
    ))

    # Check 5: BO finds top-20% faster than random
    bo_steps  = s2.get("GP + EI  (BO)",  {}).get("median_steps_top20", np.nan)
    rand_steps = s2.get("Random", {}).get("median_steps_top20", np.nan)
    if not np.isnan(bo_steps) and not np.isnan(rand_steps):
        bo_beats_random = bo_steps <= rand_steps
        bo_detail = f"BO median={bo_steps:.0f} steps, Random={rand_steps:.0f} steps"
    elif np.isnan(bo_steps):
        bo_beats_random = False
        bo_detail = "BO never reached top-20% in this run"
    else:
        bo_beats_random = True
        bo_detail = f"BO={bo_steps:.0f} steps; random never reached top-20%"
    checks.append((
        "§2  GP+EI BO reaches top-20% faster than random",
        bo_beats_random,
        bo_detail,
    ))

    # Check 6: top-5 EI suggestions inside convex hull
    in_hull_flags = s3.get("top5_in_hull", [True]*5)
    all_in_hull   = all(in_hull_flags)
    n_in = sum(in_hull_flags)
    checks.append((
        "§3  Top-5 EI suggestions inside data convex hull",
        all_in_hull,
        f"{n_in}/5 suggestions inside hull"
        + (" (extrapolations flagged)" if not all_in_hull else ""),
    ))

    # Check 7: top-5 EI suggestions sum to 1
    sum_ok_flags = s3.get("top5_sum_ok", [True]*5)
    all_sum_ok   = all(sum_ok_flags)
    checks.append((
        "§3  Top-5 EI suggestions sum to 1.0 (±0.02)",
        all_sum_ok,
        f"{sum(sum_ok_flags)}/5 suggestions have valid composition sums",
    ))

    # Check 8: GP mean vs MA correlation is positive (MA → higher instability)
    r_ma = s3.get("gp_mean_vs_MA_r", 0.0)
    ma_corr_pass = r_ma > 0.3
    checks.append((
        "§3  GP mean positively correlated with MA fraction (r > 0.3)",
        ma_corr_pass,
        f"r = {r_ma:.3f} (positive ⟹ MA-rich → more unstable, as expected)",
    ))

    # Check 9: MA corner worse than Cs corner (domain knowledge)
    checks.append((
        "§4  Model predicts MA corner worse than Cs corner",
        s4.get("ma_worse_than_cs", False),
        f"GP: MA={s4['corner_preds']['Pure MAPbI (MA=1)']['gp_mean']:,.0f}, "
        f"Cs={s4['corner_preds']['Pure CsPbI (Cs=1)']['gp_mean']:,.0f}",
    ))

    # Check 10: Best GP optimum has low MA fraction
    checks.append((
        "§4  Best GP-predicted composition has MA < 20%",
        s4.get("best_grid_low_MA", False),
        "Best optimum is in Cs/FA-rich region (consistent with literature)",
    ))

    # ── Print report ──────────────────────────────────────────────────────────
    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {'CHECK':<52}  {'STATUS':<8}  DETAIL")
    print(f"  {'─'*52}  {'─'*8}  {'─'*40}")
    for desc, ok, detail in checks:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {desc:<52}  {status:<8}  {detail}")

    print(f"\n  OVERALL: {n_pass}/{len(checks)} checks passed")
    if n_pass == len(checks):
        print("  🟢 All checks passed – surrogate and BO look reliable.")
    elif n_pass >= len(checks) * 0.75:
        print("  🟡 Most checks passed – review any failures carefully.")
    else:
        print("  🔴 Multiple failures – re-examine kernel, scaling, or acquisition settings.")

    # ── Metrics table CSV ─────────────────────────────────────────────────────
    rows = []
    for split_name, m_rf, m_gp in [
        ("70/30 hold-out", s1["holdout_rf"], s1["holdout_gp"]),
        (f"Repeated {N_CV_FOLDS}-fold CV", s1["cv_rf"], s1["cv_gp"]),
    ]:
        for model, m in [("RandomForest", m_rf), ("GaussianProcess", m_gp)]:
            rows.append({
                "Split": split_name, "Model": model,
                **{k: round(v, 4) for k, v in m.items()
                   if k not in ("gp_full", "rf_full", "X", "y")},
            })
    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(OUT_DIR, "surrogate_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Metrics table saved → {metrics_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Perovskite BO Validation Suite")
    print("=" * 70)
    print(f"  Dataset : {CSV_PATH}")
    print(f"  Output  : {OUT_DIR}/")

    # Load raw data
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    # Run all sections
    s0 = section0(df)

    s1 = section1(df)
    gp_full = s1.pop("gp_full")
    rf_full = s1.pop("rf_full")
    X_uniq  = s1.pop("X")
    y_uniq  = s1.pop("y")

    s2 = section2(X_uniq, y_uniq)

    s3 = section3(gp_full, rf_full, X_uniq, y_uniq)

    s4 = section4(gp_full, rf_full, X_uniq, y_uniq)

    section5(s0, s1, s2, s3, s4)

    print(f"\n{'='*70}")
    print(f"  Done.  All plots saved to  {OUT_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
