"""
Phase 4 — Sampler Utilities
Dead-region-aware parameter suggestion helpers for Optuna.

Strategy
--------
For parameters with multiple allowed sub-ranges, we use a two-level suggestion:
  1. Suggest WHICH sub-range to use (categorical, weighted by size).
  2. Suggest the value within that sub-range (int/float).

This keeps the internal representation transparent to TPE's Parzen estimator
and avoids slow rejection sampling.
"""
from __future__ import annotations

from typing import List, Tuple

import optuna

from parameter_config import AllowedSubRange


# ──────────────────────────────────────────────────────────────────────────────
# Public suggestion helpers
# ──────────────────────────────────────────────────────────────────────────────

def subrange_suggest_int(
    trial: optuna.Trial,
    name: str,
    subranges: List[AllowedSubRange],
    step: int = 1,
) -> int:
    """
    Suggest an integer from the union of *subranges*.

    If there is only one sub-range, behaves exactly like trial.suggest_int().
    With multiple sub-ranges a categorical choice picks the sub-range (weighted
    by the number of integers it contains), then suggest_int is called within
    the chosen range.
    """
    if not subranges:
        raise ValueError(f"No allowed sub-ranges for parameter '{name}'.")

    if len(subranges) == 1:
        return trial.suggest_int(name, int(subranges[0].low), int(subranges[0].high), step=step)

    # Weight each range by its integer count
    sizes = [max(1, (int(r.high) - int(r.low)) // step + 1) for r in subranges]
    weighted_indices = _build_weighted_list(list(range(len(subranges))), sizes)

    chosen_idx = trial.suggest_categorical(f"{name}__range_idx", weighted_indices)
    chosen = subranges[chosen_idx]
    return trial.suggest_int(
        f"{name}__in_range_{chosen_idx}",
        int(chosen.low),
        int(chosen.high),
        step=step,
    )


def subrange_suggest_float(
    trial: optuna.Trial,
    name: str,
    subranges: List[AllowedSubRange],
) -> float:
    """
    Suggest a float from the union of *subranges*.

    Same two-level strategy as subrange_suggest_int, weighted by interval width.
    """
    if not subranges:
        raise ValueError(f"No allowed sub-ranges for parameter '{name}'.")

    if len(subranges) == 1:
        return trial.suggest_float(name, subranges[0].low, subranges[0].high)

    widths = [max(1e-12, r.high - r.low) for r in subranges]
    total = sum(widths)
    # Scale to integers with resolution 1000 for the weighted categorical list
    int_weights = [max(1, round(w / total * 1000)) for w in widths]
    weighted_indices = _build_weighted_list(list(range(len(subranges))), int_weights)

    chosen_idx = trial.suggest_categorical(f"{name}__range_idx", weighted_indices)
    chosen = subranges[chosen_idx]
    return trial.suggest_float(
        f"{name}__in_range_{chosen_idx}",
        chosen.low,
        chosen.high,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dead-region computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_allowed_subranges(
    full_min: float,
    full_max: float,
    dead_regions: List[Tuple[float, float]],
) -> List[AllowedSubRange]:
    """
    Return the complement of *dead_regions* within [full_min, full_max].

    Dead regions are merged and clipped before inversion, so overlapping or
    out-of-bounds inputs are handled gracefully.

    Raises
    ------
    ValueError if the dead regions cover the entire [full_min, full_max] range.
    """
    if not dead_regions:
        return [AllowedSubRange(full_min, full_max)]

    # Clip each dead region to the full range and drop empty ones
    clipped: List[Tuple[float, float]] = []
    for lo, hi in dead_regions:
        lo = max(lo, full_min)
        hi = min(hi, full_max)
        if lo < hi:
            clipped.append((lo, hi))

    if not clipped:
        return [AllowedSubRange(full_min, full_max)]

    # Sort and merge overlapping dead regions
    clipped.sort(key=lambda r: r[0])
    merged: List[Tuple[float, float]] = []
    for lo, hi in clipped:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    # Build complement within [full_min, full_max]
    allowed: List[AllowedSubRange] = []
    cursor = full_min
    for dead_lo, dead_hi in merged:
        if cursor < dead_lo:
            allowed.append(AllowedSubRange(cursor, dead_lo))
        cursor = dead_hi
    if cursor < full_max:
        allowed.append(AllowedSubRange(cursor, full_max))

    if not allowed:
        raise ValueError(
            "Dead regions cover the entire parameter range. No allowed values remain."
        )

    return allowed


def validate_subranges(
    subranges: List[AllowedSubRange],
    full_min: float,
    full_max: float,
) -> List[str]:
    """
    Return a list of human-readable error messages for any problems found.
    An empty list means the configuration is valid.
    """
    errors: List[str] = []

    if not subranges:
        errors.append("No allowed sub-ranges defined.")
        return errors

    for i, r in enumerate(subranges):
        label = f"Sub-range {i + 1}"
        if r.low >= r.high:
            errors.append(f"{label}: low ({r.low}) must be strictly less than high ({r.high}).")
        if r.low < full_min:
            errors.append(f"{label}: low ({r.low}) is below the full minimum ({full_min}).")
        if r.high > full_max:
            errors.append(f"{label}: high ({r.high}) exceeds the full maximum ({full_max}).")

    # Check pairwise overlaps on sorted copy
    sorted_r = sorted(subranges, key=lambda r: r.low)
    for i in range(len(sorted_r) - 1):
        if sorted_r[i].high > sorted_r[i + 1].low:
            errors.append(
                f"Sub-ranges {i + 1} and {i + 2} overlap "
                f"({sorted_r[i].high} > {sorted_r[i + 1].low})."
            )

    return errors


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_weighted_list(indices: List[int], weights: List[int]) -> List[int]:
    """Return a flat list where index i appears weights[i] times."""
    result: List[int] = []
    for idx, w in zip(indices, weights):
        result.extend([idx] * w)
    return result
