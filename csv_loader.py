"""
Phase 2 — CSV Loader
Reads CSV data, infers parameter types, and extracts sensible defaults.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from parameter_config import AllowedSubRange, ObjectiveConfig, ParameterConfig, ParameterType, StudyConfig


def load_csv(path: str) -> pd.DataFrame:
    """Read a CSV file and return a cleaned DataFrame."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Cannot read CSV file '{path}': {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_column_type(series: pd.Series, categorical_threshold: int = 8) -> ParameterType:
    """
    Infer the ParameterType of a pandas Series.

    Decision order
    --------------
    1. BOOL   — dtype bool, or ≤ 2 unique values all in {0, 1, True, False}
    2. CATEGORICAL — object / string dtype
    3. CATEGORICAL — numeric but n_unique ≤ categorical_threshold
    4. INT    — numeric, all non-null values equal their integer cast
    5. FLOAT  — everything else
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return ParameterType.FLOAT

    # 1. Bool
    if series.dtype == bool or pd.api.types.is_bool_dtype(series):
        return ParameterType.BOOL
    unique_vals = set(non_null.unique())
    if unique_vals <= {0, 1, 0.0, 1.0, True, False} and len(unique_vals) <= 2:
        str_vals = {str(v).strip().lower() for v in unique_vals}
        if str_vals <= {"0", "1", "true", "false"}:
            return ParameterType.BOOL

    # 2. Object / string → categorical
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return ParameterType.CATEGORICAL

    # 3. Few unique numeric values → categorical
    n_unique = non_null.nunique()
    if n_unique <= categorical_threshold:
        return ParameterType.CATEGORICAL

    # 4. All values equal their int cast → INT
    try:
        as_int = non_null.values.astype(int).astype(float)
        if np.allclose(non_null.values.astype(float), as_int, atol=0, rtol=0):
            return ParameterType.INT
    except (ValueError, TypeError, OverflowError):
        pass

    # 5. Default float
    return ParameterType.FLOAT


def extract_param_defaults(
    df: pd.DataFrame,
    result_columns: List[str],
    categorical_threshold: int = 8,
) -> List[ParameterConfig]:
    """
    Build a ParameterConfig for each non-result column in *df*.

    - INT / FLOAT : full_min/max come from the data;
                    allowed_subranges defaults to [full_min, full_max].
    - CATEGORICAL  : all_choices = sorted unique string values; all allowed.
    - BOOL         : fixed_value = None (optimise both), choices = [True, False].
    """
    configs: List[ParameterConfig] = []
    for col in df.columns:
        if col in result_columns:
            continue

        series = df[col]
        ptype = infer_column_type(series, categorical_threshold)
        non_null = series.dropna()

        if ptype in (ParameterType.INT, ParameterType.FLOAT):
            full_min = float(non_null.min())
            full_max = float(non_null.max())
            # Guard against zero-width ranges (single unique value)
            if full_min == full_max:
                full_max = full_min + (1.0 if ptype == ParameterType.INT else 0.01)
            configs.append(
                ParameterConfig(
                    name=col,
                    ptype=ptype,
                    enabled=True,
                    full_min=full_min,
                    full_max=full_max,
                    allowed_subranges=[AllowedSubRange(full_min, full_max)],
                )
            )

        elif ptype == ParameterType.CATEGORICAL:
            choices = sorted(str(v) for v in non_null.unique())
            configs.append(
                ParameterConfig(
                    name=col,
                    ptype=ptype,
                    enabled=True,
                    all_choices=choices,
                    allowed_choices=choices[:],
                )
            )

        else:  # BOOL
            configs.append(
                ParameterConfig(
                    name=col,
                    ptype=ParameterType.BOOL,
                    enabled=True,
                    all_choices=["True", "False"],
                    allowed_choices=["True", "False"],
                    fixed_value=None,
                )
            )

    return configs


def load_trials_from_csv(df: pd.DataFrame, config: StudyConfig) -> List[dict]:
    """
    Convert rows in *df* that have non-null values in ALL objective columns
    into dicts suitable for adding to an Optuna study as historical trials.

    Returns
    -------
    list of {"params": {col: val, ...}, "values": [float, ...]}
    Values list order matches config.objectives order.
    Safe to call multiple times — the caller deduplicates before adding.
    """
    obj_cols = [o.column_name for o in config.objectives]
    param_names = [p.name for p in config.parameters if p.enabled]

    results: List[dict] = []
    for _, row in df.iterrows():
        # Skip rows with any missing objective value
        if any(pd.isna(row.get(c)) for c in obj_cols):
            continue
        params = {name: row[name] for name in param_names if name in row.index}
        values = [float(row[c]) for c in obj_cols]
        results.append({"params": params, "values": values})

    return results


def append_rows_to_csv(path: str, rows: List[dict]) -> None:
    """
    Append *rows* (list of flat dicts) to the CSV at *path*.
    Creates the file if it does not exist.
    Used to persist newly submitted batch results back to the experiment CSV.
    """
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    try:
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    except FileNotFoundError:
        combined = new_df
    combined.to_csv(path, index=False)
