"""
Phase 5 — Optuna Builder
Translates a StudyConfig into a live, persistent Optuna study and provides
the ask/tell batch interface used by the GUI worker.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import FrozenTrial, TrialState

from parameter_config import ParameterConfig, ParameterType, StudyConfig
from sampler_utils import (
    _build_weighted_list,
    subrange_suggest_float,
    subrange_suggest_int,
)

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose per-trial logging in the GUI
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Study creation
# ──────────────────────────────────────────────────────────────────────────────

def build_study(
    config: StudyConfig,
    storage_path: str,
    study_name: str,
) -> optuna.Study:
    """
    Create (or reload) a persistent Optuna study backed by a SQLite database.

    Parameters
    ----------
    config       : StudyConfig driving sampler choice and direction(s).
    storage_path : Absolute path to the SQLite .db file.
    study_name   : Unique name for this study.

    Notes
    -----
    *load_if_exists=True* means this function is safe to call on every
    application start — it will silently reuse an existing study.
    """
    storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")

    sampler_map: Dict[str, optuna.samplers.BaseSampler] = {
        "TPE":    optuna.samplers.TPESampler(seed=42, multivariate=True),
        "NSGAII": optuna.samplers.NSGAIISampler(seed=42),
        "Random": optuna.samplers.RandomSampler(seed=42),
    }
    sampler = sampler_map.get(config.sampler_name, optuna.samplers.TPESampler(seed=42))

    if len(config.objectives) == 1:
        study = optuna.create_study(
            direction=config.objectives[0].direction,
            storage=storage,
            study_name=study_name,
            sampler=sampler,
            load_if_exists=True,
        )
    else:
        directions = [o.direction for o in config.objectives]
        study = optuna.create_study(
            directions=directions,
            storage=storage,
            study_name=study_name,
            sampler=sampler,
            load_if_exists=True,
        )

    return study


# ──────────────────────────────────────────────────────────────────────────────
# Distribution dict (matches suggest calls in build_objective_fn)
# ──────────────────────────────────────────────────────────────────────────────

def build_distributions(config: StudyConfig) -> Dict[str, BaseDistribution]:
    """
    Return an Optuna distributions dict that matches exactly the parameter keys
    produced by _suggest_params().  Required for study.ask() and FrozenTrial
    construction.
    """
    dists: Dict[str, BaseDistribution] = {}

    for param in config.parameters:
        if not param.enabled:
            continue
        name = param.name

        if param.ptype == ParameterType.INT:
            step = param.step or 1
            subranges = param.allowed_subranges
            if len(subranges) == 1:
                dists[name] = IntDistribution(int(subranges[0].low), int(subranges[0].high), step=step)
            else:
                sizes = [max(1, (int(r.high) - int(r.low)) // step + 1) for r in subranges]
                weighted = _build_weighted_list(list(range(len(subranges))), sizes)
                dists[f"{name}__range_idx"] = CategoricalDistribution(weighted)
                for idx, r in enumerate(subranges):
                    dists[f"{name}__in_range_{idx}"] = IntDistribution(
                        int(r.low), int(r.high), step=step
                    )

        elif param.ptype == ParameterType.FLOAT:
            subranges = param.allowed_subranges
            if len(subranges) == 1:
                dists[name] = FloatDistribution(subranges[0].low, subranges[0].high)
            else:
                widths = [max(1e-12, r.high - r.low) for r in subranges]
                total = sum(widths)
                int_weights = [max(1, round(w / total * 1000)) for w in widths]
                weighted = _build_weighted_list(list(range(len(subranges))), int_weights)
                dists[f"{name}__range_idx"] = CategoricalDistribution(weighted)
                for idx, r in enumerate(subranges):
                    dists[f"{name}__in_range_{idx}"] = FloatDistribution(r.low, r.high)

        elif param.ptype == ParameterType.CATEGORICAL:
            choices = param.allowed_choices if param.allowed_choices else param.all_choices
            dists[name] = CategoricalDistribution(choices)

        elif param.ptype == ParameterType.BOOL:
            if param.fixed_value is None:          # optimise → treat as categorical
                dists[name] = CategoricalDistribution([True, False])
            # fixed_value is not a free parameter — omit from distributions

    return dists


# ──────────────────────────────────────────────────────────────────────────────
# Internal parameter suggestion (used by objective fn and historical loading)
# ──────────────────────────────────────────────────────────────────────────────

def _suggest_params(trial: optuna.Trial, config: StudyConfig) -> dict:
    """Register all enabled parameters with *trial* and return a params dict."""
    params: dict = {}
    for param in config.parameters:
        if not param.enabled:
            continue
        name = param.name

        if param.ptype == ParameterType.INT:
            step = param.step or 1
            params[name] = subrange_suggest_int(trial, name, param.allowed_subranges, step)

        elif param.ptype == ParameterType.FLOAT:
            params[name] = subrange_suggest_float(trial, name, param.allowed_subranges)

        elif param.ptype == ParameterType.CATEGORICAL:
            choices = param.allowed_choices if param.allowed_choices else param.all_choices
            params[name] = trial.suggest_categorical(name, choices)

        elif param.ptype == ParameterType.BOOL:
            if param.fixed_value is None:
                params[name] = trial.suggest_categorical(name, [True, False])
            else:
                params[name] = param.fixed_value  # injected directly, not registered

    return params


def build_objective_fn(config: StudyConfig) -> Callable:
    """
    Return a bare objective function skeleton for use with study.optimize().

    In batch/GUI mode the measurement is entered externally via tell_batch(),
    so this function's body is never called during normal GUI operation.
    It is provided here for headless / scripting use-cases where the caller
    supplies a *measure* callable.
    """
    def objective(trial: optuna.Trial) -> float:
        _suggest_params(trial, config)
        raise NotImplementedError(
            "Direct objective calls are not supported in batch mode. "
            "Use ask_batch() / tell_batch() instead, or override this function "
            "with your measurement logic."
        )
    return objective


# ──────────────────────────────────────────────────────────────────────────────
# Historical CSV trial loading
# ──────────────────────────────────────────────────────────────────────────────

def load_historical_trials(
    study: optuna.Study,
    trial_dicts: List[dict],
    config: StudyConfig,
) -> Tuple[int, int]:
    """
    Add completed trials from *trial_dicts* to *study* as FrozenTrial objects.

    Duplicate detection
    -------------------
    Before adding, the function builds a set of existing (param, value)
    fingerprints.  Rows with identical parameter values are skipped, making it
    safe to call after every CSV reload.

    Returns
    -------
    (added, skipped)
    """
    distributions = build_distributions(config)

    # Collect fingerprints of already-loaded trials
    existing_fingerprints = {
        _param_fingerprint(t.params)
        for t in study.trials
    }

    added = skipped = 0

    for data in trial_dicts:
        raw_params: dict = data["params"]
        values: List[float] = data["values"]

        fp = _param_fingerprint(raw_params)
        if fp in existing_fingerprints:
            skipped += 1
            continue

        # Translate raw CSV params into the internal Optuna key space
        internal_params = _csv_params_to_internal(raw_params, config)

        # Only keep distributions that appear in internal_params
        trial_dists = {k: v for k, v in distributions.items() if k in internal_params}
        trial_params = {k: internal_params[k] for k in trial_dists}

        trial_number = len(study.trials)

        # Optuna 4.x FrozenTrial requires trial_id; pass -1 so storage assigns one.
        if len(values) == 1:
            frozen = FrozenTrial(
                number=trial_number,
                trial_id=-1,
                state=TrialState.COMPLETE,
                value=values[0],
                values=None,
                datetime_start=datetime.now(),
                datetime_complete=datetime.now(),
                params=trial_params,
                distributions=trial_dists,
                user_attrs={},
                system_attrs={},
                intermediate_values={},
            )
        else:
            frozen = FrozenTrial(
                number=trial_number,
                trial_id=-1,
                state=TrialState.COMPLETE,
                value=None,
                values=values,
                datetime_start=datetime.now(),
                datetime_complete=datetime.now(),
                params=trial_params,
                distributions=trial_dists,
                user_attrs={},
                system_attrs={},
                intermediate_values={},
            )

        try:
            study.add_trial(frozen)
            existing_fingerprints.add(fp)
            added += 1
        except Exception as exc:
            logger.warning("Could not add trial (params=%s): %s", raw_params, exc)
            skipped += 1

    logger.info("Historical trials: %d added, %d skipped.", added, skipped)
    return added, skipped


def _param_fingerprint(params: dict) -> str:
    """Stable string key for a parameter dict (for duplicate detection)."""
    return str(sorted((k, str(v)) for k, v in params.items()))


def _csv_params_to_internal(raw_params: dict, config: StudyConfig) -> dict:
    """
    Convert raw CSV column values to the internal Optuna key space.

    For parameters with a single allowed sub-range the key equals the column
    name.  For multi-sub-range parameters the two-level keys are used
    ({name}__range_idx and {name}__in_range_{idx}).
    """
    internal: dict = {}

    for param in config.parameters:
        if not param.enabled or param.name not in raw_params:
            continue

        name = param.name
        raw = raw_params[name]

        if param.ptype == ParameterType.INT:
            subranges = param.allowed_subranges
            step = param.step or 1
            ival = int(float(raw))
            if len(subranges) == 1:
                internal[name] = ival
            else:
                chosen_idx = 0
                for idx, r in enumerate(subranges):
                    if int(r.low) <= ival <= int(r.high):
                        chosen_idx = idx
                        break
                sizes = [max(1, (int(r.high) - int(r.low)) // step + 1) for r in subranges]
                weighted = _build_weighted_list(list(range(len(subranges))), sizes)
                internal[f"{name}__range_idx"] = chosen_idx
                internal[f"{name}__in_range_{chosen_idx}"] = ival

        elif param.ptype == ParameterType.FLOAT:
            subranges = param.allowed_subranges
            fval = float(raw)
            if len(subranges) == 1:
                internal[name] = fval
            else:
                chosen_idx = 0
                for idx, r in enumerate(subranges):
                    if r.low <= fval <= r.high:
                        chosen_idx = idx
                        break
                widths = [max(1e-12, r.high - r.low) for r in subranges]
                total = sum(widths)
                int_weights = [max(1, round(w / total * 1000)) for w in widths]
                weighted = _build_weighted_list(list(range(len(subranges))), int_weights)
                internal[f"{name}__range_idx"] = chosen_idx
                internal[f"{name}__in_range_{chosen_idx}"] = fval

        elif param.ptype == ParameterType.CATEGORICAL:
            internal[name] = str(raw)

        elif param.ptype == ParameterType.BOOL:
            if param.fixed_value is None:
                # Store as bool; handle both Python bool and 0/1 strings
                if isinstance(raw, bool):
                    internal[name] = raw
                else:
                    internal[name] = str(raw).strip().lower() in {"true", "1", "yes"}

    return internal


# ──────────────────────────────────────────────────────────────────────────────
# Batch ask / tell
# ──────────────────────────────────────────────────────────────────────────────

def ask_batch(
    study: optuna.Study,
    config: StudyConfig,
    batch_size: int,
) -> List[optuna.Trial]:
    """
    Ask Optuna for *batch_size* parameter suggestions without running any
    objective.  Returns a list of Trial objects (each has .number and .params).

    Note: in Optuna 4.x, study.ask() creates trials in RUNNING state (not
    WAITING).  Use tell_batch() with the trial numbers to complete them.
    """
    distributions = build_distributions(config)
    trials: List[optuna.Trial] = []
    for _ in range(batch_size):
        trial = study.ask(distributions)
        trials.append(trial)
    return trials


def tell_batch(
    study: optuna.Study,
    trial_numbers: List[int],
    results: List[List[float]],
) -> None:
    """
    Report measured lab results back to Optuna for the given trial numbers.

    Uses study.tell(trial_number, values) directly — compatible with Optuna 4.x
    where study.ask() puts trials in RUNNING state and study.tell() accepts an
    integer trial number.
    """
    for trial_number, values in zip(trial_numbers, results):
        try:
            # study.tell accepts a Trial object or an integer trial number.
            # Scalar for single-objective, sequence for multi-objective.
            if len(values) == 1:
                study.tell(trial_number, values[0])
            else:
                study.tell(trial_number, values)
        except Exception as exc:
            logger.warning("Could not tell result for trial %d: %s", trial_number, exc)


# ──────────────────────────────────────────────────────────────────────────────
# Pareto front helper
# ──────────────────────────────────────────────────────────────────────────────

def get_pareto_front(study: optuna.Study) -> List[FrozenTrial]:
    """Return the best trial(s): Pareto front for multi-obj, best_trial for single-obj."""
    try:
        if hasattr(study, "directions") and len(study.directions) > 1:
            return optuna.study.get_pareto_front_trials(study)
        else:
            return [study.best_trial]
    except Exception:
        return []
