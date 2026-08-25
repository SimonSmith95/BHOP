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

from parameter_config import ParameterConfig, ParameterConstraint, ParameterType, StudyConfig
from sampler_utils import (
    _build_weighted_list,
    subrange_suggest_float,
    subrange_suggest_int,
)

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose per-trial logging in the GUI
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Constraint helpers
# ──────────────────────────────────────────────────────────────────────────────

def _residual_param_names(config: StudyConfig) -> set:
    """Return the set of parameter names that are equality-constraint residuals."""
    return {
        c.residual_param
        for c in config.constraints
        if c.operator == "=" and c.residual_param
    }


def _extract_external_params(trial_params: dict, config: StudyConfig) -> dict:
    """
    Map internal Optuna trial.params keys back to user-facing parameter names.

    For single-subrange float/int params the key IS the param name.
    For multi-subrange params the value lives under ``{name}__in_range_{idx}``.
    """
    result: dict = {}
    for param in config.parameters:
        if not param.enabled:
            continue
        name = param.name
        subranges = getattr(param, "allowed_subranges", [])
        if len(subranges) <= 1:
            if name in trial_params:
                result[name] = trial_params[name]
        else:
            range_idx = trial_params.get(f"{name}__range_idx")
            if range_idx is not None:
                val = trial_params.get(f"{name}__in_range_{range_idx}")
                if val is not None:
                    result[name] = val
    return result


def _apply_equality_constraint(params: dict, c: ParameterConstraint) -> None:
    """
    Compute the residual parameter value so that *c.expression = c.target*.

    Strategy
    --------
    For expressions that are linear in the residual variable (e.g. ``a+b+c``):
        residual = (target − eval(expr, {residual→0, others→actual})) / coefficient
    where coefficient = eval(expr, {residual→1, others→0}) − eval(expr, {residual→0, others→0}).

    If the free params already exceed the target (residual would be negative), all
    free params in the expression are scaled down proportionally so that the
    residual lands at 0 exactly.

    After this call, ``params[c.residual_param]`` is always ≥ 0.
    """
    import ast as _ast

    residual = c.residual_param
    if not residual:
        return

    # Evaluate expression with residual = 0 (offset = contribution of free params)
    params_zero = dict(params)
    params_zero[residual] = 0.0
    try:
        offset = ParameterConstraint.eval_expr(c.expression, params_zero)
    except ValueError:
        params[residual] = max(0.0, c.target)
        return

    # Determine coefficient of the residual term (for linear expressions)
    params_one = dict(params)
    params_one[residual] = 1.0
    try:
        with_one = ParameterConstraint.eval_expr(c.expression, params_one)
        coeff = with_one - offset
    except ValueError:
        coeff = 1.0

    if abs(coeff) < 1e-12:
        coeff = 1.0

    residual_val = (c.target - offset) / coeff

    # If residual < 0, scale the other free params down to just fit
    if residual_val < 0.0:
        try:
            tree = _ast.parse(c.expression.strip(), mode="eval")
            free_in_expr = list(dict.fromkeys(
                n.id for n in _ast.walk(tree)
                if isinstance(n, _ast.Name)
                and n.id in params
                and n.id != residual
            ))
        except SyntaxError:
            free_in_expr = []

        if free_in_expr and abs(offset) > 1e-15:
            scale = c.target / offset
            for p in free_in_expr:
                if isinstance(params.get(p), (int, float)):
                    params[p] = params[p] * scale
        residual_val = 0.0

    params[residual] = max(0.0, residual_val)


def _apply_inequality_projection(params: dict, c: ParameterConstraint) -> None:
    """
    If *c* is violated, project all numeric params in the expression toward
    the constraint boundary using proportional scaling.

    This is a best-effort heuristic: it scales all free numeric params in the
    expression uniformly so that *eval(expr) = target* exactly.
    """
    import ast as _ast

    try:
        actual = ParameterConstraint.eval_expr(c.expression, params)
    except ValueError:
        return

    satisfied = (
        (c.operator == "<=" and actual <= c.target) or
        (c.operator == ">=" and actual >= c.target)
    )
    if satisfied:
        return

    # Extract param names from the expression
    try:
        tree = _ast.parse(c.expression.strip(), mode="eval")
        expr_params = list(dict.fromkeys(
            n.id for n in _ast.walk(tree)
            if isinstance(n, _ast.Name) and n.id in params
            and isinstance(params[n.id], (int, float))
        ))
    except SyntaxError:
        return

    if not expr_params or abs(actual) < 1e-15:
        return

    # Scale all expression params uniformly to hit the boundary
    ratio = c.target / actual   # < 1 for "<=", > 1 for ">="
    for p in expr_params:
        params[p] = params[p] * ratio


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

    If the StudyConfig contains inequality constraints (<=, >=), a
    ``constraints_func`` is attached to the TPE / NSGAII sampler so that the
    surrogate model learns to avoid infeasible regions over time.
    """
    storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")

    # Build constraints_func for inequality constraints
    ineq_constraints = [c for c in config.constraints if c.operator in ("<=", ">=")]

    def _constraints_fn(trial: FrozenTrial) -> List[float]:
        ext = _extract_external_params(trial.params, config)
        violations: List[float] = []
        for c in ineq_constraints:
            _, v = c.is_satisfied(ext)
            # Optuna convention: positive = infeasible
            violations.append(max(0.0, v))
        return violations

    if ineq_constraints:
        sampler_map: Dict[str, optuna.samplers.BaseSampler] = {
            "TPE":    optuna.samplers.TPESampler(
                          seed=42, multivariate=True,
                          constraints_func=_constraints_fn),
            "NSGAII": optuna.samplers.NSGAIISampler(
                          seed=42, constraints_func=_constraints_fn),
            "Random": optuna.samplers.RandomSampler(seed=42),
        }
    else:
        sampler_map = {
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

    Residual parameters (auto-computed by equality constraints) are excluded
    from the distributions — they have no Optuna search space of their own.
    """
    residuals = _residual_param_names(config)
    dists: Dict[str, BaseDistribution] = {}

    for param in config.parameters:
        if not param.enabled:
            continue
        if param.name in residuals:
            continue   # computed deterministically; not a free variable
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
    """
    Register all enabled *free* parameters with *trial* and return a full
    params dict (including residual params computed from constraints).

    Constraint enforcement order
    ----------------------------
    1. All free parameters (not residuals) are suggested by Optuna.
    2. Equality-constraint residuals are computed deterministically.
    3. Inequality-constraint violations are projected to the boundary.
    """
    residuals = _residual_param_names(config)
    params: dict = {}

    for param in config.parameters:
        if not param.enabled:
            continue
        if param.name in residuals:
            continue   # will be filled in by constraint logic below

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

    # Apply equality constraints — compute residual params
    for c in config.constraints:
        if c.operator == "=" and c.residual_param:
            _apply_equality_constraint(params, c)

    # Apply inequality constraints — project to boundary if violated
    for c in config.constraints:
        if c.operator in ("<=", ">="):
            _apply_inequality_projection(params, c)

    return params


def compute_full_params(trial: optuna.Trial, config: StudyConfig) -> dict:
    """
    Build the complete user-facing parameter dict for a trial returned by
    ``study.ask()``.

    ``trial.params`` only contains the *free* parameters registered with
    Optuna (residuals are excluded from distributions).  This function:

    1. Maps internal Optuna keys → external parameter names.
    2. Applies equality constraints to fill in residual params.
    3. Projects any violated inequality constraints to the boundary.

    Called from the worker right after ``ask_batch()`` so that the
    ``BatchResultsDialog`` shows complete, constraint-satisfying suggestions.
    """
    params = _extract_external_params(dict(trial.params), config)

    for c in config.constraints:
        if c.operator == "=" and c.residual_param:
            _apply_equality_constraint(params, c)

    for c in config.constraints:
        if c.operator in ("<=", ">="):
            _apply_inequality_projection(params, c)

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

    Residual parameters (auto-computed by equality constraints) are excluded —
    they have no distribution in the study and must not be stored in trial.params.
    """
    residuals = _residual_param_names(config)
    internal: dict = {}

    for param in config.parameters:
        if not param.enabled or param.name not in raw_params:
            continue
        if param.name in residuals:
            continue   # no distribution; skip

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
    constrained_params: Optional[List[dict]] = None,
    config: Optional[StudyConfig] = None,
) -> None:
    """
    Report measured lab results back to Optuna for the given trial numbers.

    Constraint-correct mode (recommended when constraints are active)
    ---------------------------------------------------------------
    When *constrained_params* and *config* are supplied the function:

    1. **Fails** the RUNNING trial created by ``study.ask()`` — this removes
       the unconstrained raw suggestion from the surrogate's training data.
    2. **Adds** a new ``FrozenTrial(COMPLETE)`` whose params are the
       constraint-satisfying values that were actually shown to and run by
       the user.

    This ensures the surrogate always trains on feasible, constraint-
    satisfying data regardless of constraint type (equality, inequality, or
    any combination).  FAILED trials are excluded from surrogate training by
    Optuna's TPE and NSGAII samplers.

    Legacy mode
    -----------
    When *constrained_params* / *config* are omitted the function falls back
    to the simple ``study.tell(trial_number, values)`` behaviour.
    """
    for i, (trial_number, values) in enumerate(zip(trial_numbers, results)):
        try:
            if (constrained_params is not None
                    and config is not None
                    and i < len(constrained_params)
                    and constrained_params[i]):

                ext_params = constrained_params[i]

                # Step 1: fail the RUNNING trial so the surrogate ignores its
                # unconstrained internal params.
                try:
                    study.tell(trial_number, state=TrialState.FAIL)
                except Exception:
                    pass  # already finished / not found — continue to add corrected trial

                # Step 2: map constrained external params → internal Optuna key space.
                internal = _csv_params_to_internal(ext_params, config)
                dists    = build_distributions(config)
                trial_dists  = {k: v for k, v in dists.items() if k in internal}
                trial_params = {k: internal[k] for k in trial_dists}

                if not trial_params:
                    # Nothing matched distributions — fall back to simple tell on
                    # the (already failed) trial so we don't lose the result.
                    try:
                        if len(values) == 1:
                            study.tell(trial_number, values[0])
                        else:
                            study.tell(trial_number, values)
                    except Exception:
                        pass
                    continue

                # Step 3: add a corrected COMPLETE trial with constrained params.
                frozen = FrozenTrial(
                    number=len(study.trials),
                    trial_id=-1,
                    state=TrialState.COMPLETE,
                    value=values[0] if len(values) == 1 else None,
                    values=None if len(values) == 1 else list(values),
                    datetime_start=datetime.now(),
                    datetime_complete=datetime.now(),
                    params=trial_params,
                    distributions=trial_dists,
                    user_attrs={},
                    system_attrs={},
                    intermediate_values={},
                )
                study.add_trial(frozen)

            else:
                # Legacy / no-constraint path: plain tell.
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
