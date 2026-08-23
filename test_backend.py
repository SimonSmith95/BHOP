"""
Backend test suite for BHOP (no GUI required).
Run with:  pytest test_backend.py -v

Covers:
  Phase 1 — parameter_config  (serialisation round-trip)
  Phase 2 — csv_loader        (type inference, defaults, trial loading)
  Phase 3 — session_manager   (create/save/load/pending/match)
  Phase 4 — sampler_utils     (dead-region computation & validation)
  Phase 5 — optuna_builder    (study, historical trials, ask/tell batch)
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import List

import pandas as pd
import pytest

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — parameter_config
# ══════════════════════════════════════════════════════════════════════════════

from parameter_config import (
    AllowedSubRange,
    ObjectiveConfig,
    ParameterConfig,
    ParameterType,
    StudyConfig,
)


class TestAllowedSubRange:
    def test_round_trip(self):
        r = AllowedSubRange(1.5, 9.9)
        assert AllowedSubRange.from_dict(r.to_dict()) == r

    def test_repr(self):
        assert "[" in repr(AllowedSubRange(0.0, 1.0))


class TestParameterConfig:
    def _int_param(self) -> ParameterConfig:
        return ParameterConfig(
            name="speed",
            ptype=ParameterType.INT,
            full_min=100.0,
            full_max=5000.0,
            allowed_subranges=[AllowedSubRange(100, 2000), AllowedSubRange(3000, 5000)],
            step=100,
        )

    def _float_param(self) -> ParameterConfig:
        return ParameterConfig(
            name="conc",
            ptype=ParameterType.FLOAT,
            full_min=0.01,
            full_max=0.15,
        )

    def _cat_param(self) -> ParameterConfig:
        return ParameterConfig(
            name="solvent",
            ptype=ParameterType.CATEGORICAL,
            all_choices=["THF", "DMF", "NMP"],
            allowed_choices=["THF", "DMF"],
        )

    def _bool_param(self) -> ParameterConfig:
        return ParameterConfig(name="anneal", ptype=ParameterType.BOOL, fixed_value=None)

    def test_int_round_trip(self):
        p = self._int_param()
        assert ParameterConfig.from_dict(p.to_dict()).name == p.name
        assert ParameterConfig.from_dict(p.to_dict()).allowed_subranges == p.allowed_subranges
        assert ParameterConfig.from_dict(p.to_dict()).step == p.step

    def test_float_round_trip(self):
        p = self._float_param()
        p2 = ParameterConfig.from_dict(p.to_dict())
        assert p2.full_min == p.full_min
        assert p2.full_max == p.full_max

    def test_categorical_round_trip(self):
        p = self._cat_param()
        p2 = ParameterConfig.from_dict(p.to_dict())
        assert p2.allowed_choices == p.allowed_choices

    def test_bool_round_trip(self):
        p = self._bool_param()
        p2 = ParameterConfig.from_dict(p.to_dict())
        assert p2.fixed_value is None

    def test_default_subrange_populated(self):
        """__post_init__ should fill allowed_subranges from full_min/max."""
        p = ParameterConfig(name="x", ptype=ParameterType.FLOAT, full_min=0.0, full_max=1.0)
        assert len(p.allowed_subranges) == 1
        assert p.allowed_subranges[0].low == 0.0
        assert p.allowed_subranges[0].high == 1.0


class TestStudyConfig:
    def _config(self) -> StudyConfig:
        return StudyConfig(
            parameters=[
                ParameterConfig("speed", ParameterType.INT, full_min=100, full_max=5000),
                ParameterConfig("conc",  ParameterType.FLOAT, full_min=0.01, full_max=0.15),
            ],
            objectives=[
                ObjectiveConfig("thickness", "maximize"),
                ObjectiveConfig("roughness", "minimize"),
            ],
            batch_size=3,
            n_batches=20,
            sampler_name="NSGAII",
        )

    def test_round_trip(self):
        cfg = self._config()
        cfg2 = StudyConfig.from_dict(cfg.to_dict())
        assert cfg2.sampler_name == "NSGAII"
        assert cfg2.batch_size == 3
        assert len(cfg2.parameters) == 2
        assert len(cfg2.objectives) == 2
        assert cfg2.objectives[0].direction == "maximize"
        assert cfg2.objectives[1].direction == "minimize"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — csv_loader
# ══════════════════════════════════════════════════════════════════════════════

from csv_loader import (
    append_rows_to_csv,
    extract_param_defaults,
    infer_column_type,
    load_csv,
    load_trials_from_csv,
)


def _make_df(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text))


class TestInferColumnType:
    def test_bool_int(self):
        s = pd.Series([0, 1, 0, 1])
        assert infer_column_type(s) == ParameterType.BOOL

    def test_bool_true_false(self):
        s = pd.Series([True, False, True])
        assert infer_column_type(s) == ParameterType.BOOL

    def test_categorical_string(self):
        s = pd.Series(["THF", "DMF", "NMP", "THF"])
        assert infer_column_type(s) == ParameterType.CATEGORICAL

    def test_categorical_few_numeric(self):
        s = pd.Series([1, 2, 3, 1, 2])   # ≤ 8 unique → categorical
        assert infer_column_type(s) == ParameterType.CATEGORICAL

    def test_int_many_unique(self):
        s = pd.Series(list(range(50, 150)))   # 100 unique integers
        assert infer_column_type(s) == ParameterType.INT

    def test_float(self):
        s = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert infer_column_type(s) == ParameterType.FLOAT

    def test_empty_series_defaults_float(self):
        s = pd.Series([], dtype=float)
        assert infer_column_type(s) == ParameterType.FLOAT


class TestExtractParamDefaults:
    # Need >8 unique values for spin_speed and concentration so they are
    # classified as INT / FLOAT (not CATEGORICAL) by infer_column_type.
    CSV = """spin_speed,concentration,solvent,anneal,thickness
500,0.01,THF,True,100.0
1000,0.02,DMF,False,110.0
1500,0.03,NMP,True,120.0
2000,0.04,THF,False,130.0
2500,0.05,DMF,True,140.0
3000,0.06,NMP,False,150.0
3500,0.07,THF,True,160.0
4000,0.08,DMF,False,170.0
4500,0.09,NMP,True,180.0
"""

    def setup_method(self):
        self.df = _make_df(self.CSV)

    def test_result_columns_excluded(self):
        params = extract_param_defaults(self.df, ["thickness"])
        names = [p.name for p in params]
        assert "thickness" not in names
        assert "spin_speed" in names

    def test_int_range(self):
        params = extract_param_defaults(self.df, ["thickness"])
        speed = next(p for p in params if p.name == "spin_speed")
        assert speed.ptype == ParameterType.INT
        assert speed.full_min == 500.0
        assert speed.full_max == 4500.0

    def test_float_range(self):
        params = extract_param_defaults(self.df, ["thickness"])
        conc = next(p for p in params if p.name == "concentration")
        assert conc.ptype == ParameterType.FLOAT
        assert abs(conc.full_min - 0.01) < 1e-9
        assert abs(conc.full_max - 0.09) < 1e-9

    def test_categorical_choices(self):
        params = extract_param_defaults(self.df, ["thickness"])
        solvent = next(p for p in params if p.name == "solvent")
        assert solvent.ptype == ParameterType.CATEGORICAL
        assert set(solvent.all_choices) == {"THF", "DMF", "NMP"}

    def test_bool_param(self):
        params = extract_param_defaults(self.df, ["thickness"])
        anneal = next(p for p in params if p.name == "anneal")
        assert anneal.ptype == ParameterType.BOOL


class TestLoadTrialsFromCsv:
    CSV = """spin_speed,concentration,thickness
2000,0.05,125.3
3000,0.08,89.7
1500,,210.5
"""

    def test_rows_with_missing_objective_skipped(self):
        df = _make_df(self.CSV)
        config = StudyConfig(
            parameters=[
                ParameterConfig("spin_speed", ParameterType.INT, full_min=1000, full_max=5000),
                ParameterConfig("concentration", ParameterType.FLOAT, full_min=0.01, full_max=0.15),
            ],
            objectives=[ObjectiveConfig("thickness", "maximize")],
        )
        results = load_trials_from_csv(df, config)
        # Row 3 has missing concentration but objective IS present — still included
        # Row 2 has missing concentration and objective — wait, let me re-check
        # Actually: row 3 has missing concentration but thickness=210.5 → included
        # Objective missing check only looks at objective columns
        assert len(results) == 3

    def test_missing_objective_excluded(self):
        csv = "spin_speed,thickness\n2000,125.3\n3000,\n"
        df = _make_df(csv)
        config = StudyConfig(
            parameters=[ParameterConfig("spin_speed", ParameterType.INT, full_min=1000, full_max=5000)],
            objectives=[ObjectiveConfig("thickness", "maximize")],
        )
        results = load_trials_from_csv(df, config)
        assert len(results) == 1
        assert results[0]["values"] == [125.3]


class TestAppendRowsToCsv:
    def test_creates_new_file(self, tmp_path):
        path = str(tmp_path / "out.csv")
        append_rows_to_csv(path, [{"a": 1, "b": 2}])
        df = pd.read_csv(path)
        assert len(df) == 1

    def test_appends_to_existing(self, tmp_path):
        path = str(tmp_path / "out.csv")
        pd.DataFrame([{"a": 1}]).to_csv(path, index=False)
        append_rows_to_csv(path, [{"a": 2}])
        df = pd.read_csv(path)
        assert len(df) == 2

    def test_empty_rows_no_op(self, tmp_path):
        path = str(tmp_path / "out.csv")
        append_rows_to_csv(path, [])
        assert not os.path.exists(path)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — sampler_utils
# ══════════════════════════════════════════════════════════════════════════════

from sampler_utils import compute_allowed_subranges, validate_subranges


class TestComputeAllowedSubranges:
    def test_no_dead_regions(self):
        result = compute_allowed_subranges(0.0, 10.0, [])
        assert len(result) == 1
        assert result[0].low == 0.0
        assert result[0].high == 10.0

    def test_dead_region_in_middle(self):
        result = compute_allowed_subranges(0.0, 10.0, [(4.0, 6.0)])
        assert len(result) == 2
        assert result[0].low == 0.0
        assert result[0].high == 4.0
        assert result[1].low == 6.0
        assert result[1].high == 10.0

    def test_dead_region_at_start(self):
        result = compute_allowed_subranges(0.0, 10.0, [(0.0, 3.0)])
        assert len(result) == 1
        assert result[0].low == 3.0
        assert result[0].high == 10.0

    def test_dead_region_at_end(self):
        result = compute_allowed_subranges(0.0, 10.0, [(8.0, 10.0)])
        assert len(result) == 1
        assert result[0].high == 8.0

    def test_multiple_dead_regions_merged(self):
        # Two overlapping dead regions should merge
        result = compute_allowed_subranges(0.0, 10.0, [(3.0, 5.0), (4.0, 7.0)])
        assert len(result) == 2
        assert result[0].high == 3.0
        assert result[1].low == 7.0

    def test_multiple_separate_dead_regions(self):
        result = compute_allowed_subranges(1.0, 10.0, [(2.0, 3.0), (5.0, 6.0), (8.0, 9.0)])
        assert len(result) == 4

    def test_dead_region_covers_all_raises(self):
        with pytest.raises(ValueError, match="entire"):
            compute_allowed_subranges(0.0, 10.0, [(0.0, 10.0)])

    def test_dead_region_out_of_bounds_clipped(self):
        # Dead region partially outside full range — should be clipped gracefully
        result = compute_allowed_subranges(0.0, 10.0, [(-5.0, 3.0)])
        assert result[0].low == 3.0


class TestValidateSubranges:
    def test_valid_subranges(self):
        errors = validate_subranges(
            [AllowedSubRange(0.0, 5.0), AllowedSubRange(7.0, 10.0)],
            0.0, 10.0
        )
        assert errors == []

    def test_empty_subranges(self):
        errors = validate_subranges([], 0.0, 10.0)
        assert len(errors) > 0

    def test_zero_width_subrange(self):
        errors = validate_subranges([AllowedSubRange(5.0, 5.0)], 0.0, 10.0)
        assert any("less than" in e for e in errors)

    def test_out_of_bounds_low(self):
        errors = validate_subranges([AllowedSubRange(-1.0, 5.0)], 0.0, 10.0)
        assert any("below" in e for e in errors)

    def test_out_of_bounds_high(self):
        errors = validate_subranges([AllowedSubRange(5.0, 15.0)], 0.0, 10.0)
        assert any("exceeds" in e for e in errors)

    def test_overlapping_subranges(self):
        errors = validate_subranges(
            [AllowedSubRange(0.0, 6.0), AllowedSubRange(5.0, 10.0)],
            0.0, 10.0
        )
        assert any("overlap" in e for e in errors)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — session_manager
# ══════════════════════════════════════════════════════════════════════════════

from session_manager import SessionManager, SessionState


def _simple_config() -> StudyConfig:
    return StudyConfig(
        parameters=[
            ParameterConfig("speed", ParameterType.INT, full_min=1000, full_max=5000),
        ],
        objectives=[ObjectiveConfig("thickness", "maximize")],
        batch_size=2,
        n_batches=5,
        sampler_name="TPE",
    )


class TestSessionManager:
    def test_create_save_load(self, tmp_path):
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "/fake/data.csv", str(tmp_path))

        assert os.path.exists(state.session_path)
        # Manually create the DB file so load() validates it
        open(state.storage_path, "w").close()

        loaded = SessionManager.load(state.session_path)
        assert loaded.study_name == state.study_name
        assert loaded.study_config.sampler_name == "TPE"
        assert loaded.csv_path == "/fake/data.csv"

    def test_load_missing_db_raises(self, tmp_path):
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        # Do NOT create the .db file
        with pytest.raises(FileNotFoundError, match="database"):
            SessionManager.load(state.session_path)

    def test_mark_and_clear_pending(self, tmp_path):
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        open(state.storage_path, "w").close()

        # Build mock trials
        class MockTrial:
            def __init__(self, n, params):
                self.number = n
                self.params = params

        trials = [MockTrial(0, {"speed": 2000}), MockTrial(1, {"speed": 3000})]
        SessionManager.mark_batch_pending(state, trials)

        assert state.pending_batch is not None
        assert len(state.pending_batch) == 2
        pending_csv = os.path.join(str(tmp_path), "pending_batch.csv")
        assert os.path.exists(pending_csv)

        # Reload and check pending persisted
        loaded = SessionManager.load(state.session_path)
        assert loaded.pending_batch is not None

        SessionManager.clear_pending_batch(state)
        assert state.pending_batch is None
        assert not os.path.exists(pending_csv)

    def test_match_pending_to_csv(self, tmp_path):
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        open(state.storage_path, "w").close()

        class MockTrial:
            def __init__(self, n, params):
                self.number = n
                self.params = params

        SessionManager.mark_batch_pending(state, [MockTrial(0, {"speed": 2000})])

        df = pd.DataFrame([
            {"speed": 2000, "thickness": 125.3},
            {"speed": 3000, "thickness": 99.1},
        ])
        matches = SessionManager.match_pending_to_csv(state, df)
        assert matches is not None
        assert 0 in matches
        assert matches[0] == [125.3]

    def test_match_pending_no_match_returns_none(self, tmp_path):
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        open(state.storage_path, "w").close()

        class MockTrial:
            def __init__(self, n, params):
                self.number = n
                self.params = params

        SessionManager.mark_batch_pending(state, [MockTrial(0, {"speed": 9999})])

        df = pd.DataFrame([{"speed": 2000, "thickness": 125.3}])
        matches = SessionManager.match_pending_to_csv(state, df)
        assert matches is None

    def test_atomic_save_no_corruption(self, tmp_path):
        """The .tmp file should be gone after save."""
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        assert not os.path.exists(state.session_path + ".tmp")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — optuna_builder
# ══════════════════════════════════════════════════════════════════════════════

from optuna_builder import (
    ask_batch,
    build_distributions,
    build_study,
    get_pareto_front,
    load_historical_trials,
    tell_batch,
)


def _full_config() -> StudyConfig:
    return StudyConfig(
        parameters=[
            ParameterConfig("speed", ParameterType.INT, full_min=1000, full_max=5000),
            ParameterConfig("conc",  ParameterType.FLOAT, full_min=0.01, full_max=0.15),
            ParameterConfig(
                "solvent", ParameterType.CATEGORICAL,
                all_choices=["THF", "DMF"], allowed_choices=["THF", "DMF"]
            ),
        ],
        objectives=[ObjectiveConfig("thickness", "maximize")],
        batch_size=2,
        n_batches=3,
        sampler_name="Random",
    )


class TestBuildStudy:
    def test_creates_study(self, tmp_path):
        cfg = _full_config()
        db = str(tmp_path / "test.db")
        study = build_study(cfg, db, "test_study")
        assert study.study_name == "test_study"

    def test_load_if_exists(self, tmp_path):
        cfg = _full_config()
        db = str(tmp_path / "test.db")
        s1 = build_study(cfg, db, "test_study")
        s2 = build_study(cfg, db, "test_study")
        assert s1.study_name == s2.study_name

    def test_multi_objective(self, tmp_path):
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT, full_min=0, full_max=1)],
            objectives=[
                ObjectiveConfig("obj1", "minimize"),
                ObjectiveConfig("obj2", "maximize"),
            ],
            sampler_name="NSGAII",
        )
        study = build_study(cfg, str(tmp_path / "mo.db"), "mo_study")
        assert len(study.directions) == 2


class TestBuildDistributions:
    def test_single_range_int(self):
        cfg = StudyConfig(
            parameters=[ParameterConfig("speed", ParameterType.INT, full_min=100, full_max=5000)],
            objectives=[ObjectiveConfig("y", "maximize")],
        )
        dists = build_distributions(cfg)
        assert "speed" in dists

    def test_multi_range_int_uses_range_idx(self):
        cfg = StudyConfig(
            parameters=[ParameterConfig(
                "speed", ParameterType.INT, full_min=100, full_max=5000,
                allowed_subranges=[AllowedSubRange(100, 2000), AllowedSubRange(3000, 5000)],
            )],
            objectives=[ObjectiveConfig("y", "maximize")],
        )
        dists = build_distributions(cfg)
        assert "speed__range_idx" in dists
        assert "speed__in_range_0" in dists
        assert "speed__in_range_1" in dists

    def test_disabled_param_excluded(self):
        cfg = StudyConfig(
            parameters=[
                ParameterConfig("speed", ParameterType.INT, full_min=100, full_max=5000, enabled=True),
                ParameterConfig("conc",  ParameterType.FLOAT, full_min=0.01, full_max=0.15, enabled=False),
            ],
            objectives=[ObjectiveConfig("y", "maximize")],
        )
        dists = build_distributions(cfg)
        assert "speed" in dists
        assert "conc" not in dists


class TestLoadHistoricalTrials:
    def test_adds_trials(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "h.db"), "h_study")

        trials = [
            {"params": {"speed": 2000, "conc": 0.05, "solvent": "THF"}, "values": [125.3]},
            {"params": {"speed": 3000, "conc": 0.08, "solvent": "DMF"}, "values": [89.7]},
        ]
        added, skipped = load_historical_trials(study, trials, cfg)
        assert added == 2
        assert skipped == 0

    def test_duplicate_detection(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "d.db"), "d_study")
        trials = [
            {"params": {"speed": 2000, "conc": 0.05, "solvent": "THF"}, "values": [125.3]},
        ]
        load_historical_trials(study, trials, cfg)
        added, skipped = load_historical_trials(study, trials, cfg)
        assert added == 0
        assert skipped == 1


class TestAskTellBatch:
    def test_ask_returns_waiting_trials(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "ab.db"), "ab_study")
        trials = ask_batch(study, cfg, batch_size=3)
        assert len(trials) == 3
        # Optuna 4.x: study.ask() returns Trial objects (no .state attribute).
        # Verify via study.trials that 3 non-complete trials exist.
        from optuna.trial import TrialState
        active = [t for t in study.trials if t.state != TrialState.COMPLETE]
        assert len(active) == 3

    def test_tell_completes_trials(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "tb.db"), "tb_study")
        trials = ask_batch(study, cfg, batch_size=2)

        trial_numbers = [t.number for t in trials]
        results = [[100.0 + i] for i in range(len(trials))]
        tell_batch(study, trial_numbers, results)

        from optuna.trial import TrialState
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        assert len(completed) == 2

    def test_tell_with_unknown_trial_number_skips(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "sk.db"), "sk_study")
        ask_batch(study, cfg, batch_size=1)
        # Telling a non-existent trial number should not crash
        tell_batch(study, [9999], [[100.0]])


class TestGetParetoFront:
    def test_single_objective_returns_best(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "pf.db"), "pf_study")
        trials = ask_batch(study, cfg, 2)
        tell_batch(study, [t.number for t in trials], [[90.0], [120.0]])
        pareto = get_pareto_front(study)
        assert len(pareto) == 1
        assert pareto[0].value == 120.0

    def test_empty_study_returns_empty(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "empty.db"), "empty_study")
        pareto = get_pareto_front(study)
        assert pareto == []


# ══════════════════════════════════════════════════════════════════════════════
# Integration smoke test — full headless loop
# ══════════════════════════════════════════════════════════════════════════════

class TestHeadlessIntegration:
    def test_full_loop(self, tmp_path):
        """
        Seed historical data → ask a batch → mock-measure → tell → check best.
        This exercises parameter_config, session_manager, optuna_builder,
        csv_loader, and sampler_utils together.
        """
        cfg = StudyConfig(
            parameters=[
                ParameterConfig("speed", ParameterType.INT, full_min=1000, full_max=5000,
                                allowed_subranges=[AllowedSubRange(1000, 2500),
                                                   AllowedSubRange(3500, 5000)]),
                ParameterConfig("conc",  ParameterType.FLOAT, full_min=0.01, full_max=0.15),
            ],
            objectives=[ObjectiveConfig("thickness", "maximize")],
            batch_size=2,
            n_batches=2,
            sampler_name="Random",
        )
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        study = build_study(cfg, state.storage_path, state.study_name)

        # Seed history
        historical = [
            {"params": {"speed": 1500, "conc": 0.05}, "values": [125.3]},
            {"params": {"speed": 4000, "conc": 0.10}, "values": [89.7]},
        ]
        added, _ = load_historical_trials(study, historical, cfg)
        assert added == 2

        # Ask + tell
        trials = ask_batch(study, cfg, 2)
        assert len(trials) == 2

        # Verify suggested speeds are NOT in the dead region 2500–3500
        for t in trials:
            # Reconstruct the actual speed from internal params
            if "speed" in t.params:
                speed = t.params["speed"]
            else:
                # find in_range keys
                speed_keys = [k for k in t.params if k.startswith("speed__in_range_")]
                speed = t.params[speed_keys[0]] if speed_keys else None
            if speed is not None:
                assert not (2500 < speed < 3500), f"Speed {speed} is in dead region!"

        tell_batch(study, [t.number for t in trials], [[200.0], [210.0]])

        from optuna.trial import TrialState
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        assert len(completed) == 4   # 2 historical + 2 batch

        pareto = get_pareto_front(study)
        assert pareto[0].value == 210.0

        # Config serialization round-trip
        cfg_dict = cfg.to_dict()
        cfg2 = StudyConfig.from_dict(cfg_dict)
        assert cfg2.parameters[0].allowed_subranges[0].low == 1000.0
