"""
BHOP — Bayesian Hyperparameter Optimization for the Lab
========================================================

GUI entry point
---------------
    python main.py

Headless / scripting example
-----------------------------
This file shows how to use the backend modules without the GUI, useful for
batch processing or integration into scripts.  Replace the TODO sections with
your actual experiment runner.
"""
from __future__ import annotations

import os
import tempfile

from csv_loader import load_csv, load_trials_from_csv
from optuna_builder import ask_batch, build_study, load_historical_trials, tell_batch
from parameter_config import (
    AllowedSubRange,
    ObjectiveConfig,
    ParameterConfig,
    ParameterType,
    StudyConfig,
)
from session_manager import SessionManager


def run_headless_example() -> None:
    """
    Minimal headless example: seed 3 historical trials from a hard-coded
    dataset, ask for 2 more, then 'measure' them with a mock function.

    In a real lab workflow:
      1. Replace the mock data with a real CSV file.
      2. Replace the `mock_measure` function with your actual experiment runner.
      3. Use the GUI (python main.py) for the interactive version.
    """

    # ── 1. Define the search space ─────────────────────────────────────────
    config = StudyConfig(
        parameters=[
            ParameterConfig(
                name="spin_speed",
                ptype=ParameterType.INT,
                full_min=1000,
                full_max=5000,
                allowed_subranges=[
                    AllowedSubRange(1000, 2800),   # dead region: 2800–3200
                    AllowedSubRange(3200, 5000),
                ],
            ),
            ParameterConfig(
                name="concentration",
                ptype=ParameterType.FLOAT,
                full_min=0.01,
                full_max=0.15,
            ),
            ParameterConfig(
                name="hotplate_temp",
                ptype=ParameterType.INT,
                full_min=40,
                full_max=150,
            ),
            ParameterConfig(
                name="spin_time",
                ptype=ParameterType.INT,
                full_min=10,
                full_max=90,
            ),
        ],
        objectives=[ObjectiveConfig(column_name="thickness_nm", direction="maximize")],
        batch_size=2,
        n_batches=1,
        sampler_name="TPE",
    )

    # ── 2. Create a temporary session (uses a temp directory) ──────────────
    session_dir = tempfile.mkdtemp(prefix="bhop_headless_")
    state = SessionManager.create_new_session(config, csv_path="", session_dir=session_dir)
    study = build_study(config, state.storage_path, state.study_name)

    # ── 3. Seed 3 historical data points ──────────────────────────────────
    historical = [
        {"params": {"spin_speed": 2000, "concentration": 0.05, "hotplate_temp": 80,  "spin_time": 30}, "values": [125.3]},
        {"params": {"spin_speed": 3000, "concentration": 0.08, "hotplate_temp": 100, "spin_time": 45}, "values": [89.7]},
        {"params": {"spin_speed": 1500, "concentration": 0.03, "hotplate_temp": 60,  "spin_time": 60}, "values": [210.5]},
    ]
    added, skipped = load_historical_trials(study, historical, config)
    print(f"Seeded {added} historical trials ({skipped} skipped).")

    # ── 4. Ask for a batch ─────────────────────────────────────────────────
    trials = ask_batch(study, config, batch_size=config.batch_size)
    print(f"\nNext {len(trials)} suggested experiment(s):")
    for t in trials:
        print(f"  Trial {t.number}: {t.params}")

    # ── 5. Simulate measuring the result (replace with real lab measurement) ─
    def mock_measure(params: dict) -> float:
        """Toy model: higher spin speed → thinner film."""
        speed = params.get("spin_speed", 2000)
        conc  = params.get("concentration", 0.05)
        return 300.0 - speed * 0.04 + conc * 500

    results = [
        {"trial_number": t.number, "values": [mock_measure(t.params)]}
        for t in trials
    ]

    # ── 6. Tell the study ─────────────────────────────────────────────────
    tell_batch(study, [r["trial_number"] for r in results], [r["values"] for r in results])
    SessionManager.clear_pending_batch(state)

    from optuna.trial import TrialState
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"\nStudy now has {len(completed)} completed trials.")
    print(f"Best trial: {study.best_trial.params}  →  {study.best_trial.value:.2f} nm")

    # Clean up temp files
    import shutil
    shutil.rmtree(session_dir, ignore_errors=True)


if __name__ == "__main__":
    print("Running headless BHOP example...\n")
    run_headless_example()
    print("\nDone.  For the full GUI, run:  python main.py")
