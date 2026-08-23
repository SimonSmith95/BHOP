"""
Phase 3 — Session Manager
Persists the full application state (StudyConfig + pending batch) to a JSON
file alongside the Optuna SQLite database, enabling close-and-resume workflow.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from parameter_config import StudyConfig

# Registry of recent sessions stored in the user's home directory
RECENT_SESSIONS_PATH = os.path.join(os.path.expanduser("~"), ".bhop_sessions.json")


# ──────────────────────────────────────────────────────────────────────────────
# SessionState dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """
    Represents everything needed to save and restore a BHOP session.

    pending_batch
        List of trials that have been *asked* from Optuna but whose lab
        results have not yet been *told* back.  Each entry is:
            {"trial_number": int, "params": {col: val, ...}}
    """
    session_path: str          # absolute path to this .json file
    storage_path: str          # absolute path to the Optuna SQLite .db file
    study_name: str            # unique Optuna study name
    csv_path: str              # path to the experiment data CSV
    study_config: StudyConfig
    pending_batch: Optional[List[dict]] = None

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "session_path": self.session_path,
            "storage_path": self.storage_path,
            "study_name": self.study_name,
            "csv_path": self.csv_path,
            "study_config": self.study_config.to_dict(),
            "pending_batch": self.pending_batch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SessionState:
        return cls(
            session_path=d["session_path"],
            storage_path=d["storage_path"],
            study_name=d["study_name"],
            csv_path=d.get("csv_path", ""),
            study_config=StudyConfig.from_dict(d["study_config"]),
            pending_batch=d.get("pending_batch"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# SessionManager
# ──────────────────────────────────────────────────────────────────────────────

class SessionManager:
    """
    Static helper class.  All methods are @staticmethod so the class can be
    used without instantiation (matching usage in main_window.py).
    """

    @staticmethod
    def create_new_session(
        config: StudyConfig,
        csv_path: str,
        session_dir: str,
    ) -> SessionState:
        """
        Create a brand-new session directory structure and return the initial
        SessionState.  The SQLite DB is *not* created here — Optuna creates it
        lazily when build_study() is first called.
        """
        os.makedirs(session_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"bhop_study_{timestamp}"
        storage_path = os.path.join(session_dir, f"{study_name}.db")
        session_path = os.path.join(session_dir, f"{study_name}_session.json")

        state = SessionState(
            session_path=session_path,
            storage_path=storage_path,
            study_name=study_name,
            csv_path=csv_path,
            study_config=config,
            pending_batch=None,
        )
        SessionManager.save(state)
        SessionManager._register_recent(session_path, study_name)
        return state

    @staticmethod
    def save(state: SessionState) -> None:
        """
        Write *state* to disk atomically (write to .tmp then rename) to avoid
        corruption if the process is killed mid-write.
        """
        tmp = state.session_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2)
        shutil.move(tmp, state.session_path)

    @staticmethod
    def load(session_path: str) -> SessionState:
        """
        Load a SessionState from a *_session.json file.

        Raises
        ------
        FileNotFoundError  if the SQLite DB referenced in the file is missing.
        """
        with open(session_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        state = SessionState.from_dict(data)
        if not os.path.exists(state.storage_path):
            raise FileNotFoundError(
                f"Optuna database not found: {state.storage_path}\n"
                "The session file may have been moved without its .db companion."
            )
        return state

    @staticmethod
    def list_recent_sessions(n: int = 5) -> List[dict]:
        """
        Return the *n* most recently used sessions from the global registry,
        filtered to only those whose session file still exists on disk.
        """
        if not os.path.exists(RECENT_SESSIONS_PATH):
            return []
        try:
            with open(RECENT_SESSIONS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            sessions: List[dict] = data.get("sessions", [])
            valid = [s for s in sessions if os.path.exists(s.get("session_path", ""))]
            return list(reversed(valid[-n:]))          # newest first
        except Exception:
            return []

    @staticmethod
    def mark_batch_pending(state: SessionState, trials) -> None:
        """
        Record *trials* (list of optuna.Trial) as the currently pending batch,
        export their suggested parameters to *pending_batch.csv* for lab use,
        and save the session.
        """
        pending = [
            {"trial_number": t.number, "params": dict(t.params)}
            for t in trials
        ]
        state.pending_batch = pending
        SessionManager.save(state)

        # Write lab-friendly CSV
        session_dir = os.path.dirname(state.session_path)
        pending_csv = os.path.join(session_dir, "pending_batch.csv")
        rows = [{"trial_number": p["trial_number"], **p["params"]} for p in pending]
        pd.DataFrame(rows).to_csv(pending_csv, index=False)

    @staticmethod
    def clear_pending_batch(state: SessionState) -> None:
        """Mark the pending batch as resolved and clean up the lab CSV."""
        state.pending_batch = None
        SessionManager.save(state)

        session_dir = os.path.dirname(state.session_path)
        pending_csv = os.path.join(session_dir, "pending_batch.csv")
        if os.path.exists(pending_csv):
            os.remove(pending_csv)

    @staticmethod
    def match_pending_to_csv(
        state: SessionState,
        df: pd.DataFrame,
        float_tol: float = 1e-6,
    ) -> Optional[Dict[int, List[float]]]:
        """
        Try to find rows in *df* that match each pending trial's parameters.

        Returns
        -------
        dict  {trial_number: [objective_value, ...]}  for matched rows, or
        None  if no rows matched at all.

        Partial matches (only some trials found) are included; the caller
        displays unmatched trials as blank cells in the results dialog.
        """
        if not state.pending_batch:
            return None

        obj_cols = [o.column_name for o in state.study_config.objectives]
        matches: Dict[int, List[float]] = {}

        for pending in state.pending_batch:
            trial_num = pending["trial_number"]
            params = pending["params"]

            for _, row in df.iterrows():
                # Row must have all objective values
                if any(pd.isna(row.get(c)) for c in obj_cols):
                    continue

                matched = True
                for col, expected in params.items():
                    if col not in row.index:
                        matched = False
                        break
                    actual = row[col]
                    # Always try numeric comparison first — pandas may upcast int
                    # columns to float64 when the DataFrame has mixed dtypes, so
                    # str("2000.0") != str("2000") even though the values match.
                    try:
                        if abs(float(actual) - float(expected)) > float_tol:
                            matched = False
                            break
                    except (TypeError, ValueError):
                        if str(actual).strip() != str(expected).strip():
                            matched = False
                            break

                if matched:
                    matches[trial_num] = [float(row[c]) for c in obj_cols]
                    break   # first matching row wins

        return matches if matches else None

    # ── Internal ───────────────────────────────────────────────────────────

    @staticmethod
    def _register_recent(session_path: str, study_name: str) -> None:
        sessions: List[dict] = []
        if os.path.exists(RECENT_SESSIONS_PATH):
            try:
                with open(RECENT_SESSIONS_PATH, "r", encoding="utf-8") as fh:
                    sessions = json.load(fh).get("sessions", [])
            except Exception:
                sessions = []

        # Remove duplicate (same path)
        sessions = [s for s in sessions if s.get("session_path") != session_path]
        sessions.append({
            "session_path": session_path,
            "study_name": study_name,
            "timestamp": datetime.now().isoformat(),
        })
        # Cap at 20 entries
        sessions = sessions[-20:]

        with open(RECENT_SESSIONS_PATH, "w", encoding="utf-8") as fh:
            json.dump({"sessions": sessions}, fh, indent=2)
