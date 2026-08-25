"""
Phase 11 — Optimization Worker
QThread subclass that runs the Optuna ask/tell batch loop in the background,
keeping the PySide6 GUI responsive while the study DB is being accessed.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import optuna
from PySide6.QtCore import QThread, Signal

from optuna_builder import ask_batch, compute_full_params, tell_batch
from parameter_config import StudyConfig
from session_manager import SessionManager, SessionState


class OptimizationWorker(QThread):
    """
    Background thread that drives the batch optimization loop.

    Signals
    -------
    batch_ready(list)
        Emitted after ask_batch() completes.
        Payload: list of {"trial_number": int, "params": dict}
        The main thread must show BatchResultsDialog and call submit_results().

    batch_complete(int, int)
        Emitted after tell_batch() completes.
        Payload: (batches_done, total_batches)

    optimization_done()
        Emitted when all batches have been processed (or stop() was called).

    error(str)
        Emitted if an unhandled exception occurs.

    Thread-safety
    -------------
    *submit_results()* is called from the main thread and uses a threading.Event
    to unblock the worker after the user has submitted results.  The worker
    never touches Qt widgets directly.
    """

    batch_ready = Signal(list)        # list of {"trial_number": int, "params": dict}
    batch_complete = Signal(int, int) # (batches_done, total_batches)
    optimization_done = Signal()
    error = Signal(str)

    def __init__(
        self,
        study: optuna.Study,
        config: StudyConfig,
        session_manager: type,        # SessionManager class (static methods only)
        session_state: SessionState,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._study = study
        self._config = config
        self._session_manager = session_manager
        self._session_state = session_state

        # Pause / stop control
        self._pause_event = threading.Event()
        self._pause_event.set()       # start in running state
        self._stop_flag = False

        # Synchronisation between main thread and worker
        self._results_ready = threading.Event()
        self._pending_results: Optional[List[dict]] = None  # set by submit_results()

    # ── QThread entry point ────────────────────────────────────────────────

    def run(self) -> None:
        try:
            n_batches = self._config.n_batches
            batch_size = self._config.batch_size

            for batch_idx in range(n_batches):

                # ── Pause / stop check ─────────────────────────────────────
                self._pause_event.wait()
                if self._stop_flag:
                    break

                # ── Ask ────────────────────────────────────────────────────
                trials = ask_batch(self._study, self._config, batch_size)

                # compute_full_params applies all equality and inequality
                # constraints so that BatchResultsDialog shows the exact
                # feasible values the user should prepare in the lab.
                # This must happen BEFORE mark_batch_pending so the session
                # JSON stores the constraint-satisfying params, not the raw
                # unconstrained Optuna suggestions.
                param_dicts = [
                    {"trial_number": t.number,
                     "params": compute_full_params(t, self._config)}
                    for t in trials
                ]

                SessionManager.mark_batch_pending(self._session_state, param_dicts)

                # Reset the handshake event and signal the GUI
                self._results_ready.clear()
                self._pending_results = None
                self.batch_ready.emit(param_dicts)

                # ── Wait for the user to submit lab results ────────────────
                self._results_ready.wait()

                if self._stop_flag:
                    break
                if self._pending_results is None:
                    # User cancelled the dialog
                    break

                # ── Tell ───────────────────────────────────────────────────
                trial_numbers = [r["trial_number"] for r in self._pending_results]
                values_list   = [r["values"]        for r in self._pending_results]

                # Look up the constrained params for each trial from the pending
                # batch (what was actually shown to / run by the user).  These
                # are passed to tell_batch so Optuna's surrogate trains on the
                # correct feasible values rather than the raw internal suggestions.
                pending_by_num = {
                    p["trial_number"]: p["params"]
                    for p in (self._session_state.pending_batch or [])
                }
                constrained = [pending_by_num.get(n, {}) for n in trial_numbers]

                tell_batch(self._study, trial_numbers, values_list,
                           constrained, self._config)
                SessionManager.clear_pending_batch(self._session_state)

                self.batch_complete.emit(batch_idx + 1, n_batches)

        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
            return

        self.optimization_done.emit()

    # ── Control methods (called from the main thread) ──────────────────────

    def pause(self) -> None:
        """Block the worker loop at the next pause-check point."""
        self._pause_event.clear()

    def resume(self) -> None:
        """Unblock the worker loop."""
        self._pause_event.set()

    def stop(self) -> None:
        """
        Request a clean stop after the current operation finishes.
        Also unblocks any waiting events so the thread can exit.
        """
        self._stop_flag = True
        self._pause_event.set()
        self._results_ready.set()   # unblock if waiting for results

    def submit_results(self, results: List[dict]) -> None:
        """
        Called by the main thread once the user has confirmed batch results.

        Parameters
        ----------
        results : list of {"trial_number": int, "values": list[float]}
        """
        self._pending_results = results
        self._results_ready.set()

    def cancel_results(self) -> None:
        """Called by the main thread when the user cancels the results dialog."""
        self._pending_results = None
        self._results_ready.set()
