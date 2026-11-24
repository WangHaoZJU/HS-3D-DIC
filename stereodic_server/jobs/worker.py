"""Job worker: runs one pipeline stage in a spawned subprocess.

Messages back to the API process (``multiprocessing.Queue``), all JSON-serialisable::

    {"type": "state", "job": id, "state": "running" | "succeeded" | "failed" | "cancelled", "error": str}
    {"type": "progress", "job": id, "stage": s, "phase": p, "done": n, "total": m,
     "message": str, "rate": float | None, "eta_s": float | None}
    {"type": "summary", "job": id, "stage": s, ...}
    {"type": "log", "job": id, "level": "INFO", "text": str}
"""

from __future__ import annotations

import logging
import time
import traceback
from collections import deque
from typing import Any

from stereodic.core.progress import JobCancelled

# Stages the worker can run. analyze / uncertainty / noise_floor follow in U5 (docs §9 B2/B3).
RUNNABLE = ("calibrate", "stereo", "temporal", "reconstruct", "rbm", "pipeline", "noop")
PIPELINE_DEFAULT = ("stereo", "temporal", "reconstruct", "rbm")


class QueueProgress:
    """ProgressReporter that forwards to the API process, rate-limited."""

    def __init__(self, job_id: str, queue, cancel_event, min_interval_s: float = 0.2) -> None:
        self.job_id = job_id
        self.queue = queue
        self.cancel_event = cancel_event
        self.min_interval_s = min_interval_s
        self.stage = ""
        self.current_phase = ""
        self._last_sent = 0.0
        self._history: deque[tuple[float, int]] = deque(maxlen=50)

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.current_phase = ""
        self._history.clear()
        self._last_sent = 0.0

    def _send(self, msg: dict[str, Any]) -> None:
        msg["job"] = self.job_id
        self.queue.put(msg)

    def phase(self, name: str) -> None:
        self.current_phase = name
        self._history.clear()
        self._send({"type": "progress", "stage": self.stage, "phase": name, "done": 0, "total": 0,
                    "message": "", "rate": None, "eta_s": None})

    def update(self, done: int, total: int, message: str = "") -> None:
        now = time.monotonic()
        self._history.append((now, done))
        final = total > 0 and done >= total
        if not final and now - self._last_sent < self.min_interval_s:
            return
        self._last_sent = now
        rate = eta = None
        if len(self._history) >= 2:
            (t0, d0), (t1, d1) = self._history[0], self._history[-1]
            if t1 > t0 and d1 > d0:
                rate = (d1 - d0) / (t1 - t0)
                eta = max(total - done, 0) / rate if total else None
        self._send({"type": "progress", "stage": self.stage, "phase": self.current_phase, "done": int(done),
                    "total": int(total), "message": message, "rate": rate, "eta_s": eta})

    def summary(self, data: dict[str, Any]) -> None:
        self._send({"type": "summary", "stage": self.stage, **data})

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled()


class _QueueLogHandler(logging.Handler):
    def __init__(self, job_id: str, queue) -> None:
        super().__init__(logging.INFO)
        self.job_id = job_id
        self.queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put({"type": "log", "job": self.job_id, "level": record.levelname, "text": self.format(record)})
        except Exception:  # noqa: BLE001
            pass


def _setup_logging(job_id: str, queue, log_path: str) -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    qh = _QueueLogHandler(job_id, queue)
    qh.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root.addHandler(fh)
    root.addHandler(qh)
    return logging.getLogger("stereodic_server.worker")


def _noop(params: dict[str, Any], progress: QueueProgress) -> None:
    """Test job: ``steps`` updates spaced by ``seconds / steps``."""
    steps = int(params.get("steps", 20))
    delay = float(params.get("seconds", 2.0)) / max(steps, 1)
    progress.phase("working")
    for i in range(steps):
        progress.check_cancel()
        time.sleep(delay)
        progress.update(i + 1, steps, f"step {i + 1}")
        progress.summary({"row": i, "value": (i + 1) / steps})
    if params.get("fail"):
        raise RuntimeError("noop job asked to fail")


def run_stage(stage: str, cfg, force: bool, params: dict[str, Any], progress: QueueProgress, log) -> None:
    from stereodic import pipeline

    progress.set_stage(stage)
    log.info("stage %s started", stage)
    if stage == "calibrate":
        pipeline.run_calibration(cfg, plot=True, progress=progress)
    elif stage == "stereo":
        pipeline.run_stereo(cfg, force=force, progress=progress)
    elif stage == "temporal":
        pipeline.run_temporal(cfg, force=force, progress=progress)
    elif stage == "reconstruct":
        pipeline.run_reconstruct(cfg, force=force, progress=progress)
    elif stage == "rbm":
        pipeline.run_rbm(cfg, force=force, progress=progress)
    elif stage == "noop":
        _noop(params, progress)
    else:
        raise ValueError(f"stage {stage!r} cannot run in the worker")
    log.info("stage %s finished", stage)


def worker_main(job: dict[str, Any], settings, queue, cancel_event) -> None:
    """Entry point of the spawned process."""
    job_id = job["id"]
    log = _setup_logging(job_id, queue, str(settings.log_path(job_id)))
    progress = QueueProgress(job_id, queue, cancel_event)
    queue.put({"type": "state", "job": job_id, "state": "running"})
    try:
        stage = job["stage"]
        params = job.get("params") or {}
        if stage == "noop":
            run_stage("noop", None, False, params, progress, log)
        else:
            from ..workspace import Workspace

            cfg = Workspace(settings).load(job["experiment"])
            stages = list(params.get("stages") or PIPELINE_DEFAULT) if stage == "pipeline" else [stage]
            for s in stages:
                progress.check_cancel()
                run_stage(s, cfg, bool(job.get("force")), params, progress, log)
        queue.put({"type": "state", "job": job_id, "state": "succeeded"})
    except JobCancelled:
        log.info("cancelled")
        queue.put({"type": "state", "job": job_id, "state": "cancelled"})
    except BaseException as exc:  # noqa: BLE001 - report everything, including KeyboardInterrupt
        log.error("failed: %s\n%s", exc, traceback.format_exc())
        queue.put({"type": "state", "job": job_id, "state": "failed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        logging.shutdown()
