"""Job scheduling: a serial queue (one GPU) feeding spawned worker processes."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import queue as queue_mod
import threading
import time
from collections import deque
from typing import Any

from ..settings import Settings
from .store import ACTIVE, JobStore, now_utc
from .worker import RUNNABLE, worker_main

log = logging.getLogger(__name__)

FINAL = ("succeeded", "failed", "cancelled")


class EventHub:
    """Fan-out of JSON messages to WebSocket subscribers."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, msg: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:  # slow client: drop the oldest message
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass


class _Running:
    def __init__(self, job: dict[str, Any], process, cancel_event) -> None:
        self.job = job
        self.process = process
        self.cancel_event = cancel_event
        self.final_state: str | None = None
        self.error = ""
        self.summaries: deque[dict[str, Any]] = deque(maxlen=20000)
        self.last_db_write = 0.0
        self.cancel_requested_at: float | None = None


class JobManager:
    def __init__(self, settings: Settings, store: JobStore, hub: EventHub) -> None:
        self.settings = settings
        self.store = store
        self.hub = hub
        self.ctx = mp.get_context("spawn")
        self.messages = self.ctx.Queue()
        self.running: _Running | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._scheduler_task: asyncio.Task | None = None
        self._pump: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        n = self.store.fail_interrupted()
        if n:
            log.warning("%d job(s) were running when the server stopped; marked failed", n)
        self._pump = threading.Thread(target=self._pump_messages, name="job-pump", daemon=True)
        self._pump.start()
        self._scheduler_task = asyncio.create_task(self._scheduler())
        self._wake.set()

    async def stop(self) -> None:
        self._stop.set()
        if self.running is not None and self.running.process.is_alive():
            self.running.cancel_event.set()
            await asyncio.to_thread(self.running.process.join, 10)
            if self.running.process.is_alive():
                self.running.process.kill()
        if self._scheduler_task:
            self._scheduler_task.cancel()
        if self._pump:
            self._pump.join(timeout=2)

    # ------------------------------------------------------------------ API
    def submit(self, experiment: str, stage: str, params: dict[str, Any] | None = None, force: bool = False) -> dict:
        if stage not in RUNNABLE:
            raise ValueError(f"stage {stage!r} is not available yet; runnable: {', '.join(RUNNABLE)}")
        job = self.store.create(experiment, stage, params or {}, force)
        self._publish_job(job)
        self._kick()
        return job

    def cancel(self, job_id: str) -> dict:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["state"] == "queued":
            job = self.store.update(job_id, state="cancelled", finished_utc=now_utc())
        elif self.running is not None and self.running.job["id"] == job_id and job["state"] == "running":
            self.running.cancel_event.set()
            self.running.cancel_requested_at = time.monotonic()
            job = self.store.update(job_id, state="cancelling")
        self._publish_job(job)
        return job

    def kill(self, job_id: str) -> dict:
        """Terminate a running worker immediately (may leave the HDF5 file inconsistent)."""
        r = self.running
        if r is None or r.job["id"] != job_id:
            raise KeyError(job_id)
        r.final_state = "cancelled"
        r.error = "terminated by user"
        r.process.kill()
        return self.store.get(job_id)  # type: ignore[return-value]

    def retry(self, job_id: str) -> dict:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["state"] in ACTIVE:
            raise ValueError("job is still active")
        return self.submit(job["experiment"], job["stage"], job["params"], job["force"])

    def is_busy(self, experiment: str) -> bool:
        r = self.running
        return r is not None and r.job["experiment"] == experiment and r.job["stage"] != "noop"

    def summaries(self, job_id: str, since: int = 0) -> list[dict]:
        r = self.running
        if r is None or r.job["id"] != job_id:
            return []
        return list(r.summaries)[since:]

    # ------------------------------------------------------------------ internals
    def _kick(self) -> None:
        if self._loop is not None and self._wake is not None:
            self._loop.call_soon_threadsafe(self._wake.set)

    def _publish_job(self, job: dict | None) -> None:
        if job is not None:
            self.hub.publish({"type": "job", "job": job})

    async def _scheduler(self) -> None:
        assert self._wake is not None
        while not self._stop.is_set():
            await self._wake.wait()
            self._wake.clear()
            if self.running is not None:
                await self._check_running()
                continue
            queued = self.store.queued()
            if not queued:
                continue
            self._launch(queued[0])

    def _launch(self, job: dict[str, Any]) -> None:
        cancel_event = self.ctx.Event()
        proc = self.ctx.Process(
            target=worker_main,
            args=(job, self.settings, self.messages, cancel_event),
            name=f"stereodic-job-{job['id']}",
            daemon=False,
        )
        job = self.store.update(job["id"], state="running", started_utc=now_utc(), phase="starting") or job
        self.running = _Running(job, proc, cancel_event)
        proc.start()
        self._publish_job(job)

    async def _check_running(self) -> None:
        r = self.running
        if r is None:
            return
        if r.process.is_alive():
            return
        await asyncio.to_thread(r.process.join, 1)
        state = r.final_state
        error = r.error
        if state is None:
            state = "failed"
            error = f"worker exited unexpectedly (exit code {r.process.exitcode})"
        current = self.store.get(r.job["id"]) or r.job
        if state == "succeeded" and current["total"]:
            self.store.update(r.job["id"], done=current["total"])
        job = self.store.update(r.job["id"], state=state, error=error, finished_utc=now_utc())
        self.running = None
        self._publish_job(job)
        self._wake.set()  # type: ignore[union-attr]

    def _pump_messages(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self.messages.get(timeout=0.25)
            except queue_mod.Empty:
                if self.running is not None and not self.running.process.is_alive():
                    self._kick()
                continue
            except (EOFError, OSError):
                return
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._handle, msg)

    def _handle(self, msg: dict[str, Any]) -> None:
        r = self.running
        job_id = msg.get("job")
        if r is None or r.job["id"] != job_id:
            return
        kind = msg.get("type")
        if kind == "state":
            state = msg["state"]
            if state in FINAL:
                r.final_state = state
                r.error = msg.get("error", "")
                self._kick()
            return
        if kind == "progress":
            now = time.monotonic()
            phase_changed = msg.get("phase") != r.job.get("phase") or msg.get("stage") != r.job.get("current_stage")
            r.job["phase"] = msg.get("phase", "")
            r.job["current_stage"] = msg.get("stage", "")
            if phase_changed or now - r.last_db_write > 1.0 or (msg["total"] and msg["done"] >= msg["total"]):
                phase = f"{msg.get('stage', '')}:{msg.get('phase', '')}" if r.job["stage"] == "pipeline" else msg.get("phase", "")
                self.store.update(job_id, phase=phase, done=msg["done"], total=msg["total"], message=msg.get("message", ""))
                r.last_db_write = now
        elif kind == "summary":
            r.summaries.append({k: v for k, v in msg.items() if k not in ("type", "job")})
        self.hub.publish(msg)
