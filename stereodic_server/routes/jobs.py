"""Jobs: submit, list, cancel, logs, live events over WebSocket."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ..workspace import WorkspaceError

router = APIRouter(prefix="/api", tags=["jobs"])


class SubmitJob(BaseModel):
    stage: str
    params: dict[str, Any] = Field(default_factory=dict)
    force: bool = False


class SubmitAnyJob(SubmitJob):
    experiment: str = ""


def _jobs(request: Request):
    return request.app.state.jobs


def _job_or_404(request: Request, job_id: str) -> dict[str, Any]:
    job = _jobs(request).store.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return job


def _submit(request: Request, experiment: str, body: SubmitJob) -> dict[str, Any]:
    if body.stage != "noop":
        request.app.state.workspace.require(experiment)
    try:
        return _jobs(request).submit(experiment, body.stage, body.params, body.force)
    except ValueError as exc:
        raise WorkspaceError(str(exc), 400) from exc


@router.post("/experiments/{name}/jobs", status_code=201)
def submit_for_experiment(request: Request, name: str, body: SubmitJob) -> dict[str, Any]:
    return _submit(request, name, body)


@router.post("/jobs", status_code=201)
def submit_job(request: Request, body: SubmitAnyJob) -> dict[str, Any]:
    return _submit(request, body.experiment, body)


@router.get("/jobs")
def list_jobs(request: Request, experiment: str | None = None, active: bool = False, limit: int = 100) -> list[dict]:
    return _jobs(request).store.list(limit=min(limit, 1000), experiment=experiment, active_only=active)


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict[str, Any]:
    return _job_or_404(request, job_id)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict[str, Any]:
    _job_or_404(request, job_id)
    return _jobs(request).cancel(job_id)


@router.post("/jobs/{job_id}/kill")
def kill_job(request: Request, job_id: str) -> dict[str, Any]:
    _job_or_404(request, job_id)
    try:
        return _jobs(request).kill(job_id)
    except KeyError as exc:
        raise HTTPException(409, "job is not running") from exc


@router.post("/jobs/{job_id}/retry", status_code=201)
def retry_job(request: Request, job_id: str) -> dict[str, Any]:
    _job_or_404(request, job_id)
    try:
        return _jobs(request).retry(job_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/jobs/{job_id}/log")
def job_log(request: Request, job_id: str, offset: int = 0, limit: int = 256_000) -> dict[str, Any]:
    _job_or_404(request, job_id)
    path = request.app.state.settings.log_path(job_id)
    if not path.exists():
        return {"text": "", "offset": 0, "size": 0}
    size = path.stat().st_size
    offset = max(0, min(offset, size))
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(limit)
    return {"text": data.decode("utf-8", errors="replace"), "offset": offset + len(data), "size": size}


@router.get("/jobs/{job_id}/summaries")
def job_summaries(request: Request, job_id: str, since: int = 0) -> list[dict[str, Any]]:
    _job_or_404(request, job_id)
    return _jobs(request).summaries(job_id, since)


@router.websocket("/ws")
async def events(ws: WebSocket) -> None:
    hub = ws.app.state.hub
    await ws.accept()
    q = hub.subscribe()
    try:
        await ws.send_json({"type": "hello"})
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=20)
            except asyncio.TimeoutError:
                msg = {"type": "ping"}
            await ws.send_json(msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unsubscribe(q)
