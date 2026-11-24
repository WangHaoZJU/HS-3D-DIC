"""Experiments: list, create, read/write configuration, validation, stage status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from stereodic.core.config import ExperimentConfig
from stereodic.pipeline import STAGES, stage_status

from ..workspace import Workspace, WorkspaceError

router = APIRouter(prefix="/api", tags=["experiments"])

_status_cache: dict[str, dict[str, Any]] = {}


class CreateExperiment(BaseModel):
    name: str
    copy_from: str | None = None
    description: str = ""


class ConfigPayload(BaseModel):
    """Either a full/partial config mapping or YAML text."""

    config: dict[str, Any] | None = None
    yaml: str | None = None


class ExperimentDetail(BaseModel):
    name: str
    path: str
    yaml: str
    config: dict[str, Any]


class ValidationResult(BaseModel):
    valid: bool
    errors: list[dict[str, Any]] = Field(default_factory=list)
    config: dict[str, Any] | None = None
    invalidates: list[str] = Field(default_factory=list)


def _ws(request: Request) -> Workspace:
    return request.app.state.workspace


def _payload_dict(ws: Workspace, body: ConfigPayload) -> dict[str, Any]:
    if body.yaml is not None:
        return ws.parse_yaml(body.yaml)
    if body.config is not None:
        return body.config
    raise WorkspaceError("send either `config` or `yaml`")


def _detail(ws: Workspace, name: str) -> ExperimentDetail:
    cfg = ws.load(name)
    return ExperimentDetail(name=name, path=str(ws.config_path(name)), yaml=ws.raw_yaml(name), config=ws.effective(cfg))


@router.get("/schema/experiment")
def experiment_schema() -> dict[str, Any]:
    return ExperimentConfig.model_json_schema()


@router.get("/experiments")
def list_experiments(request: Request) -> list[dict[str, Any]]:
    return _ws(request).list()


@router.post("/experiments", status_code=201)
def create_experiment(request: Request, body: CreateExperiment) -> ExperimentDetail:
    ws = _ws(request)
    ws.create(body.name, body.copy_from, body.description)
    return _detail(ws, body.name)


@router.get("/experiments/{name}")
def get_experiment(request: Request, name: str) -> ExperimentDetail:
    ws = _ws(request)
    ws.require(name)
    try:
        return _detail(ws, name)
    except WorkspaceError:
        raise
    except Exception as exc:  # invalid YAML on disk: still return the text so it can be fixed
        return ExperimentDetail(name=name, path=str(ws.config_path(name)), yaml=ws.raw_yaml(name), config={"_error": str(exc)})


@router.put("/experiments/{name}/config")
def save_config(request: Request, name: str, body: ConfigPayload) -> ExperimentDetail:
    ws = _ws(request)
    ws.save(name, _payload_dict(ws, body))
    _status_cache.pop(name, None)
    return _detail(ws, name)


@router.post("/experiments/{name}/validate")
def validate_config(request: Request, name: str, body: ConfigPayload) -> ValidationResult:
    ws = _ws(request)
    ws.require(name)
    try:
        data = _payload_dict(ws, body)
    except WorkspaceError as exc:
        return ValidationResult(valid=False, errors=[{"loc": [], "msg": str(exc), "type": "yaml"}])
    cfg, errors = ws.validate(name, data)
    if cfg is None:
        return ValidationResult(valid=False, errors=errors)
    invalidates: list[str] = []
    if not request.app.state.jobs.is_busy(name):
        try:
            before = stage_status(ws.load(name))
            after = stage_status(cfg)
            invalidates = [s for s in STAGES if before[s]["state"] in ("done", "partial") and after[s]["state"] != before[s]["state"]]
        except Exception:  # noqa: BLE001 - preview only
            invalidates = []
    return ValidationResult(valid=True, config=ws.effective(cfg), invalidates=invalidates)


@router.delete("/experiments/{name}")
def delete_experiment(request: Request, name: str) -> dict[str, str]:
    if request.app.state.jobs.is_busy(name):
        raise WorkspaceError("a job is running for this experiment", 409)
    dst = _ws(request).delete(name)
    return {"moved_to": str(dst)}


@router.get("/experiments/{name}/status")
def experiment_status(request: Request, name: str) -> dict[str, Any]:
    ws = _ws(request)
    ws.require(name)
    jobs = request.app.state.jobs
    running = jobs.running.job if jobs.running is not None and jobs.running.job["experiment"] == name else None
    if jobs.is_busy(name):
        # never open a results file that a worker is writing; serve the last known status
        stages = {k: dict(v) for k, v in _status_cache.get(name, {s: {"state": "unknown", "detail": "",
                                                                        "done_utc": None, "fraction": None}
                                                                    for s in STAGES}).items()}
    else:
        try:
            stages = stage_status(ws.load(name))
        except Exception as exc:  # noqa: BLE001
            stages = {s: {"state": "unknown", "detail": str(exc), "done_utc": None, "fraction": None} for s in STAGES}
        _status_cache[name] = stages
    if running is not None:
        current = running.get("current_stage") or running["stage"]
        if current in stages:
            stages[current] = {**stages[current], "state": "running"}
    return {"stages": stages, "busy": jobs.is_busy(name), "running_job": running["id"] if running else None}


@router.get("/experiments/{name}/export.yaml", response_class=PlainTextResponse)
def export_yaml(request: Request, name: str) -> PlainTextResponse:
    text = _ws(request).raw_yaml(name)
    return PlainTextResponse(text, headers={"Content-Disposition": f'attachment; filename="{name}.yaml"'})
