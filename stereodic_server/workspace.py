"""Experiment workspace: one directory per experiment under ``<results_root>/experiments``.

Layout (docs/UI_Design.md §4.1)::

    <workspace>/<name>/experiment.yaml   same format as configs/experiments/*.yaml
    <workspace>/<name>/masks/            masks drawn in the UI
    <workspace>/<name>/calibration.h5    default calibration_file
    <workspace>/<name>/results.h5        default output_file
    <workspace>/<name>/exports/

Experiments live outside the git checkout so the UI never leaves uncommitted changes
on the Linux machine; "export YAML" gives a file that can be committed by hand.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from stereodic.core.config import DEFAULTS_DIR, REPO_ROOT, ExperimentConfig, deep_merge, load_local_config, load_yaml

from .settings import Settings

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
CONFIG_NAME = "experiment.yaml"
TEMPLATE = REPO_ROOT / "configs" / "experiments" / "example.yaml"


class WorkspaceError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _defaults() -> dict[str, Any]:
    path = Path(DEFAULTS_DIR) / "dic.yaml"
    return {"dic": load_yaml(path)} if path.exists() else {}


def validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]


class Workspace:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.workspace

    # ------------------------------------------------------------------ paths
    def check_name(self, name: str) -> str:
        if not NAME_RE.match(name or ""):
            raise WorkspaceError("experiment name: letters, digits, '_', '-', '.'; up to 64 characters")
        return name

    def dir(self, name: str) -> Path:
        return self.root / self.check_name(name)

    def config_path(self, name: str) -> Path:
        return self.dir(name) / CONFIG_NAME

    def exists(self, name: str) -> bool:
        return self.config_path(name).exists()

    def require(self, name: str) -> Path:
        path = self.config_path(name)
        if not path.exists():
            raise WorkspaceError(f"experiment {name!r} not found", 404)
        return path

    # ------------------------------------------------------------------ read
    def list(self) -> list[dict[str, Any]]:
        out = []
        if not self.root.exists():
            return out
        for d in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            cfg_path = d / CONFIG_NAME
            if not cfg_path.exists():
                continue
            item: dict[str, Any] = {"name": d.name, "modified_utc": _utc(cfg_path.stat().st_mtime)}
            try:
                raw = load_yaml(cfg_path)
                item["description"] = str(raw.get("description", ""))
                cfg = self.load(d.name)
                item["valid"] = True
                item["frame_rate_hz"] = cfg.cameras.frame_rate_hz
            except Exception as exc:  # noqa: BLE001 - listing must not fail on one bad file
                item.setdefault("description", "")
                item["valid"] = False
                item["error"] = str(exc).splitlines()[0][:300]
            out.append(item)
        return out

    def raw_yaml(self, name: str) -> str:
        return self.require(name).read_text(encoding="utf-8")

    def load(self, name: str) -> ExperimentConfig:
        data = load_yaml(self.require(name))
        cfg = ExperimentConfig.model_validate(deep_merge(_defaults(), data))
        cfg.local = load_local_config(self.settings.local_path)
        return cfg

    @staticmethod
    def effective(cfg: ExperimentConfig) -> dict[str, Any]:
        return cfg.model_dump(mode="json", exclude={"local"})

    # ------------------------------------------------------------------ validate / write
    def validate(self, name: str, data: dict[str, Any]) -> tuple[ExperimentConfig | None, list[dict[str, Any]]]:
        data = dict(data)
        data["name"] = name
        data.pop("local", None)
        try:
            cfg = ExperimentConfig.model_validate(deep_merge(_defaults(), data))
        except ValidationError as exc:
            return None, validation_errors(exc)
        cfg.local = load_local_config(self.settings.local_path)
        return cfg, []

    @staticmethod
    def parse_yaml(text: str) -> dict[str, Any]:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise WorkspaceError(f"YAML syntax error: {exc}") from exc
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise WorkspaceError("YAML top level must be a mapping")
        return data

    def write(self, name: str, cfg: ExperimentConfig) -> Path:
        path = self.config_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.effective(cfg), sort_keys=False, allow_unicode=True)
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        return path

    def save(self, name: str, data: dict[str, Any]) -> ExperimentConfig:
        self.require(name)
        cfg, errors = self.validate(name, data)
        if cfg is None:
            raise WorkspaceError(f"invalid configuration: {errors[0]['loc']}: {errors[0]['msg']}", 422)
        self.write(name, cfg)
        return cfg

    # ------------------------------------------------------------------ create / delete
    def create(self, name: str, copy_from: str | None = None, description: str = "") -> ExperimentConfig:
        self.check_name(name)
        if self.dir(name).exists():
            raise WorkspaceError(f"experiment {name!r} already exists", 409)
        target = self.dir(name)
        if copy_from:
            data = load_yaml(self.require(copy_from))
            # results belong to the new experiment; the calibration file stays shared
            data["output_file"] = str(target / "results.h5")
        else:
            data = load_yaml(TEMPLATE)
            data["calibration_file"] = str(target / "calibration.h5")
            data["output_file"] = str(target / "results.h5")
        data["name"] = name
        if description:
            data["description"] = description
        cfg, errors = self.validate(name, data)
        if cfg is None:
            raise WorkspaceError(f"template is invalid: {errors[0]['loc']}: {errors[0]['msg']}", 500)
        (target / "masks").mkdir(parents=True, exist_ok=True)
        self.write(name, cfg)
        return cfg

    def delete(self, name: str) -> Path:
        """Move the experiment directory to ``<workspace>/.trash`` (recoverable)."""
        src = self.dir(name)
        self.require(name)
        trash = self.root / ".trash"
        trash.mkdir(parents=True, exist_ok=True)
        dst = trash / f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        shutil.move(str(src), str(dst))
        return dst
