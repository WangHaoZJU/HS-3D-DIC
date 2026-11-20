"""Typed configuration: defaults YAML + experiment YAML + machine-local paths.

Merge order (later wins): configs/defaults/*.yaml -> experiment file.
Machine-specific roots come from configs/local.yaml or STEREODIC_DATA_ROOT /
STEREODIC_RESULTS_ROOT environment variables (env vars win).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_DIR = REPO_ROOT / "configs" / "defaults"
LOCAL_CONFIG = REPO_ROOT / "configs" / "local.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ----------------------------------------------------------------------------- DIC
class PoiConfig(_Strict):
    subset_size: int = Field(31, ge=5)
    step: int = Field(5, ge=1)

    @field_validator("subset_size")
    @classmethod
    def _odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("subset_size must be odd")
        return v


class InterpolationConfig(_Strict):
    order: Literal[3, 5] = 3


class SolverConfig(_Strict):
    max_iterations: int = Field(50, ge=1, le=255)  # stored as uint8
    convergence_threshold_px: float = Field(1e-3, gt=0)


class StereoConfig(_Strict):
    shape_order: Literal[1, 2] = 2
    initialization: Literal["sift_affine", "epipolar_zncc"] = "sift_affine"
    reliability_guided: bool = True


class TemporalConfig(_Strict):
    shape_order: Literal[0, 1, 2] = 1
    initialization: Literal["previous_solution", "zero"] = "previous_solution"
    frame_batch: int = Field(64, ge=1)


class QualityConfig(_Strict):
    fb_stride: int = Field(50, ge=0)


class StorageConfig(_Strict):
    save_params: bool = False


class DICConfig(_Strict):
    backend: Literal["auto", "numpy", "numba", "cuda"] = "auto"
    poi: PoiConfig = PoiConfig()
    interpolation: InterpolationConfig = InterpolationConfig()
    solver: SolverConfig = SolverConfig()
    stereo: StereoConfig = StereoConfig()
    temporal: TemporalConfig = TemporalConfig()
    quality: QualityConfig = QualityConfig()
    storage: StorageConfig = StorageConfig()


# ---------------------------------------------------------------------- experiment
class CameraSource(_Strict):
    path: Path
    pattern: str = "*.tif"


class CamerasConfig(_Strict):
    left: CameraSource
    right: CameraSource
    frame_rate_hz: float = Field(gt=0)


class FramesConfig(_Strict):
    start: int = Field(0, ge=0)
    stop: int | None = None
    step: int = Field(1, ge=1)


class LocalConfig(_Strict):
    data_root: Path | None = None
    results_root: Path | None = None


class ExperimentConfig(_Strict):
    name: str
    description: str = ""
    cameras: CamerasConfig
    roi: list[int] | None = None
    frames: FramesConfig = FramesConfig()
    reference_frame: int = Field(0, ge=0)
    calibration_file: Path | None = None
    output_file: Path | None = None
    dic: DICConfig = DICConfig()
    local: LocalConfig = LocalConfig()

    @field_validator("roi")
    @classmethod
    def _roi(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and (len(v) != 4 or v[2] <= 0 or v[3] <= 0 or v[0] < 0 or v[1] < 0):
            raise ValueError("roi must be [x0, y0, width, height] with positive size")
        return v

    # Path resolution ----------------------------------------------------------
    def data_path(self, p: Path) -> Path:
        return _resolve(p, self.local.data_root, "data_root")

    def results_path(self, p: Path) -> Path:
        return _resolve(p, self.local.results_root, "results_root")

    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable snapshot (stored in HDF5 metadata)."""
        return self.model_dump(mode="json")


def _resolve(p: Path, root: Path | None, root_name: str) -> Path:
    p = Path(p).expanduser()
    if p.is_absolute():
        return p
    if root is None:
        raise ValueError(f"relative path {p} needs {root_name} (configs/local.yaml or env var)")
    return Path(root).expanduser() / p


# ------------------------------------------------------------------------- loading
def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; values in `override` win. Inputs are not modified."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_local_config(path: str | Path | None = None) -> LocalConfig:
    path = Path(path) if path is not None else LOCAL_CONFIG
    data = load_yaml(path) if path.exists() else {}
    env = {
        "data_root": os.environ.get("STEREODIC_DATA_ROOT"),
        "results_root": os.environ.get("STEREODIC_RESULTS_ROOT"),
    }
    data.update({k: v for k, v in env.items() if v})
    return LocalConfig.model_validate(data)


def load_experiment(
    path: str | Path,
    defaults_dir: str | Path | None = DEFAULTS_DIR,
    local_path: str | Path | None = None,
) -> ExperimentConfig:
    """Load an experiment file merged on top of the defaults directory."""
    merged: dict[str, Any] = {}
    if defaults_dir is not None:
        dic_defaults = Path(defaults_dir) / "dic.yaml"
        if dic_defaults.exists():
            merged["dic"] = load_yaml(dic_defaults)
    merged = deep_merge(merged, load_yaml(path))
    cfg = ExperimentConfig.model_validate(merged)
    cfg.local = load_local_config(local_path)
    return cfg


def config_hash(obj: BaseModel | dict[str, Any]) -> str:
    """Stable SHA-256 of a config (or sub-config), used for cache invalidation."""
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
