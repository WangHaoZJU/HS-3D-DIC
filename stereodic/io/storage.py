"""HDF5 result store (schema v1, see docs §12.1).

Single writer. Float time series are pre-filled with NaN; completion markers are
written only after the data has been flushed, so an interrupted run never looks
finished.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..core.provenance import git_info, run_environment
from ..core.types import CameraModel, StereoCalibration

SCHEMA_VERSION = "1"

UNITS = {"length": "mm", "time": "s", "frequency": "Hz", "phase": "rad", "image": "px"}
CONVENTIONS = {
    "image_xy": "x right, y down, zero-based, integer = pixel centre, full-sensor frame",
    "array_index": "image[y, x]",
    "world_frame": "left camera, right-handed",
    "extrinsics": "X_right = R_RL @ X_left + t_RL",
    "displacement": "X_t - X_0 relative to the fixed reference frame",
    "time_series_axes": "(T, N, C)",
    "invalid": "NaN + status code",
}

DEFAULT_CHUNK_T = 64
DEFAULT_CHUNK_N = 2048


def _json_attr(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class ResultStore:
    """Thin wrapper around an HDF5 results file.

    Use as a context manager::

        with ResultStore.create(path, config_snapshot=cfg.snapshot()) as store:
            ds = store.create_timeseries("dic/left/xy", T, N, 2)
    """

    def __init__(self, h5: h5py.File) -> None:
        self.h5 = h5

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def create(
        cls,
        path: str | Path,
        config_snapshot: dict[str, Any] | None = None,
        input_manifest: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> ResultStore:
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists (pass overwrite=True or open with ResultStore.open)")
        path.parent.mkdir(parents=True, exist_ok=True)
        h5 = h5py.File(path, "w")
        store = cls(h5)
        meta = h5.require_group("metadata")
        meta.attrs["schema_version"] = SCHEMA_VERSION
        meta.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        git = git_info()
        meta.attrs["git_commit"] = git["commit"] or ""
        meta.attrs["git_dirty"] = "unknown" if git["dirty"] is None else str(git["dirty"]).lower()
        meta.attrs["environment"] = _json_attr(run_environment())
        meta.attrs["units"] = _json_attr(UNITS)
        meta.attrs["conventions"] = _json_attr(CONVENTIONS)
        meta.attrs["config_snapshot"] = _json_attr(config_snapshot or {})
        meta.attrs["input_manifest"] = _json_attr(input_manifest or {})
        h5.require_group("progress")
        h5.flush()
        return store

    @classmethod
    def open(cls, path: str | Path, mode: str = "r+") -> ResultStore:
        h5 = h5py.File(path, mode)
        version = h5["metadata"].attrs.get("schema_version")
        if version != SCHEMA_VERSION:
            h5.close()
            raise ValueError(f"{path}: schema_version {version!r}, expected {SCHEMA_VERSION!r}")
        return cls(h5)

    def close(self) -> None:
        if self.h5.id.valid:
            self.h5.close()

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def flush(self) -> None:
        self.h5.flush()

    # ------------------------------------------------------------------ metadata
    def metadata(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.h5["metadata"].attrs.items():
            if isinstance(v, str) and v.startswith(("{", "[")):
                out[k] = json.loads(v)
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------ datasets
    def write_array(self, name: str, data: np.ndarray, **attrs: Any) -> h5py.Dataset:
        """Write (or replace) a small dataset in one go."""
        if name in self.h5:
            del self.h5[name]
        ds = self.h5.create_dataset(name, data=np.asarray(data))
        for k, v in attrs.items():
            ds.attrs[k] = v
        return ds

    def read_array(self, name: str) -> np.ndarray:
        return self.h5[name][()]

    def create_timeseries(
        self,
        name: str,
        n_frames: int,
        n_points: int,
        n_components: int | None = None,
        dtype: np.dtype | type = np.float64,
        fill_value: Any = None,
        chunk_t: int = DEFAULT_CHUNK_T,
        chunk_n: int = DEFAULT_CHUNK_N,
    ) -> h5py.Dataset:
        """Create a (T, N[, C]) dataset chunked for both per-frame writes and per-point reads.

        Float datasets default to NaN fill (= not computed / invalid).
        """
        dtype = np.dtype(dtype)
        shape: tuple[int, ...] = (n_frames, n_points) if n_components is None else (n_frames, n_points, n_components)
        chunks: tuple[int, ...] = (min(chunk_t, n_frames), min(chunk_n, n_points)) + shape[2:]
        if fill_value is None:
            fill_value = np.nan if dtype.kind in "fc" else 0
        if name in self.h5:
            raise ValueError(f"dataset {name} already exists")
        return self.h5.create_dataset(
            name, shape=shape, dtype=dtype, chunks=chunks, compression="lzf", fillvalue=fill_value
        )

    # ------------------------------------------------------------------ calibration
    def write_calibration(self, calib: StereoCalibration, roi_origin: tuple[int, int] = (0, 0)) -> None:
        grp = self.h5.require_group("calibration")
        for side, cam in (("left", calib.left), ("right", calib.right)):
            for key, value in (("K", cam.K), ("D", cam.dist), ("image_size", np.array(cam.image_size))):
                name = f"{key}_{side}"
                if name in grp:
                    del grp[name]
                grp.create_dataset(name, data=value)
        for name, value in (("R_RL", calib.R_RL), ("t_RL", calib.t_RL), ("roi_origin", np.array(roi_origin))):
            if name in grp:
                del grp[name]
            grp.create_dataset(name, data=value)

    def read_calibration(self) -> StereoCalibration:
        g = self.h5["calibration"]
        cams = {
            side: CameraModel(K=g[f"K_{side}"][()], dist=g[f"D_{side}"][()], image_size=tuple(g[f"image_size_{side}"][()]))
            for side in ("left", "right")
        }
        return StereoCalibration(left=cams["left"], right=cams["right"], R_RL=g["R_RL"][()], t_RL=g["t_RL"][()])

    # ------------------------------------------------------------------ progress
    def mark_stage_done(self, stage: str, config_hash: str) -> None:
        """Flush data, then record the stage as complete for this config hash."""
        self.h5.flush()
        grp = self.h5.require_group("progress")
        grp.attrs[f"{stage}_done"] = True
        grp.attrs[f"{stage}_config_hash"] = config_hash
        grp.attrs[f"{stage}_done_utc"] = datetime.now(timezone.utc).isoformat()
        self.h5.flush()

    def invalidate_stage(self, stage: str) -> None:
        grp = self.h5.require_group("progress")
        grp.attrs[f"{stage}_done"] = False

    def stage_done(self, stage: str, config_hash: str) -> bool:
        """True only if the stage finished with exactly this config hash."""
        attrs = self.h5.require_group("progress").attrs
        return bool(attrs.get(f"{stage}_done", False)) and attrs.get(f"{stage}_config_hash") == config_hash
