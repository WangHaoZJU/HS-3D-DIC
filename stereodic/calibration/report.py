"""Calibration file I/O and report figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.types import StereoCalibration
from ..io.storage import ResultStore
from .calibrate import StereoResult
from .detect import BoardDetection


def save_calibration(
    path: str | Path,
    result: StereoResult | StereoCalibration,
    roi_origin: tuple[int, int] = (0, 0),
    extra: dict | None = None,
) -> Path:
    """Write a calibration HDF5 file (calibration/ group + JSON report) and a sidecar .json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    calib = result.calibration if isinstance(result, StereoResult) else result
    report = result.report() if isinstance(result, StereoResult) else {}
    report.update(extra or {})
    with ResultStore.create(path, config_snapshot=extra or {}, overwrite=True) as store:
        store.write_calibration(calib, roi_origin)
        store.h5["calibration"].attrs["report"] = json.dumps(report, sort_keys=True, default=float)
        if isinstance(result, StereoResult) and result.pair_ids:
            _write_details(store.h5.require_group("details"), result)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"calibration": calib.to_dict(), "roi_origin": list(roi_origin), "report": report}, indent=2, default=float),
        encoding="utf-8",
    )
    return path


def _ragged(group, name: str, arrays: list[np.ndarray]) -> None:
    """Store a list of (K_i, ...) arrays as one concatenated dataset plus counts."""
    counts = np.asarray([len(a) for a in arrays], dtype=np.int64)
    trailing = arrays[0].shape[1:] if arrays else ()
    data = np.concatenate(arrays) if arrays else np.zeros((0, *trailing))
    group.create_dataset(name, data=data)
    group.create_dataset(name + "_counts", data=counts)


def _read_ragged(group, name: str) -> list[np.ndarray]:
    data = group[name][()]
    counts = group[name + "_counts"][()]
    return np.split(data, np.cumsum(counts)[:-1]) if counts.size else []


def _write_details(g, result: StereoResult) -> None:
    """Per stereo pair: corner ids, measured points, residuals and board pose (for the UI)."""
    g.create_dataset("pairs", data=np.asarray(result.pairs, dtype=np.int64))
    g.create_dataset("view_rms_left", data=np.asarray(result.view_rms_left))
    g.create_dataset("view_rms_right", data=np.asarray(result.view_rms_right))
    _ragged(g, "ids", [np.asarray(i, dtype=np.int64) for i in result.pair_ids])
    _ragged(g, "points_left", result.pair_points_left)
    _ragged(g, "points_right", result.pair_points_right)
    _ragged(g, "residuals_left", result.residuals_left)
    _ragged(g, "residuals_right", result.residuals_right)
    g.create_dataset("pose_rvec", data=np.asarray([p[0] for p in result.poses]).reshape(-1, 3))
    g.create_dataset("pose_tvec", data=np.asarray([p[1] for p in result.poses]).reshape(-1, 3))
    for side, m in (("left", result.left), ("right", result.right)):
        g.create_dataset(f"mono_used_{side}", data=m.used)
        g.create_dataset(f"mono_view_rms_{side}", data=m.view_rms)


def load_calibration_details(path: str | Path) -> dict | None:
    """Per-pair details written by `save_calibration` (None for files without them)."""
    import h5py

    with h5py.File(path, "r") as f:
        if "details" not in f:
            return None
        g = f["details"]
        out = {k: g[k][()] for k in ("pairs", "view_rms_left", "view_rms_right", "pose_rvec", "pose_tvec")}
        for k in ("ids", "points_left", "points_right", "residuals_left", "residuals_right"):
            out[k] = _read_ragged(g, k)
        for side in ("left", "right"):
            out[f"mono_used_{side}"] = g[f"mono_used_{side}"][()]
            out[f"mono_view_rms_{side}"] = g[f"mono_view_rms_{side}"][()]
        return out


# ------------------------------------------------------------------ detections
def detections_path(calibration_file: str | Path) -> Path:
    p = Path(calibration_file)
    return p.with_name(p.stem + "_detections.h5")


def save_detections(
    path: str | Path,
    left: list[BoardDetection | None],
    right: list[BoardDetection | None],
    names: list[tuple[str, str]],
    meta: dict,
) -> Path:
    """ChArUco detections of every calibration view (both cameras), reusable by `calibrate`."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as f:
        f.attrs["meta"] = json.dumps(meta, sort_keys=True, default=str)
        f.create_dataset("names_left", data=np.asarray([n[0] for n in names], dtype=object), dtype=h5py.string_dtype())
        f.create_dataset("names_right", data=np.asarray([n[1] for n in names], dtype=object), dtype=h5py.string_dtype())
        for side, dets in (("left", left), ("right", right)):
            g = f.create_group(side)
            found = np.asarray([d is not None for d in dets])
            g.create_dataset("found", data=found)
            _ragged(g, "ids", [d.ids if d is not None else np.zeros(0, np.int64) for d in dets])
            _ragged(g, "points", [d.image_points if d is not None else np.zeros((0, 2)) for d in dets])
            _ragged(g, "object_points", [d.object_points if d is not None else np.zeros((0, 3)) for d in dets])
    tmp.replace(path)
    return path


def load_detections(path: str | Path) -> tuple[list, list, list[tuple[str, str]], dict]:
    import h5py

    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["meta"])
        names = list(zip([s.decode() if isinstance(s, bytes) else s for s in f["names_left"][()]],
                         [s.decode() if isinstance(s, bytes) else s for s in f["names_right"][()]]))
        out = []
        for side in ("left", "right"):
            g = f[side]
            found = g["found"][()]
            ids = _read_ragged(g, "ids")
            pts = _read_ragged(g, "points")
            obj = _read_ragged(g, "object_points")
            out.append(
                [BoardDetection(ids[i].astype(np.int64), pts[i], obj[i]) if found[i] else None for i in range(found.size)]
            )
    return out[0], out[1], names, meta


def load_calibration(path: str | Path) -> StereoCalibration:
    """Read a calibration from an HDF5 file (calibration/ group) or a sidecar/plain .json."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return StereoCalibration.from_dict(data.get("calibration", data))
    with ResultStore.open(path, "r") as store:
        return store.read_calibration()


def plot_residuals(result: StereoResult, out_path: str | Path) -> Path:
    """Residual vectors and per-view RMS for both cameras."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, (name, res, vr) in enumerate(
        (("left", result.residuals_left, result.view_rms_left), ("right", result.residuals_right, result.view_rms_right))
    ):
        ax = axes[0, col]
        allr = np.vstack(res)
        ax.scatter(allr[:, 0], allr[:, 1], s=3, alpha=0.5)
        lim = max(0.05, 1.1 * np.abs(allr).max())
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="dx [px]", ylabel="dy [px]", title=f"{name}: residuals")
        ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax = axes[1, col]
        ax.bar(result.pairs, vr)
        ax.set(xlabel="view", ylabel="RMS [px]", title=f"{name}: per-view RMS")
    fig.suptitle(f"stereo RMS {result.rms:.4f} px, baseline {result.calibration.baseline_mm:.3f} mm")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
