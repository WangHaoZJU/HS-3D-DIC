"""Calibration file I/O and report figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.types import StereoCalibration
from ..io.storage import ResultStore
from .calibrate import StereoResult


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
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"calibration": calib.to_dict(), "roi_origin": list(roi_origin), "report": report}, indent=2, default=float),
        encoding="utf-8",
    )
    return path


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
