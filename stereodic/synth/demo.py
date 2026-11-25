"""Demo dataset: synthetic stereo calibration views + a vibrating curved blade (S6-like).

Produces the same on-disk layout as real exported camera data, so the web UI and the
command-line pipeline can be exercised end to end without cameras or a GPU::

    <data_root>/<name>/
        calib/left/view0000.tif ...   ChArUco views (16-bit)
        calib/right/view0000.tif ...
        left/img00000.tif ...         vibration sequence (16-bit)
        right/img00000.tif ...
        timestamps_left.csv, timestamps_right.csv
        truth.json                    modal frequencies, amplitudes, phases
        experiment.yaml               ready-to-run experiment (absolute paths)

Motion: torsion about the target span axis at ``f_torsion`` plus a rigid out-of-plane
translation at ``f_bending``, both harmonic, so the spectrum has two known peaks. Either
mode is switched off with a zero amplitude. With ``exposure_us`` > 0 each frame is the
average of ``blur_samples`` instants spread over the exposure (motion blur, docs §3.2).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CHARUCO = {"squares_x": 9, "squares_y": 7, "square_mm": 5.0, "marker_mm": 3.75, "dictionary": "DICT_5X5_100"}


@dataclass(frozen=True)
class DemoSpec:
    name: str = "demo_s6"
    image_size: tuple[int, int] = (512, 416)
    n_frames: int = 160
    frame_rate_hz: float = 2000.0
    f_torsion_hz: float = 137.0
    torsion_deg: float = 0.5
    phase_torsion_rad: float = 0.4
    f_bending_hz: float = 412.0
    bending_mm: float = 0.02
    phase_bending_rad: float = -1.1
    n_calib_views: int = 12
    noise_std: float = 3.0
    oversample: int = 2
    poi_step: int = 12
    seed: int = 7
    exposure_us: float = 0.0  # 0: instantaneous frames; otherwise motion blur over this exposure
    blur_samples: int = 5


def _scene(spec: DemoSpec):
    from .stereo import cylinder_height, make_stereo_scene

    return make_stereo_scene(image_size=spec.image_size, height=cylinder_height(80.0), seed=spec.seed)


def _target_at(spec: DemoSpec, scene, t: float):
    from .stereo import rotation

    t0 = 0.0
    ang = spec.torsion_deg * (
        np.cos(2 * np.pi * spec.f_torsion_hz * t + spec.phase_torsion_rad)
        - np.cos(2 * np.pi * spec.f_torsion_hz * t0 + spec.phase_torsion_rad)
    )
    dz = spec.bending_mm * (
        np.cos(2 * np.pi * spec.f_bending_hz * t + spec.phase_bending_rad)
        - np.cos(2 * np.pi * spec.f_bending_hz * t0 + spec.phase_bending_rad)
    )
    pivot = scene.target.t
    R_rel = scene.target.R @ rotation(0.0, float(ang), 0.0) @ scene.target.R.T
    normal = scene.target.R[:, 2]
    t_rel = pivot - R_rel @ pivot + dz * normal
    return scene.target.moved(R_rel, t_rel)


def _render_frame(args: tuple[DemoSpec, int, str, str]) -> int:
    import tifffile

    from .render import SensorModel, apply_sensor
    from .stereo import render_view

    spec, k, left_dir, right_dir = args
    scene = _scene(spec)
    t = k / spec.frame_rate_hz
    sensor = SensorModel(noise_std=spec.noise_std)
    rng = np.random.default_rng(spec.seed * 1000 + k)
    if spec.exposure_us > 0 and spec.blur_samples > 1:
        e = spec.exposure_us * 1e-6
        times = t + np.linspace(-0.5 * e, 0.5 * e, spec.blur_samples)
    else:
        times = np.array([t])
    for cam, out_dir in ((scene.left, left_dir), (scene.right, right_dir)):
        trans = np.mean(
            [render_view(scene.texture, _target_at(spec, scene, tt), cam, spec.oversample) for tt in times], axis=0
        )
        tifffile.imwrite(Path(out_dir) / f"img{k:05d}.tif", apply_sensor(trans, sensor, rng))
    return k


def _board_pose(rng: np.random.Generator, center_z: float = 250.0):
    from .stereo import HeightFieldTarget, rotation

    R = rotation(*rng.uniform(-30, 30, 2), rng.uniform(-20, 20)) @ rotation(0, -10, 0)
    C = np.array([*rng.uniform(-4, 4, 2), center_z + rng.uniform(-15, 15)])
    sx, sy = CHARUCO["squares_x"] * CHARUCO["square_mm"], CHARUCO["squares_y"] * CHARUCO["square_mm"]
    return HeightFieldTarget(R=R, t=C - R @ np.array([sx / 2, sy / 2, 0.0]))


def _render_calibration(spec: DemoSpec, scene, out: Path) -> None:
    import tifffile

    from ..calibration.detect import CharucoBoardSpec
    from .charuco import board_raster, render_board_view
    from .render import SensorModel

    board = CharucoBoardSpec(**CHARUCO)
    raster = board_raster(board.board(), px_per_mm=40, blur_mm=0.05)
    rng = np.random.default_rng(spec.seed + 1)
    for side in ("left", "right"):
        (out / "calib" / side).mkdir(parents=True, exist_ok=True)
    for k in range(spec.n_calib_views):
        pose = _board_pose(rng)
        for side, cam in (("left", scene.left), ("right", scene.right)):
            img = render_board_view(raster, pose, cam, SensorModel(noise_std=4.0), 2, rng)
            tifffile.imwrite(out / "calib" / side / f"view{k:04d}.tif", img)


def experiment_dict(spec: DemoSpec, data_dir: Path, workspace: Path) -> dict[str, Any]:
    """Experiment config for the demo data with absolute paths (results go to ``workspace``)."""
    w, h = spec.image_size
    margin = 24
    return {
        "name": spec.name,
        "description": (
            f"Synthetic demo: curved target, torsion {spec.f_torsion_hz:g} Hz ({spec.torsion_deg:g} deg) + "
            f"out-of-plane {spec.f_bending_hz:g} Hz ({spec.bending_mm * 1e3:g} um), {spec.n_frames} frames at "
            f"{spec.frame_rate_hz:g} fps" + (f", exposure {spec.exposure_us:g} us" if spec.exposure_us > 0 else "")
        ),
        "cameras": {
            "left": {"path": str(data_dir / "left"), "pattern": "*.tif", "timestamp_path": str(data_dir / "timestamps_left.csv")},
            "right": {"path": str(data_dir / "right"), "pattern": "*.tif", "timestamp_path": str(data_dir / "timestamps_right.csv")},
            "frame_rate_hz": spec.frame_rate_hz,
        },
        "roi": [margin, margin, w - 2 * margin, h - 2 * margin],
        "frames": {"start": 0, "stop": None, "step": 1},
        "reference_frame": 0,
        "calibration_file": str(workspace / "calibration.h5"),
        "calibration": {
            "board": dict(CHARUCO),
            "left": {"path": str(data_dir / "calib" / "left"), "pattern": "*.tif"},
            "right": {"path": str(data_dir / "calib" / "right"), "pattern": "*.tif"},
            "distortion": "k1k2p1p2k3",
        },
        "output_file": str(workspace / "results.h5"),
        "dic": {"poi": {"subset_size": 31, "step": spec.poi_step}},
    }


def make_demo_dataset(
    data_root: str | Path,
    spec: DemoSpec = DemoSpec(),
    workspace: str | Path | None = None,
    workers: int | None = None,
    exact_calibration: bool = False,
) -> Path:
    """Render the demo dataset into ``data_root / spec.name`` and return that directory.

    ``workspace`` is where ``experiment.yaml`` points its calibration and results files
    (default: the data directory). ``exact_calibration`` additionally writes the true
    calibration to the workspace, so the pipeline can run without the calibration stage.
    """
    import yaml

    from ..calibration.report import save_calibration

    out = Path(data_root) / spec.name
    ws = Path(workspace) if workspace is not None else out
    for side in ("left", "right"):
        (out / side).mkdir(parents=True, exist_ok=True)
    ws.mkdir(parents=True, exist_ok=True)
    scene = _scene(spec)

    jobs = [(spec, k, str(out / "left"), str(out / "right")) for k in range(spec.n_frames)]
    workers = workers if workers is not None else min(8, os.cpu_count() or 1)
    if workers > 1 and spec.n_frames > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_render_frame, jobs, chunksize=max(1, spec.n_frames // (4 * workers))))
    else:
        for j in jobs:
            _render_frame(j)

    if spec.n_calib_views > 0:
        _render_calibration(spec, scene, out)

    t = np.arange(spec.n_frames) / spec.frame_rate_hz
    for side in ("left", "right"):
        np.savetxt(out / f"timestamps_{side}.csv", t, fmt="%.9f")

    truth = asdict(spec) | {
        "baseline_mm": float(scene.calibration().baseline_mm),
        "mm_per_px": scene.mm_per_px,
        "notes": "displacements are relative to frame 0; torsion about the target span (y) axis",
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    if exact_calibration:
        save_calibration(ws / "calibration.h5", scene.calibration())
    cfg = experiment_dict(spec, out, ws)
    (ws / "experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out
