"""Calibration stage (U2 / B5): cached detections, per-view details, view exclusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from stereodic.calibration.report import detections_path, load_calibration, load_calibration_details, load_detections
from stereodic.core.config import load_experiment
from stereodic.pipeline import run_calibration, run_calibration_detect, stage_status
from stereodic.synth.demo import DemoSpec, _scene, make_demo_dataset


def test_detect_calibrate_exclude(tmp_path: Path):
    spec = DemoSpec(name="cal", n_frames=1, n_calib_views=9, oversample=1)
    make_demo_dataset(tmp_path / "data", spec, workspace=tmp_path / "ws", workers=1)
    exp = tmp_path / "ws" / "experiment.yaml"
    local = tmp_path / "none.yaml"
    cfg = load_experiment(exp, local_path=local)
    assert stage_status(cfg)["calibrate"]["state"] == "missing"

    det = run_calibration_detect(cfg)
    dl, dr, names, meta = load_detections(det)
    assert len(dl) == len(dr) == 9 and len(names) == 9
    assert sum(d is not None for d in dl) >= 8 and meta["n_corners_board"] == 48
    mtime = det.stat().st_mtime_ns
    assert run_calibration_detect(cfg) == det and det.stat().st_mtime_ns == mtime  # cached

    res = run_calibration(cfg, plot=False)
    truth = _scene(spec).calibration()
    assert res.rms < 0.1
    assert abs(res.calibration.baseline_mm / truth.baseline_mm - 1) < 5e-3
    cal_file = cfg.data_path(cfg.calibration_file)
    np.testing.assert_allclose(load_calibration(cal_file).R_RL, res.calibration.R_RL)
    d = load_calibration_details(cal_file)
    assert d is not None and list(d["pairs"]) == list(res.pairs)
    assert len(d["ids"]) == len(d["residuals_left"]) == d["pose_tvec"].shape[0] == len(res.pairs)
    assert all(len(i) == len(r) for i, r in zip(d["ids"], d["residuals_right"]))
    assert 200 < np.linalg.norm(d["pose_tvec"], axis=1).mean() < 300  # board ~250 mm away
    assert stage_status(cfg)["calibrate"]["state"] == "done"

    # exclude two views: detections are reused, those views disappear from the pairs
    raw = yaml.safe_load(exp.read_text())
    raw["calibration"]["exclude_views"] = [0, 3]
    exp.write_text(yaml.safe_dump(raw))
    cfg = load_experiment(exp, local_path=local)
    assert stage_status(cfg)["calibrate"]["state"] == "stale"
    res2 = run_calibration(cfg, plot=False)
    assert det.stat().st_mtime_ns == mtime
    assert 0 not in res2.pairs and 3 not in res2.pairs and len(res2.pairs) == len(res.pairs) - 2
    report = json.loads(cal_file.with_suffix(".json").read_text())["report"]
    assert report["excluded_views"] == [0, 3]
    assert stage_status(cfg)["calibrate"]["state"] == "done"
    assert detections_path(cal_file) == det
