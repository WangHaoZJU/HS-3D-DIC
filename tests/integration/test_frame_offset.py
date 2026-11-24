"""Per-camera frame offset (trigger alignment, B9): shifted export gives identical results."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from stereodic.core.config import load_experiment
from stereodic.io.storage import ResultStore
from stereodic.pipeline import run_reconstruct, run_stereo, run_temporal
from stereodic.synth.demo import DemoSpec, make_demo_dataset


def test_frame_offset_matches_aligned_run(tmp_path: Path):
    spec = DemoSpec(name="off", image_size=(320, 256), n_frames=5, n_calib_views=0, poi_step=24)
    make_demo_dataset(tmp_path / "data", spec, workspace=tmp_path / "ws", workers=1, exact_calibration=True)
    base = yaml.safe_load((tmp_path / "ws" / "experiment.yaml").read_text())
    local = tmp_path / "none.yaml"

    # left camera exported 2 extra pre-trigger frames (+ timestamps), right camera 1 extra post-trigger frame
    d = tmp_path / "data" / "off"
    shifted = d / "left_shifted"
    shifted.mkdir()
    src = sorted((d / "left").glob("*.tif"))
    for k, p in enumerate([src[-1], src[-2], *src]):
        shutil.copy(p, shifted / f"img{k:05d}.tif")
    ts = np.loadtxt(d / "timestamps_left.csv")
    np.savetxt(d / "timestamps_left_shifted.csv", np.r_[ts[0] - 2e-3, ts[0] - 1e-3, ts], fmt="%.9f")
    shutil.copy(sorted((d / "right").glob("*.tif"))[-1], d / "right" / "img99999.tif")

    results = {}
    for tag, left_path, off, ts_path, stop in (
        ("aligned", base["cameras"]["left"]["path"], 0, base["cameras"]["left"]["timestamp_path"], 5),
        ("offset", str(shifted), 2, str(d / "timestamps_left_shifted.csv"), 5),
    ):
        cfg_d = yaml.safe_load(yaml.safe_dump(base))
        cfg_d["cameras"]["left"].update(path=left_path, frame_offset=off, timestamp_path=ts_path)
        cfg_d["cameras"]["right"]["timestamp_path"] = None  # right has an extra frame; use the frame rate
        cfg_d["cameras"]["left"]["timestamp_path"] = None if tag == "aligned" else ts_path
        cfg_d["frames"] = {"start": 0, "stop": stop, "step": 1}
        cfg_d["output_file"] = str(tmp_path / f"{tag}.h5")
        p = tmp_path / f"{tag}.yaml"
        p.write_text(yaml.safe_dump(cfg_d))
        cfg = load_experiment(p, local_path=local)
        out = run_stereo(cfg)
        run_temporal(cfg)
        run_reconstruct(cfg)
        with ResultStore.open(out, "r") as store:
            results[tag] = (store.read_array("reconstruction/displacement_local"), store.read_array("time/frame_ids"))

    np.testing.assert_allclose(results["offset"][0], results["aligned"][0], atol=1e-9)
    np.testing.assert_array_equal(results["offset"][1][:, 0], np.arange(5) + 2)  # raw export indices
    np.testing.assert_array_equal(results["offset"][1][:, 1], np.arange(5))

    # unequal lengths without limiting frames.stop are refused with a hint
    cfg_d["frames"] = {"start": 0, "stop": None, "step": 1}
    cfg_d["output_file"] = str(tmp_path / "bad.h5")
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg_d))
    cfg = load_experiment(p, local_path=local)
    run_stereo(cfg)
    with pytest.raises(ValueError, match="frame_offset"):
        run_temporal(cfg)
