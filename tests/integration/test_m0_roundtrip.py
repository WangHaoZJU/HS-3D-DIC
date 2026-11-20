"""M0 end-to-end: synthetic frames -> TIFF sequences -> reader -> HDF5 store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import yaml

from stereodic.core.config import REPO_ROOT, config_hash, load_experiment
from stereodic.core.types import ROI
from stereodic.io.sequence import frame_indices, open_sequence
from stereodic.io.storage import ResultStore
from stereodic.synth import SensorModel, SpeckleTexture, Translation, render_frame


def test_synthetic_sequence_roundtrip(tmp_path: Path):
    data_root, results_root = tmp_path / "data", tmp_path / "results"
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({"data_root": str(data_root), "results_root": str(results_root)}))
    exp = yaml.safe_load((REPO_ROOT / "configs" / "experiments" / "example.yaml").read_text())
    exp["roi"] = [4, 6, 40, 30]
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(yaml.safe_dump(exp))
    cfg = load_experiment(exp_path, local_path=local)

    # 8 frames of a rigid sinusoidal motion, rendered independently for left/right
    tex = SpeckleTexture.random((-20, -20, 84, 68), seed=4)
    t = np.arange(8) / cfg.cameras.frame_rate_hz
    u = 0.4 * np.sin(2 * np.pi * 1000.0 * t)
    sensor = SensorModel(bits=12, noise_std=1.0)
    for side in ("left", "right"):
        d = cfg.data_path(getattr(cfg.cameras, side).path)
        d.mkdir(parents=True)
        for k, uk in enumerate(u):
            tifffile.imwrite(d / f"img{k}.tif", render_frame(tex, (48, 64), Translation(uk, 0.0), sensor, 2, seed=k))

    roi = ROI.from_list(cfg.roi)
    left = open_sequence(cfg.data_path(cfg.cameras.left.path), cfg.cameras.left.pattern)
    idx = frame_indices(len(left), cfg.frames.start, cfg.frames.stop, cfg.frames.step)
    frames = [img for _, img in left.iter_frames(idx, roi=roi, workers=2)]
    assert len(frames) == 8 and frames[0].shape == (30, 40) and frames[0].dtype == np.uint16

    out = cfg.results_path(cfg.output_file)
    with ResultStore.create(out, config_snapshot=cfg.snapshot()) as st:
        st.write_array("time/timestamps", t, units="s")
        st.create_timeseries("dic/left/xy", len(idx), 10, 2)
        st.mark_stage_done("temporal", config_hash(cfg.dic))
    with ResultStore.open(out, "r") as st:
        assert st.stage_done("temporal", config_hash(cfg.dic))
        np.testing.assert_array_equal(st.read_array("time/timestamps"), t)
        assert st.metadata()["config_snapshot"]["roi"] == [4, 6, 40, 30]
