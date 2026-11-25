"""run_analyze / run_noise_floor / run_uncertainty on the synthetic vibrating demo (U5, B2/B3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stereodic.analysis import AnalyzeOptions, run_analyze, run_noise_floor, run_uncertainty
from stereodic.core.config import load_experiment
from stereodic.io.storage import ResultStore
from stereodic.pipeline import run_reconstruct, run_stereo, run_temporal, stage_status
from stereodic.synth.demo import DemoSpec, make_demo_dataset


def test_analysis_stages(tmp_path: Path):
    spec = DemoSpec(name="ana", image_size=(320, 256), n_frames=48, n_calib_views=0, poi_step=28)
    make_demo_dataset(tmp_path / "data", spec, workspace=tmp_path / "ws", workers=4, exact_calibration=True)
    cfg = load_experiment(tmp_path / "ws" / "experiment.yaml", local_path=tmp_path / "none.yaml")
    out = run_stereo(cfg)
    run_temporal(cfg)
    run_reconstruct(cfg)

    s = run_analyze(cfg, AnalyzeOptions(direction=2, fit_peaks=2, fdd=True, n_modes=2))
    fitted = sorted(h["frequency_hz"] for h in s["harmonic"])
    # torsion 137 Hz and bending 412 Hz; the 41.7 Hz bin spacing is refined by the field energy search
    assert abs(fitted[0] - spec.f_torsion_hz) < 1.0 and abs(fitted[1] - spec.f_bending_hz) < 1.0
    with ResultStore.open(out, "r") as store:
        h5 = store.h5
        assert h5["vibration/spectrum_amplitude_all"].shape[1] == 3
        assert h5["vibration/damping"].shape == (h5["vibration/frequencies"].shape[0], 5)
        keys = list(h5["vibration/harmonic"].keys())
        amp = h5[f"vibration/harmonic/{keys[0]}/amplitude"][()]
        assert amp.shape == (1, h5["reference/points_3d"].shape[0], 3) and np.isfinite(amp).all()
        assert h5["vibration/fdd/peak_mode_shapes"].shape[1] == amp.shape[1]
    st = stage_status(cfg)
    assert st["analyze"]["state"] == "done" and st["noise_floor"]["state"] == "missing"

    nf = run_noise_floor(cfg)
    assert nf["n_frames"] == 48 and len(nf["median_std_mm"]) == 3
    un = run_uncertainty(cfg, pixel_sigma=0.01, monte_carlo=50)
    assert un["key"] == "reference" and 0 < un["median_std_mm"][2] < 0.01  # out-of-plane, mm
    st = stage_status(cfg)
    assert st["noise_floor"]["state"] == "done" and st["uncertainty"]["state"] == "done"
