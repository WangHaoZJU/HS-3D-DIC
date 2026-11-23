"""S6 end-to-end synthetic verification (docs §14.1):
curved stereo surface undergoing known 3D antisymmetric (torsional) harmonic vibration,
verified through stereo -> temporal -> reconstruct -> field PSD -> ODS -> FDD -> MAC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import yaml

from stereodic.calibration.report import save_calibration
from stereodic.core.config import REPO_ROOT, load_experiment
from stereodic.core.status import Status
from stereodic.io.storage import ResultStore
from stereodic.pipeline import run_reconstruct, run_stereo, run_temporal
from stereodic.recon.frames import to_local
from stereodic.synth import SensorModel
from stereodic.synth.stereo import cylinder_height, make_stereo_scene, rotation
from stereodic.vibration.fdd import frequency_domain_decomposition
from stereodic.vibration.mac import modal_assurance
from stereodic.vibration.ods import extract_ods
from stereodic.vibration.spectrum import dominant_peaks, field_average_psd


def test_s6_curved_stereo_harmonic_vibration_end_to_end(tmp_path: Path):
    fs = 1000.0
    n_frames = 24  # 3 full cycles at 125 Hz
    f0 = 125.0
    phase0 = 0.45
    theta0_deg = 0.6  # torsional rocking amplitude around target span axis (~0.1 mm at edges)
    times = np.arange(n_frames, dtype=np.float64) / fs

    full_size = (280, 220)
    origin_l = (16, 12)
    origin_r = (10, 14)
    crop_w, crop_h = 240, 188
    scene = make_stereo_scene(image_size=full_size, height=cylinder_height(80.0), seed=19)

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({"data_root": str(data_root), "results_root": str(results_root)}))
    exp = yaml.safe_load((REPO_ROOT / "configs/experiments/example.yaml").read_text())
    exp["cameras"]["frame_rate_hz"] = fs
    exp["cameras"]["left"]["sensor_origin"] = list(origin_l)
    exp["cameras"]["right"]["sensor_origin"] = list(origin_r)
    exp["roi"] = [14, 14, crop_w - 28, crop_h - 28]
    exp["frames"] = {"start": 0, "stop": n_frames, "step": 1}
    exp["reference_frame"] = 0
    exp["dic"] = {
        "poi": {"subset_size": 31, "step": 22},
        "temporal": {"shape_order": 1, "max_znssd": 0.4, "frame_batch": 8},
    }
    exp_path = tmp_path / "s6.yaml"
    exp_path.write_text(yaml.safe_dump(exp))
    cfg = load_experiment(exp_path, local_path=local)

    for side in ("left", "right"):
        cfg.data_path(getattr(cfg.cameras, side).path).mkdir(parents=True)

    # Frame 0 is the undeformed reference pose; harmonic motion is centered around the target origin.
    # For frame 0, evaluate the exact harmonic angle so t=0 is part of the sinusoid; shift the
    # scene reference pose to angle(t=0) so frame 0 == reference pose.
    angles = theta0_deg * np.cos(2.0 * np.pi * f0 * times + phase0)
    pivot = scene.target.t
    targets = []
    for k, ang in enumerate(angles):
        # Rotate around the target centre along its local y (span) axis relative to frame 0
        R_rel = scene.target.R @ rotation(0.0, float(ang - angles[0]), 0.0) @ scene.target.R.T
        t_rel = pivot - R_rel @ pivot
        tgt = scene.target.moved(R_rel, t_rel)
        targets.append(tgt)
        left, right = scene.render(target=tgt, sensor=SensorModel(noise_std=3.0), oversample=3, seed=100 + k)
        tifffile.imwrite(
            cfg.data_path(cfg.cameras.left.path) / f"img{k:04d}.tif",
            left[origin_l[1] : origin_l[1] + crop_h, origin_l[0] : origin_l[0] + crop_w],
        )
        tifffile.imwrite(
            cfg.data_path(cfg.cameras.right.path) / f"img{k:04d}.tif",
            right[origin_r[1] : origin_r[1] + crop_h, origin_r[0] : origin_r[0] + crop_w],
        )

    calibration_path = cfg.data_path(cfg.calibration_file)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(calibration_path, scene.calibration())

    out = run_stereo(cfg)
    run_temporal(cfg)
    run_reconstruct(cfg)

    with ResultStore.open(out, "r") as store:
        pts_left_full = store.read_array("reference/points_left")
        frames_local = store.read_array("reference/local_frames")
        recon_status = store.read_array("reconstruction/status")
        disp_local = store.read_array("reconstruction/displacement_local")
        timestamps = store.read_array("time/timestamps")

    assert np.all(recon_status == Status.OK)

    # Exact synthetic ground truth in blade-local coordinates
    truth_world = np.stack([scene.truth_from_left(pts_left_full, tgt)["X"] for tgt in targets], axis=0)
    truth_disp_world = truth_world - truth_world[0:1]
    truth_disp_local = to_local(frames_local, truth_disp_world)

    # 1. Frequency identification from full-field averaged Welch PSD (must detect antisymmetric mode)
    spec = field_average_psd(disp_local[:, :, 2], timestamps, nperseg=16)
    peaks, _ = dominant_peaks(spec, min_frequency_hz=30.0, max_peaks=3, refine=True)
    assert peaks.size >= 1
    assert abs(peaks[0] - f0) < 2.0

    # 2. ODS harmonic fit: verify full-field amplitude and phase against exact truth
    ods_meas = extract_ods(disp_local[:, :, 2], timestamps, f0)
    ods_true = extract_ods(truth_disp_local[:, :, 2], timestamps, f0)
    amp_err = np.sqrt(np.mean((ods_meas.amplitude[0] - ods_true.amplitude[0]) ** 2))
    assert amp_err < 0.005  # < 5 um RMSE on curved blade

    strong = ods_true.amplitude[0] > 0.02
    assert strong.sum() >= 10
    phase_diff = np.angle(np.exp(1j * (ods_meas.phase[0, strong] - ods_true.phase[0, strong])))
    assert np.max(np.abs(phase_diff)) < 0.05  # < 0.05 rad

    # 3. FDD thin SVD: frequency and MAC against true complex 3D mode shape
    fdd = frequency_domain_decomposition(disp_local[:, :, 2], timestamps, nperseg=16, n_modes=2)
    peak_bin = int(np.argmax(fdd.singular_values[:, 0]))
    assert abs(fdd.frequencies[peak_bin] - f0) < 1e-9
    mac = float(modal_assurance(fdd.mode_shapes[peak_bin, 0], ods_true.response[0]))
    assert mac > 0.995
