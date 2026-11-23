from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import yaml

from stereodic.calibration.report import save_calibration
from stereodic.core.config import REPO_ROOT, load_experiment
from stereodic.core.status import Status
from stereodic.io.storage import ResultStore
from stereodic.pipeline import run_rbm, run_reconstruct, run_stereo, run_temporal
from stereodic.synth import SensorModel
from stereodic.synth.stereo import cylinder_height, make_stereo_scene


def test_temporal_and_reconstruction_pipeline(tmp_path: Path):
    scene = make_stereo_scene(image_size=(320, 256), height=cylinder_height(80.0), seed=3)
    data_root, results_root = tmp_path / "data", tmp_path / "results"
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({"data_root": str(data_root), "results_root": str(results_root)}))
    exp = yaml.safe_load((REPO_ROOT / "configs/experiments/example.yaml").read_text())
    exp["roi"] = [20, 20, 280, 216]
    exp["frames"] = {"start": 0, "stop": 4, "step": 1}
    exp["dic"] = {"poi": {"subset_size": 31, "step": 12}, "temporal": {"shape_order": 1, "max_znssd": 0.8}}
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(yaml.safe_dump(exp))
    cfg = load_experiment(exp_path, local_path=local)

    for side in ("left", "right"):
        cfg.data_path(getattr(cfg.cameras, side).path).mkdir(parents=True)
    for k, shift in enumerate((0.0, 0.08, 0.16, 0.24)):
        target = scene.target.moved(np.eye(3), np.array([0.0, 0.0, shift]))
        left, right = scene.render(target=target, sensor=SensorModel(noise_std=0.0), oversample=3, seed=k)
        tifffile.imwrite(cfg.data_path(cfg.cameras.left.path) / f"img{k:04d}.tif", left)
        tifffile.imwrite(cfg.data_path(cfg.cameras.right.path) / f"img{k:04d}.tif", right)
    calibration_path = cfg.data_path(cfg.calibration_file)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(calibration_path, scene.calibration())

    out = run_stereo(cfg)
    run_temporal(cfg)
    run_reconstruct(cfg)
    run_rbm(cfg)
    with ResultStore.open(out, "r") as store:
        assert store.stage_done("temporal", store.h5["progress"].attrs["temporal_config_hash"])
        assert store.stage_done("reconstruct", store.h5["progress"].attrs["reconstruct_config_hash"])
        left_status = store.read_array("dic/left/status")
        recon_status = store.read_array("reconstruction/status")
        assert left_status.shape == recon_status.shape
        assert left_status.shape[0] == 4
        assert np.all(left_status == Status.OK)
        assert np.all(recon_status == Status.OK)
        assert np.isfinite(store.read_array("reconstruction/displacement_raw")).all()
        assert store.read_array("reconstruction/displacement_rbm").shape == store.read_array("reconstruction/displacement_raw").shape

    # A second invocation uses the completion markers and leaves the result intact.
        before = store.read_array("reconstruction/displacement_raw")
    run_temporal(cfg)
    run_reconstruct(cfg)
    with ResultStore.open(out, "r") as store:
        np.testing.assert_array_equal(store.read_array("reconstruction/displacement_raw"), before)


def test_reconstruct_applies_sensor_origin_and_recovers_from_transient_failure(tmp_path: Path):
    """Non-zero sensor_origin must be added before triangulation, left errors must be preserved,
    and a transient single-frame failure must not poison previous_params on subsequent frames."""
    scene = make_stereo_scene(image_size=(320, 256), height=cylinder_height(80.0), seed=5)
    origin_l = (24, 16)
    origin_r = (12, 20)
    crop_w, crop_h = 260, 210

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({"data_root": str(data_root), "results_root": str(results_root)}))
    exp = yaml.safe_load((REPO_ROOT / "configs/experiments/example.yaml").read_text())
    exp["cameras"]["left"]["sensor_origin"] = list(origin_l)
    exp["cameras"]["right"]["sensor_origin"] = list(origin_r)
    exp["roi"] = [16, 16, crop_w - 32, crop_h - 32]
    exp["frames"] = {"start": 0, "stop": 3, "step": 1}
    exp["dic"] = {"poi": {"subset_size": 31, "step": 18}, "temporal": {"shape_order": 1, "max_znssd": 0.4}}
    exp_path = tmp_path / "exp_crop.yaml"
    exp_path.write_text(yaml.safe_dump(exp))
    cfg = load_experiment(exp_path, local_path=local)

    for side in ("left", "right"):
        cfg.data_path(getattr(cfg.cameras, side).path).mkdir(parents=True)
    shifts = (0.0, 0.08, 0.16)
    for k, shift in enumerate(shifts):
        target = scene.target.moved(np.eye(3), np.array([0.0, 0.0, shift]))
        left, right = scene.render(target=target, sensor=SensorModel(noise_std=0.0), oversample=3, seed=10 + k)
        left_crop = left[origin_l[1] : origin_l[1] + crop_h, origin_l[0] : origin_l[0] + crop_w].copy()
        right_crop = right[origin_r[1] : origin_r[1] + crop_h, origin_r[0] : origin_r[0] + crop_w].copy()
        if k == 1:
            # Blank out a central patch on the left image for a single frame.
            left_crop[70:140, 90:170] = 0
        tifffile.imwrite(cfg.data_path(cfg.cameras.left.path) / f"img{k:04d}.tif", left_crop)
        tifffile.imwrite(cfg.data_path(cfg.cameras.right.path) / f"img{k:04d}.tif", right_crop)
    calibration_path = cfg.data_path(cfg.calibration_file)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(calibration_path, scene.calibration())

    out = run_stereo(cfg)
    run_temporal(cfg)
    run_reconstruct(cfg)
    with ResultStore.open(out, "r") as store:
        st_l = store.read_array("dic/left/status")
        st_rec = store.read_array("reconstruction/status")
        disp = store.read_array("reconstruction/displacement_raw")

        # Frame 1 must have occluded left POIs, and their exact left status must propagate (never NOT_COMPUTED).
        failed_l = st_l[1] != Status.OK
        assert failed_l.any()
        assert np.all(st_rec[1, failed_l] == st_l[1, failed_l])
        assert not np.any(st_rec[1, failed_l] == Status.NOT_COMPUTED)

        # Frame 2 must recover all occluded POIs back to OK with accurate 3D displacement.
        assert np.all(st_l[2] == Status.OK)
        assert np.all(st_rec[2] == Status.OK)
        np.testing.assert_allclose(disp[2, :, 2], shifts[2], atol=0.01)
        np.testing.assert_allclose(disp[2, :, :2], 0.0, atol=0.01)

    # Config change without force=True must invalidate and recompute temporal & reconstruct.
    cfg.dic.temporal.max_znssd = 1e-12
    run_temporal(cfg, force=False)
    run_reconstruct(cfg, force=False)
    with ResultStore.open(out, "r") as store:
        st_l_strict = store.read_array("dic/left/status")
        st_rec_strict = store.read_array("reconstruction/status")
        assert np.all(st_l_strict[2] == Status.HIGH_RESIDUAL)
        assert np.all(st_rec_strict[2] == Status.HIGH_RESIDUAL)
