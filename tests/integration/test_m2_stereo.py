"""M2: synthetic stereo scene -> stereo matching -> reference 3D shape (S4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml

from stereodic.calibration.report import save_calibration
from stereodic.core.config import REPO_ROOT, load_experiment
from stereodic.core.status import Status
from stereodic.dic.icgn import ICGNOptions
from stereodic.dic.mesh import make_poi_grid
from stereodic.dic.stereo_match import StereoMatchOptions, stereo_match
from stereodic.io.storage import ResultStore
from stereodic.pipeline import run_stereo
from stereodic.recon.frames import to_local
from stereodic.recon.shape import reference_shape
from stereodic.synth import SensorModel
from stereodic.synth.stereo import cylinder_height, make_stereo_scene

SIZE = (320, 256)


@pytest.fixture(scope="module")
def curved_scene():
    scene = make_stereo_scene(image_size=SIZE, height=cylinder_height(80.0), seed=3)
    left, right = scene.render(sensor=SensorModel(noise_std=2.0), oversample=3, seed=5)
    return scene, left, right


def test_stereo_match_and_shape(curved_scene):
    scene, left, right = curved_scene
    pois = make_poi_grid(left.shape, 31, 8, margin=12)
    truth = scene.truth_from_left(pois.xy)
    errs = {}
    for order in (1, 2):
        res = stereo_match(left, right, pois, scene.calibration(), StereoMatchOptions(icgn=ICGNOptions(31, order)))
        ok = res.result.status == Status.OK
        assert ok.mean() > 0.9
        e = res.xy_right[ok] - truth["xy_right"][ok]
        errs[order] = float(np.sqrt((e**2).sum(1).mean()))
        # invalid points carry a reason, never a position
        assert np.isnan(res.xy_right[~ok]).all()
        assert set(np.unique(res.result.status[~ok])) <= {Status.OUT_OF_BOUNDS, Status.HIGH_RESIDUAL, Status.NOT_COMPUTED}
    assert errs[2] < 0.005, errs  # px
    assert errs[1] > 3 * errs[2], errs  # perspective + curvature need the second-order warp

    shape = reference_shape(scene.calibration(), pois.xy, res.xy_right, res.result.status)
    ok = shape.status == Status.OK
    d = shape.points_3d[ok] - truth["X0"][ok]
    assert np.sqrt((d**2).sum(1).mean()) < 2e-3  # mm (field ~32 mm, depth ~250 mm)
    assert np.median(shape.reprojection_error[ok]) < 0.01
    assert len(shape.connectivity) > 0.8 * 2 * ok.sum()
    # normals: compare with the analytic cylinder normal in the world frame
    st = truth["st"][ok]
    _, hs, ht = scene.target.height(st[:, 0], st[:, 1])
    n_local = np.column_stack([-hs, -ht, np.ones_like(hs)])
    n_true = n_local @ scene.target.R.T
    n_true /= np.linalg.norm(n_true, axis=1, keepdims=True)
    n_true *= -np.sign((n_true * truth["X0"][ok]).sum(1))[:, None]  # face the camera
    cosang = np.abs((shape.normals[ok] * n_true).sum(1))
    assert np.nanpercentile(np.degrees(np.arccos(np.clip(cosang, -1, 1))), 95) < 1.0
    # local frames are orthonormal and right-handed
    F = shape.local_frames[ok]
    np.testing.assert_allclose(F @ np.transpose(F, (0, 2, 1)), np.broadcast_to(np.eye(3), F.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(F), 1.0, atol=1e-12)
    np.testing.assert_allclose(to_local(F, shape.normals[ok]), np.tile([0, 0, 1.0], (ok.sum(), 1)), atol=1e-12)


def test_run_stereo_pipeline(tmp_path: Path, curved_scene):
    scene, left, right = curved_scene
    data_root, results_root = tmp_path / "data", tmp_path / "results"
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({"data_root": str(data_root), "results_root": str(results_root)}))
    exp = yaml.safe_load((REPO_ROOT / "configs" / "experiments" / "example.yaml").read_text())
    exp["roi"] = [20, 20, 280, 216]
    exp["dic"] = {"poi": {"subset_size": 31, "step": 10}}
    # the recorded window starts at (8, 4) on the left sensor: crop the frame accordingly
    exp["cameras"]["left"]["sensor_origin"] = [8, 4]
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(yaml.safe_dump(exp))
    cfg = load_experiment(exp_path, local_path=local)

    for side, img in (("left", left[4:, 8:]), ("right", right)):
        d = cfg.data_path(getattr(cfg.cameras, side).path)
        d.mkdir(parents=True)
        tifffile.imwrite(d / "img0000.tif", img)
    save_calibration(cfg.data_path(cfg.calibration_file), scene.calibration())

    out = run_stereo(cfg)
    with ResultStore.open(out, "r") as store:
        assert store.stage_done("stereo", store.h5["progress"].attrs["stereo_config_hash"])
        pl = store.read_array("reference/points_left")
        X = store.read_array("reference/points_3d")
        st = store.read_array("reference/stereo_status")
        assert store.read_array("reference/point_ids").size == st.size == len(pl)
        assert store.read_array("reference/local_frames").shape == (st.size, 3, 3)
    ok = st == Status.OK
    assert ok.mean() > 0.9
    truth = scene.truth_from_left(pl)  # full-sensor left coordinates
    d = X[ok] - truth["X0"][ok]
    assert np.sqrt((d**2).sum(1).mean()) < 2e-3
    # second run is a no-op (stage up to date)
    mtime = out.stat().st_mtime_ns
    run_stereo(cfg)
    assert out.stat().st_mtime_ns == mtime


@pytest.mark.slow
def test_run_calibration_pipeline(tmp_path: Path):
    """Rendered ChArUco views on disk -> run_calibration -> calibration file usable by run_stereo."""
    from stereodic.calibration.detect import CharucoBoardSpec
    from stereodic.calibration.report import load_calibration
    from stereodic.pipeline import run_calibration
    from stereodic.synth.charuco import board_raster, render_board_view
    from stereodic.synth.stereo import HeightFieldTarget, rotation

    scene = make_stereo_scene()
    data_root = tmp_path / "data"
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({"data_root": str(data_root), "results_root": str(tmp_path / "results")}))
    exp = yaml.safe_load((REPO_ROOT / "configs" / "experiments" / "example.yaml").read_text())
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(yaml.safe_dump(exp))
    cfg = load_experiment(exp_path, local_path=local)
    b = cfg.calibration.board
    spec = CharucoBoardSpec(b.squares_x, b.squares_y, b.square_mm, b.marker_mm, b.dictionary)
    raster = board_raster(spec.board(), px_per_mm=40, blur_mm=0.05)
    rng = np.random.default_rng(7)
    dl = cfg.data_path(cfg.calibration.left.path)
    dr = cfg.data_path(cfg.calibration.right.path)
    dl.mkdir(parents=True)
    dr.mkdir(parents=True)
    w, h = spec.size_mm
    for k in range(9):
        R = rotation(*rng.uniform(-30, 30, 2), rng.uniform(-20, 20)) @ rotation(0, -10, 0)
        C = np.array([*rng.uniform(-4, 4, 2), 250 + rng.uniform(-15, 15)])
        pose = HeightFieldTarget(R=R, t=C - R @ np.array([w / 2, h / 2, 0.0]))
        tifffile.imwrite(dl / f"c{k:02d}.tif", render_board_view(raster, pose, scene.left, SensorModel(noise_std=4.0), 2, rng))
        tifffile.imwrite(dr / f"c{k:02d}.tif", render_board_view(raster, pose, scene.right, SensorModel(noise_std=4.0), 2, rng))
    res = run_calibration(cfg)
    assert res.rms < 0.06 and len(res.pairs) == 9
    path = cfg.data_path(cfg.calibration_file)
    assert path.exists() and path.with_suffix(".png").exists() and path.with_suffix(".json").exists()
    back = load_calibration(path)
    assert abs(back.baseline_mm / scene.calibration().baseline_mm - 1) < 2e-3
