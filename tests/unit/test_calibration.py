"""Camera model, ChArUco detection and stereo calibration on synthetic data."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from stereodic.calibration.calibrate import CalibrationOptions, calibrate_mono, calibrate_stereo
from stereodic.calibration.camera_model import (
    fundamental_matrix,
    optical_axis_angle_deg,
    pixel_to_normalized,
    project,
    project_stereo,
)
from stereodic.calibration.detect import BoardDetection, CharucoBoardSpec, detect_charuco
from stereodic.calibration.report import load_calibration, save_calibration
from stereodic.core.types import CameraModel
from stereodic.recon.triangulate import triangulate
from stereodic.synth import SensorModel
from stereodic.synth.charuco import board_raster, render_board_view
from stereodic.synth.stereo import HeightFieldTarget, make_stereo_scene, rotation

SPEC = CharucoBoardSpec(9, 7, 5.0, 3.75)


@pytest.fixture(scope="module")
def scene():
    return make_stereo_scene()


# ------------------------------------------------------------------ camera model
@pytest.mark.parametrize(
    "dist",
    [
        [-0.12, 0.3, 2e-4, -1e-4],
        [-0.12, 0.3, 2e-4, -1e-4, 0.5],
        [-0.12, 0.3, 2e-4, -1e-4, 0.5, 0.1, -0.2, 0.3],
    ],
)
def test_projection_matches_opencv_and_jacobian(dist):
    K = np.array([[2500.0, 0.0, 260.0], [0.0, 2502.0, 205.0], [0.0, 0.0, 1.0]])
    cam = CameraModel(K=K, dist=np.array(dist), image_size=(512, 416))
    X = np.random.default_rng(0).uniform([-15, -12, 230], [15, 12, 270], (300, 3))
    uv, J = project(cam, X, with_jacobian=True)
    ref, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K, np.array(dist))
    np.testing.assert_allclose(uv, ref.reshape(-1, 2), atol=1e-9)
    e = 1e-5
    Jn = np.stack([(project(cam, X + e * np.eye(3)[k]) - project(cam, X - e * np.eye(3)[k])) / (2 * e) for k in range(3)], -1)
    np.testing.assert_allclose(J, Jn, atol=1e-6)
    np.testing.assert_allclose(pixel_to_normalized(cam, uv), X[:, :2] / X[:, 2:], atol=1e-14)


def test_unsupported_distortion_terms():
    cam = CameraModel(K=np.eye(3), dist=np.r_[np.zeros(8), 1e-3, 0, 0, 0], image_size=(10, 10))
    with pytest.raises(NotImplementedError):
        project(cam, np.array([[0.0, 0.0, 1.0]]))


def test_scene_calibration_geometry(scene):
    cal = scene.calibration()
    assert optical_axis_angle_deg(cal) == pytest.approx(20.0, abs=1e-9)
    # epipolar constraint on undistorted pixels
    X = np.random.default_rng(1).uniform([-10, -8, 240], [10, 8, 260], (50, 3))
    pl, pr = project_stereo(cal, X)
    ul = cv2.undistortPoints(pl.reshape(-1, 1, 2), cal.left.K, cal.left.dist, P=cal.left.K).reshape(-1, 2)
    ur = cv2.undistortPoints(pr.reshape(-1, 1, 2), cal.right.K, cal.right.dist, P=cal.right.K).reshape(-1, 2)
    F = fundamental_matrix(cal)
    alg = np.einsum("ni,ij,nj->n", np.c_[ur, np.ones(50)], F, np.c_[ul, np.ones(50)])
    assert np.abs(alg).max() < 1e-4


# ------------------------------------------------------------------ detection
def _board_pose(rng, center_z=250.0):
    R = rotation(*rng.uniform(-30, 30, 2), rng.uniform(-20, 20)) @ rotation(0, -10, 0)
    C = np.array([*rng.uniform(-4, 4, 2), center_z + rng.uniform(-15, 15)])
    return HeightFieldTarget(R=R, t=C - R @ np.array([22.5, 17.5, 0.0]))


def test_charuco_detection_subpixel(scene):
    raster = board_raster(SPEC.board(), px_per_mm=40, blur_mm=0.05)
    rng = np.random.default_rng(3)
    pose = _board_pose(rng)
    for cam in (scene.left, scene.right):
        img = render_board_view(raster, pose, cam, SensorModel(noise_std=4.0), 2, rng)
        det = detect_charuco(img, SPEC)
        assert det is not None and len(det) >= 30
        truth = cam.project(pose.point(det.object_points[:, :2]))
        err = det.image_points - truth
        assert np.abs(err.mean(axis=0)).max() < 0.02
        assert np.sqrt((err**2).sum(1).mean()) < 0.08
    assert detect_charuco(np.full((100, 100), 1000, np.uint16), SPEC) is None


# ------------------------------------------------------------------ calibration
def _synthetic_detections(scene, n_views=12, noise_px=0.03, seed=0):
    """Projected board corners + noise (fast path without rendering)."""
    rng = np.random.default_rng(seed)
    obj = SPEC.corner_points()
    L, R = [], []
    for _ in range(n_views):
        pose = _board_pose(rng)
        X = pose.point(obj[:, :2])
        for cam, out in ((scene.left, L), (scene.right, R)):
            uv = cam.project(X) + rng.normal(0, noise_px, (len(obj), 2))
            w, h = cam.image_size
            inside = (uv[:, 0] > 5) & (uv[:, 0] < w - 6) & (uv[:, 1] > 5) & (uv[:, 1] < h - 6)
            ids = np.nonzero(inside)[0]
            out.append(BoardDetection(ids, uv[ids], obj[ids]) if len(ids) >= 8 else None)
    return L, R


def _check_3d(cal, truth):
    X = np.random.default_rng(5).uniform([-15, -12, 240], [15, 12, 260], (1000, 3))
    pl, pr = project_stereo(truth, X)
    tri = triangulate(cal, pl, pr)
    i, j = np.random.default_rng(6).integers(0, 1000, (2, 3000))
    keep = i != j
    d_true = np.linalg.norm(X[i[keep]] - X[j[keep]], axis=1)
    d_est = np.linalg.norm(tri.X[i[keep]] - tri.X[j[keep]], axis=1)
    rel = d_est / d_true - 1
    U = np.array([0.03, -0.02, 0.1])
    tri2 = triangulate(cal, *project_stereo(truth, X + U))
    du = tri2.X - tri.X - U
    return np.abs(rel).max(), np.abs(du).max()


def test_mono_outlier_rejection(scene):
    L, _ = _synthetic_detections(scene, n_views=10, seed=1)
    bad = L[3]
    L[3] = BoardDetection(bad.ids, bad.image_points + np.random.default_rng(0).normal(0, 2.0, bad.image_points.shape), bad.object_points)
    res = calibrate_mono(L, scene.left.image_size)
    assert not res.used[3] and res.used.sum() == 9
    assert res.rms < 0.06


def test_stereo_calibration_recovers_3d_geometry(scene, tmp_path):
    L, R = _synthetic_detections(scene, seed=2)
    res = calibrate_stereo(L, R, scene.left.image_size)
    truth = scene.calibration()
    assert res.rms < 0.06
    assert abs(res.calibration.baseline_mm / truth.baseline_mm - 1) < 2e-3
    rel_dist, disp_err = _check_3d(res.calibration, truth)
    assert rel_dist < 5e-4  # 3D distances within 0.05 %
    assert disp_err < 2e-4  # mm, for a ~0.1 mm displacement
    rep = res.report()
    assert rep["n_corners"] > 300 and rep["left"]["views_used"] == 12
    # file round trip (HDF5 and sidecar JSON)
    path = save_calibration(tmp_path / "calib.h5", res)
    for p in (path, path.with_suffix(".json")):
        back = load_calibration(p)
        np.testing.assert_allclose(back.R_RL, res.calibration.R_RL, atol=1e-15)
        np.testing.assert_allclose(back.left.dist, res.calibration.left.dist, atol=1e-15)


@pytest.mark.slow
def test_stereo_calibration_from_rendered_images(scene):
    """Full chain: rendered board images -> detection -> calibration -> 3D accuracy."""
    raster = board_raster(SPEC.board(), px_per_mm=40, blur_mm=0.05)
    rng = np.random.default_rng(1)
    L, R = [], []
    for _ in range(12):
        pose = _board_pose(rng)
        L.append(detect_charuco(render_board_view(raster, pose, scene.left, SensorModel(noise_std=4.0), 2, rng), SPEC))
        R.append(detect_charuco(render_board_view(raster, pose, scene.right, SensorModel(noise_std=4.0), 2, rng), SPEC))
    res = calibrate_stereo(L, R, scene.left.image_size, options=CalibrationOptions())
    assert res.rms < 0.06
    rel_dist, disp_err = _check_3d(res.calibration, scene.calibration())
    assert rel_dist < 5e-4 and disp_err < 2e-4
