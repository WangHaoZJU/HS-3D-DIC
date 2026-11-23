from __future__ import annotations

import numpy as np

from stereodic.calibration.camera_model import project
from stereodic.core.types import CameraModel, StereoCalibration
from stereodic.quality.noise_floor import estimate_noise_floor
from stereodic.uncertainty.monte_carlo import monte_carlo_triangulation
from stereodic.uncertainty.propagation import displacement_covariance, triangulation_uncertainty


def _calibration() -> StereoCalibration:
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    cam_l = CameraModel(K, np.zeros(5), (640, 480))
    cam_r = CameraModel(K, np.zeros(5), (640, 480))
    return StereoCalibration(cam_l, cam_r, np.eye(3), np.array([50.0, 0.0, 0.0]))


def test_first_order_triangulation_uncertainty_and_difference_covariance():
    calibration = _calibration()
    points = np.array([[20.0, -10.0, 800.0], [0.0, 5.0, 900.0]])
    left = project(calibration.left, points)
    right = project(calibration.right, points + np.array([50.0, 0.0, 0.0]))
    covariance = triangulation_uncertainty(calibration, left, right, np.eye(2) * 0.01**2)
    assert covariance.valid.all()
    assert np.all(covariance.standard_deviation > 0)
    np.testing.assert_allclose(covariance.covariance, np.swapaxes(covariance.covariance, 1, 2), atol=1e-12)
    difference = displacement_covariance(covariance.covariance, covariance.covariance)
    np.testing.assert_allclose(difference, 2.0 * covariance.covariance)


def test_monte_carlo_matches_linear_scale():
    calibration = _calibration()
    points = np.array([[10.0, 0.0, 700.0]])
    left = project(calibration.left, points)
    right = project(calibration.right, points + np.array([50.0, 0.0, 0.0]))
    cov = np.eye(2) * 0.01**2
    linear = triangulation_uncertainty(calibration, left, right, cov)
    result = monte_carlo_triangulation(calibration, left, right, cov, samples=500, seed=4)
    assert result.valid_samples[0] == 500
    np.testing.assert_allclose(result.standard_deviation[0], linear.standard_deviation[0], rtol=0.15)


def test_monte_carlo_excludes_geometry_inconsistent_samples():
    calibration = _calibration()
    # Point with a very small triangulation angle (< min_angle_deg) must be rejected
    far_point = np.array([[0.0, 0.0, 1e5]])
    left = project(calibration.left, far_point)
    right = project(calibration.right, far_point + np.array([50.0, 0.0, 0.0]))
    result = monte_carlo_triangulation(calibration, left, right, np.eye(2) * 0.01**2, samples=20, seed=0, min_angle_deg=1.0)
    assert result.valid_samples[0] == 0
    assert np.all(np.isnan(result.standard_deviation[0]))


def test_noise_floor_statistics():
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 0.2, (200, 4, 3))
    values[0, 2] = np.nan
    result = estimate_noise_floor(values)
    assert result.mean.shape == (4, 3)
    assert result.standard_deviation.shape == (4, 3)
    assert np.nanmedian(result.standard_deviation) > 0.1
    assert result.valid_samples[2] == 199
