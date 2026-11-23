"""Monte Carlo validation for stereo measurement uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import StereoCalibration
from ..recon.triangulate import triangulate


@dataclass(frozen=True)
class MonteCarloResult:
    mean: np.ndarray  # (N,3), mm
    covariance: np.ndarray  # (N,3,3), mm^2
    standard_deviation: np.ndarray  # (N,3), mm
    valid_samples: np.ndarray  # (N,)
    samples: int


def monte_carlo_triangulation(
    calibration: StereoCalibration,
    xy_left: np.ndarray,
    xy_right: np.ndarray,
    pixel_covariance: np.ndarray,
    right_pixel_covariance: np.ndarray | None = None,
    *,
    samples: int = 1000,
    seed: int | None = 0,
) -> MonteCarloResult:
    """Sample pixel observations and estimate empirical 3D covariance."""
    left = np.asarray(xy_left, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(xy_right, dtype=np.float64).reshape(-1, 2)
    if left.shape != right.shape:
        raise ValueError("xy_left and xy_right must have the same shape")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    n = len(left)
    left_cov = np.asarray(pixel_covariance, dtype=np.float64)
    right_cov = left_cov if right_pixel_covariance is None else np.asarray(right_pixel_covariance, dtype=np.float64)
    if left_cov.shape == (2, 2):
        left_cov = np.broadcast_to(left_cov, (n, 2, 2))
    if right_cov.shape == (2, 2):
        right_cov = np.broadcast_to(right_cov, (n, 2, 2))
    if left_cov.shape != (n, 2, 2) or right_cov.shape != (n, 2, 2):
        raise ValueError("pixel covariance must have shape (2,2) or (N,2,2)")
    rng = np.random.default_rng(seed)
    draws = np.full((samples, n, 3), np.nan, dtype=np.float64)
    for k in range(samples):
        noise_l = np.stack([rng.multivariate_normal(np.zeros(2), left_cov[i]) for i in range(n)])
        noise_r = np.stack([rng.multivariate_normal(np.zeros(2), right_cov[i]) for i in range(n)])
        result = triangulate(calibration, left + noise_l, right + noise_r)
        draws[k] = result.X
    valid_samples = np.sum(np.all(np.isfinite(draws), axis=2), axis=0)
    mean = np.full((n, 3), np.nan)
    covariance = np.full((n, 3, 3), np.nan)
    for i in range(n):
        values = draws[:, i]
        values = values[np.all(np.isfinite(values), axis=1)]
        if len(values) < 2:
            continue
        mean[i] = values.mean(axis=0)
        covariance[i] = np.cov(values, rowvar=False, ddof=1)
    std = np.full((n, 3), np.nan)
    ok = valid_samples >= 2
    std[ok] = np.sqrt(np.maximum(np.diagonal(covariance[ok], axis1=1, axis2=2), 0.0))
    return MonteCarloResult(mean, covariance, std, valid_samples, samples)
