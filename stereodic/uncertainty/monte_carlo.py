"""Monte Carlo validation for stereo measurement uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.status import Status
from ..core.types import StereoCalibration
from ..recon.triangulate import triangulate


@dataclass(frozen=True)
class MonteCarloResult:
    mean: np.ndarray  # (N,3), mm
    covariance: np.ndarray  # (N,3,3), mm^2
    standard_deviation: np.ndarray  # (N,3), mm
    valid_samples: np.ndarray  # (N,)
    samples: int


def _cov_factors(cov: np.ndarray) -> np.ndarray:
    """Return L (N, 2, 2) such that L @ L^T == cov for symmetric PSD 2x2 matrices."""
    sym = 0.5 * (cov + np.swapaxes(cov, 1, 2))
    w, v = np.linalg.eigh(sym)
    if np.any(w < -1e-12):
        raise ValueError("pixel covariance must be positive semi-definite")
    return v * np.sqrt(np.maximum(w, 0.0))[:, None, :]


def monte_carlo_triangulation(
    calibration: StereoCalibration,
    xy_left: np.ndarray,
    xy_right: np.ndarray,
    pixel_covariance: np.ndarray,
    right_pixel_covariance: np.ndarray | None = None,
    *,
    samples: int = 1000,
    seed: int | None = 0,
    min_angle_deg: float = 1.0,
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
    fac_l = _cov_factors(left_cov)
    fac_r = _cov_factors(right_cov)
    rng = np.random.default_rng(seed)
    z_l = rng.standard_normal((samples, n, 2))
    z_r = rng.standard_normal((samples, n, 2))
    noise_l_all = np.einsum("nij,snj->sni", fac_l, z_l)
    noise_r_all = np.einsum("nij,snj->sni", fac_r, z_r)
    draws = np.full((samples, n, 3), np.nan, dtype=np.float64)
    for k in range(samples):
        result = triangulate(calibration, left + noise_l_all[k], right + noise_r_all[k], min_angle_deg=min_angle_deg)
        ok = (result.status == Status.OK) & np.all(np.isfinite(result.X), axis=1)
        draws[k, ok] = result.X[ok]
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
