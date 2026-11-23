"""First-order uncertainty propagation for stereo triangulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.status import Status
from ..core.types import StereoCalibration
from ..recon.triangulate import triangulate


@dataclass(frozen=True)
class UncertaintyResult:
    covariance: np.ndarray  # (N, 3, 3), mm^2
    standard_deviation: np.ndarray  # (N, 3), mm
    jacobian: np.ndarray  # (N, 3, 4), mm/px
    valid: np.ndarray  # (N,)


def _observation_covariance(
    n: int,
    pixel_covariance: np.ndarray,
    right_pixel_covariance: np.ndarray | None,
) -> np.ndarray:
    left = np.asarray(pixel_covariance, dtype=np.float64)
    if left.shape == (2, 2):
        left = np.broadcast_to(left, (n, 2, 2))
    elif left.shape != (n, 2, 2):
        raise ValueError("pixel_covariance must have shape (2,2) or (N,2,2)")
    right = left if right_pixel_covariance is None else np.asarray(right_pixel_covariance, dtype=np.float64)
    if right.shape == (2, 2):
        right = np.broadcast_to(right, (n, 2, 2))
    elif right.shape != (n, 2, 2):
        raise ValueError("right_pixel_covariance must have shape (2,2) or (N,2,2)")
    out = np.zeros((n, 4, 4), dtype=np.float64)
    out[:, :2, :2] = left
    out[:, 2:, 2:] = right
    return out


def triangulation_uncertainty(
    calibration: StereoCalibration,
    xy_left: np.ndarray,
    xy_right: np.ndarray,
    pixel_covariance: np.ndarray,
    right_pixel_covariance: np.ndarray | None = None,
    *,
    finite_difference_px: float = 1e-4,
    min_angle_deg: float = 1.0,
) -> UncertaintyResult:
    """Propagate independent 2D pixel covariance through triangulation.

    The Jacobian is evaluated numerically in the same raw distorted pixel
    model used by the reconstruction code. Calibration-parameter covariance
    is intentionally a separate input in future extensions, so this result
    currently represents observation noise only.
    """
    left = np.asarray(xy_left, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(xy_right, dtype=np.float64).reshape(-1, 2)
    if left.shape != right.shape:
        raise ValueError("xy_left and xy_right must have the same shape")
    if finite_difference_px <= 0:
        raise ValueError("finite_difference_px must be positive")
    n = len(left)
    cov_p = _observation_covariance(n, pixel_covariance, right_pixel_covariance)
    base = triangulate(calibration, left, right, min_angle_deg=min_angle_deg)
    jac = np.full((n, 3, 4), np.nan, dtype=np.float64)
    valid = base.status == Status.OK
    for k in range(4):
        lp = left.copy()
        rp = right.copy()
        if k < 2:
            lp[:, k] += finite_difference_px
        else:
            rp[:, k - 2] += finite_difference_px
        plus = triangulate(calibration, lp, rp, min_angle_deg=min_angle_deg)
        lp = left.copy()
        rp = right.copy()
        if k < 2:
            lp[:, k] -= finite_difference_px
        else:
            rp[:, k - 2] -= finite_difference_px
        minus = triangulate(calibration, lp, rp, min_angle_deg=min_angle_deg)
        derivative_ok = valid & (plus.status == Status.OK) & (minus.status == Status.OK)
        jac[derivative_ok, :, k] = (plus.X[derivative_ok] - minus.X[derivative_ok]) / (2.0 * finite_difference_px)
        valid &= derivative_ok
    covariance = np.full((n, 3, 3), np.nan, dtype=np.float64)
    if np.any(valid):
        covariance[valid] = np.einsum("nik,nkl,njl->nij", jac[valid], cov_p[valid], jac[valid])
        covariance[valid] = 0.5 * (covariance[valid] + np.swapaxes(covariance[valid], 1, 2))
    std = np.full((n, 3), np.nan, dtype=np.float64)
    std[valid] = np.sqrt(np.maximum(np.diagonal(covariance[valid], axis1=1, axis2=2), 0.0))
    return UncertaintyResult(covariance, std, jac, valid)


def displacement_covariance(
    covariance_dynamic: np.ndarray,
    covariance_reference: np.ndarray,
    cross_covariance: np.ndarray | None = None,
) -> np.ndarray:
    """Propagate covariance of ``U = X_dynamic - X_reference``.

    ``cross_covariance`` is the covariance of dynamic and reference position
    errors. It is required when the shared reference image induces correlation.
    """
    dynamic = np.asarray(covariance_dynamic, dtype=np.float64)
    reference = np.asarray(covariance_reference, dtype=np.float64)
    if dynamic.shape != reference.shape or dynamic.shape[-2:] != (3, 3):
        raise ValueError("covariance arrays must have matching (...,3,3) shapes")
    if cross_covariance is None:
        cross = np.zeros_like(dynamic)
    else:
        cross = np.asarray(cross_covariance, dtype=np.float64)
        if cross.shape != dynamic.shape:
            raise ValueError("cross_covariance must match covariance shapes")
    return dynamic + reference - cross - np.swapaxes(cross, -1, -2)
