"""Stereo triangulation — the single implementation used everywhere (docs §9.1).

1. Remove distortion (Newton on the lens model) and intersect the two rays with the
   midpoint method for an initial point.
2. Gauss–Newton on the reprojection error in the *raw* (distorted) pixel domain:
       X = argmin |pi_L(X) - p_L|^2 + |pi_R(R X + t) - p_R|^2
   (3 unknowns, 4 residuals per point, vectorized over all points).
3. Checks: positive depth in both cameras, triangulation angle, reprojection error.
   The reprojection error measures the epipolar inconsistency of the pair, so this also
   serves as the per-frame epipolar check.

World frame = left camera, lengths in mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calibration.camera_model import pixel_to_normalized, project
from ..core.status import STATUS_DTYPE, Status
from ..core.types import StereoCalibration


@dataclass
class Triangulation:
    X: np.ndarray  # (N, 3) mm, NaN where not computed
    residual_left: np.ndarray  # (N, 2) px, projection - observation
    residual_right: np.ndarray  # (N, 2) px
    angle_deg: np.ndarray  # (N,) angle between the two rays at X
    status: np.ndarray  # (N,) uint8: OK, GEOMETRY_INCONSISTENT or NOT_COMPUTED (non-finite input)

    @property
    def reprojection_error(self) -> np.ndarray:
        """RMS over both cameras of the reprojection residual norm, px."""
        return np.sqrt(0.5 * ((self.residual_left**2).sum(1) + (self.residual_right**2).sum(1)))


def _midpoint(calib: StereoCalibration, nl: np.ndarray, nr: np.ndarray) -> np.ndarray:
    dl = np.column_stack([nl, np.ones(len(nl))])
    dr = np.column_stack([nr, np.ones(len(nr))]) @ calib.R_RL  # R^T d in the left frame
    c = -calib.R_RL.T @ calib.t_RL  # right camera centre in the left frame
    # minimize |a dl - (c + b dr)|^2
    a11 = (dl * dl).sum(1)
    a12 = -(dl * dr).sum(1)
    a22 = (dr * dr).sum(1)
    b1 = (dl * c).sum(1)
    b2 = -(dr * c).sum(1)
    det = a11 * a22 - a12 * a12
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (b1 * a22 - a12 * b2) / det
        b = (a11 * b2 - a12 * b1) / det
    return 0.5 * (a[:, None] * dl + c + b[:, None] * dr)


def triangulate(
    calib: StereoCalibration,
    xy_left: np.ndarray,
    xy_right: np.ndarray,
    max_reprojection_px: float | None = None,
    min_angle_deg: float = 1.0,
    iterations: int = 10,
) -> Triangulation:
    """Triangulate corresponding pixels (N, 2) in the left and right images."""
    pl = np.asarray(xy_left, dtype=np.float64).reshape(-1, 2)
    pr = np.asarray(xy_right, dtype=np.float64).reshape(-1, 2)
    if pl.shape != pr.shape:
        raise ValueError("xy_left and xy_right must have the same shape")
    n = pl.shape[0]
    X = np.full((n, 3), np.nan)
    res_l = np.full((n, 2), np.nan)
    res_r = np.full((n, 2), np.nan)
    angle = np.full(n, np.nan)
    status = np.full(n, Status.NOT_COMPUTED, dtype=STATUS_DTYPE)
    ok = np.all(np.isfinite(pl), axis=1) & np.all(np.isfinite(pr), axis=1)
    if not ok.any():
        return Triangulation(X, res_l, res_r, angle, status)

    R, t = calib.R_RL, calib.t_RL
    a, b = pl[ok], pr[ok]
    x = _midpoint(calib, pixel_to_normalized(calib.left, a), pixel_to_normalized(calib.right, b))
    for _ in range(iterations):
        ul, Jl = project(calib.left, x, with_jacobian=True)
        ur, Jr = project(calib.right, x @ R.T + t, with_jacobian=True)
        Jr = Jr @ R
        r = np.concatenate([ul - a, ur - b], axis=1)  # (M, 4)
        J = np.concatenate([Jl, Jr], axis=1)  # (M, 4, 3)
        JtJ = np.einsum("mki,mkj->mij", J, J)
        Jtr = np.einsum("mki,mk->mi", J, r)
        with np.errstate(invalid="ignore"):
            dx = np.linalg.solve(JtJ, -Jtr[..., None])[..., 0]
        x = x + dx
        if not np.any(np.abs(dx) > 1e-12):
            break
    ul = project(calib.left, x)
    ur = project(calib.right, x @ R.T + t)

    X[ok] = x
    res_l[ok] = ul - a
    res_r[ok] = ur - b
    c_r = -R.T @ t
    v1 = x / np.linalg.norm(x, axis=1, keepdims=True)
    v2 = (x - c_r) / np.linalg.norm(x - c_r, axis=1, keepdims=True)
    angle[ok] = np.degrees(np.arccos(np.clip((v1 * v2).sum(1), -1.0, 1.0)))

    depth_ok = (x[:, 2] > 0) & ((x @ R.T + t)[:, 2] > 0)
    good = depth_ok & np.all(np.isfinite(x), axis=1) & (angle[ok] >= min_angle_deg)
    if max_reprojection_px is not None:
        rep = np.sqrt(0.5 * (((ul - a) ** 2).sum(1) + ((ur - b) ** 2).sum(1)))
        good &= rep <= max_reprojection_px
    st = np.where(good, Status.OK, Status.GEOMETRY_INCONSISTENT).astype(STATUS_DTYPE)
    status[ok] = st  # X is kept for diagnostics; consumers must honour status
    return Triangulation(X, res_l, res_r, angle, status)
