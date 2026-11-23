"""Rigid-body motion estimation and compensation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidMotion:
    rotation: np.ndarray
    translation: np.ndarray
    residual_rms: float
    valid: bool
    n_points: int


def estimate_rigid_motion(
    reference_points: np.ndarray,
    current_points: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> RigidMotion:
    """Fit ``current ~= R @ reference + t`` with weighted Kabsch/SVD."""
    ref = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    cur = np.asarray(current_points, dtype=np.float64).reshape(-1, 3)
    if ref.shape != cur.shape:
        raise ValueError("reference_points and current_points must have the same shape")
    mask = np.all(np.isfinite(ref), axis=1) & np.all(np.isfinite(cur), axis=1)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool).reshape(-1)
    if weights is None:
        w = np.ones(len(ref), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != len(ref):
            raise ValueError("weights must have one value per point")
        mask &= np.isfinite(w) & (w > 0)
    if int(mask.sum()) < 3:
        return RigidMotion(np.full((3, 3), np.nan), np.full(3, np.nan), np.nan, False, int(mask.sum()))

    r = ref[mask]
    c = cur[mask]
    ww = w[mask]
    ww /= ww.sum()
    r_mean = np.sum(r * ww[:, None], axis=0)
    c_mean = np.sum(c * ww[:, None], axis=0)
    r0 = r - r_mean
    c0 = c - c_mean
    _, s_ref, _ = np.linalg.svd(r0 * np.sqrt(ww)[:, None], full_matrices=False)
    if s_ref.size < 2 or s_ref[0] <= 0.0 or s_ref[1] <= 1e-12 * s_ref[0]:
        return RigidMotion(np.full((3, 3), np.nan), np.full(3, np.nan), np.nan, False, int(mask.sum()))
    covariance = (c0 * ww[:, None]).T @ r0
    U, _, Vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
    R = U @ correction @ Vt
    t = c_mean - R @ r_mean
    predicted = r @ R.T + t
    residual = np.sqrt(np.sum((predicted - c) ** 2, axis=1))
    return RigidMotion(R, t, float(np.sqrt(np.sum(ww * residual**2))), True, int(mask.sum()))


@dataclass(frozen=True)
class RigidCompensation:
    rotations: np.ndarray
    translations: np.ndarray
    displacement: np.ndarray
    residual_rms: np.ndarray
    valid: np.ndarray


def compensate_trajectory(
    reference_points: np.ndarray,
    trajectory: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
) -> RigidCompensation:
    """Remove frame-wise rigid motion from a ``(T, N, 3)`` trajectory.

    When ``reference_mask`` (shape ``(N,)``) is given, the rigid-body transform
    is estimated only on those reference points (e.g. blade-root fixture or
    static background) and then applied to compensate all points.
    """
    ref = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    xyz = np.asarray(trajectory, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[1:] != ref.shape:
        raise ValueError("trajectory must have shape (T, N, 3) matching reference_points")
    T, N, _ = xyz.shape
    if valid_mask is None:
        masks = np.ones((T, N), dtype=bool)
    else:
        masks = np.asarray(valid_mask, dtype=bool)
        if masks.shape != (T, N):
            raise ValueError("valid_mask must have shape (T, N)")
    if reference_mask is None:
        fit_masks = masks
    else:
        rmask = np.asarray(reference_mask, dtype=bool).reshape(-1)
        if rmask.size != N:
            raise ValueError("reference_mask must have length N")
        fit_masks = masks & rmask[None, :]
    R_all = np.full((T, 3, 3), np.nan)
    t_all = np.full((T, 3), np.nan)
    disp = np.full_like(xyz, np.nan)
    rms = np.full(T, np.nan)
    good = np.zeros(T, dtype=bool)
    for k in range(T):
        motion = estimate_rigid_motion(ref, xyz[k], valid=fit_masks[k])
        R_all[k], t_all[k], rms[k], good[k] = motion.rotation, motion.translation, motion.residual_rms, motion.valid
        if not motion.valid:
            continue
        finite = np.all(np.isfinite(xyz[k]), axis=1) & np.all(np.isfinite(ref), axis=1) & masks[k]
        compensated = (xyz[k] - motion.translation) @ motion.rotation
        disp[k, finite] = compensated[finite] - ref[finite]
    return RigidCompensation(R_all, t_all, disp, rms, good)
