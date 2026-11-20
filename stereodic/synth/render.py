"""Supersampled rendering of a (deformed) speckle texture, plus a simple sensor model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter

from .speckle import SpeckleTexture

# Mahalanobis^2 cutoff: exp(-0.5 * 60) ~ 1e-13 relative contribution.
_Q_MAX = 60.0


# ----------------------------------------------------------------------------- maps
class Mapping(Protocol):
    """Material (reference) coordinates X -> image coordinates x."""

    def forward(self, X: np.ndarray) -> np.ndarray: ...

    def jacobian(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Translation:
    u: float
    v: float

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) + np.array([self.u, self.v])

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.eye(2), (np.asarray(X).shape[0], 2, 2)).copy()

    def displacement(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X) - np.asarray(X, dtype=np.float64)


@dataclass(frozen=True)
class Affine:
    """x = center + A (X - center) + t."""

    A: np.ndarray
    t: np.ndarray = field(default_factory=lambda: np.zeros(2))
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        A = np.asarray(self.A, dtype=np.float64)
        c = np.asarray(self.center, dtype=np.float64)
        return c + (X - c) @ A.T + np.asarray(self.t, dtype=np.float64)

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.A, dtype=np.float64), (np.asarray(X).shape[0], 2, 2)).copy()

    def displacement(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X) - np.asarray(X, dtype=np.float64)


# ------------------------------------------------------------------------- kernel
@njit(parallel=True, cache=True)
def _render_kernel(
    height, width, os, cx, cy, ixx, ixy, iyy, amp, qmax,
    gx0, gy0, cell, ncx, ncy, cell_start, cell_items, out,
):  # pragma: no cover - compiled
    inv_os = 1.0 / os
    n_sub = os * os
    for r in prange(height):
        s = np.empty(n_sub)
        for c in range(width):
            for k in range(n_sub):
                s[k] = 0.0
            ci = int(np.floor((c - gx0) / cell))
            cj = int(np.floor((r - gy0) / cell))
            for dj in range(-1, 2):
                jj = cj + dj
                if jj < 0 or jj >= ncy:
                    continue
                for di in range(-1, 2):
                    ii = ci + di
                    if ii < 0 or ii >= ncx:
                        continue
                    cid = jj * ncx + ii
                    for p in range(cell_start[cid], cell_start[cid + 1]):
                        b = cell_items[p]
                        bx = cx[b]
                        by = cy[b]
                        a = amp[b]
                        axx = ixx[b]
                        axy = ixy[b]
                        ayy = iyy[b]
                        for sy in range(os):
                            dy = r - 0.5 + (sy + 0.5) * inv_os - by
                            for sx in range(os):
                                dx = c - 0.5 + (sx + 0.5) * inv_os - bx
                                q = axx * dx * dx + 2.0 * axy * dx * dy + ayy * dy * dy
                                if q < qmax:
                                    s[sy * os + sx] += a * np.exp(-0.5 * q)
            acc = 0.0
            for k in range(n_sub):
                acc += np.exp(-s[k])
            out[r, c] = acc / n_sub


def render_transmission(
    texture: SpeckleTexture,
    shape: tuple[int, int],
    mapping: Mapping | None = None,
    oversample: int = 8,
) -> np.ndarray:
    """Pixel-integrated transmission t in (0, 1] of the (deformed) texture, shape (H, W).

    Pixel (r, c) integrates over [c-0.5, c+0.5] x [r-0.5, r+0.5] with oversample^2 samples.
    """
    if oversample < 1:
        raise ValueError("oversample must be >= 1")
    height, width = (int(v) for v in shape)
    centers = texture.centers
    cov = texture.covariances
    if mapping is not None:
        J = mapping.jacobian(centers)
        centers = mapping.forward(centers)
        cov = J @ cov @ np.transpose(J, (0, 2, 1))
    inv = np.linalg.inv(cov)
    eig_max = np.linalg.eigvalsh(cov)[:, -1]
    max_radius = float(np.sqrt(_Q_MAX * eig_max.max())) if len(eig_max) else 1.0
    cell = max_radius + 1.0
    gx0 = -0.5 - cell
    gy0 = -0.5 - cell
    ncx = int(np.ceil((width + 2 * cell) / cell)) + 1
    ncy = int(np.ceil((height + 2 * cell) / cell)) + 1
    ci = np.floor((centers[:, 0] - gx0) / cell).astype(np.int64)
    cj = np.floor((centers[:, 1] - gy0) / cell).astype(np.int64)
    keep = (ci >= 0) & (ci < ncx) & (cj >= 0) & (cj < ncy)
    idx = np.nonzero(keep)[0]
    cell_id = cj[idx] * ncx + ci[idx]
    order = np.argsort(cell_id, kind="stable")
    cell_items = idx[order].astype(np.int64)
    counts = np.bincount(cell_id, minlength=ncx * ncy)
    cell_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    out = np.empty((height, width), dtype=np.float64)
    _render_kernel(
        height, width, int(oversample),
        np.ascontiguousarray(centers[:, 0]), np.ascontiguousarray(centers[:, 1]),
        np.ascontiguousarray(inv[:, 0, 0]), np.ascontiguousarray(inv[:, 0, 1]), np.ascontiguousarray(inv[:, 1, 1]),
        np.ascontiguousarray(texture.amplitudes), _Q_MAX,
        gx0, gy0, cell, ncx, ncy, cell_start, cell_items, out,
    )
    return out


# ------------------------------------------------------------------------- sensor
@dataclass(frozen=True)
class SensorModel:
    """Maps transmission to digital numbers: gray = full_scale * (black + (white - black) * t)."""

    bits: int = 12
    white: float = 0.85  # fraction of full scale for t = 1 (background)
    black: float = 0.08  # fraction of full scale for t -> 0
    noise_std: float = 0.0  # additive Gaussian noise, gray levels
    psf_sigma_px: float = 0.0  # optional optical blur after pixel integration

    @property
    def full_scale(self) -> int:
        return 2**self.bits - 1

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8) if self.bits <= 8 else np.dtype(np.uint16)


def apply_sensor(transmission: np.ndarray, sensor: SensorModel, rng: np.random.Generator | None = None) -> np.ndarray:
    t = np.asarray(transmission, dtype=np.float64)
    if sensor.psf_sigma_px > 0:
        t = gaussian_filter(t, sensor.psf_sigma_px, mode="nearest")
    gray = sensor.full_scale * (sensor.black + (sensor.white - sensor.black) * t)
    if sensor.noise_std > 0:
        rng = rng if rng is not None else np.random.default_rng()
        gray = gray + sensor.noise_std * rng.standard_normal(gray.shape)
    return np.clip(np.rint(gray), 0, sensor.full_scale).astype(sensor.dtype)


def render_frame(
    texture: SpeckleTexture,
    shape: tuple[int, int],
    mapping: Mapping | None = None,
    sensor: SensorModel | None = None,
    oversample: int = 8,
    seed: int | None = None,
) -> np.ndarray:
    """Render a quantized camera frame (uint8/uint16)."""
    sensor = sensor or SensorModel()
    t = render_transmission(texture, shape, mapping, oversample)
    return apply_sensor(t, sensor, np.random.default_rng(seed))


def s1_translation(
    shape: tuple[int, int] = (256, 256),
    shift: tuple[float, float] = (0.3, -0.7),
    diameter_px: float = 4.0,
    sensor: SensorModel | None = None,
    oversample: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, Translation]:
    """S1 benchmark pair: reference image and image translated by `shift` = (u, v) px.

    Returns (reference, deformed, mapping); mapping.displacement(X) is the exact truth.
    Independent noise realisations are used for the two frames.
    """
    h, w = shape
    margin = 20.0 + abs(shift[0]) + abs(shift[1])
    texture = SpeckleTexture.random((-margin, -margin, w + margin, h + margin), diameter_px=diameter_px, seed=seed)
    mapping = Translation(*shift)
    ref = render_frame(texture, shape, None, sensor, oversample, seed=seed + 1)
    deformed = render_frame(texture, shape, mapping, sensor, oversample, seed=seed + 2)
    return ref, deformed, mapping
