"""Numba multi-core IC-GN (FP64). Must match `icgn.solve_reference` (tests/backends).

Same algorithm as the reference solver; differences are limited to floating-point
summation order. Reference preparation stays in NumPy (one-time cost per POI).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ...core.status import Status
from ...core.types import DICBatchResult
from .. import shape_functions as sf
from ..icgn import ReferenceSubsets
from ..interp import BSplineImage

_OK = int(Status.OK)
_OOB = int(Status.OUT_OF_BOUNDS)
_LOW = int(Status.LOW_TEXTURE)
_NOT_CONV = int(Status.NOT_CONVERGED)


@njit(cache=True)
def _weights(frac, order, w):  # pragma: no cover - compiled
    if order == 3:
        for i in range(4):
            a = abs(frac - (i - 1))
            if a < 1.0:
                w[i] = 2.0 / 3.0 - a * a + 0.5 * a * a * a
            elif a < 2.0:
                b = 2.0 - a
                w[i] = b * b * b / 6.0
            else:
                w[i] = 0.0
    else:
        for i in range(6):
            a = abs(frac - (i - 2))
            if a < 1.0:
                w[i] = 11.0 / 20.0 - a * a / 2.0 + a**4 / 4.0 - a**5 / 12.0
            elif a < 2.0:
                b = 3.0 - a
                c = 2.0 - a
                w[i] = (b**5 - 6.0 * c**5) / 120.0
            elif a < 3.0:
                b = 3.0 - a
                w[i] = b**5 / 120.0
            else:
                w[i] = 0.0


@njit(cache=True)
def _interp(coeffs, pad, order, x, y, wx, wy):  # pragma: no cover - compiled
    bx = int(np.floor(x))
    by = int(np.floor(y))
    _weights(x - bx, order, wx)
    _weights(y - by, order, wy)
    off = pad - (order - 1) // 2
    s = 0.0
    for j in range(order + 1):
        row = by + off + j
        sx = 0.0
        for i in range(order + 1):
            sx += coeffs[row, bx + off + i] * wx[i]
        s += sx * wy[j]
    return s


@njit(cache=True)
def _p_to_m(p, order, M):  # pragma: no cover - compiled
    M[0, 0] = 1.0
    M[0, 1] = 0.0
    M[1, 0] = 0.0
    M[1, 1] = 1.0
    M[2, 0] = 0.0
    M[2, 1] = 0.0
    M[2, 2] = 1.0
    if order == 0:
        M[0, 2] = p[0]
        M[1, 2] = p[1]
    else:
        M[0, 0] += p[1]
        M[0, 1] = p[2]
        M[0, 2] = p[0]
        M[1, 0] = p[4]
        M[1, 1] += p[5]
        M[1, 2] = p[3]


@njit(cache=True)
def _m_to_p(M, order, p):  # pragma: no cover - compiled
    if order == 0:
        p[0] = M[0, 2]
        p[1] = M[1, 2]
    else:
        p[0] = M[0, 2]
        p[1] = M[0, 0] - 1.0
        p[2] = M[0, 1]
        p[3] = M[1, 2]
        p[4] = M[1, 0]
        p[5] = M[1, 1] - 1.0


@njit(cache=True)
def _compose_inverse(M, D, out):  # pragma: no cover - compiled
    """out = M @ inv(D) for affine D."""
    det = D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0]
    a = D[1, 1] / det
    b = -D[0, 1] / det
    c = -D[1, 0] / det
    d = D[0, 0] / det
    tx = -(a * D[0, 2] + b * D[1, 2])
    ty = -(c * D[0, 2] + d * D[1, 2])
    for r in range(2):
        m0 = M[r, 0]
        m1 = M[r, 1]
        out[r, 0] = m0 * a + m1 * c
        out[r, 1] = m0 * b + m1 * d
        out[r, 2] = m0 * tx + m1 * ty + M[r, 2]
    out[2, 0] = 0.0
    out[2, 1] = 0.0
    out[2, 2] = 1.0


@njit(cache=True)
def _corner_motion(D, h):  # pragma: no cover - compiled
    best = 0.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            x = sx * h
            y = sy * h
            ux = D[0, 0] * x + D[0, 1] * y + D[0, 2] - x
            uy = D[1, 0] * x + D[1, 1] * y + D[1, 2] - y
            m = np.sqrt(ux * ux + uy * uy)
            if m > best:
                best = m
    return best


@njit(cache=True)
def _sample(coeffs, pad, order, border, width, height, cx, cy, dx, dy, M, g, wx, wy):  # pragma: no cover - compiled
    lo = float(border)
    hi_x = width - 1.0 - border
    hi_y = height - 1.0 - border
    for k in range(dx.shape[0]):
        x = cx + M[0, 0] * dx[k] + M[0, 1] * dy[k] + M[0, 2]
        y = cy + M[1, 0] * dx[k] + M[1, 1] * dy[k] + M[1, 2]
        if not (x >= lo and x <= hi_x and y >= lo and y <= hi_y):
            return False
        g[k] = _interp(coeffs, pad, order, x, y, wx, wy)
    return True


@njit(cache=True)
def _mean_norm(g):  # pragma: no cover - compiled
    q = g.shape[0]
    m = 0.0
    for k in range(q):
        m += g[k]
    m /= q
    s = 0.0
    for k in range(q):
        d = g[k] - m
        s += d * d
    return m, np.sqrt(s)


@njit(parallel=True, cache=True)
def _icgn_kernel(
    centers, dx, dy, f_tilde, f_norm, J, H_inv, ref_status,
    coeffs, pad, interp_order, border, width, height,
    p0, shape_order, max_iter, thr, half,
    out_xy, out_p, out_znssd, out_iter, out_status,
):  # pragma: no cover - compiled
    n = centers.shape[0]
    q = dx.shape[0]
    n_p = p0.shape[1]
    for i in prange(n):
        if ref_status[i] != _OK:
            out_status[i] = ref_status[i]
            continue
        g = np.empty(q)
        wx = np.empty(interp_order + 1)
        wy = np.empty(interp_order + 1)
        M = np.empty((3, 3))
        D = np.empty((3, 3))
        Mn = np.empty((3, 3))
        b = np.empty(n_p)
        dp = np.empty(n_p)
        p = np.empty(n_p)
        cx = centers[i, 0]
        cy = centers[i, 1]
        fn = f_norm[i]
        _p_to_m(p0[i], shape_order, M)
        status = _NOT_CONV
        it = 0
        while it < max_iter:
            it += 1
            if not _sample(coeffs, pad, interp_order, border, width, height, cx, cy, dx, dy, M, g, wx, wy):
                status = _OOB
                break
            gm, gn = _mean_norm(g)
            if gn == 0.0:
                status = _LOW
                break
            ratio = fn / gn
            for j in range(n_p):
                b[j] = 0.0
            for k in range(q):
                e = f_tilde[i, k] - ratio * (g[k] - gm)
                for j in range(n_p):
                    b[j] += J[i, k, j] * e
            for r in range(n_p):
                s = 0.0
                for j in range(n_p):
                    s += H_inv[i, r, j] * b[j]
                dp[r] = -s
            _p_to_m(dp, shape_order, D)
            _compose_inverse(M, D, Mn)
            for r in range(3):
                for c in range(3):
                    M[r, c] = Mn[r, c]
            if _corner_motion(D, half) < thr:
                status = _OK
                break
        out_iter[i] = it
        if status == _OK or status == _NOT_CONV:
            if _sample(coeffs, pad, interp_order, border, width, height, cx, cy, dx, dy, M, g, wx, wy):
                gm, gn = _mean_norm(g)
                if gn > 0.0:
                    z = 0.0
                    for k in range(q):
                        r_k = f_tilde[i, k] / fn - (g[k] - gm) / gn
                        z += r_k * r_k
                    out_znssd[i] = z
                _m_to_p(M, shape_order, p)
                for j in range(n_p):
                    out_p[i, j] = p[j]
                out_xy[i, 0] = cx + M[0, 2]
                out_xy[i, 1] = cy + M[1, 2]
            else:
                status = _OOB
        out_status[i] = status


def solve_numba(refs: ReferenceSubsets, target: BSplineImage, p0: np.ndarray) -> DICBatchResult:
    opt = refs.options
    n = len(refs)
    n_p = sf.n_params(opt.shape_order)
    p0 = np.ascontiguousarray(np.asarray(p0, dtype=np.float64).reshape(n, n_p))
    out = DICBatchResult.empty(n, n_params=n_p)
    znssd = np.full(n, np.nan)
    _icgn_kernel(
        refs.centers, refs.dx, refs.dy, refs.f_tilde, refs.f_norm, refs.J, refs.H_inv, refs.status,
        target.coeffs, target.pad, target.order, target.border, target.width, target.height,
        p0, opt.shape_order, opt.max_iterations, opt.convergence_threshold_px, float(opt.half_size),
        out.xy, out.params, znssd, out.iterations, out.status,
    )
    out.znssd[:] = znssd
    return out
