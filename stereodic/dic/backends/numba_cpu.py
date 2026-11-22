"""Numba multi-core IC-GN (FP64). Must match `icgn.solve_reference` (tests/backends).

Same algorithm as the reference solver; differences are limited to floating-point
summation order. Reference preparation stays in NumPy (one-time cost per POI).

Warps are handled generically through the lifted basis xi (Q, m) and an m x m warp
matrix (m = 3 for orders 0/1, m = 6 for order 2, see shape_functions): the warped
local position is x' = M[rx] . xi, y' = M[ry] . xi.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ...core.status import Status
from ...core.types import DICBatchResult
from .. import shape_functions as sf
from ..icgn import ReferenceSubsets, _check_indices
from ..interp import BSplineImage

_OK = int(Status.OK)
_OOB = int(Status.OUT_OF_BOUNDS)
_LOW = int(Status.LOW_TEXTURE)
_NOT_CONV = int(Status.NOT_CONVERGED)
_ILL = int(Status.ILL_CONDITIONED)


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
    m = M.shape[0]
    for r in range(m):
        for c in range(m):
            M[r, c] = 0.0
    if order == 0:
        M[0, 0] = 1.0
        M[1, 1] = 1.0
        M[2, 2] = 1.0
        M[0, 2] = p[0]
        M[1, 2] = p[1]
    elif order == 1:
        M[0, 0] = 1.0 + p[1]
        M[0, 1] = p[2]
        M[0, 2] = p[0]
        M[1, 0] = p[4]
        M[1, 1] = 1.0 + p[5]
        M[1, 2] = p[3]
        M[2, 2] = 1.0
    else:
        U = p[0]
        A = 1.0 + p[1]
        B = p[2]
        C = 0.5 * p[3]
        D = p[4]
        E = 0.5 * p[5]
        V = p[6]
        a = p[7]
        b = 1.0 + p[8]
        c = 0.5 * p[9]
        d = p[10]
        e = 0.5 * p[11]
        M[0, 0] = A * A + 2 * U * C
        M[0, 1] = 2 * A * B + 2 * U * D
        M[0, 2] = B * B + 2 * U * E
        M[0, 3] = 2 * U * A
        M[0, 4] = 2 * U * B
        M[0, 5] = U * U
        M[1, 0] = A * a + U * c + V * C
        M[1, 1] = A * b + B * a + U * d + V * D
        M[1, 2] = B * b + U * e + V * E
        M[1, 3] = U * a + V * A
        M[1, 4] = U * b + V * B
        M[1, 5] = U * V
        M[2, 0] = a * a + 2 * V * c
        M[2, 1] = 2 * a * b + 2 * V * d
        M[2, 2] = b * b + 2 * V * e
        M[2, 3] = 2 * V * a
        M[2, 4] = 2 * V * b
        M[2, 5] = V * V
        M[3, 0] = C
        M[3, 1] = D
        M[3, 2] = E
        M[3, 3] = A
        M[3, 4] = B
        M[3, 5] = U
        M[4, 0] = c
        M[4, 1] = d
        M[4, 2] = e
        M[4, 3] = a
        M[4, 4] = b
        M[4, 5] = V
        M[5, 5] = 1.0


@njit(cache=True)
def _m_to_p(M, order, p):  # pragma: no cover - compiled
    if order == 0:
        p[0] = M[0, 2]
        p[1] = M[1, 2]
    elif order == 1:
        p[0] = M[0, 2]
        p[1] = M[0, 0] - 1.0
        p[2] = M[0, 1]
        p[3] = M[1, 2]
        p[4] = M[1, 0]
        p[5] = M[1, 1] - 1.0
    else:
        p[0] = M[3, 5]
        p[1] = M[3, 3] - 1.0
        p[2] = M[3, 4]
        p[3] = 2.0 * M[3, 0]
        p[4] = M[3, 1]
        p[5] = 2.0 * M[3, 2]
        p[6] = M[4, 5]
        p[7] = M[4, 3]
        p[8] = M[4, 4] - 1.0
        p[9] = 2.0 * M[4, 0]
        p[10] = M[4, 1]
        p[11] = 2.0 * M[4, 2]


@njit(cache=True)
def _invert(D, inv, work):  # pragma: no cover - compiled
    """Gauss–Jordan inversion with partial pivoting; returns False if singular."""
    m = D.shape[0]
    for r in range(m):
        for c in range(m):
            work[r, c] = D[r, c]
            inv[r, c] = 1.0 if r == c else 0.0
    for col in range(m):
        piv = col
        best = abs(work[col, col])
        for r in range(col + 1, m):
            if abs(work[r, col]) > best:
                best = abs(work[r, col])
                piv = r
        if best == 0.0:
            return False
        if piv != col:
            for c in range(m):
                t = work[col, c]
                work[col, c] = work[piv, c]
                work[piv, c] = t
                t = inv[col, c]
                inv[col, c] = inv[piv, c]
                inv[piv, c] = t
        d = work[col, col]
        for c in range(m):
            work[col, c] /= d
            inv[col, c] /= d
        for r in range(m):
            if r != col:
                f = work[r, col]
                if f != 0.0:
                    for c in range(m):
                        work[r, c] -= f * work[col, c]
                        inv[r, c] -= f * inv[col, c]
    return True


@njit(cache=True)
def _compose_inverse(M, D, out, inv, work):  # pragma: no cover - compiled
    """out = M @ inv(D); returns False if D is singular."""
    m = M.shape[0]
    if m == 3:
        det = D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0]
        if det == 0.0:
            return False
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
        return True
    if not _invert(D, inv, work):
        return False
    for r in range(m):
        for c in range(m):
            s = 0.0
            for k in range(m):
                s += M[r, k] * inv[k, c]
            out[r, c] = s
    return True


@njit(cache=True)
def _corner_motion(D, corners, rx, ry):  # pragma: no cover - compiled
    """Largest motion of the subset corners (corners: (4, m) lifted basis, local x/y at m-3, m-2)."""
    m = D.shape[0]
    best = 0.0
    for k in range(corners.shape[0]):
        x = 0.0
        y = 0.0
        for j in range(m):
            x += D[rx, j] * corners[k, j]
            y += D[ry, j] * corners[k, j]
        ux = x - corners[k, m - 3]
        uy = y - corners[k, m - 2]
        mag = np.sqrt(ux * ux + uy * uy)
        if mag > best:
            best = mag
    return best


@njit(cache=True)
def _sample(coeffs, pad, order, border, width, height, cx, cy, xi, M, rx, ry, g, wx, wy):  # pragma: no cover
    lo = float(border)
    hi_x = width - 1.0 - border
    hi_y = height - 1.0 - border
    m = xi.shape[1]
    for k in range(xi.shape[0]):
        x = cx
        y = cy
        for j in range(m):
            x += M[rx, j] * xi[k, j]
            y += M[ry, j] * xi[k, j]
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
    indices, centers, xi, corners, f_tilde, f_norm, J, H_inv, ref_status,
    coeffs, pad, interp_order, border, width, height,
    p0, shape_order, rx, ry, max_iter, thr,
    out_xy, out_p, out_znssd, out_iter, out_status,
):  # pragma: no cover - compiled
    q = xi.shape[0]
    m = xi.shape[1]
    n_p = p0.shape[1]
    for k_out in prange(indices.shape[0]):
        i = indices[k_out]
        if ref_status[i] != _OK:
            out_status[k_out] = ref_status[i]
            continue
        g = np.empty(q)
        wx = np.empty(interp_order + 1)
        wy = np.empty(interp_order + 1)
        M = np.empty((m, m))
        D = np.empty((m, m))
        Mn = np.empty((m, m))
        inv = np.empty((m, m))
        work = np.empty((m, m))
        b = np.empty(n_p)
        dp = np.empty(n_p)
        p = np.empty(n_p)
        cx = centers[i, 0]
        cy = centers[i, 1]
        fn = f_norm[i]
        _p_to_m(p0[k_out], shape_order, M)
        status = _NOT_CONV
        it = 0
        while it < max_iter:
            it += 1
            if not _sample(coeffs, pad, interp_order, border, width, height, cx, cy, xi, M, rx, ry, g, wx, wy):
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
            if not _compose_inverse(M, D, Mn, inv, work):
                status = _ILL
                break
            for r in range(m):
                for c in range(m):
                    M[r, c] = Mn[r, c]
            if _corner_motion(D, corners, rx, ry) < thr:
                status = _OK
                break
        out_iter[k_out] = it
        if status == _OK or status == _NOT_CONV:
            if _sample(coeffs, pad, interp_order, border, width, height, cx, cy, xi, M, rx, ry, g, wx, wy):
                gm, gn = _mean_norm(g)
                if gn > 0.0:
                    z = 0.0
                    for k in range(q):
                        r_k = f_tilde[i, k] / fn - (g[k] - gm) / gn
                        z += r_k * r_k
                    out_znssd[k_out] = z
                _m_to_p(M, shape_order, p)
                for j in range(n_p):
                    out_p[k_out, j] = p[j]
                out_xy[k_out, 0] = cx + M[rx, m - 1]
                out_xy[k_out, 1] = cy + M[ry, m - 1]
            else:
                status = _OOB
        out_status[k_out] = status


def solve_numba(
    refs: ReferenceSubsets, target: BSplineImage, p0: np.ndarray, indices: np.ndarray | None = None
) -> DICBatchResult:
    opt = refs.options
    order = opt.shape_order
    idx, p0 = _check_indices(refs, p0, indices)
    n_p = sf.n_params(order)
    out = DICBatchResult.empty(idx.size, n_params=n_p)
    znssd = np.full(idx.size, np.nan)
    h = float(opt.half_size)
    xi = np.ascontiguousarray(sf.basis(refs.dx, refs.dy, order))
    corners = np.ascontiguousarray(sf.basis(np.array([-h, h, -h, h]), np.array([-h, -h, h, h]), order))
    _icgn_kernel(
        idx, refs.centers, xi, corners, refs.f_tilde, refs.f_norm, refs.J, refs.H_inv, refs.status,
        target.coeffs, target.pad, target.order, target.border, target.width, target.height,
        p0, order, sf._ROW_X[order], sf._ROW_Y[order], opt.max_iterations, opt.convergence_threshold_px,
        out.xy, out.params, znssd, out.iterations, out.status,
    )
    out.znssd[:] = znssd
    return out
