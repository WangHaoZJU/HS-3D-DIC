"""CuPy CUDA implementation of the fixed-reference IC-GN solve.

The implementation keeps the reference arrays resident on the device for one
target frame and performs the interpolation, residual reduction and affine
warp update with CuPy array operations. It intentionally uses float64 for the
first production path so the CUDA result has a clear correctness baseline.
"""

from __future__ import annotations

import numpy as np

from ...core.status import Status
from ...core.types import DICBatchResult
from .. import shape_functions as sf
from ..icgn import ReferenceSubsets, _check_indices
from ..interp import BSplineImage, support_offsets


def _weights(cp, frac, order: int):
    offsets = cp.asarray(support_offsets(order), dtype=cp.float64)
    t = frac[..., None] - offsets
    a = cp.abs(t)
    if order == 3:
        return cp.where(a < 1.0, 2.0 / 3.0 - a * a + 0.5 * a**3, cp.where(a < 2.0, (2.0 - a) ** 3 / 6.0, 0.0))
    if order == 5:
        b, c = 3.0 - a, 2.0 - a
        return cp.where(
            a < 1.0,
            11.0 / 20.0 - a * a / 2.0 + a**4 / 4.0 - a**5 / 12.0,
            cp.where(a < 2.0, (b**5 - 6.0 * c**5) / 120.0, cp.where(a < 3.0, b**5 / 120.0, 0.0)),
        )
    raise ValueError(f"unsupported interpolation order {order}")


def _sample(cp, coeff, target: BSplineImage, x, y):
    """Evaluate the padded B-spline coefficient image at an (N,Q) grid."""
    order = target.order
    bx = cp.floor(x).astype(cp.int64)
    by = cp.floor(y).astype(cp.int64)
    wx = _weights(cp, x - bx, order)
    wy = _weights(cp, y - by, order)
    flat = coeff.ravel()
    width = coeff.shape[1]
    out = cp.zeros_like(x, dtype=cp.float64)
    offsets = support_offsets(order)
    for jy, oy in enumerate(offsets):
        iy = by + int(oy) + target.pad
        for ix, ox in enumerate(offsets):
            col = bx + int(ox) + target.pad
            out += flat[iy * width + col] * wy[..., jy] * wx[..., ix]
    return out


def _warp(cp, centers, dx, dy, p, order: int):
    if order == 0:
        x = centers[:, None, 0] + p[:, None, 0] + dx[None, :]
        y = centers[:, None, 1] + p[:, None, 1] + dy[None, :]
        return x, y
    if order != 1:
        raise NotImplementedError("CUDA IC-GN currently supports temporal orders 0 and 1")
    x = centers[:, None, 0] + p[:, None, 0] + (1.0 + p[:, None, 1]) * dx[None, :] + p[:, None, 2] * dy[None, :]
    y = centers[:, None, 1] + p[:, None, 3] + p[:, None, 4] * dx[None, :] + (1.0 + p[:, None, 5]) * dy[None, :]
    return x, y


def _compose_inverse_order1(cp, p, dp):
    a, b, tx = 1.0 + dp[:, 1], dp[:, 2], dp[:, 0]
    c, d, ty = dp[:, 4], 1.0 + dp[:, 5], dp[:, 3]
    det = a * d - b * c
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    itx = -(ia * tx + ib * ty)
    ity = -(ic * tx + id_ * ty)
    pa, pb, ptx = 1.0 + p[:, 1], p[:, 2], p[:, 0]
    pc, pd, pty = p[:, 4], 1.0 + p[:, 5], p[:, 3]
    na, nb = pa * ia + pb * ic, pa * ib + pb * id_
    nc, nd = pc * ia + pd * ic, pc * ib + pd * id_
    ntx, nty = pa * itx + pb * ity + ptx, pc * itx + pd * ity + pty
    return cp.column_stack([ntx, na - 1.0, nb, nty, nc, nd - 1.0])


_CUDA_CHUNK = 1024  # POIs per vectorized GPU batch to bound temporary memory


def _to_host(cp, arr) -> np.ndarray:
    return cp.asnumpy(arr) if hasattr(cp, "asnumpy") else np.asarray(arr)


def _any_true(cp, mask) -> bool:
    val = cp.any(mask)
    return bool(val.get()) if hasattr(val, "get") else bool(val)


def _safe_coords(cp, x, y, width: int, height: int):
    sx = cp.where(cp.isfinite(x), cp.clip(x, 0.0, float(width - 1)), 0.0)
    sy = cp.where(cp.isfinite(y), cp.clip(y, 0.0, float(height - 1)), 0.0)
    return sx, sy


def _solve_chunk(
    cp,
    refs: ReferenceSubsets,
    target: BSplineImage,
    coeff,
    dx,
    dy,
    idx: np.ndarray,
    p0_np: np.ndarray,
    out: DICBatchResult,
    offset: int,
) -> None:
    opt = refs.options
    order = opt.shape_order
    n = idx.size
    centers = cp.asarray(refs.centers[idx], dtype=cp.float64)
    f_tilde = cp.asarray(refs.f_tilde[idx], dtype=cp.float64)
    f_norm = cp.asarray(refs.f_norm[idx], dtype=cp.float64)
    J = cp.asarray(refs.J[idx], dtype=cp.float64)
    H_inv = cp.asarray(refs.H_inv[idx], dtype=cp.float64)
    p = cp.asarray(p0_np.copy(), dtype=cp.float64)

    ref_status = cp.asarray(refs.status[idx].astype(np.uint8))
    active = ref_status == int(Status.OK)
    status = cp.where(active, int(Status.NOT_CONVERGED), ref_status)
    out_xy = cp.full((n, 2), cp.nan, dtype=cp.float64)
    out_params = cp.full((n, sf.n_params(order)), cp.nan, dtype=cp.float64)
    out_znssd = cp.full(n, cp.nan, dtype=cp.float64)
    out_iter = cp.zeros(n, dtype=cp.uint8)
    lo_x, lo_y = float(target.border), float(target.border)
    hi_x, hi_y = float(target.width - 1 - target.border), float(target.height - 1 - target.border)

    h = float(opt.half_size)
    cx = cp.asarray([-h, h, -h, h], dtype=cp.float64)
    cy = cp.asarray([-h, -h, h, h], dtype=cp.float64)
    zeros_c = cp.zeros((n, 2), dtype=cp.float64)

    for iteration in range(1, opt.max_iterations + 1):
        if not _any_true(cp, active):
            break
        out_iter = cp.where(active, iteration, out_iter)
        x, y = _warp(cp, centers, dx, dy, p, order)
        inside = cp.all(
            cp.isfinite(x) & cp.isfinite(y) & (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y),
            axis=1,
        )
        oob = active & ~inside
        status = cp.where(oob, int(Status.OUT_OF_BOUNDS), status)
        active &= inside
        if not _any_true(cp, active):
            break
        sx, sy = _safe_coords(cp, x, y, target.width, target.height)
        g = _sample(cp, coeff, target, sx, sy)
        gm = cp.mean(g, axis=1)
        gt = g - gm[:, None]
        gn = cp.sqrt(cp.sum(gt * gt, axis=1))
        low = active & ~(gn > 0.0)
        status = cp.where(low, int(Status.LOW_TEXTURE), status)
        active &= ~low
        if not _any_true(cp, active):
            break
        safe_gn = cp.where(gn > 0.0, gn, 1.0)
        ratio = f_norm / safe_gn
        e = f_tilde - ratio[:, None] * gt
        b = cp.einsum("nqp,nq->np", J, e)
        dp = -cp.einsum("npq,nq->np", H_inv, b)
        if order == 0:
            p_next = p - dp
            motion = cp.sqrt(dp[:, 0] ** 2 + dp[:, 1] ** 2)
        else:
            det = (1.0 + dp[:, 1]) * (1.0 + dp[:, 5]) - dp[:, 2] * dp[:, 4]
            bad = active & (~cp.isfinite(det) | (cp.abs(det) <= 1e-15))
            status = cp.where(bad, int(Status.ILL_CONDITIONED), status)
            active &= ~bad
            safe_dp = cp.where(active[:, None], dp, 0.0)
            p_next = _compose_inverse_order1(cp, p, safe_dp)
            wx, wy = _warp(cp, zeros_c, cx, cy, safe_dp, order)
            motion = cp.max(cp.sqrt((wx - cx[None, :]) ** 2 + (wy - cy[None, :]) ** 2), axis=1)
        p = cp.where(active[:, None], p_next, p)
        converged = active & (motion < opt.convergence_threshold_px)
        status = cp.where(converged, int(Status.OK), status)
        active &= ~converged

    candidate = (status == int(Status.OK)) | (status == int(Status.NOT_CONVERGED))
    if _any_true(cp, candidate):
        x, y = _warp(cp, centers, dx, dy, p, order)
        inside = cp.all(
            cp.isfinite(x) & cp.isfinite(y) & (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y),
            axis=1,
        )
        final = candidate & inside
        status = cp.where(candidate & ~inside, int(Status.OUT_OF_BOUNDS), status)
        sx, sy = _safe_coords(cp, x, y, target.width, target.height)
        g = _sample(cp, coeff, target, sx, sy)
        gm = cp.mean(g, axis=1)
        gt = g - gm[:, None]
        gn = cp.sqrt(cp.sum(gt * gt, axis=1))
        znssd_ok = final & (gn > 0.0) & (f_norm > 0.0)
        safe_gn = cp.where(znssd_ok, gn, 1.0)
        safe_fn = cp.where(znssd_ok, f_norm, 1.0)
        out_znssd = cp.where(
            znssd_ok,
            cp.sum((f_tilde / safe_fn[:, None] - gt / safe_gn[:, None]) ** 2, axis=1),
            out_znssd,
        )
        if order == 0:
            out_xy = cp.where(final[:, None], centers + p[:, :2], out_xy)
        else:
            out_xy = cp.where(final[:, None], centers + p[:, [0, 3]], out_xy)
        out_params = cp.where(final[:, None], p, out_params)

    sl = slice(offset, offset + n)
    out.xy[sl] = _to_host(cp, out_xy)
    out.params[sl] = _to_host(cp, out_params)
    out.znssd[sl] = _to_host(cp, out_znssd).astype(np.float32)
    out.iterations[sl] = _to_host(cp, out_iter)
    out.status[sl] = _to_host(cp, status).astype(np.uint8)


def _solve_with_xp(
    cp,
    refs: ReferenceSubsets,
    target: BSplineImage,
    p0: np.ndarray,
    indices: np.ndarray | None = None,
    chunk_size: int = _CUDA_CHUNK,
) -> DICBatchResult:
    opt = refs.options
    order = opt.shape_order
    if order not in (0, 1):
        raise NotImplementedError("CUDA IC-GN currently supports temporal orders 0 and 1")
    idx, p0_np = _check_indices(refs, p0, indices)
    n = idx.size
    out = DICBatchResult.empty(n, n_params=sf.n_params(order))
    if n == 0:
        return out
    coeff = cp.asarray(target.coeffs, dtype=cp.float64)
    dx = cp.asarray(refs.dx, dtype=cp.float64)
    dy = cp.asarray(refs.dy, dtype=cp.float64)
    for start in range(0, n, max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), n)
        _solve_chunk(cp, refs, target, coeff, dx, dy, idx[start:stop], p0_np[start:stop], out, start)
    return out


def solve_cuda(
    refs: ReferenceSubsets, target: BSplineImage, p0: np.ndarray, indices: np.ndarray | None = None
) -> DICBatchResult:
    """Solve selected POIs on a CUDA device using CuPy."""
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - availability is checked by backend selection
        raise RuntimeError(f"CUDA backend requires CuPy: {exc}") from exc
    return _solve_with_xp(cp, refs, target, p0, indices)
