"""ZNSSD + inverse-compositional Gauss–Newton (IC-GN) — FP64 reference solver.

Criterion (zero-mean normalised SSD):
    C(p) = sum_q [ f~_q / |f~| - g~_q(p) / |g~| ]^2,  f~ = f - mean(f),  |f~| = sqrt(sum f~^2)

IC-GN (Pan et al. 2013): the Hessian depends only on the reference subset, so it
is computed and inverted once per POI. Each iteration:
    dp = -H^-1 sum_q J~_q^T [ f~_q - (|f~|/|g~|) g~_q(p) ],   W(p) <- W(p) o W(dp)^-1
J~ are the zero-mean steepest-descent images (grad f . dW/dp minus their subset mean),
which is the exact linearisation of the zero-mean reference term; the stationary
point is identical to using the raw J, only convergence is slightly better.

Note: with zero residual the IC-GN fixed point is the ZNSSD minimum. With model
mismatch (aliasing, edge effects) it can differ slightly from the forward-additive
minimum; this was observed only for subsets touching the image edge, which the
interpolation safety band (interp.BORDER) now excludes.

The reference subset is always sampled through the B-spline model, so POIs with
non-integer centres (right camera after stereo matching) use the same code path.

Subset masks (partial subsets at blade edges) are not implemented yet; POI grids keep\nwhole subsets inside the mask instead (mesh.make_poi_grid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.config import DICConfig
from ..core.status import STATUS_DTYPE, Status
from ..core.types import DICBatchResult
from . import shape_functions as sf
from .interp import BSplineImage

_PREPARE_CHUNK = 512  # POIs per vectorised chunk in prepare_reference


@dataclass(frozen=True)
class ICGNOptions:
    subset_size: int = 31
    shape_order: int = 1
    max_iterations: int = 50
    convergence_threshold_px: float = 1e-3
    min_gray_std: float = 1.0  # LOW_TEXTURE below this subset std (gray levels)
    max_condition: float = 1e10  # ILL_CONDITIONED above this (diagonally scaled) Hessian condition

    def __post_init__(self) -> None:
        if self.subset_size % 2 == 0 or self.subset_size < 5:
            raise ValueError("subset_size must be odd and >= 5")
        if not 1 <= self.max_iterations <= 255:
            raise ValueError("max_iterations must be in [1, 255] (stored as uint8)")
        sf.n_params(self.shape_order)  # validates order

    @property
    def half_size(self) -> int:
        return self.subset_size // 2

    @classmethod
    def from_config(cls, cfg: DICConfig, stage: str = "temporal") -> ICGNOptions:
        order = cfg.temporal.shape_order if stage == "temporal" else cfg.stereo.shape_order
        return cls(
            subset_size=cfg.poi.subset_size,
            shape_order=order,
            max_iterations=cfg.solver.max_iterations,
            convergence_threshold_px=cfg.solver.convergence_threshold_px,
            min_gray_std=cfg.solver.min_gray_std,
            max_condition=cfg.solver.max_condition,
        )


@dataclass
class ReferenceSubsets:
    """Precomputed reference data for N POIs (Q pixels per subset, P parameters)."""

    options: ICGNOptions
    centers: np.ndarray  # (N, 2) float64
    dx: np.ndarray  # (Q,) local x offsets
    dy: np.ndarray  # (Q,) local y offsets
    f_tilde: np.ndarray  # (N, Q) zero-mean reference intensities
    f_norm: np.ndarray  # (N,)
    J: np.ndarray  # (N, Q, P) zero-mean steepest-descent images
    H_inv: np.ndarray  # (N, P, P)
    status: np.ndarray  # (N,) uint8: OK or reason the POI cannot be solved

    def __len__(self) -> int:
        return int(self.centers.shape[0])


def subset_offsets(half_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-major local offsets (dy outer, dx inner) of a square subset."""
    r = np.arange(-half_size, half_size + 1, dtype=np.float64)
    gy, gx = np.meshgrid(r, r, indexing="ij")
    return gx.ravel(), gy.ravel()


def prepare_reference(ref: BSplineImage, centers: np.ndarray, options: ICGNOptions) -> ReferenceSubsets:
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    n = centers.shape[0]
    dx, dy = subset_offsets(options.half_size)
    q = dx.size
    jac = sf.jacobian(dx, dy, options.shape_order)  # (Q, 2, P)
    p_n = jac.shape[2]

    f_tilde = np.zeros((n, q))
    f_norm = np.zeros(n)
    J = np.zeros((n, q, p_n))
    H_inv = np.zeros((n, p_n, p_n))
    status = np.full(n, Status.OK, dtype=STATUS_DTYPE)

    xs = centers[:, 0:1] + dx  # (N, Q)
    ys = centers[:, 1:2] + dy
    inside = ref.in_domain(xs, ys).all(axis=1) & np.isfinite(centers).all(axis=1)
    status[~inside] = Status.OUT_OF_BOUNDS

    idx = np.nonzero(inside)[0]
    for start in range(0, idx.size, _PREPARE_CHUNK):
        sel = idx[start : start + _PREPARE_CHUNK]
        f, gx, gy = ref.evaluate_with_gradient(xs[sel], ys[sel])  # (n_sel, Q)
        ft = f - f.mean(axis=1, keepdims=True)
        fn = np.sqrt(np.sum(ft * ft, axis=1))
        Jc = gx[:, :, None] * jac[None, :, 0, :] + gy[:, :, None] * jac[None, :, 1, :]
        Jc -= Jc.mean(axis=1, keepdims=True)
        H = np.einsum("nqi,nqj->nij", Jc, Jc)
        f_tilde[sel], f_norm[sel], J[sel] = ft, fn, Jc

        low = fn / np.sqrt(q) < options.min_gray_std
        d = np.sqrt(np.einsum("nii->ni", H))
        with np.errstate(divide="ignore", invalid="ignore"):
            Hs = H / (d[:, :, None] * d[:, None, :])
            cond = np.where(np.all(d > 0, axis=1), np.linalg.cond(np.where(np.isfinite(Hs), Hs, 0.0)), np.inf)
        ill = ~low & ~(cond < options.max_condition)
        ok = ~low & ~ill
        status[sel[low]] = Status.LOW_TEXTURE
        status[sel[ill]] = Status.ILL_CONDITIONED
        if ok.any():
            H_inv[sel[ok]] = np.linalg.inv(H[ok])

    return ReferenceSubsets(options, centers, dx, dy, f_tilde, f_norm, J, H_inv, status)


def _znssd(ft: np.ndarray, fn: float, gt: np.ndarray, gn: float) -> float:
    r = ft / fn - gt / gn
    return float(r @ r)


def _check_indices(refs: ReferenceSubsets, p0: np.ndarray, indices: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    n_p = sf.n_params(refs.options.shape_order)
    idx = np.arange(len(refs), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64).ravel()
    if idx.size and (idx.min() < 0 or idx.max() >= len(refs)):
        raise IndexError("POI index out of range")
    p0 = np.ascontiguousarray(np.asarray(p0, dtype=np.float64).reshape(idx.size, n_p))
    return idx, p0


def solve_reference(
    refs: ReferenceSubsets, target: BSplineImage, p0: np.ndarray, indices: np.ndarray | None = None
) -> DICBatchResult:
    """Plain NumPy IC-GN, one POI at a time (readable baseline, not for production volumes).

    indices: optional subset of POIs to solve; p0 and the result then follow that order.
    """
    opt = refs.options
    order = opt.shape_order
    idx, p0 = _check_indices(refs, p0, indices)
    out = DICBatchResult.empty(idx.size, n_params=sf.n_params(order))
    for k, i in enumerate(idx):
        if refs.status[i] != Status.OK:
            out.status[k] = refs.status[i]
            continue
        c = refs.centers[i]
        ft, fn, Ji, Hi = refs.f_tilde[i], refs.f_norm[i], refs.J[i], refs.H_inv[i]
        p = p0[k].copy()
        status = Status.NOT_CONVERGED
        it = 0
        gt, gn = None, 0.0
        while it < opt.max_iterations:
            it += 1
            wx, wy = sf.warp(p, refs.dx, refs.dy, order)
            xs, ys = c[0] + wx, c[1] + wy
            if not target.in_domain(xs, ys).all():
                status = Status.OUT_OF_BOUNDS
                break
            g = target.evaluate(xs, ys)
            gt = g - g.mean()
            gn = float(np.sqrt(gt @ gt))
            if gn == 0.0:
                status = Status.LOW_TEXTURE
                break
            e = ft - (fn / gn) * gt
            dp = -Hi @ (Ji.T @ e)
            try:
                p = sf.compose_inverse(p, dp, order)
            except np.linalg.LinAlgError:
                status = Status.ILL_CONDITIONED
                break
            if sf.max_corner_motion(dp, opt.half_size, order) < opt.convergence_threshold_px:
                status = Status.OK
                break
        out.iterations[k] = it
        out.status[k] = status
        if status in (Status.OK, Status.NOT_CONVERGED):
            wx, wy = sf.warp(p, refs.dx, refs.dy, order)
            xs, ys = c[0] + wx, c[1] + wy
            if target.in_domain(xs, ys).all():
                g = target.evaluate(xs, ys)
                gt = g - g.mean()
                gn = float(np.sqrt(gt @ gt))
                out.znssd[k] = _znssd(ft, fn, gt, gn) if gn > 0 else np.nan
            else:
                out.status[k] = Status.OUT_OF_BOUNDS
                continue
            out.params[k] = p
            out.xy[k] = c + sf.center_displacement(p, order)
    return out


class ICGNSolver:
    """Front end: reference preparation (NumPy) + backend-specific solve."""

    def __init__(self, options: ICGNOptions, backend: str = "numpy") -> None:
        from .backends import resolve_backend

        self.options = options
        self.backend = resolve_backend(backend)

    def prepare_reference(self, ref: BSplineImage, centers: np.ndarray) -> ReferenceSubsets:
        return prepare_reference(ref, centers, self.options)

    def make_target(self, image: np.ndarray, interp_order: int):
        """Interpolation model of a target frame for this backend.

        The CUDA backend prefilters on the GPU and keeps the coefficients there
        (`backends.cuda.CudaBSplineImage`); the CPU backends use `BSplineImage`.
        """
        if self.backend == "cuda":
            from .backends.cuda import make_target

            return make_target(image, interp_order)
        return BSplineImage(image, interp_order)

    def solve(
        self,
        refs: ReferenceSubsets,
        target: BSplineImage,
        p0: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> DICBatchResult:
        """Solve all POIs, or only `indices` (result and p0 then follow that order)."""
        if refs.options != self.options:
            raise ValueError("reference subsets were prepared with different options")
        if p0 is None:
            n = len(refs) if indices is None else np.asarray(indices).size
            p0 = np.zeros((n, sf.n_params(self.options.shape_order)))
        if self.backend == "numpy":
            return solve_reference(refs, target, p0, indices)
        if self.backend == "cuda":
            from .backends.cuda import solve_cuda

            return solve_cuda(refs, target, p0, indices)
        from .backends.numba_cpu import solve_numba

        return solve_numba(refs, target, p0, indices)
