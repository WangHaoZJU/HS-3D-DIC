"""CUDA IC-GN (docs §8.3): one thread block per POI, reference data resident on the GPU.

Design
------
* One kernel launch per target frame solves every POI to convergence: the IC-GN loop
  runs inside the kernel, so there are no per-iteration launches or host syncs.
* Each POI gets one thread block; its threads share the subset pixels. Per iteration:
  pass 1 samples the warped subset (B-spline) into shared memory and reduces the sum;
  pass 2 reduces sum (g - mean)^2 and J^T (g - mean). Thread 0 then forms
  dp = -H^-1 (J^T f~ - |f~|/|g~| J^T g~), composes W(p) o W(dp)^-1 and tests convergence.
  J^T f~ is constant per POI and precomputed once in FP64.
* Mixed precision (docs §6.3): POI centres, warp parameters, warped positions and all
  reductions are FP64; the fractional part of each position, the B-spline weights,
  the coefficient image, f~ and J are FP32.
* Reference arrays (f~, J, H^-1, J^T f~ ...) are uploaded once per ReferenceSubsets and
  cached on it; per frame only the image (uint16) goes up and is prefiltered on the GPU
  (`CudaBSplineImage`), and only the per-POI results come back.

Must agree with the FP64 reference within P99 < 1e-4 px (tests/backends, cuda marker).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.status import Status
from ...core.types import DICBatchResult
from .. import shape_functions as sf
from ..icgn import ReferenceSubsets, _check_indices
from ..interp import BORDER, SUPPORTED_ORDERS, BSplineImage

THREADS = 128  # threads per block (one block per POI)
_SHARED_LIMIT = 48 * 1024

_KERNEL_SRC = r"""
// Compile-time constants: INTERP (3|5), SHAPE (0|1|2), MD (3|6), NP (2|6|12), RX, RY, NT.
#define NWARP (NT / 32)
#define NW (INTERP + 1)
#define ST_OK 1
#define ST_OOB 2
#define ST_LOW 3
#define ST_ILL 4
#define ST_NOT_CONV 5
#define ST_CONTINUE 255

__device__ __forceinline__ void bweights(float t, float* w) {
#if INTERP == 3
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = fabsf(t - (float)(i - 1));
        float r = 0.f;
        if (a < 1.f) r = 2.f / 3.f - a * a + 0.5f * a * a * a;
        else if (a < 2.f) { float b = 2.f - a; r = b * b * b / 6.f; }
        w[i] = r;
    }
#else
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        float a = fabsf(t - (float)(i - 2));
        float r = 0.f;
        if (a < 1.f) { float a2 = a * a; r = 11.f / 20.f - a2 / 2.f + a2 * a2 / 4.f - a2 * a2 * a / 12.f; }
        else if (a < 2.f) { float b = 3.f - a, c = 2.f - a; float b2 = b * b, c2 = c * c;
                            r = (b2 * b2 * b - 6.f * c2 * c2 * c) / 120.f; }
        else if (a < 3.f) { float b = 3.f - a; float b2 = b * b; r = b2 * b2 * b / 120.f; }
        w[i] = r;
    }
#endif
}

__device__ __forceinline__ float interp(const float* __restrict__ c, long long stride, int off, double x, double y) {
    double fx = floor(x), fy = floor(y);
    float wx[NW], wy[NW];
    bweights((float)(x - fx), wx);
    bweights((float)(y - fy), wy);
    const float* base = c + ((long long)fy + off) * stride + ((long long)fx + off);
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < NW; ++j) {
        const float* row = base + j * stride;
        float sx = 0.f;
#pragma unroll
        for (int i = 0; i < NW; ++i) sx += __ldg(row + i) * wx[i];
        s += sx * wy[j];
    }
    return s;
}

__device__ void p_to_m(const double* p, double* M) {
    for (int r = 0; r < MD * MD; ++r) M[r] = 0.0;
#if SHAPE == 0
    M[0] = 1.0; M[4] = 1.0; M[8] = 1.0; M[2] = p[0]; M[5] = p[1];
#elif SHAPE == 1
    M[0] = 1.0 + p[1]; M[1] = p[2]; M[2] = p[0];
    M[3] = p[4]; M[4] = 1.0 + p[5]; M[5] = p[3];
    M[8] = 1.0;
#else
    double U = p[0], A = 1.0 + p[1], B = p[2], C = 0.5 * p[3], D = p[4], E = 0.5 * p[5];
    double V = p[6], a = p[7], b = 1.0 + p[8], c = 0.5 * p[9], d = p[10], e = 0.5 * p[11];
#define m6(r, cc) M[(r) * 6 + (cc)]
    m6(0,0) = A*A + 2*U*C; m6(0,1) = 2*A*B + 2*U*D; m6(0,2) = B*B + 2*U*E; m6(0,3) = 2*U*A; m6(0,4) = 2*U*B; m6(0,5) = U*U;
    m6(1,0) = A*a + U*c + V*C; m6(1,1) = A*b + B*a + U*d + V*D; m6(1,2) = B*b + U*e + V*E;
    m6(1,3) = U*a + V*A; m6(1,4) = U*b + V*B; m6(1,5) = U*V;
    m6(2,0) = a*a + 2*V*c; m6(2,1) = 2*a*b + 2*V*d; m6(2,2) = b*b + 2*V*e; m6(2,3) = 2*V*a; m6(2,4) = 2*V*b; m6(2,5) = V*V;
    m6(3,0) = C; m6(3,1) = D; m6(3,2) = E; m6(3,3) = A; m6(3,4) = B; m6(3,5) = U;
    m6(4,0) = c; m6(4,1) = d; m6(4,2) = e; m6(4,3) = a; m6(4,4) = b; m6(4,5) = V;
    m6(5,5) = 1.0;
#undef m6
#endif
}

__device__ void m_to_p(const double* M, double* p) {
#if SHAPE == 0
    p[0] = M[2]; p[1] = M[5];
#elif SHAPE == 1
    p[0] = M[2]; p[1] = M[0] - 1.0; p[2] = M[1]; p[3] = M[5]; p[4] = M[3]; p[5] = M[4] - 1.0;
#else
    p[0] = M[3*6+5]; p[1] = M[3*6+3] - 1.0; p[2] = M[3*6+4]; p[3] = 2.0 * M[3*6+0]; p[4] = M[3*6+1]; p[5] = 2.0 * M[3*6+2];
    p[6] = M[4*6+5]; p[7] = M[4*6+3]; p[8] = M[4*6+4] - 1.0; p[9] = 2.0 * M[4*6+0]; p[10] = M[4*6+1]; p[11] = 2.0 * M[4*6+2];
#endif
}

// out = M @ inv(D); false if D is singular (same arithmetic as the Numba backend)
__device__ bool compose_inverse(const double* M, const double* D, double* out) {
#if MD == 3
    double det = D[0] * D[4] - D[1] * D[3];
    if (det == 0.0) return false;
    double a = D[4] / det, b = -D[1] / det, c = -D[3] / det, d = D[0] / det;
    double tx = -(a * D[2] + b * D[5]);
    double ty = -(c * D[2] + d * D[5]);
    for (int r = 0; r < 2; ++r) {
        double m0 = M[r * 3], m1 = M[r * 3 + 1];
        out[r * 3] = m0 * a + m1 * c;
        out[r * 3 + 1] = m0 * b + m1 * d;
        out[r * 3 + 2] = m0 * tx + m1 * ty + M[r * 3 + 2];
    }
    out[6] = 0.0; out[7] = 0.0; out[8] = 1.0;
    return true;
#else
    double work[36], inv[36];
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) { work[r * 6 + c] = D[r * 6 + c]; inv[r * 6 + c] = (r == c) ? 1.0 : 0.0; }
    for (int col = 0; col < 6; ++col) {
        int piv = col;
        double best = fabs(work[col * 6 + col]);
        for (int r = col + 1; r < 6; ++r)
            if (fabs(work[r * 6 + col]) > best) { best = fabs(work[r * 6 + col]); piv = r; }
        if (best == 0.0) return false;
        if (piv != col)
            for (int c = 0; c < 6; ++c) {
                double t = work[col * 6 + c]; work[col * 6 + c] = work[piv * 6 + c]; work[piv * 6 + c] = t;
                t = inv[col * 6 + c]; inv[col * 6 + c] = inv[piv * 6 + c]; inv[piv * 6 + c] = t;
            }
        double dd = work[col * 6 + col];
        for (int c = 0; c < 6; ++c) { work[col * 6 + c] /= dd; inv[col * 6 + c] /= dd; }
        for (int r = 0; r < 6; ++r) {
            if (r == col) continue;
            double f = work[r * 6 + col];
            if (f != 0.0)
                for (int c = 0; c < 6; ++c) { work[r * 6 + c] -= f * work[col * 6 + c]; inv[r * 6 + c] -= f * inv[col * 6 + c]; }
        }
    }
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) {
            double s = 0.0;
            for (int k = 0; k < 6; ++k) s += M[r * 6 + k] * inv[k * 6 + c];
            out[r * 6 + c] = s;
        }
    return true;
#endif
}

__device__ double corner_motion(const double* D, const double* __restrict__ corners) {
    double best = 0.0;
    for (int k = 0; k < 4; ++k) {
        double x = 0.0, y = 0.0;
        for (int j = 0; j < MD; ++j) { x += D[RX * MD + j] * corners[k * MD + j]; y += D[RY * MD + j] * corners[k * MD + j]; }
        double ux = x - corners[k * MD + MD - 3], uy = y - corners[k * MD + MD - 2];
        double mag = sqrt(ux * ux + uy * uy);
        if (mag > best) best = mag;
    }
    return best;
}

// Sum K per-thread values over the block; result in out[0..K). Contains __syncthreads().
template <int K>
__device__ __forceinline__ void block_sum(double* v, double* sred, double* out) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        double x = v[k];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
        v[k] = x;
    }
    if (lane == 0)
        for (int k = 0; k < K; ++k) sred[wid * K + k] = v[k];
    __syncthreads();
    if (threadIdx.x < K) {
        double s = 0.0;
        for (int w = 0; w < NWARP; ++w) s += sred[w * K + threadIdx.x];
        out[threadIdx.x] = s;
    }
    __syncthreads();
}

// Sample the warped subset of POI (cx, cy) into g[]; returns the block-wide sum of g
// and sets *oob (uniform) if any pixel leaves the reliable domain.
__device__ __forceinline__ double sample_subset(
    const double* sM, const float* __restrict__ xi, int Q, double cx, double cy,
    const float* __restrict__ coeffs, long long stride, int off, double lo, double hi_x, double hi_y,
    float* g, int* s_oob, double* sred, double* sres, bool* oob)
{
    double mrx[MD], mry[MD];
#pragma unroll
    for (int j = 0; j < MD; ++j) { mrx[j] = sM[RX * MD + j]; mry[j] = sM[RY * MD + j]; }
    double v[1] = {0.0};
    bool bad = false;
    for (int k = threadIdx.x; k < Q; k += NT) {
        const float* xk = xi + k * MD;
        double lx = 0.0, ly = 0.0;
#pragma unroll
        for (int j = 0; j < MD; ++j) { lx += mrx[j] * (double)xk[j]; ly += mry[j] * (double)xk[j]; }
        double x = cx + lx, y = cy + ly;
        float gv = 0.f;
        if (!(x >= lo && x <= hi_x && y >= lo && y <= hi_y)) bad = true;  // NaN-safe
        else gv = interp(coeffs, stride, off, x, y);
        g[k] = gv;
        v[0] += (double)gv;
    }
    if (bad) atomicOr(s_oob, 1);
    block_sum<1>(v, sred, sres);
    *oob = (*s_oob != 0);
    return sres[0];
}

extern "C" __global__ void __launch_bounds__(NT) icgn_kernel(
    const long long* __restrict__ idx, const double* __restrict__ centers, const float* __restrict__ xi,
    const double* __restrict__ corners, const float* __restrict__ f_tilde, const double* __restrict__ f_norm,
    const float* __restrict__ J, const double* __restrict__ Jf, const double* __restrict__ H_inv,
    const unsigned char* __restrict__ ref_status,
    const float* __restrict__ coeffs, long long stride, int off, double lo, double hi_x, double hi_y,
    const double* __restrict__ p0, int Q, int max_iter, double thr,
    double* __restrict__ out_xy, double* __restrict__ out_p, double* __restrict__ out_znssd,
    unsigned char* __restrict__ out_iter, unsigned char* __restrict__ out_status)
{
    extern __shared__ float g[];
    __shared__ double sM[MD * MD];
    __shared__ double sred[NWARP * (NP + 1)];
    __shared__ double sres[NP + 1];
    __shared__ int s_oob, s_state;

    const int k_out = blockIdx.x;
    const long long i = idx[k_out];
    const int tid = threadIdx.x;
    if (ref_status[i] != ST_OK) {
        if (tid == 0) out_status[k_out] = ref_status[i];
        return;
    }
    const double cx = centers[2 * i], cy = centers[2 * i + 1];
    const double fn = f_norm[i];
    const float* fti = f_tilde + i * (long long)Q;
    const float* Ji = J + i * (long long)Q * NP;

    if (tid == 0) {
        double p[NP];
        for (int j = 0; j < NP; ++j) p[j] = p0[(long long)k_out * NP + j];
        p_to_m(p, sM);
    }

    int state = ST_NOT_CONV;
    int iters = max_iter;
    bool oob;
    for (int it = 1; it <= max_iter; ++it) {
        if (tid == 0) s_oob = 0;
        __syncthreads();
        double gsum = sample_subset(sM, xi, Q, cx, cy, coeffs, stride, off, lo, hi_x, hi_y, g, &s_oob, sred, sres, &oob);
        if (oob) { state = ST_OOB; iters = it; break; }
        const double gm = gsum / Q;
        double w[NP + 1];
#pragma unroll
        for (int j = 0; j <= NP; ++j) w[j] = 0.0;
        for (int k = tid; k < Q; k += NT) {
            double d = (double)g[k] - gm;
            w[NP] += d * d;
            const float* Jk = Ji + (long long)k * NP;
#pragma unroll
            for (int j = 0; j < NP; ++j) w[j] += (double)Jk[j] * d;
        }
        block_sum<NP + 1>(w, sred, sres);
        if (tid == 0) {
            int st = ST_CONTINUE;
            double gn = sqrt(sres[NP]);
            if (gn == 0.0) {
                st = ST_LOW;
            } else {
                double ratio = fn / gn, b[NP], dp[NP];
                for (int j = 0; j < NP; ++j) b[j] = Jf[i * NP + j] - ratio * sres[j];
                for (int r = 0; r < NP; ++r) {
                    double s = 0.0;
                    for (int j = 0; j < NP; ++j) s += H_inv[(i * NP + r) * NP + j] * b[j];
                    dp[r] = -s;
                }
                double D[MD * MD], Mn[MD * MD];
                p_to_m(dp, D);
                if (!compose_inverse(sM, D, Mn)) {
                    st = ST_ILL;
                } else {
                    for (int r = 0; r < MD * MD; ++r) sM[r] = Mn[r];
                    if (corner_motion(D, corners) < thr) st = ST_OK;
                }
            }
            s_state = st;
        }
        __syncthreads();
        if (s_state != ST_CONTINUE) { state = s_state; iters = it; break; }
    }

    double z = nan("");
    bool write_params = false;
    if (state == ST_OK || state == ST_NOT_CONV) {
        if (tid == 0) s_oob = 0;
        __syncthreads();
        double gsum = sample_subset(sM, xi, Q, cx, cy, coeffs, stride, off, lo, hi_x, hi_y, g, &s_oob, sred, sres, &oob);
        if (oob) {
            state = ST_OOB;
        } else {
            write_params = true;
            const double gm = gsum / Q;
            double v[1] = {0.0};
            for (int k = tid; k < Q; k += NT) { double d = (double)g[k] - gm; v[0] += d * d; }
            block_sum<1>(v, sred, sres);
            const double gn = sqrt(sres[0]);
            if (gn > 0.0) {
                double u[1] = {0.0};
                for (int k = tid; k < Q; k += NT) {
                    double r = (double)fti[k] / fn - ((double)g[k] - gm) / gn;
                    u[0] += r * r;
                }
                block_sum<1>(u, sred, sres);
                z = sres[0];
            }
        }
    }
    if (tid == 0) {
        out_iter[k_out] = (unsigned char)iters;
        out_status[k_out] = (unsigned char)state;
        if (write_params) {
            double p[NP];
            m_to_p(sM, p);
            for (int j = 0; j < NP; ++j) out_p[(long long)k_out * NP + j] = p[j];
            out_xy[2 * k_out] = cx + sM[RX * MD + MD - 1];
            out_xy[2 * k_out + 1] = cy + sM[RY * MD + MD - 1];
            out_znssd[k_out] = z;
        }
    }
}
"""

_modules: dict[tuple[int, int], Any] = {}


def _cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - availability is checked by backend selection
        raise RuntimeError(f"CUDA backend requires CuPy: {exc}") from exc
    return cp


def _kernel(interp_order: int, shape_order: int):
    key = (interp_order, shape_order)
    if key not in _modules:
        cp = _cupy()
        md = 6 if shape_order == 2 else 3
        defines = {
            "INTERP": interp_order,
            "SHAPE": shape_order,
            "MD": md,
            "NP": sf.n_params(shape_order),
            "RX": sf._ROW_X[shape_order],
            "RY": sf._ROW_Y[shape_order],
            "NT": THREADS,
        }
        options = ("--std=c++14",) + tuple(f"-D{k}={v}" for k, v in defines.items())
        # no --use_fast_math: docs §6.3 forbids it
        _modules[key] = cp.RawModule(code=_KERNEL_SRC, options=options).get_function("icgn_kernel")
    return _modules[key]


# --------------------------------------------------------------------------- target image
class CudaBSplineImage:
    """B-spline coefficient image on the GPU (FP32 storage, FP64 prefilter).

    Same model as `interp.BSplineImage` (mirror prefilter, reflect padding, safety band),
    built on the device so only the raw frame is transferred per target.
    """

    def __init__(self, image: Any, order: int = 3) -> None:
        if order not in SUPPORTED_ORDERS:
            raise ValueError(f"unsupported B-spline order {order}; expected one of {SUPPORTED_ORDERS}")
        cp = _cupy()
        from cupyx.scipy.ndimage import spline_filter

        img = cp.asarray(image).astype(cp.float64, copy=False)
        if img.ndim != 2:
            raise ValueError("expected a 2D image")
        self.order = order
        self.height, self.width = (int(s) for s in img.shape)
        self.pad = order // 2 + 2
        if min(img.shape) <= self.pad:
            raise ValueError(f"image too small for order-{order} interpolation")
        coeffs = spline_filter(img, order=order, mode="mirror", output=cp.float64)
        self.coeffs = cp.ascontiguousarray(cp.pad(coeffs, self.pad, mode="reflect").astype(cp.float32))

    @classmethod
    def from_host(cls, target: BSplineImage) -> CudaBSplineImage:
        """Wrap an existing CPU model (coefficients are uploaded, not recomputed)."""
        cp = _cupy()
        obj = cls.__new__(cls)
        obj.order, obj.height, obj.width, obj.pad = target.order, target.height, target.width, target.pad
        obj.coeffs = cp.ascontiguousarray(cp.asarray(target.coeffs, dtype=cp.float32))
        return obj

    @property
    def border(self) -> int:
        return BORDER[self.order]

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


# --------------------------------------------------------------------------- reference data
@dataclass
class _DeviceReference:
    n: int
    q: int
    order: int
    centers: Any
    xi: Any
    corners: Any
    f_tilde: Any
    f_norm: Any
    J: Any
    Jf: Any
    H_inv: Any
    status: Any


def _device_reference(refs: ReferenceSubsets) -> _DeviceReference:
    """Upload the reference subsets once and cache them on the object."""
    cached = refs.__dict__.get("_cuda_device")
    if cached is not None and cached.n == len(refs):
        return cached
    cp = _cupy()
    order = refs.options.shape_order
    h = float(refs.options.half_size)
    xi = sf.basis(refs.dx, refs.dy, order)
    corners = sf.basis(np.array([-h, h, -h, h]), np.array([-h, -h, h, h]), order)
    Jf = np.einsum("nqp,nq->np", refs.J, refs.f_tilde)  # J^T f~ in FP64, constant per POI
    dev = _DeviceReference(
        n=len(refs),
        q=int(refs.dx.size),
        order=order,
        centers=cp.asarray(np.ascontiguousarray(refs.centers, dtype=np.float64)),
        xi=cp.asarray(np.ascontiguousarray(xi, dtype=np.float32)),
        corners=cp.asarray(np.ascontiguousarray(corners, dtype=np.float64)),
        f_tilde=cp.asarray(np.ascontiguousarray(refs.f_tilde, dtype=np.float32)),
        f_norm=cp.asarray(np.ascontiguousarray(refs.f_norm, dtype=np.float64)),
        J=cp.asarray(np.ascontiguousarray(refs.J, dtype=np.float32)),
        Jf=cp.asarray(np.ascontiguousarray(Jf, dtype=np.float64)),
        H_inv=cp.asarray(np.ascontiguousarray(refs.H_inv, dtype=np.float64)),
        status=cp.asarray(np.ascontiguousarray(refs.status, dtype=np.uint8)),
    )
    refs.__dict__["_cuda_device"] = dev
    return dev


# --------------------------------------------------------------------------- solve
def solve_cuda(
    refs: ReferenceSubsets,
    target: BSplineImage | CudaBSplineImage,
    p0: np.ndarray,
    indices: np.ndarray | None = None,
) -> DICBatchResult:
    """Solve the selected POIs on the GPU (same contract as `icgn.solve_reference`)."""
    cp = _cupy()
    opt = refs.options
    order = opt.shape_order
    idx, p0 = _check_indices(refs, p0, indices)
    n_p = sf.n_params(order)
    out = DICBatchResult.empty(idx.size, n_params=n_p)
    n = idx.size
    if n == 0:
        return out
    tgt = target if isinstance(target, CudaBSplineImage) else CudaBSplineImage.from_host(target)
    dev = _device_reference(refs)
    shared = dev.q * 4
    if shared > _SHARED_LIMIT:
        raise ValueError(f"subset of {dev.q} pixels exceeds the CUDA shared-memory budget")

    d_idx = cp.asarray(idx)
    d_p0 = cp.asarray(p0)
    xy = cp.full((n, 2), cp.nan, dtype=cp.float64)
    params = cp.full((n, n_p), cp.nan, dtype=cp.float64)
    znssd = cp.full(n, cp.nan, dtype=cp.float64)
    iters = cp.zeros(n, dtype=cp.uint8)
    status = cp.zeros(n, dtype=cp.uint8)
    off = tgt.pad - (tgt.order - 1) // 2
    border = float(tgt.border)
    kernel = _kernel(tgt.order, order)
    kernel(
        (n,),
        (THREADS,),
        (
            d_idx, dev.centers, dev.xi, dev.corners, dev.f_tilde, dev.f_norm, dev.J, dev.Jf, dev.H_inv, dev.status,
            tgt.coeffs, np.int64(tgt.coeffs.shape[1]), np.int32(off),
            np.float64(border), np.float64(tgt.width - 1.0 - border), np.float64(tgt.height - 1.0 - border),
            d_p0, np.int32(dev.q), np.int32(opt.max_iterations), np.float64(opt.convergence_threshold_px),
            xy, params, znssd, iters, status,
        ),
        shared_mem=shared,
    )
    out.xy[:] = cp.asnumpy(xy)
    out.params[:] = cp.asnumpy(params)
    out.znssd[:] = cp.asnumpy(znssd).astype(np.float32)
    out.iterations[:] = cp.asnumpy(iters)
    out.status[:] = cp.asnumpy(status)
    return out


def make_target(image: np.ndarray, order: int) -> CudaBSplineImage:
    return CudaBSplineImage(image, order)


__all__ = ["CudaBSplineImage", "make_target", "solve_cuda", "THREADS", "Status"]
