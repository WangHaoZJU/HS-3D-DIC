"""Backend consistency (BK): accelerated backends must reproduce the FP64 NumPy reference.

A tight convergence threshold makes both solvers converge to the same fixed point,
so the comparison is not sensitive to which iteration crosses the threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from stereodic.core.status import Status
from stereodic.dic import shape_functions as sf
from stereodic.dic.icgn import ICGNOptions, ICGNSolver
from stereodic.dic.interp import BSplineImage
from stereodic.dic.mesh import make_poi_grid
from stereodic.synth import Affine, SensorModel, SpeckleTexture, render_frame

SHAPE = (110, 130)


@pytest.fixture(scope="module")
def images():
    tex = SpeckleTexture.random((-30, -30, 160, 140), seed=33)
    sensor = SensorModel(bits=12, noise_std=8.0)
    m = Affine(A=np.array([[1.003, -0.002], [0.001, 0.998]]), t=np.array([0.43, -0.27]), center=np.array([65.0, 55.0]))
    ref = render_frame(tex, SHAPE, None, sensor, oversample=4, seed=1)
    tgt = render_frame(tex, SHAPE, m, sensor, oversample=4, seed=2)
    return ref, tgt


@pytest.mark.parametrize("shape_order", [0, 1, 2])
@pytest.mark.parametrize("interp_order", [3, 5])
def test_numba_matches_numpy(images, shape_order, interp_order):
    ref_img, tgt_img = (BSplineImage(im, interp_order) for im in images)
    opt = ICGNOptions(subset_size=21, shape_order=shape_order, convergence_threshold_px=1e-9, max_iterations=100)
    centers = make_poi_grid(SHAPE, 21, 9, margin=10).xy
    centers = np.vstack([centers, [[3.0, 50.0], [60.5, 60.25]]])  # one OOB, one non-integer
    a = ICGNSolver(opt, "numpy")
    b = ICGNSolver(opt, "numba")
    refs = a.prepare_reference(ref_img, centers)
    p0 = sf.from_displacement(np.tile([0.3, -0.2], (len(centers), 1)), shape_order)
    ra = a.solve(refs, tgt_img, p0)
    rb = b.solve(refs, tgt_img, p0)
    np.testing.assert_array_equal(ra.status, rb.status)
    assert (ra.status == Status.OK).sum() >= len(centers) - 1
    ok = ra.status == Status.OK
    assert np.abs(ra.xy[ok] - rb.xy[ok]).max() < 1e-8
    assert np.abs(ra.params[ok] - rb.params[ok]).max() < 1e-8
    np.testing.assert_allclose(ra.znssd[ok], rb.znssd[ok], rtol=1e-6, atol=1e-12)
    assert np.abs(ra.iterations[ok].astype(int) - rb.iterations[ok].astype(int)).max() <= 1


def test_index_subset_matches_full_solve(images):
    ref_img, tgt_img = (BSplineImage(im, 5) for im in images)
    opt = ICGNOptions(subset_size=21, shape_order=2, convergence_threshold_px=1e-9, max_iterations=100)
    centers = make_poi_grid(SHAPE, 21, 9, margin=10).xy
    p0 = sf.from_displacement(np.tile([0.3, -0.2], (len(centers), 1)), 2)
    for backend in ("numpy", "numba"):
        solver = ICGNSolver(opt, backend)
        refs = solver.prepare_reference(ref_img, centers)
        full = solver.solve(refs, tgt_img, p0)
        idx = np.array([7, 2, 11])
        part = solver.solve(refs, tgt_img, p0[idx], indices=idx)
        np.testing.assert_array_equal(part.status, full.status[idx])
        np.testing.assert_allclose(part.xy, full.xy[idx], atol=1e-12)


# ----------------------------------------------------------------------------- CUDA (BK)
# Mixed precision (FP32 coefficients / J) puts a ~1e-6 px noise floor on each IC-GN update,
# so the GPU tests use a convergence threshold above it; both solvers use the same options.
CUDA_THR = 1e-5


def _cuda_case(images, shape_order, interp_order, subset=21, step=9, thr=CUDA_THR, max_iter=100):
    ref_img, tgt_img = (BSplineImage(im, interp_order) for im in images)
    opt = ICGNOptions(subset_size=subset, shape_order=shape_order, convergence_threshold_px=thr, max_iterations=max_iter)
    centers = make_poi_grid(SHAPE, subset, step, margin=10).xy
    centers = np.vstack([centers, [[3.0, 50.0], [60.5, 60.25], [np.nan, np.nan]]])  # OOB, non-integer, invalid
    a = ICGNSolver(opt, "numpy")
    b = ICGNSolver(opt, "cuda")
    refs = a.prepare_reference(ref_img, centers)
    p0 = sf.from_displacement(np.tile([0.3, -0.2], (len(centers), 1)), shape_order)
    return a, b, refs, tgt_img, p0, images[1]


@pytest.mark.cuda
@pytest.mark.parametrize("shape_order", [0, 1, 2])
@pytest.mark.parametrize("interp_order", [3, 5])
def test_cuda_matches_numpy(images, shape_order, interp_order):
    """GPU consistency (BK): P99 of the position difference to FP64 NumPy < 1e-4 px."""
    a, b, refs, tgt, p0, _ = _cuda_case(images, shape_order, interp_order)
    p0[0] = np.nan  # a NaN guess must become OUT_OF_BOUNDS cleanly
    ra = a.solve(refs, tgt, p0)
    rb = b.solve(refs, tgt, p0)
    np.testing.assert_array_equal(ra.status, rb.status)
    ok = ra.status == Status.OK
    assert ok.sum() >= len(ok) - 4
    d = np.linalg.norm(ra.xy[ok] - rb.xy[ok], axis=1)
    assert np.percentile(d, 99) < 1e-4 and d.max() < 1e-3
    np.testing.assert_allclose(ra.znssd[ok], rb.znssd[ok], rtol=1e-3, atol=1e-6)
    assert np.all(np.isnan(rb.xy[~ok]))
    assert np.abs(ra.iterations[ok].astype(int) - rb.iterations[ok].astype(int)).max() <= 2


@pytest.mark.cuda
def test_cuda_gpu_target_indices_and_not_converged(images):
    from stereodic.dic.backends.cuda import CudaBSplineImage

    a, b, refs, tgt, p0, raw = _cuda_case(images, 1, 5)
    # prefilter on the GPU == CPU model (FP32 storage)
    gpu = CudaBSplineImage(raw, 5)
    np.testing.assert_allclose(gpu.coeffs.get(), tgt.coeffs, rtol=0, atol=1e-3)
    full = b.solve(refs, gpu, p0)
    host = b.solve(refs, tgt, p0)
    np.testing.assert_array_equal(full.status, host.status)
    np.testing.assert_allclose(full.xy, host.xy, atol=1e-6)
    idx = np.array([7, 2, 11])
    part = b.solve(refs, gpu, p0[idx], indices=idx)
    np.testing.assert_array_equal(part.status, full.status[idx])
    np.testing.assert_allclose(part.xy, full.xy[idx], atol=1e-12)

    one = ICGNOptions(subset_size=21, shape_order=1, convergence_threshold_px=1e-12, max_iterations=1)
    refs1 = ICGNSolver(one, "numpy").prepare_reference(BSplineImage(images[0], 5), refs.centers[:4])
    r_np = ICGNSolver(one, "numpy").solve(refs1, tgt, np.zeros((4, 6)))
    r_cu = ICGNSolver(one, "cuda").solve(refs1, tgt, np.zeros((4, 6)))
    assert np.all(r_np.status == Status.NOT_CONVERGED)
    np.testing.assert_array_equal(r_cu.status, r_np.status)
    np.testing.assert_array_equal(r_cu.iterations, r_np.iterations)
    np.testing.assert_allclose(r_cu.xy, r_np.xy, atol=1e-4)


@pytest.mark.cuda
def test_cuda_large_subset(images):
    a, b, refs, tgt, p0, _ = _cuda_case(images, 1, 3, subset=41, step=17)
    ra, rb = a.solve(refs, tgt, p0), b.solve(refs, tgt, p0)
    np.testing.assert_array_equal(ra.status, rb.status)
    ok = ra.status == Status.OK
    assert ok.any() and np.abs(ra.xy[ok] - rb.xy[ok]).max() < 1e-4
