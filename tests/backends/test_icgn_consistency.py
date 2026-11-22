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
