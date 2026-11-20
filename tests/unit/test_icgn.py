"""IC-GN reference solver on independent synthetic images (S1)."""

from __future__ import annotations

import numpy as np
import pytest

from stereodic.core.config import DICConfig
from stereodic.core.status import Status
from stereodic.dic import shape_functions as sf
from stereodic.dic.icgn import ICGNOptions, ICGNSolver, subset_offsets
from stereodic.dic.interp import BSplineImage
from stereodic.dic.mesh import make_poi_grid
from stereodic.synth import Affine, SensorModel, SpeckleTexture, Translation, render_frame

SHAPE = (120, 140)


@pytest.fixture(scope="module")
def texture() -> SpeckleTexture:
    return SpeckleTexture.random((-30, -30, SHAPE[1] + 30, SHAPE[0] + 30), seed=21)


def _img(texture, mapping=None, noise=0.0, seed=0, order=3):
    frame = render_frame(texture, SHAPE, mapping, SensorModel(bits=12, noise_std=noise), oversample=6, seed=seed)
    return BSplineImage(frame, order)


def test_subset_offsets_row_major():
    dx, dy = subset_offsets(1)
    assert dx.tolist() == [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    assert dy.tolist() == [-1, -1, -1, 0, 0, 0, 1, 1, 1]


def test_options_from_config():
    opt = ICGNOptions.from_config(DICConfig())
    assert opt.subset_size == 31 and opt.shape_order == 1 and opt.max_iterations == 50
    with pytest.raises(ValueError):
        ICGNOptions(subset_size=30)


def test_identical_images_give_zero(texture):
    ref = _img(texture)
    solver = ICGNSolver(ICGNOptions(subset_size=21))
    pois = make_poi_grid(SHAPE, 21, 25)
    refs = solver.prepare_reference(ref, pois.xy)
    res = solver.solve(refs, ref)
    assert (res.status == Status.OK).all()
    np.testing.assert_allclose(res.xy, pois.xy, atol=1e-10)
    assert np.nanmax(res.znssd) < 1e-20


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("interp_order", [3, 5])
def test_subpixel_translation(texture, order, interp_order):
    shift = (0.37, -0.62)
    ref = _img(texture, order=interp_order)
    tgt = _img(texture, Translation(*shift), order=interp_order)
    solver = ICGNSolver(ICGNOptions(subset_size=31, shape_order=order))
    pois = make_poi_grid(SHAPE, 31, 20, margin=10)
    res = solver.solve(solver.prepare_reference(ref, pois.xy), tgt)
    assert (res.status == Status.OK).all()
    err = res.xy - pois.xy - np.array(shift)
    assert np.abs(err).max() < 0.01
    assert np.all(res.iterations < 10)


def test_affine_deformation_first_order(texture):
    A = np.array([[1.004, 0.002], [-0.003, 0.997]])
    m = Affine(A=A, t=np.array([0.8, 0.3]), center=np.array([70.0, 60.0]))
    ref = _img(texture, noise=0.0)
    tgt = _img(texture, m)
    solver = ICGNSolver(ICGNOptions(subset_size=31, shape_order=1))
    pois = make_poi_grid(SHAPE, 31, 20, margin=10)
    res = solver.solve(solver.prepare_reference(ref, pois.xy), tgt)
    assert (res.status == Status.OK).all()
    truth = m.forward(pois.xy)
    assert np.abs(res.xy - truth).max() < 0.01
    # gradients recovered: u_x = A00 - 1, u_y = A01, v_x = A10, v_y = A11 - 1
    np.testing.assert_allclose(np.median(res.params[:, [1, 2, 4, 5]], axis=0), [0.004, 0.002, -0.003, -0.003], atol=3e-4)


def test_non_integer_reference_centres(texture):
    shift = (0.21, 0.44)
    ref = _img(texture)
    tgt = _img(texture, Translation(*shift))
    solver = ICGNSolver(ICGNOptions(subset_size=31))
    centers = make_poi_grid(SHAPE, 31, 20, margin=10).xy + np.array([0.37, -0.81])
    res = solver.solve(solver.prepare_reference(ref, centers), tgt)
    assert (res.status == Status.OK).all()
    assert np.abs(res.xy - centers - np.array(shift)).max() < 0.01


def test_initial_guess_needed_for_large_shift(texture):
    ref = _img(texture)
    tgt = _img(texture, Translation(6.3, -5.2))
    solver = ICGNSolver(ICGNOptions(subset_size=31))
    pois = make_poi_grid(SHAPE, 31, 30, margin=16)
    refs = solver.prepare_reference(ref, pois.xy)
    p0 = sf.from_displacement(np.tile([6.0, -5.0], (len(pois), 1)), 1)
    res = solver.solve(refs, tgt, p0)
    assert (res.status == Status.OK).all()
    np.testing.assert_allclose(res.xy - pois.xy, np.tile([6.3, -5.2], (len(pois), 1)), atol=0.01)


def test_failure_statuses(texture):
    ref_frame = render_frame(texture, SHAPE, sensor=SensorModel(bits=12), oversample=2).astype(float)
    ref_frame[:, 100:] = 1000.0  # flat region -> no texture
    ref = BSplineImage(ref_frame, 3)
    solver = ICGNSolver(ICGNOptions(subset_size=21, max_iterations=3, convergence_threshold_px=1e-12))
    centers = np.array([[5.0, 60.0], [120.0, 60.0], [50.0, 60.0]])  # OOB, flat, normal
    refs = solver.prepare_reference(ref, centers)
    assert refs.status.tolist() == [Status.OUT_OF_BOUNDS, Status.LOW_TEXTURE, Status.OK]
    tgt = _img(texture, Translation(0.5, 0.0))
    res = solver.solve(refs, tgt)
    assert res.status[0] == Status.OUT_OF_BOUNDS and res.status[1] == Status.LOW_TEXTURE
    assert res.status[2] == Status.NOT_CONVERGED and res.iterations[2] == 3  # unreachable threshold
    assert np.isfinite(res.xy[2]).all() and np.isnan(res.xy[:2]).all()
    # a subset pushed out of the image by the initial guess
    far = solver.solve(refs, tgt, sf.from_displacement(np.array([[0, 0], [0, 0], [200.0, 0]]), 1))
    assert far.status[2] == Status.OUT_OF_BOUNDS
