"""Interpolation, shape functions, POI grid and initial-guess search."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from stereodic.core.types import ROI
from stereodic.dic import shape_functions as sf
from stereodic.dic.init_guess import global_shift, zncc_search
from stereodic.dic.interp import BSplineImage, bspline_weights
from stereodic.dic.mesh import make_poi_grid
from stereodic.synth import SensorModel, SpeckleTexture, Translation, render_frame

ORDERS = [3, 5]


@pytest.fixture(scope="module")
def speckle_image() -> np.ndarray:
    tex = SpeckleTexture.random((-20, -20, 100, 90), seed=11)
    return render_frame(tex, (70, 80), sensor=SensorModel(bits=12), oversample=4).astype(np.float64)


# -------------------------------------------------------------------- interp
@pytest.mark.parametrize("order", ORDERS)
def test_weights_partition_of_unity(order):
    f = np.linspace(0, 1, 101, endpoint=False)
    w, dw = bspline_weights(f, order)
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-14)
    np.testing.assert_allclose(dw.sum(axis=-1), 0.0, atol=1e-13)


@pytest.mark.parametrize("order", ORDERS)
def test_matches_scipy_map_coordinates(speckle_image, order):
    img = BSplineImage(speckle_image, order)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, img.width - 1, 500)
    y = rng.uniform(0, img.height - 1, 500)
    from scipy.ndimage import spline_filter

    c = spline_filter(speckle_image, order=order, mode="mirror")
    ref = map_coordinates(c, [y, x], order=order, mode="mirror", prefilter=False)
    np.testing.assert_allclose(img.evaluate(x, y), ref, rtol=0, atol=1e-9)


@pytest.mark.parametrize("order", ORDERS)
def test_interpolates_pixels(speckle_image, order):
    img = BSplineImage(speckle_image, order)
    yy, xx = np.mgrid[0 : img.height, 0 : img.width].astype(float)
    np.testing.assert_allclose(img.evaluate(xx, yy), speckle_image, rtol=0, atol=1e-8)


@pytest.mark.parametrize("order", ORDERS)
def test_gradient_matches_finite_difference(speckle_image, order):
    img = BSplineImage(speckle_image, order)
    rng = np.random.default_rng(1)
    x = rng.uniform(5, img.width - 6, 300)
    y = rng.uniform(5, img.height - 6, 300)
    _, gx, gy = img.evaluate_with_gradient(x, y)
    e = 1e-5
    fdx = (img.evaluate(x + e, y) - img.evaluate(x - e, y)) / (2 * e)
    fdy = (img.evaluate(x, y + e) - img.evaluate(x, y - e)) / (2 * e)
    scale = np.abs(speckle_image).max()
    np.testing.assert_allclose(gx, fdx, rtol=0, atol=1e-6 * scale)
    np.testing.assert_allclose(gy, fdy, rtol=0, atol=1e-6 * scale)


@pytest.mark.parametrize("order", ORDERS)
def test_reproduces_polynomial_in_interior(order):
    yy, xx = np.mgrid[0:60, 0:60].astype(float)
    img = BSplineImage(3.0 * xx - 2.0 * yy + 0.01 * xx * yy, order)
    x = np.array([25.3, 30.71, 33.5])
    y = np.array([28.9, 31.2, 27.05])
    v, gx, gy = img.evaluate_with_gradient(x, y)
    np.testing.assert_allclose(v, 3 * x - 2 * y + 0.01 * x * y, atol=1e-9)
    np.testing.assert_allclose(gx, 3 + 0.01 * y, atol=1e-9)
    np.testing.assert_allclose(gy, -2 + 0.01 * x, atol=1e-9)


def test_domain_checks(speckle_image):
    img = BSplineImage(speckle_image, 3)  # 80 x 70, border 5 -> reliable x in [5, 74], y in [5, 64]
    assert img.border == 5 and BSplineImage(speckle_image, 5).border == 8
    x = np.array([5.0, 74.0, 74.01, 4.99, 40.0])
    y = np.array([5.0, 64.0, 30.0, 30.0, 64.5])
    assert img.in_domain(x, y).tolist() == [True, True, False, False, False]
    with pytest.raises(ValueError):
        img.evaluate(np.array([-1.5]), np.array([3.0]))
    with pytest.raises(ValueError):
        BSplineImage(speckle_image, 4)


# ------------------------------------------------------------ shape functions
@pytest.mark.parametrize("order", [0, 1])
def test_matrix_roundtrip_and_inverse_compose(order):
    rng = np.random.default_rng(2)
    p = rng.normal(0, 0.05, sf.n_params(order))
    np.testing.assert_allclose(sf.from_matrix(sf.to_matrix(p, order), order), p, atol=1e-15)
    np.testing.assert_allclose(sf.compose_inverse(p, p, order), np.zeros_like(p), atol=1e-14)
    # composing with identity is a no-op
    np.testing.assert_allclose(sf.compose_inverse(p, sf.identity(order), order), p, atol=1e-15)


@pytest.mark.parametrize("order", [0, 1])
def test_jacobian_matches_finite_difference(order):
    dx = np.array([-3.0, 0.0, 2.0, 5.0])
    dy = np.array([1.0, -4.0, 0.0, 2.0])
    J = sf.jacobian(dx, dy, order)
    e = 1e-7
    for k in range(sf.n_params(order)):
        dp = np.zeros(sf.n_params(order))
        dp[k] = e
        xp, yp = sf.warp(dp, dx, dy, order)
        np.testing.assert_allclose((xp - dx) / e, J[:, 0, k], atol=1e-6)
        np.testing.assert_allclose((yp - dy) / e, J[:, 1, k], atol=1e-6)


def test_second_order_not_yet():
    with pytest.raises(NotImplementedError):
        sf.n_params(2)


def test_center_displacement_and_corner_motion():
    p = sf.from_displacement(np.array([0.3, -0.2]), 1)
    np.testing.assert_array_equal(sf.center_displacement(p, 1), [0.3, -0.2])
    assert sf.max_corner_motion(p, 15, 1) == pytest.approx(np.hypot(0.3, 0.2))
    p2 = np.array([0, 0.01, 0, 0, 0, 0.0])  # u_x = 1% strain -> corners move 0.15 px
    assert sf.max_corner_motion(p2, 15, 1) == pytest.approx(0.15)


# -------------------------------------------------------------------- mesh
def test_poi_grid_inside_image_and_mask():
    pois = make_poi_grid((100, 120), subset_size=21, step=10)
    assert pois.xy[:, 0].min() >= 11 and pois.xy[:, 0].max() <= 120 - 1 - 11
    assert np.array_equal(pois.point_ids, np.arange(len(pois)))
    roi_pois = make_poi_grid((100, 120), 21, 10, roi=ROI(30, 20, 60, 50))
    assert roi_pois.xy[:, 0].min() >= 30 + 11 and roi_pois.xy[:, 1].max() <= 20 + 50 - 1 - 11
    mask = np.zeros((100, 120), bool)
    mask[:, :60] = True
    m_pois = make_poi_grid((100, 120), 21, 10, mask=mask)
    assert len(m_pois) > 0 and m_pois.xy[:, 0].max() + 10 <= 59


# ---------------------------------------------------------------- init guess
def test_zncc_search_integer_shift():
    tex = SpeckleTexture.random((-30, -30, 130, 130), seed=5)
    ref = render_frame(tex, (100, 100), oversample=2, seed=1)
    tgt = render_frame(tex, (100, 100), Translation(7.0, -4.0), oversample=2, seed=2)
    centers = np.array([[40.0, 50.0], [55.4, 45.6], [5.0, 5.0]])
    uv, z = zncc_search(ref, tgt, centers, 21, search_radius=10)
    np.testing.assert_allclose(uv[:2], [[7, -4], [7, -4]], atol=0.15)
    assert np.all(z[:2] > 0.9)
    assert np.isnan(uv[2]).all()  # template does not fit


def test_global_shift():
    tex = SpeckleTexture.random((-30, -30, 160, 160), seed=6)
    ref = render_frame(tex, (128, 128), oversample=2)
    tgt = render_frame(tex, (128, 128), Translation(3.0, 2.0), oversample=2)
    uv, _ = global_shift(ref, tgt)
    np.testing.assert_allclose(uv, [3.0, 2.0], atol=0.1)
