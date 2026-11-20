from __future__ import annotations

import numpy as np
import pytest

from stereodic.synth import (
    Affine,
    SensorModel,
    SpeckleTexture,
    Translation,
    render_frame,
    render_transmission,
    s1_translation,
)


@pytest.fixture(scope="module")
def texture() -> SpeckleTexture:
    return SpeckleTexture.random((-20, -20, 100, 84), diameter_px=4.0, seed=1)


def test_speckle_count_matches_density():
    ext = (0, 0, 400, 400)
    tex = SpeckleTexture.random(ext, diameter_px=4.0, coverage=0.5, seed=3)
    expected = -np.log(0.5) / (np.pi * 4.0) * 400 * 400
    assert abs(len(tex) - expected) < 5 * np.sqrt(expected)
    assert len(SpeckleTexture.random(ext, seed=3)) == len(tex)  # deterministic with seed


def test_integer_shift_is_exact(texture):
    f = render_transmission(texture, (64, 80), oversample=4)
    g = render_transmission(texture, (64, 80), Translation(3.0, 2.0), oversample=4)
    # g(x) = f(x - u): pixel (r, c) of g equals pixel (r-2, c-3) of f
    np.testing.assert_allclose(g[2:, 3:], f[:-2, :-3], rtol=0, atol=1e-12)


def test_affine_matches_exact_texture():
    tex = SpeckleTexture.random((-15, -15, 40, 40), diameter_px=4.0, seed=7)
    A = np.array([[1.02, 0.03], [-0.01, 0.97]])
    m = Affine(A=A, t=np.array([0.37, -0.81]), center=np.array([12.0, 12.0]))
    t = render_transmission(tex, (24, 24), m, oversample=1)  # one sample per pixel = point sampling
    yy, xx = np.mgrid[0:24, 0:24]
    x = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    X = m.center + np.linalg.solve(A, (x - m.center - m.t).T).T  # inverse map to material coords
    expected = np.exp(-tex.density_at(X)).reshape(24, 24)
    np.testing.assert_allclose(t, expected, rtol=0, atol=1e-10)


def test_oversampling_converges(texture):
    ref = render_transmission(texture, (48, 48), Translation(0.37, 0.61), oversample=16)
    err8 = np.abs(render_transmission(texture, (48, 48), Translation(0.37, 0.61), oversample=8) - ref).max()
    err2 = np.abs(render_transmission(texture, (48, 48), Translation(0.37, 0.61), oversample=2) - ref).max()
    assert err8 < err2
    assert err8 < 5e-3


def test_sensor_quantization_and_noise(texture):
    sensor = SensorModel(bits=12, noise_std=2.0)
    a = render_frame(texture, (32, 32), sensor=sensor, oversample=2, seed=5)
    b = render_frame(texture, (32, 32), sensor=sensor, oversample=2, seed=5)
    c = render_frame(texture, (32, 32), sensor=sensor, oversample=2, seed=6)
    assert a.dtype == np.uint16 and a.max() <= 4095
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert render_frame(texture, (8, 8), sensor=SensorModel(bits=8), oversample=1).dtype == np.uint8


def test_speckle_contrast_reasonable(texture):
    img = render_frame(texture, (64, 64), sensor=SensorModel(bits=12), oversample=4).astype(float)
    # a usable speckle pattern spans a large part of the dynamic range
    assert np.percentile(img, 95) - np.percentile(img, 5) > 0.4 * 4095


def test_s1_pair_truth():
    ref, deformed, mapping = s1_translation(shape=(40, 50), shift=(0.25, -0.5), oversample=2, seed=2)
    assert ref.shape == deformed.shape == (40, 50)
    np.testing.assert_array_equal(mapping.displacement(np.zeros((3, 2))), [[0.25, -0.5]] * 3)
    assert not np.array_equal(ref, deformed)
