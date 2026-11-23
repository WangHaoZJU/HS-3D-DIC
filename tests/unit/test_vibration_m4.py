from __future__ import annotations

import numpy as np

from stereodic.recon.rbm import compensate_trajectory, estimate_rigid_motion
from stereodic.vibration.damping import half_power_damping
from stereodic.vibration.fdd import frequency_domain_decomposition
from stereodic.vibration.frf import estimate_frf
from stereodic.vibration.harmonic import fit_harmonic
from stereodic.vibration.mac import mac_matrix, modal_assurance
from stereodic.vibration.ods import extract_ods


def test_harmonic_fit_recovers_amplitude_phase_with_irregular_time():
    rng = np.random.default_rng(2)
    t = np.cumsum(0.001 + rng.uniform(-2e-5, 2e-5, 1200))
    f = 37.0
    phase = 0.42
    values = np.column_stack(
        [2.5 * np.cos(2 * np.pi * f * t + phase) + 0.7, 1.2 * np.cos(2 * np.pi * f * t - 0.2)]
    )
    result = fit_harmonic(values, t, f)
    np.testing.assert_allclose(result.amplitude[0], [2.5, 1.2], atol=1e-10)
    np.testing.assert_allclose(result.phase[0], [phase, -0.2], atol=1e-10)
    assert np.nanmax(result.residual_rms) < 1e-10


def test_ods_response_matches_cosine_convention():
    t = np.arange(1024) / 2000.0
    f = 83.984375
    y = (1.7 * np.cos(2 * np.pi * f * t + 0.3))[:, None, None]
    ods = extract_ods(y, t, f)
    np.testing.assert_allclose(ods.amplitude[0, 0, 0], 1.7, atol=1e-10)
    np.testing.assert_allclose(ods.phase[0, 0, 0], 0.3, atol=1e-10)
    np.testing.assert_allclose(ods.response[0, 0, 0], 1.7 * np.exp(1j * 0.3), atol=1e-10)


def test_fdd_finds_shared_output_frequency():
    fs = 1000.0
    t = np.arange(4096) / fs
    f = 61.03515625
    x = np.column_stack([np.sin(2 * np.pi * f * t), 0.5 * np.sin(2 * np.pi * f * t + 0.4)])
    result = frequency_domain_decomposition(x, t, nperseg=1024, n_modes=2)
    peak = int(np.argmax(result.singular_values[:, 0]))
    assert abs(result.frequencies[peak] - f) <= fs / 1024
    assert result.mode_shapes.shape == (513, 2, 2)


def test_frf_and_coherence_for_linear_response():
    fs = 1000.0
    t = np.arange(2048) / fs
    f = 49.8046875
    force = np.sin(2 * np.pi * f * t)
    response = (2.0 * force)[:, None]
    result = estimate_frf(response, force, t, nperseg=1024)
    k = int(round(f / (fs / 1024)))
    assert abs(result.frequencies[k] - f) <= fs / 1024
    assert abs(abs(result.H1[k, 0]) - 2.0) < 1e-10
    assert result.coherence[k, 0] > 0.999999


def test_half_power_damping_and_mac():
    f0, zeta = 100.0, 0.02
    frequencies = np.linspace(80.0, 120.0, 4001)
    amplitude = 1.0 / np.sqrt(1.0 + ((frequencies - f0) / (zeta * f0)) ** 2)
    result = half_power_damping(frequencies, amplitude)
    assert abs(result.peak_frequency_hz - f0) < 0.02
    assert abs(result.damping_ratio - zeta) < 5e-4
    a = np.array([[1.0, 2.0j, 3.0], [2.0, -1.0j, 0.5]])
    b = np.array([[2.0, 4.0j, 6.0], [1.0, 0.0j, 0.0]])
    np.testing.assert_allclose(modal_assurance(a[0], b[0]), 1.0)
    assert mac_matrix(a, b).shape == (2, 2)


def test_rigid_motion_estimation_and_compensation():
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 2.0, 3.0]])
    angle = 0.23
    R = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([3.0, -2.0, 0.7])
    current = reference @ R.T + translation
    motion = estimate_rigid_motion(reference, current)
    np.testing.assert_allclose(motion.rotation, R, atol=1e-12)
    np.testing.assert_allclose(motion.translation, translation, atol=1e-12)
    trajectory = np.stack([reference, current])
    compensated = compensate_trajectory(reference, trajectory)
    np.testing.assert_allclose(compensated.displacement, 0.0, atol=1e-12)
    assert compensated.valid.all()
