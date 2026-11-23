from __future__ import annotations

import numpy as np

from stereodic.vibration.spectrum import dominant_peaks, field_average_psd, one_sided_spectrum, welch_spectrum


def test_one_sided_spectrum_finds_known_frequency():
    fs = 2000.0
    n = 2048
    t = np.arange(n) / fs
    frequency = 123.046875
    spectrum = one_sided_spectrum(np.sin(2 * np.pi * frequency * t), t)
    peaks, amplitudes = dominant_peaks(spectrum, min_frequency_hz=20.0, max_frequency_hz=300.0, max_peaks=3)
    assert len(peaks) >= 1
    assert abs(peaks[0] - frequency) <= fs / n
    assert amplitudes[0] > 0.9


def test_welch_spectrum_amplitude_is_enbw_accurate():
    fs = 2000.0
    t = np.arange(8192) / fs
    f0 = 125.0  # exact bin for nperseg=1024
    y = 2.0 * np.cos(2.0 * np.pi * f0 * t)
    spec = welch_spectrum(y, t, nperseg=1024)
    peaks, amps = dominant_peaks(spec, min_frequency_hz=10.0, max_peaks=1)
    np.testing.assert_allclose(peaks[0], f0, atol=1e-6)
    np.testing.assert_allclose(amps[0], 2.0, rtol=1e-3)


def test_field_average_psd_detects_antisymmetric_torsional_mode():
    fs = 2000.0
    t = np.arange(4096) / fs
    f_torsion = 177.3  # off-bin frequency to also verify parabolic sub-bin refinement
    base = np.cos(2.0 * np.pi * f_torsion * t)
    # Antisymmetric mode: half +1, half -1 (time-domain spatial median/mean is identically 0)
    field = np.column_stack([base, base, -base, -base, np.full_like(base, np.nan)])
    assert np.allclose(np.nanmedian(field, axis=1), 0.0)
    spec = field_average_psd(field, t, nperseg=1024)
    peaks, amps = dominant_peaks(spec, min_frequency_hz=20.0, max_peaks=1, refine=True)
    assert len(peaks) == 1
    # Sub-bin parabolic refinement should be well within half a bin (df = 2000/1024 ~= 1.95 Hz)
    assert abs(peaks[0] - f_torsion) < 0.25
    assert amps[0] > 0.8
