from __future__ import annotations

import numpy as np

from stereodic.vibration.spectrum import dominant_peaks, one_sided_spectrum


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
