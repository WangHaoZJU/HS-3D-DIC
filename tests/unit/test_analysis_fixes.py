"""Analysis-layer fixes from the code review (U5): time axis, frequency refinement, damping flag, force."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stereodic.analysis import load_force
from stereodic.vibration.damping import half_power_damping
from stereodic.vibration.harmonic import refine_frequency
from stereodic.vibration.ods import extract_ods
from stereodic.vibration.spectrum import _uniform_sample_rate, field_average_psd, one_sided_spectrum


def test_quantized_timestamps_accepted_gaps_rejected():
    fs = 7000.0
    t = np.round(np.arange(5000) / fs, 6)  # 1 µs timestamp resolution (0.7 % of the interval)
    assert _uniform_sample_rate(t) == pytest.approx(fs, rel=1e-6)
    with pytest.raises(ValueError, match="dropped frames"):
        _uniform_sample_rate(np.delete(t, 1234))
    with pytest.raises(ValueError, match="jitter"):
        _uniform_sample_rate(t + np.where(np.arange(t.size) == 10, 0.4 / fs, 0.0))


@pytest.mark.parametrize("duration", [1.0, 10.0])
def test_refined_frequency_removes_amplitude_bias(duration):
    fs, f_true = 2000.0, 137.3
    t = np.arange(int(duration * fs)) / fs
    rng = np.random.default_rng(0)
    amp = np.linspace(0.01, 0.05, 40)
    phase = rng.uniform(-np.pi, np.pi, 40)
    y = amp * np.cos(2 * np.pi * f_true * t[:, None] + phase) + rng.normal(0, 2e-4, (t.size, 40))
    f_peak = f_true + 0.4 / duration  # typical Welch-peak error (fraction of a bin)
    biased = extract_ods(y, t, f_peak).amplitude[0]
    f_ref = refine_frequency(y, t, f_peak)
    assert abs(f_ref - f_true) < 2e-3 / duration
    fixed = extract_ods(y, t, f_ref).amplitude[0]
    assert np.max(np.abs(fixed / amp - 1)) < 0.01
    assert np.max(np.abs(biased / amp - 1)) > 0.1  # what the old path produced


def test_damping_flags_resolution_limited_peaks():
    fs = 1000.0
    t = np.arange(512) / fs
    s = one_sided_spectrum(np.sin(2 * np.pi * 100.0 * t), t)  # undamped: width is the window main lobe
    r = half_power_damping(s.frequencies, s.amplitude)
    assert r.resolution_limited
    # a broad (heavily damped) resonance spanning many bins is not flagged
    rng = np.random.default_rng(1)
    from scipy import signal

    b, a = signal.iirpeak(100.0, Q=5, fs=fs)
    x = signal.lfilter(b, a, rng.normal(size=200_000))
    tt = np.arange(x.size) / fs
    spec = field_average_psd(x, tt, nperseg=8192)
    r2 = half_power_damping(spec.frequencies, spec.amplitude)
    assert not r2.resolution_limited and r2.damping_ratio > 0.02  # many bins wide: a real damping estimate


def test_force_resampled_to_frames(tmp_path: Path):
    t = np.arange(10) / 100.0
    p = tmp_path / "f.csv"
    tf = np.arange(0, 0.2, 1 / 1000.0)
    np.savetxt(p, np.column_stack([tf, np.sin(2 * np.pi * 5 * tf)]), delimiter=",")
    np.testing.assert_allclose(load_force(p, t), np.sin(2 * np.pi * 5 * t), atol=1e-3)
    np.savetxt(p, np.arange(10.0), delimiter=",")
    np.testing.assert_array_equal(load_force(p, t), np.arange(10.0))
    with pytest.raises(ValueError, match="one per frame"):
        load_force(p, np.arange(5) / 100.0)
