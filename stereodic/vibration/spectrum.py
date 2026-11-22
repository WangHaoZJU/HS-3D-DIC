"""FFT/Welch spectra and dominant frequency extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class Spectrum:
    frequencies: np.ndarray
    amplitude: np.ndarray
    power: np.ndarray
    sample_rate_hz: float
    n_samples: int


def _uniform_sample_rate(timestamps: np.ndarray, tolerance: float = 1e-3) -> float:
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if t.size < 2:
        raise ValueError("at least two timestamps are required")
    dt = np.diff(t)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    median = float(np.median(dt))
    if np.max(np.abs(dt - median)) > tolerance * median:
        raise ValueError("FFT/Welch requires nearly uniform timestamps")
    return 1.0 / median


def _prepare_signal(values: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if x.size != t.size:
        raise ValueError("values and timestamps must have the same length")
    fs = _uniform_sample_rate(t)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        raise ValueError("values contain fewer than two finite samples")
    if not finite.all():
        x = np.interp(t, t[finite], x[finite])
    return signal.detrend(x, type="constant"), fs


def one_sided_spectrum(values: np.ndarray, timestamps: np.ndarray, window: str = "hann") -> Spectrum:
    """Return a single-sided amplitude and power spectrum."""
    x, fs = _prepare_signal(values, timestamps)
    n = x.size
    w = signal.get_window(window, n, fftbins=True)
    coherent_gain = np.sum(w)
    y = np.fft.rfft(x * w)
    amp = 2.0 * np.abs(y) / coherent_gain
    if n % 2 == 0:
        amp[-1] *= 0.5
    amp[0] *= 0.5
    return Spectrum(np.fft.rfftfreq(n, 1.0 / fs), amp, amp * amp, fs, n)


def welch_spectrum(
    values: np.ndarray,
    timestamps: np.ndarray,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> Spectrum:
    """Return a Welch power spectrum and its amplitude-equivalent square root."""
    x, fs = _prepare_signal(values, timestamps)
    if nperseg is None:
        nperseg = min(1024, x.size)
    nperseg = max(2, min(int(nperseg), x.size))
    f, p = signal.welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    amp = np.sqrt(np.maximum(p, 0.0) * 2.0 * fs / nperseg)
    return Spectrum(f, amp, p, fs, x.size)


def dominant_peaks(
    spectrum: Spectrum,
    min_frequency_hz: float = 0.0,
    max_frequency_hz: float | None = None,
    prominence: float | None = None,
    max_peaks: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dominant peak frequencies and amplitudes, sorted by amplitude."""
    f = spectrum.frequencies
    a = spectrum.amplitude
    keep = f >= min_frequency_hz
    if max_frequency_hz is not None:
        keep &= f <= max_frequency_hz
    idx = np.nonzero(keep)[0]
    if idx.size < 3:
        return np.empty(0), np.empty(0)
    kwargs = {} if prominence is None else {"prominence": prominence}
    local, _ = signal.find_peaks(a[idx], **kwargs)
    peaks = idx[local]
    order = np.argsort(a[peaks])[::-1][:max_peaks]
    return f[peaks[order]], a[peaks[order]]
