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


def _psd_to_amplitude(psd: np.ndarray, fs: float, nperseg: int, window: str = "hann") -> np.ndarray:
    """Convert one-sided Welch PSD (unit^2/Hz) to peak sinusoidal amplitude using window ENBW."""
    w = signal.get_window(window, nperseg, fftbins=True)
    enbw = fs * float(np.sum(w * w)) / float(np.sum(w) ** 2)
    p_spec = np.maximum(psd, 0.0) * enbw
    scale = np.full_like(p_spec, 2.0)
    scale[0] = 1.0
    if nperseg % 2 == 0:
        scale[-1] = 1.0
    return np.sqrt(p_spec * scale)


def welch_spectrum(
    values: np.ndarray,
    timestamps: np.ndarray,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> Spectrum:
    """Return a Welch PSD and its ENBW-corrected sinusoidal peak amplitude."""
    x, fs = _prepare_signal(values, timestamps)
    if nperseg is None:
        nperseg = min(1024, x.size)
    nperseg = max(2, min(int(nperseg), x.size))
    f, p = signal.welch(
        x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density"
    )
    amp = _psd_to_amplitude(p, fs, nperseg, window="hann")
    return Spectrum(f, amp, p, fs, x.size)


def field_average_psd(
    values: np.ndarray,
    timestamps: np.ndarray,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> Spectrum:
    """Average Welch PSD across all valid spatial points (docs §11.2).

    Averaging power spectra rather than time-domain signals prevents opposite-phase
    regions (e.g. torsional or higher bending modes) from cancelling out.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return welch_spectrum(arr, timestamps, nperseg=nperseg, noverlap=noverlap)
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if arr.shape[0] != t.size:
        raise ValueError("values and timestamps must have the same time length")
    fs = _uniform_sample_rate(t)
    flat = arr.reshape(t.size, -1)
    valid_cols: list[np.ndarray] = []
    for j in range(flat.shape[1]):
        col = flat[:, j]
        finite = np.isfinite(col)
        if int(finite.sum()) < 2:
            continue
        if not finite.all():
            col = np.interp(t, t[finite], col[finite])
        valid_cols.append(signal.detrend(col, type="constant"))
    if not valid_cols:
        raise ValueError("no valid spatial channels with at least two finite samples")
    X = np.column_stack(valid_cols)
    if nperseg is None:
        nperseg = min(1024, t.size)
    nperseg = max(2, min(int(nperseg), t.size))
    f, p_all = signal.welch(
        X, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density", axis=0
    )
    p_mean = np.mean(p_all, axis=1)
    amp = _psd_to_amplitude(p_mean, fs, nperseg, window="hann")
    return Spectrum(f, amp, p_mean, fs, t.size)


def dominant_peaks(
    spectrum: Spectrum,
    min_frequency_hz: float = 0.0,
    max_frequency_hz: float | None = None,
    prominence: float | None = None,
    max_peaks: int = 10,
    refine: bool = True,
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
    chosen = peaks[order]
    peak_freqs = f[chosen].astype(np.float64, copy=True)
    peak_amps = a[chosen].astype(np.float64, copy=True)
    if refine and chosen.size:
        df = float(f[1] - f[0]) if f.size > 1 else 0.0
        for m, k in enumerate(chosen):
            if 0 < k < f.size - 1 and a[k - 1] > 0 and a[k] > 0 and a[k + 1] > 0:
                ym, y0, yp = np.log(a[k - 1]), np.log(a[k]), np.log(a[k + 1])
                denom = ym - 2.0 * y0 + yp
                if denom < 0.0:
                    delta = float(np.clip(0.5 * (ym - yp) / denom, -0.5, 0.5))
                    peak_freqs[m] = f[k] + delta * df
    return peak_freqs, peak_amps
