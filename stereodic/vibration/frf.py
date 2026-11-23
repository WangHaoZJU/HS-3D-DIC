"""Frequency response functions and coherence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .fdd import _prepare_channels
from .spectrum import _uniform_sample_rate


@dataclass(frozen=True)
class FRFResult:
    frequencies: np.ndarray
    H1: np.ndarray
    coherence: np.ndarray
    sample_rate_hz: float
    n_samples: int


def estimate_frf(
    response: np.ndarray,
    force: np.ndarray,
    timestamps: np.ndarray,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> FRFResult:
    """Estimate H1 FRFs ``G_yF / G_FF`` for one or more response channels."""
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    y = _prepare_channels(response)
    f = np.asarray(force, dtype=np.float64).reshape(-1)
    if y.shape[0] != t.size or f.size != t.size:
        raise ValueError("response, force, and timestamps must have the same length")
    fs = _uniform_sample_rate(t)
    if nperseg is None:
        nperseg = min(1024, t.size)
    nperseg = max(2, min(int(nperseg), t.size))
    if noverlap is None:
        noverlap = nperseg // 2
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    finite = np.isfinite(f)
    if finite.sum() < 2:
        raise ValueError("force contains fewer than two finite samples")
    if not finite.all():
        idx = np.arange(f.size, dtype=np.float64)
        f = np.interp(idx, idx[finite], f[finite])
    f = f - f.mean()
    step = nperseg - noverlap
    starts = np.arange(0, t.size - nperseg + 1, step, dtype=int)
    if starts.size == 0:
        raise ValueError("not enough samples for the requested nperseg")
    window = signal.get_window("hann", nperseg, fftbins=True)
    response_spectra = []
    force_spectra = []
    for start in starts:
        response_spectra.append(np.fft.rfft(y[start : start + nperseg] * window[:, None], axis=0))
        force_spectra.append(np.fft.rfft(f[start : start + nperseg] * window))
    Y = np.stack(response_spectra, axis=0)
    F = np.stack(force_spectra, axis=0)
    G_yf = np.mean(Y * np.conj(F[:, :, None]), axis=0)
    G_ff = np.mean(np.abs(F) ** 2, axis=0)
    G_yy = np.mean(np.abs(Y) ** 2, axis=0)
    threshold = np.max(G_ff) * 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        H1 = G_yf / G_ff[:, None]
        coherence = np.abs(G_yf) ** 2 / (G_yy * G_ff[:, None])
    H1[G_ff[:, None] <= threshold] = np.nan + 1j * np.nan
    coherence[G_ff[:, None] <= threshold] = np.nan
    H1[~np.isfinite(H1)] = np.nan + 1j * np.nan
    coherence = np.clip(np.real(coherence), 0.0, 1.0)
    coherence[~np.isfinite(coherence)] = np.nan
    return FRFResult(np.fft.rfftfreq(nperseg, 1.0 / fs), H1, coherence, fs, t.size)
