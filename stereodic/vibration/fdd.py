"""Frequency-domain decomposition (FDD) for output-only vibration data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .spectrum import _uniform_sample_rate


@dataclass(frozen=True)
class FDDResult:
    frequencies: np.ndarray
    singular_values: np.ndarray
    mode_shapes: np.ndarray
    sample_rate_hz: float
    n_samples: int


def _prepare_channels(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("values must have shape (T, channels) or (T, ...)")
    x = x.reshape(x.shape[0], -1).copy()
    for j in range(x.shape[1]):
        finite = np.isfinite(x[:, j])
        if finite.sum() < 2:
            x[:, j] = 0.0
        elif not finite.all():
            t = np.arange(x.shape[0], dtype=np.float64)
            x[:, j] = np.interp(t, t[finite], x[finite, j])
        x[:, j] -= np.mean(x[:, j])
    return x


def frequency_domain_decomposition(
    values: np.ndarray,
    timestamps: np.ndarray,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    n_modes: int = 3,
) -> FDDResult:
    """Compute singular values and singular vectors of output spectral matrices."""
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    x = _prepare_channels(values)
    if x.shape[0] != t.size:
        raise ValueError("values and timestamps must have the same time length")
    fs = _uniform_sample_rate(t)
    if nperseg is None:
        nperseg = min(1024, x.shape[0])
    nperseg = max(2, min(int(nperseg), x.shape[0]))
    if noverlap is None:
        noverlap = nperseg // 2
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    step = nperseg - noverlap
    starts = np.arange(0, x.shape[0] - nperseg + 1, step, dtype=int)
    if starts.size == 0:
        raise ValueError("not enough samples for the requested nperseg")
    window = signal.get_window("hann", nperseg, fftbins=True)
    scale = np.sqrt(np.sum(window * window))
    spectra = []
    for start in starts:
        spectra.append(np.fft.rfft(x[start : start + nperseg] * window[:, None], axis=0) / scale)
    Y = np.stack(spectra, axis=0)  # (segments, frequencies, channels)
    frequencies = np.fft.rfftfreq(nperseg, 1.0 / fs)
    modes = min(max(1, n_modes), x.shape[1])
    singular_values = np.zeros((frequencies.size, modes), dtype=np.float64)
    mode_shapes = np.zeros((frequencies.size, modes, x.shape[1]), dtype=np.complex128)
    for k in range(frequencies.size):
        Z = Y[:, k, :].T  # channels x segments
        spectral = (Z @ Z.conj().T) / max(1, Z.shape[1])
        values_k, vectors_k = np.linalg.eigh(spectral)
        order = np.argsort(values_k)[::-1][:modes]
        singular_values[k] = np.maximum(values_k[order].real, 0.0)
        mode_shapes[k] = vectors_k[:, order].T
    return FDDResult(frequencies, singular_values, mode_shapes, fs, x.shape[0])
