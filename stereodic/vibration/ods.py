"""Operating deflection shape extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harmonic import HarmonicFit, fit_harmonic


@dataclass(frozen=True)
class ODSResult:
    frequency_hz: float
    response: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    residual_rms: np.ndarray
    valid: np.ndarray
    fit: HarmonicFit


def extract_ods(
    values: np.ndarray,
    timestamps: np.ndarray,
    frequency_hz: float,
    *,
    n_harmonics: int = 1,
    weights: np.ndarray | None = None,
) -> ODSResult:
    """Fit a complex response field at ``frequency_hz``.

    The returned response follows ``Re(response * exp(i*omega*t))`` and is
    therefore ``a - i*b`` for the cosine/sine coefficients.
    """
    fit = fit_harmonic(values, timestamps, frequency_hz, n_harmonics=n_harmonics, weights=weights)
    response = fit.amplitudes * np.exp(1j * fit.phases)
    return ODSResult(
        frequency_hz=fit.frequency_hz,
        response=response,
        amplitude=fit.amplitudes,
        phase=fit.phases,
        residual_rms=fit.residual_rms,
        valid=fit.valid,
        fit=fit,
    )
