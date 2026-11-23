"""Time-domain harmonic fitting for displacement fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarmonicFit:
    """Least-squares fit results.

    ``amplitude`` and ``phase`` have shape ``(H, ...)`` where ``H`` is the
    number of fitted harmonics. Phase follows ``A*cos(omega*t + phase)``.
    """

    frequency_hz: float
    amplitudes: np.ndarray
    phases: np.ndarray
    coefficients: np.ndarray
    residual_rms: np.ndarray
    valid: np.ndarray
    n_samples: np.ndarray

    @property
    def amplitude(self) -> np.ndarray:
        """Alias for the plural field used by result schemas."""
        return self.amplitudes

    @property
    def phase(self) -> np.ndarray:
        return self.phases


def fit_harmonic(
    values: np.ndarray,
    timestamps: np.ndarray,
    frequency_hz: float,
    *,
    n_harmonics: int = 1,
    include_offset: bool = True,
    include_trend: bool = False,
    weights: np.ndarray | None = None,
) -> HarmonicFit:
    """Fit one or more harmonics to a scalar or full-field time series.

    ``values`` has time on axis 0. Missing samples are ignored independently
    for every spatial point and component. The fitted model is

    ``sum_h [a_h*cos(h*w*t) + b_h*sin(h*w*t)] + offset + trend*t``.
    """
    y = np.asarray(values, dtype=np.float64)
    if y.ndim == 0:
        raise ValueError("values must have a time axis")
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if y.shape[0] != t.size:
        raise ValueError("values and timestamps must have the same time length")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be positive")
    if t.size < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError("timestamps must be finite and strictly increasing")

    field_shape = y.shape[1:]
    flat = y.reshape(y.shape[0], -1)
    if weights is None:
        w_flat = np.ones_like(flat)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 1 and w.size == t.size:
            w_flat = np.broadcast_to(w[:, None], flat.shape)
        elif w.shape == y.shape:
            w_flat = w.reshape(flat.shape)
        else:
            raise ValueError("weights must have shape (T,) or values.shape")

    t0 = t - t.mean()
    columns = []
    for h in range(1, n_harmonics + 1):
        omega_t = 2.0 * np.pi * frequency_hz * h * t0
        columns.extend((np.cos(omega_t), np.sin(omega_t)))
    if include_offset:
        columns.append(np.ones_like(t))
    if include_trend:
        columns.append(t0)
    design = np.column_stack(columns)
    n_coeff = design.shape[1]

    coeff = np.full((n_coeff, flat.shape[1]), np.nan, dtype=np.float64)
    residual = np.full(flat.shape[1], np.nan, dtype=np.float64)
    valid = np.zeros(flat.shape[1], dtype=bool)
    n_used = np.zeros(flat.shape[1], dtype=np.int64)
    for j in range(flat.shape[1]):
        mask = np.isfinite(flat[:, j]) & np.isfinite(w_flat[:, j]) & (w_flat[:, j] > 0)
        if int(mask.sum()) < n_coeff:
            continue
        A = design[mask]
        b = flat[mask, j]
        sw = np.sqrt(w_flat[mask, j])
        try:
            solution, _, rank, _ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank < n_coeff or not np.all(np.isfinite(solution)):
            continue
        coeff[:, j] = solution
        residual[j] = float(np.sqrt(np.mean((A @ solution - b) ** 2)))
        valid[j] = True
        n_used[j] = int(mask.sum())

    amplitudes = np.full((n_harmonics, flat.shape[1]), np.nan, dtype=np.float64)
    phases = np.full_like(amplitudes, np.nan)
    for h in range(n_harmonics):
        a = coeff[2 * h]
        b = coeff[2 * h + 1]
        amplitudes[h] = np.hypot(a, b)
        phase_centered = np.arctan2(-b, a)
        phase_original = phase_centered - 2.0 * np.pi * frequency_hz * (h + 1) * t.mean()
        phases[h] = (phase_original + np.pi) % (2.0 * np.pi) - np.pi
    amplitudes[:, ~valid] = np.nan
    phases[:, ~valid] = np.nan
    return HarmonicFit(
        frequency_hz=float(frequency_hz),
        amplitudes=amplitudes.reshape((n_harmonics,) + field_shape),
        phases=phases.reshape((n_harmonics,) + field_shape),
        coefficients=coeff.reshape((n_coeff,) + field_shape),
        residual_rms=residual.reshape(field_shape),
        valid=valid.reshape(field_shape),
        n_samples=n_used.reshape(field_shape),
    )
