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

    # Fast vectorized path when weights are shared across channels (None or 1D)
    shared_weights = weights is None or np.asarray(weights).ndim == 1
    full_mask = np.all(np.isfinite(flat), axis=0)
    if shared_weights and t.size >= n_coeff and np.any(full_mask):
        w1 = np.ones(t.size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
        w_ok = np.isfinite(w1) & (w1 > 0)
        if int(w_ok.sum()) >= n_coeff:
            A_full = design[w_ok]
            sw = np.sqrt(w1[w_ok])
            B_full = flat[w_ok][:, full_mask]
            try:
                sol_full, _, rank_full, _ = np.linalg.lstsq(A_full * sw[:, None], B_full * sw[:, None], rcond=None)
            except np.linalg.LinAlgError:
                rank_full = 0
                sol_full = None
            if sol_full is not None and rank_full >= n_coeff:
                ok_cols = np.all(np.isfinite(sol_full), axis=0)
                full_idx = np.nonzero(full_mask)[0][ok_cols]
                if full_idx.size:
                    sol_ok = sol_full[:, ok_cols]
                    err = A_full @ sol_ok - B_full[:, ok_cols]
                    w_norm = w1[w_ok] / np.sum(w1[w_ok])
                    coeff[:, full_idx] = sol_ok
                    residual[full_idx] = np.sqrt(np.sum(w_norm[:, None] * (err * err), axis=0))
                    valid[full_idx] = True
                    n_used[full_idx] = int(w_ok.sum())
            remaining_idx = np.nonzero(~full_mask)[0]
        else:
            remaining_idx = np.arange(flat.shape[1])
    else:
        remaining_idx = np.arange(flat.shape[1])

    for j in remaining_idx:
        mask = np.isfinite(flat[:, j]) & np.isfinite(w_flat[:, j]) & (w_flat[:, j] > 0)
        if int(mask.sum()) < n_coeff:
            continue
        A = design[mask]
        b = flat[mask, j]
        wj = w_flat[mask, j]
        sw = np.sqrt(wj)
        try:
            solution, _, rank, _ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank < n_coeff or not np.all(np.isfinite(solution)):
            continue
        coeff[:, j] = solution
        err = A @ solution - b
        residual[j] = float(np.sqrt(np.sum(wj * (err * err)) / np.sum(wj)))
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


def _harmonic_energy(flat: np.ndarray, t0: np.ndarray, frequency_hz: float) -> float:
    """Total fitted energy sum_j (a_j^2 + b_j^2) at one frequency (offset included in the model)."""
    w = 2.0 * np.pi * frequency_hz * t0
    design = np.column_stack([np.cos(w), np.sin(w), np.ones_like(t0)])
    sol, *_ = np.linalg.lstsq(design, flat, rcond=None)
    return float(np.sum(sol[0] ** 2 + sol[1] ** 2))


def refine_frequency(
    values: np.ndarray,
    timestamps: np.ndarray,
    frequency_hz: float,
    *,
    halfwidth_hz: float | None = None,
    max_channels: int = 4000,
    tolerance_hz: float | None = None,
) -> float:
    """Frequency that maximizes the fitted harmonic energy summed over the whole field.

    A frequency error df accumulates 2*pi*df*T of phase over the record and biases the
    fitted amplitude (review: 3 % at 1 s, 21 % at 10 s with a Welch-peak frequency). The
    energy is a smooth peak of width ~1/T around the true frequency; it is searched on a
    grid over +-halfwidth (default 1.5/T) and refined by golden-section search. Works for
    undersampled data too, provided the aliased frequency is away from 0 and fs/2.
    """
    y = np.asarray(values, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if y.shape[0] != t.size:
        raise ValueError("values and timestamps must have the same time length")
    flat = y.reshape(t.size, -1)
    flat = flat[:, np.all(np.isfinite(flat), axis=0)]
    if flat.shape[1] == 0:
        raise ValueError("no channel without missing samples")
    if flat.shape[1] > max_channels:
        flat = flat[:, np.linspace(0, flat.shape[1] - 1, max_channels).astype(int)]
    t0 = t - t.mean()
    duration = float(t[-1] - t[0])
    if halfwidth_hz is None:
        halfwidth_hz = 1.5 / duration
    if tolerance_hz is None:
        tolerance_hz = 1e-5 / duration
    grid = np.linspace(frequency_hz - halfwidth_hz, frequency_hz + halfwidth_hz, 61)
    grid = grid[grid > 0]
    e = np.array([_harmonic_energy(flat, t0, f) for f in grid])
    k = int(np.argmax(e))
    a = grid[max(k - 1, 0)]
    b = grid[min(k + 1, grid.size - 1)]
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    ec, ed = _harmonic_energy(flat, t0, c), _harmonic_energy(flat, t0, d)
    for _ in range(80):
        if b - a < tolerance_hz:
            break
        if ec > ed:
            b, d, ed = d, c, ec
            c = b - g * (b - a)
            ec = _harmonic_energy(flat, t0, c)
        else:
            a, c, ec = c, d, ed
            d = a + g * (b - a)
            ed = _harmonic_energy(flat, t0, d)
    return float(0.5 * (a + b))
