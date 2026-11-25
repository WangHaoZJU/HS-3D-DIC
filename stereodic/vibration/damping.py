"""Damping estimates from frequency response or output spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HalfPowerResult:
    peak_frequency_hz: float
    left_frequency_hz: float
    right_frequency_hz: float
    damping_ratio: float
    peak_index: int
    resolution_limited: bool = False  # half-power band narrower than ~4 frequency bins: zeta is an upper bound


def _crossing(f0: float, a0: float, f1: float, a1: float, level: float) -> float:
    if a1 == a0:
        return float(f0)
    return float(f0 + (level - a0) * (f1 - f0) / (a1 - a0))


def half_power_damping(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    peak_index: int | None = None,
) -> HalfPowerResult:
    """Estimate damping ratio from the half-power bandwidth."""
    f = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    a = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    if f.size != a.size or f.size < 3 or not np.all(np.diff(f) > 0):
        raise ValueError("frequencies and amplitudes must be increasing arrays of length >= 3")
    if peak_index is None:
        peak_index = int(np.nanargmax(a))
    if peak_index <= 0 or peak_index >= f.size - 1 or not np.isfinite(a[peak_index]) or a[peak_index] <= 0:
        raise ValueError("peak must be an interior finite positive value")
    level = a[peak_index] / np.sqrt(2.0)
    i_left = None
    for i in range(peak_index - 1, -1, -1):
        if not np.isfinite(a[i]):
            break
        if a[i] <= level:
            i_left = i
            break
    i_right = None
    for i in range(peak_index + 1, f.size):
        if not np.isfinite(a[i]):
            break
        if a[i] <= level:
            i_right = i
            break
    if i_left is None or i_right is None:
        return HalfPowerResult(float(f[peak_index]), np.nan, np.nan, np.nan, peak_index)
    f_left = _crossing(f[i_left], a[i_left], f[i_left + 1], a[i_left + 1], level)
    f_right = _crossing(f[i_right - 1], a[i_right - 1], f[i_right], a[i_right], level)
    zeta = (f_right - f_left) / (2.0 * f[peak_index])
    df = float(np.median(np.diff(f)))
    # a Hann window alone gives a -3 dB width of ~1.44 bins; below ~4 bins the bandwidth
    # measures the spectral resolution, not the damping (review: 0.017 for a true 0.005)
    limited = (f_right - f_left) < 4.0 * df
    return HalfPowerResult(float(f[peak_index]), f_left, f_right, float(zeta), peak_index, bool(limited))
