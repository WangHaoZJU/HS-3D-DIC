"""Static displacement noise-floor estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseFloor:
    mean: np.ndarray  # (N,C)
    standard_deviation: np.ndarray  # (N,C)
    rms: np.ndarray  # (N,C)
    valid_samples: np.ndarray  # (N,)


def estimate_noise_floor(displacement: np.ndarray, valid_mask: np.ndarray | None = None) -> NoiseFloor:
    """Estimate per-point component noise from a stationary displacement series."""
    values = np.asarray(displacement, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("displacement must have shape (T,N,C)")
    finite = np.all(np.isfinite(values), axis=2)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != finite.shape:
            raise ValueError("valid_mask must have shape (T,N)")
        finite &= mask
    masked = np.where(finite[:, :, None], values, np.nan)
    mean = np.nanmean(masked, axis=0)
    std = np.nanstd(masked, axis=0, ddof=1)
    rms = np.sqrt(np.nanmean(masked * masked, axis=0))
    count = finite.sum(axis=0)
    mean[count < 1] = np.nan
    std[count < 2] = np.nan
    rms[count < 1] = np.nan
    return NoiseFloor(mean, std, rms, count)
