"""Frame timestamps and left/right synchronization checks.

Image sequences do not always carry trustworthy timestamps in their filenames.
This module keeps timestamp handling explicit so downstream stages can use the
recorded time axis when it is available and fall back to the configured frame
rate only when necessary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SyncReport:
    """Diagnostics for a paired left/right acquisition."""

    left_indices: np.ndarray
    right_indices: np.ndarray
    left_timestamps: np.ndarray
    right_timestamps: np.ndarray
    delta_seconds: np.ndarray

    @property
    def max_abs_delta_seconds(self) -> float:
        return float(np.max(np.abs(self.delta_seconds))) if self.delta_seconds.size else 0.0

    @property
    def mean_abs_delta_seconds(self) -> float:
        return float(np.mean(np.abs(self.delta_seconds))) if self.delta_seconds.size else 0.0


def load_timestamps(path: str | Path) -> np.ndarray:
    """Load one timestamp per frame from ``.npy``, text, or CSV input."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        values = np.load(path)
    else:
        values = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    validate_timestamps(values, path.name)
    return values


def regular_timestamps(n_frames: int, frame_rate_hz: float, start_seconds: float = 0.0) -> np.ndarray:
    """Generate an explicit evenly spaced time axis."""
    if n_frames < 1 or frame_rate_hz <= 0:
        raise ValueError("n_frames must be positive and frame_rate_hz must be positive")
    return float(start_seconds) + np.arange(n_frames, dtype=np.float64) / float(frame_rate_hz)


def validate_timestamps(values: np.ndarray, name: str = "timestamps") -> None:
    """Validate finite, strictly increasing timestamps."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name}: timestamp array is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: timestamps must be finite")
    if values.size > 1 and not np.all(np.diff(values) > 0):
        raise ValueError(f"{name}: timestamps must be strictly increasing")


def pair_timestamps(
    left_timestamps: np.ndarray,
    right_timestamps: np.ndarray,
    max_delta_seconds: float | None = None,
) -> SyncReport:
    """Pair each left frame with the nearest right frame without reuse.

    The common hardware-triggered case has equal-length arrays and returns
    index-aligned pairs. For unequal or jittered arrays, a monotonic nearest
    pairing is used and frames that cannot be paired are omitted.
    """
    left = np.asarray(left_timestamps, dtype=np.float64).reshape(-1)
    right = np.asarray(right_timestamps, dtype=np.float64).reshape(-1)
    validate_timestamps(left, "left_timestamps")
    validate_timestamps(right, "right_timestamps")
    if max_delta_seconds is not None and max_delta_seconds < 0:
        raise ValueError("max_delta_seconds must be non-negative")

    if left.size == right.size:
        li = np.arange(left.size, dtype=np.int64)
        ri = np.arange(right.size, dtype=np.int64)
    else:
        li_list: list[int] = []
        ri_list: list[int] = []
        next_right = 0
        for i, t in enumerate(left):
            if next_right >= right.size:
                break
            j = next_right + int(np.argmin(np.abs(right[next_right:] - t)))
            if max_delta_seconds is None or abs(right[j] - t) <= max_delta_seconds:
                li_list.append(i)
                ri_list.append(j)
                next_right = j + 1
        li = np.asarray(li_list, dtype=np.int64)
        ri = np.asarray(ri_list, dtype=np.int64)

    delta = right[ri] - left[li]
    if max_delta_seconds is not None and np.any(np.abs(delta) > max_delta_seconds):
        keep = np.abs(delta) <= max_delta_seconds
        li, ri, delta = li[keep], ri[keep], delta[keep]
    return SyncReport(li, ri, left[li], right[ri], delta)


def sequence_timestamps(
    n_frames: int,
    frame_rate_hz: float,
    timestamp_path: str | Path | None = None,
) -> np.ndarray:
    """Return recorded timestamps or an explicit frame-rate fallback."""
    if timestamp_path is None:
        return regular_timestamps(n_frames, frame_rate_hz)
    values = load_timestamps(timestamp_path)
    if values.size != n_frames:
        raise ValueError(f"timestamp count {values.size} differs from frame count {n_frames}")
    return values
