"""Small image utilities shared by detection and feature matching."""

from __future__ import annotations

import numpy as np


def to_uint8(image: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Contrast-stretch any bit depth to uint8 (for OpenCV detectors only, never for DIC)."""
    img = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(img, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip(np.rint((img - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)
