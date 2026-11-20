"""Integer-pixel initial guesses (ZNCC template search) for IC-GN.

Only used to start the sub-pixel solver: the result is never reported as a
measurement. OpenCV's `matchTemplate(TM_CCOEFF_NORMED)` is exactly ZNCC and uses
an FFT internally for large templates, so it serves as the FFT-CC step.
"""

from __future__ import annotations

import cv2
import numpy as np


def _parabolic_offset(cm: float, c0: float, cp: float) -> float:
    denom = cm - 2.0 * c0 + cp
    if denom >= 0:  # not a maximum
        return 0.0
    return float(np.clip(0.5 * (cm - cp) / denom, -0.5, 0.5))


def zncc_search(
    ref_image: np.ndarray,
    target_image: np.ndarray,
    centers: np.ndarray,
    subset_size: int,
    search_radius: int,
    center_guess: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Search each subset of `ref_image` in `target_image` within +-search_radius px.

    centers: (N, 2) subset centres in the reference image (rounded to pixels for the template).
    center_guess: optional (N, 2) predicted displacement around which to search.
    Returns (uv, zncc): displacement (N, 2) with parabolic sub-pixel refinement, NaN where
    the template or the search window does not fit; and the peak ZNCC (N,).
    """
    if subset_size % 2 == 0:
        raise ValueError("subset_size must be odd")
    ref = np.asarray(ref_image, dtype=np.float32)
    tgt = np.asarray(target_image, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    n = centers.shape[0]
    guess = np.zeros((n, 2)) if center_guess is None else np.asarray(center_guess, dtype=np.float64).reshape(n, 2)
    h = subset_size // 2
    H, W = tgt.shape
    uv = np.full((n, 2), np.nan)
    score = np.full(n, np.nan)
    for i in range(n):
        if not np.all(np.isfinite(centers[i])) or not np.all(np.isfinite(guess[i])):
            continue
        cx, cy = (int(round(v)) for v in centers[i])
        if cx - h < 0 or cy - h < 0 or cx + h >= ref.shape[1] or cy + h >= ref.shape[0]:
            continue
        tpl = ref[cy - h : cy + h + 1, cx - h : cx + h + 1]
        if float(tpl.std()) == 0.0:
            continue
        gx, gy = (int(round(v)) for v in guess[i])
        x0 = max(cx + gx - h - search_radius, 0)
        y0 = max(cy + gy - h - search_radius, 0)
        x1 = min(cx + gx + h + search_radius + 1, W)
        y1 = min(cy + gy + h + search_radius + 1, H)
        if x1 - x0 < subset_size or y1 - y0 < subset_size:
            continue
        res = cv2.matchTemplate(tgt[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
        iy, ix = np.unravel_index(int(np.argmax(res)), res.shape)
        sx = _parabolic_offset(res[iy, ix - 1], res[iy, ix], res[iy, ix + 1]) if 0 < ix < res.shape[1] - 1 else 0.0
        sy = _parabolic_offset(res[iy - 1, ix], res[iy, ix], res[iy + 1, ix]) if 0 < iy < res.shape[0] - 1 else 0.0
        # matched template top-left (x0+ix, y0+iy) -> centre; displacement of the (rounded) template centre
        uv[i, 0] = x0 + ix + h + sx - cx
        uv[i, 1] = y0 + iy + h + sy - cy
        score[i] = float(res[iy, ix])
    return uv, score


def global_shift(ref_image: np.ndarray, target_image: np.ndarray) -> tuple[np.ndarray, float]:
    """Whole-image translation (u, v) by phase correlation; returns (uv, response)."""
    ref = np.asarray(ref_image, dtype=np.float64)
    tgt = np.asarray(target_image, dtype=np.float64)
    window = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_64F)
    (u, v), response = cv2.phaseCorrelate(ref, tgt, window)
    return np.array([u, v]), float(response)
