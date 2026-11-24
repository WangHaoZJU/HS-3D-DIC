"""Image quality checks for acquisition (T0, docs §2.2 / §10.1).

All functions take a single raw frame (native dtype) and are cheap enough to run
interactively on a 1280 x 1024 frame. They describe the speckle image, not the DIC
result: saturation, histogram, contrast, mean intensity gradient (MIG), speckle size
and a focus / motion-blur measure.
"""

from __future__ import annotations

import numpy as np


def effective_bits(values: np.ndarray | int) -> int:
    """Smallest bit depth that holds the maximum value (at least 1)."""
    vmax = int(np.max(values))
    return max(1, int(vmax).bit_length())


def trailing_zero_bits(images: list[np.ndarray]) -> int:
    """Low bits that are zero in every pixel (e.g. 12-bit data stored MSB-aligned in 16 bits)."""
    acc = 0
    for img in images:
        if np.issubdtype(img.dtype, np.integer):
            acc |= int(np.bitwise_or.reduce(img.ravel().astype(np.uint64)))
        else:
            return 0
    if acc == 0:
        return 0
    return (acc & -acc).bit_length() - 1


def histogram(img: np.ndarray, full_scale: int, bins: int = 128) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(img, bins=bins, range=(0, full_scale + 1))
    return counts, edges


def saturation_fraction(img: np.ndarray, level: float) -> float:
    return float(np.mean(img >= level))


def _crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape
    s_y, s_x = min(size, h), min(size, w)
    y0, x0 = (h - s_y) // 2, (w - s_x) // 2
    return img[y0 : y0 + s_y, x0 : x0 + s_x].astype(np.float64)


def mean_intensity_gradient(img: np.ndarray, crop: int = 512) -> float:
    """MIG (Pan et al. 2010): mean |grad I| with central differences, gray levels per px."""
    a = _crop(img, crop)
    gy, gx = np.gradient(a)
    return float(np.mean(np.hypot(gx, gy)))


def sharpness(img: np.ndarray, crop: int = 512) -> float:
    """Gradient energy normalised by intensity variance: mean |grad I|^2 / var(I).

    Independent of brightness and gain; motion blur and defocus lower it, so the
    sharpest frames are good reference-frame candidates. Units 1/px^2.
    """
    a = _crop(img, crop)
    var = a.var()
    if var <= 0:
        return 0.0
    gy, gx = np.gradient(a)
    return float(np.mean(gx * gx + gy * gy) / var)


# FWHM of the autocorrelation peak -> speckle diameter. For Gaussian blobs of diameter D
# (2r, intensity e^-1 at r: the synth.SpeckleTexture convention) the autocorrelation FWHM
# is ~1.35 D; calibrated on rendered speckle with D = 3..8 px (tests/unit/test_quality_image.py).
_AC_FWHM_PER_DIAMETER = 1.35


def speckle_size(img: np.ndarray, crop: int = 256) -> float:
    """Speckle diameter in px estimated from the image autocorrelation.

    The full width at half maximum of the autocorrelation peak (averaged over eight
    directions) divided by 1.35, i.e. the diameter in the e^-1 convention used by the
    synthetic speckle and the design rule of 3–5 px (docs §3.1). Real speckle patterns
    are not Gaussian blobs, so treat it as a +-20 % estimate.
    """
    a = _crop(img, crop)
    a = a - a.mean()
    if not np.any(a):
        return float("nan")
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    f = np.fft.rfft2(a * win)
    ac = np.fft.fftshift(np.fft.irfft2(np.abs(f) ** 2, s=a.shape))
    ac /= ac.max()
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    # half maximum along the four axis directions and the diagonals, then average
    radii = []
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        step = float(np.hypot(dy, dx))
        prev = 1.0
        for k in range(1, min(cy, cx)):
            v = ac[cy + k * dy, cx + k * dx]
            if v < 0.5:
                frac = (prev - 0.5) / (prev - v) if prev != v else 0.0
                radii.append((k - 1 + frac) * step)
                break
            prev = v
    return float(2.0 * np.mean(radii) / _AC_FWHM_PER_DIAMETER) if radii else float("nan")


def frame_report(img: np.ndarray, full_scale: int, bins: int = 128) -> dict:
    """All per-frame checks as a JSON-serialisable dict."""
    counts, edges = histogram(img, full_scale, bins)
    a = img.astype(np.float64)
    mean = float(a.mean())
    return {
        "full_scale": int(full_scale),
        "min": int(np.min(img)) if np.issubdtype(img.dtype, np.integer) else float(np.min(img)),
        "max": int(np.max(img)) if np.issubdtype(img.dtype, np.integer) else float(np.max(img)),
        "mean": mean,
        "std": float(a.std()),
        "p01": float(np.percentile(a, 0.1)),
        "p999": float(np.percentile(a, 99.9)),
        "saturation_fraction": saturation_fraction(img, full_scale),
        "dark_fraction": float(np.mean(img <= 0)),
        "contrast": float(a.std() / mean) if mean > 0 else 0.0,
        "mig": mean_intensity_gradient(img),
        "speckle_px": speckle_size(img),
        "sharpness": sharpness(img),
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
    }
