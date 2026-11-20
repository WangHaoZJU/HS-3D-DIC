"""B-spline image interpolation — FP64 reference implementation.

The image is prefiltered once into B-spline coefficients (SciPy `spline_filter`,
mirror boundary) and then evaluated with explicit basis functions, so values and
gradients come from the same continuous model (gradients are exact derivatives of
the interpolant, not finite differences).

Boundary handling: coefficients are mirror-extended (SciPy 'mirror' == NumPy
'reflect'). The mirror condition is not the true continuation of the scene, so
the interpolant is inaccurate within a few pixels of the image edge (measured on
synthetic speckle: ~4 px for cubic, ~7 px for quintic). `in_domain` therefore
excludes a safety band `BORDER[order]`; DIC code treats points outside it as
out of bounds. Points may still be evaluated in [-1, W] x [-1, H].
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import spline_filter

SUPPORTED_ORDERS = (3, 5)
# Pixels next to the image edge excluded from the reliable domain, per spline order.
BORDER = {3: 5, 5: 8}


# --------------------------------------------------------------------- basis
def _beta3(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.abs(t)
    v = np.where(a < 1.0, 2.0 / 3.0 - a * a + 0.5 * a**3, np.where(a < 2.0, (2.0 - a) ** 3 / 6.0, 0.0))
    d = np.where(a < 1.0, -2.0 * a + 1.5 * a * a, np.where(a < 2.0, -0.5 * (2.0 - a) ** 2, 0.0))
    return v, np.sign(t) * d


def _beta5(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.abs(t)
    b, c = 3.0 - a, 2.0 - a
    v = np.where(
        a < 1.0,
        11.0 / 20.0 - a * a / 2.0 + a**4 / 4.0 - a**5 / 12.0,
        np.where(a < 2.0, (b**5 - 6.0 * c**5) / 120.0, np.where(a < 3.0, b**5 / 120.0, 0.0)),
    )
    d = np.where(
        a < 1.0,
        -a + a**3 - 5.0 * a**4 / 12.0,
        np.where(a < 2.0, (-5.0 * b**4 + 30.0 * c**4) / 120.0, np.where(a < 3.0, -(b**4) / 24.0, 0.0)),
    )
    return v, np.sign(t) * d


def support_offsets(order: int) -> np.ndarray:
    """Integer offsets k of the basis support relative to floor(x)."""
    return np.arange(-(order - 1) // 2, (order + 1) // 2 + 1)


def bspline_weights(frac: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Weights and derivative weights for nodes floor(x)+k, given frac = x - floor(x).

    Returns arrays of shape frac.shape + (order + 1,).
    """
    if order not in SUPPORTED_ORDERS:
        raise ValueError(f"unsupported B-spline order {order}; expected one of {SUPPORTED_ORDERS}")
    t = np.asarray(frac, dtype=np.float64)[..., None] - support_offsets(order)
    return _beta3(t) if order == 3 else _beta5(t)


# --------------------------------------------------------------------- image
class BSplineImage:
    """Continuous B-spline model of a grayscale image (FP64)."""

    def __init__(self, image: np.ndarray, order: int = 3) -> None:
        if order not in SUPPORTED_ORDERS:
            raise ValueError(f"unsupported B-spline order {order}; expected one of {SUPPORTED_ORDERS}")
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 2:
            raise ValueError("expected a 2D image")
        self.order = order
        self.height, self.width = img.shape
        coeffs = spline_filter(img, order=order, mode="mirror", output=np.float64)
        # pad covers the basis support for points in [-1, W] x [-1, H]
        self.pad = order // 2 + 2
        if min(img.shape) <= self.pad:
            raise ValueError(f"image too small for order-{order} interpolation")
        self.coeffs = np.ascontiguousarray(np.pad(coeffs, self.pad, mode="reflect"))
        self._offsets = support_offsets(order)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def border(self) -> int:
        return BORDER[self.order]

    def in_domain(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """True where (x, y) lies in the reliable domain [b, W-1-b] x [b, H-1-b], b = border."""
        x = np.asarray(x)
        y = np.asarray(y)
        b = self.border
        return (x >= b) & (x <= self.width - 1 - b) & (y >= b) & (y <= self.height - 1 - b)

    def _patch(self, x: np.ndarray, y: np.ndarray, grad: bool):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        if (
            x.size
            and (np.min(x) < -1 or np.max(x) > self.width or np.min(y) < -1 or np.max(y) > self.height)
            or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
        ):
            raise ValueError("evaluation point outside [-1, W] x [-1, H] or not finite")
        shape = x.shape
        x = x.ravel()
        y = y.ravel()
        bx = np.floor(x).astype(np.int64)
        by = np.floor(y).astype(np.int64)
        wx, dwx = bspline_weights(x - bx, self.order)
        wy, dwy = bspline_weights(y - by, self.order)
        ix = bx[:, None] + self._offsets + self.pad
        iy = by[:, None] + self._offsets + self.pad
        patch = self.coeffs[iy[:, :, None], ix[:, None, :]]  # (P, rows=y, cols=x)
        return shape, patch, wx, wy, (dwx, dwy) if grad else None

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        shape, patch, wx, wy, _ = self._patch(x, y, grad=False)
        return np.einsum("pj,pji,pi->p", wy, patch, wx).reshape(shape)

    def evaluate_with_gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Value, d/dx and d/dy of the interpolant."""
        shape, patch, wx, wy, (dwx, dwy) = self._patch(x, y, grad=True)
        v = np.einsum("pj,pji,pi->p", wy, patch, wx).reshape(shape)
        gx = np.einsum("pj,pji,pi->p", wy, patch, dwx).reshape(shape)
        gy = np.einsum("pj,pji,pi->p", dwy, patch, wx).reshape(shape)
        return v, gx, gy
