"""POI grid generation on the left reference image."""

from __future__ import annotations

import numpy as np

from ..core.types import ROI, POISet


def make_poi_grid(
    image_shape: tuple[int, int],
    subset_size: int,
    step: int,
    roi: ROI | None = None,
    mask: np.ndarray | None = None,
    margin: int | None = None,
) -> POISet:
    """Regular POI grid whose subsets lie fully inside the image (and ROI).

    image_shape: (height, width) of the full frame; roi in full-frame coordinates.
    mask: optional boolean (H, W) array; a POI is kept only if its whole subset is inside the mask.
    margin: pixels kept free between subset and image edge; default is the largest
        interpolation safety band (interp.BORDER), so subsets are valid for any spline order.
    IDs are assigned row-major and are stable for a given set of arguments.
    """
    if subset_size % 2 == 0:
        raise ValueError("subset_size must be odd")
    h_img, w_img = image_shape
    x0, y0, x1, y1 = 0, 0, w_img, h_img
    if roi is not None:
        x0, y0, x1, y1 = roi.x0, roi.y0, min(w_img, roi.x0 + roi.width), min(h_img, roi.y0 + roi.height)
    if margin is None:
        from .interp import BORDER

        margin = max(BORDER.values())
    half = subset_size // 2
    lo_x, hi_x = x0 + half + margin, x1 - 1 - half - margin
    lo_y, hi_y = y0 + half + margin, y1 - 1 - half - margin
    if lo_x > hi_x or lo_y > hi_y:
        raise ValueError("region too small for the requested subset size")
    xs = np.arange(lo_x, hi_x + 1, step)
    ys = np.arange(lo_y, hi_y + 1, step)
    gx, gy = np.meshgrid(xs, ys)
    xy = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != tuple(image_shape):
            raise ValueError("mask shape must equal image_shape")
        # subset fully inside mask <=> min over the subset window is True
        from scipy.ndimage import minimum_filter

        inside = minimum_filter(mask.astype(np.uint8), size=subset_size, mode="constant", cval=0).astype(bool)
        keep = inside[xy[:, 1].astype(int), xy[:, 0].astype(int)]
        xy = xy[keep]
    return POISet(point_ids=np.arange(len(xy), dtype=np.int64), xy=xy)
