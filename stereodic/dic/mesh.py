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


def grid_neighbors(xy: np.ndarray, step: float | None = None) -> list[np.ndarray]:
    """4-neighbourhood of POIs on a regular grid (indices into xy).

    step: grid spacing; default is the median nearest-neighbour distance. Points closer
    than 1.01 x step are neighbours, so diagonal points are excluded and gaps in the grid
    (masked regions) break the connectivity, which keeps RG-DIC from jumping across them.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    tree = cKDTree(xy)
    if step is None:
        if len(xy) < 2:
            return [np.zeros(0, dtype=np.int64) for _ in range(len(xy))]
        d, _ = tree.query(xy, k=2)
        step = float(np.median(d[:, 1]))
    pairs = tree.query_pairs(1.01 * step, output_type="ndarray")
    nb: list[list[int]] = [[] for _ in range(len(xy))]
    for i, j in pairs:
        nb[i].append(int(j))
        nb[j].append(int(i))
    return [np.asarray(sorted(n), dtype=np.int64) for n in nb]


def grid_triangles(xy: np.ndarray, valid: np.ndarray | None = None, step: float | None = None) -> np.ndarray:
    """Triangles (M, 3) of POI indices: Delaunay on the reference positions, keeping only
    triangles whose edges are at most sqrt(2) x step long (no bridging over gaps)."""
    from scipy.spatial import Delaunay, cKDTree

    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    idx = np.arange(len(xy)) if valid is None else np.nonzero(np.asarray(valid, dtype=bool))[0]
    if idx.size < 3:
        return np.zeros((0, 3), dtype=np.int64)
    pts = xy[idx]
    if step is None:
        d, _ = cKDTree(pts).query(pts, k=2)
        step = float(np.median(d[:, 1]))
    tri = Delaunay(pts).simplices
    e = np.stack([np.linalg.norm(pts[tri[:, a]] - pts[tri[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=1)
    keep = e.max(axis=1) <= np.sqrt(2) * step * 1.01
    return idx[tri[keep]].astype(np.int64)
