"""Polygon masks (U3): rasterization order, storage, use by the POI grid."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from stereodic.dic.masks import load_shapes, rasterize, save_mask
from stereodic.dic.mesh import make_poi_grid

SQUARE = [[10, 10], [89, 10], [89, 69], [10, 69]]
HOLE = [[40, 30], [59, 30], [59, 49], [40, 49]]


def test_rasterize_add_subtract_order():
    m = rasterize((80, 100), [{"op": "add", "points": SQUARE}, {"op": "subtract", "points": HOLE}])
    assert m[20, 20] and not m[40, 50] and not m[5, 5]
    assert m.sum() == 80 * 60 - 20 * 20
    only_hole = rasterize((80, 100), [{"op": "subtract", "points": HOLE}])  # starts from the full image
    assert only_hole.sum() == 80 * 100 - 400 and not only_hole[40, 50]
    with pytest.raises(ValueError):
        rasterize((80, 100), [{"op": "add", "points": [[0, 0], [1, 1]]}])
    with pytest.raises(ValueError):
        rasterize((80, 100), [{"op": "xor", "points": SQUARE}])


def test_save_load_and_poi_grid(tmp_path: Path):
    shapes = [{"op": "add", "points": SQUARE}]
    p = tmp_path / "masks" / "poi.png"
    mask = save_mask(p, (80, 100), shapes)
    assert cv2.imread(str(p), cv2.IMREAD_UNCHANGED).max() == 255
    assert load_shapes(p) == shapes and load_shapes(tmp_path / "none.png") is None
    pois = make_poi_grid((80, 100), 11, 5, mask=mask, margin=0)
    x, y = pois.xy[:, 0], pois.xy[:, 1]
    assert len(pois) > 0 and x.min() >= 15 and x.max() <= 84 and y.min() >= 15 and y.max() <= 64
    assert np.all(mask[y.astype(int), x.astype(int)])
