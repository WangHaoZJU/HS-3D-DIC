"""Region masks drawn as polygons (UI, U3): rasterization and storage.

A mask is described by an ordered list of shapes, each ``{"op": "add" | "subtract",
"points": [[x, y], ...]}`` in image pixel coordinates (x right, y down, integer = pixel
centre). Shapes are applied in order; if the first shape subtracts, the mask starts
from the whole image. The rasterized mask is stored as an 8-bit PNG (nonzero = inside,
the format `poi_mask` / `rbm_mask` expect) with the shapes in a JSON sidecar so they can
be edited again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_SHIFT = 4  # fixed-point bits for sub-pixel polygon vertices


def validate_shapes(shapes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in shapes or []:
        op = s.get("op", "add")
        if op not in ("add", "subtract"):
            raise ValueError(f"shape op must be 'add' or 'subtract', got {op!r}")
        pts = np.asarray(s.get("points", []), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.all(np.isfinite(pts)):
            raise ValueError("each shape needs at least 3 finite [x, y] points")
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        if area < 1.0:
            raise ValueError("degenerate shape (area < 1 px^2)")
        out.append({"op": op, "points": np.round(pts, 3).tolist()})
    return out


def rasterize(shape_hw: tuple[int, int], shapes: list[dict[str, Any]]) -> np.ndarray:
    """Boolean (H, W) mask from the ordered shapes."""
    shapes = validate_shapes(shapes)
    h, w = shape_hw
    start = 1 if shapes and shapes[0]["op"] == "subtract" else 0
    m = np.full((h, w), start, dtype=np.uint8)
    for s in shapes:
        pts = np.round(np.asarray(s["points"]) * (1 << _SHIFT)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 1 if s["op"] == "add" else 0, lineType=cv2.LINE_8, shift=_SHIFT)
    return m.astype(bool)


def sidecar(path: str | Path) -> Path:
    p = Path(path)
    return p.with_suffix(".json")


def save_mask(path: str | Path, shape_hw: tuple[int, int], shapes: list[dict[str, Any]]) -> np.ndarray:
    """Write the PNG mask and its shape sidecar; returns the boolean mask."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shapes = validate_shapes(shapes)
    mask = rasterize(shape_hw, shapes)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise OSError(f"cannot write {path}")
    sidecar(path).write_text(
        json.dumps({"image_shape": list(shape_hw), "shapes": shapes}, indent=1), encoding="utf-8"
    )
    return mask


def load_shapes(path: str | Path) -> list[dict[str, Any]] | None:
    """Shapes stored next to a mask PNG (None for masks not drawn in the UI)."""
    p = sidecar(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("shapes", [])
