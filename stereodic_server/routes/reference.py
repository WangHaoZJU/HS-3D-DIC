"""Reference page (U3): ROI and masks, POI grid preview, stereo results and 3D shape."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from stereodic.core.status import Status, status_counts
from stereodic.core.types import ROI
from stereodic.dic.masks import load_shapes, rasterize, save_mask, validate_shapes
from stereodic.dic.mesh import make_poi_grid
from stereodic.io.storage import ResultStore
from stereodic.pipeline import open_cameras

from ..workspace import WorkspaceError

router = APIRouter(prefix="/api/experiments/{name}", tags=["reference"])

MASK_FIELDS = {"poi": "poi_mask", "rbm": "rbm_mask"}


class Shape(BaseModel):
    op: str = "add"
    points: list[list[float]]


class MasksPayload(BaseModel):
    roi: list[int] | None = None
    poi: list[Shape] = Field(default_factory=list)
    rbm: list[Shape] = Field(default_factory=list)
    subset_size: int | None = None
    step: int | None = None


class PreviewPayload(BaseModel):
    roi: list[int] | None = None
    poi: list[Shape] | None = None  # None: use the saved poi_mask
    subset_size: int
    step: int


def _image_shape(cfg) -> tuple[int, int]:
    try:
        seq_l, _ = open_cameras(cfg)
    except (FileNotFoundError, ValueError) as exc:
        raise WorkspaceError(f"cannot open the left sequence: {exc}", 422) from exc
    return tuple(int(v) for v in seq_l.frame_shape)  # type: ignore[return-value]


def _mask_info(cfg, field: str) -> dict[str, Any]:
    p = getattr(cfg, field)
    if p is None:
        return {"path": None, "shapes": [], "drawn": False}
    path = cfg.data_path(p)
    shapes = load_shapes(path)
    return {"path": str(path), "shapes": shapes or [], "drawn": shapes is not None, "exists": path.exists()}


@router.get("/masks")
def get_masks(request: Request, name: str) -> dict[str, Any]:
    cfg = request.app.state.workspace.load(name)
    return {
        "roi": cfg.roi,
        "image_shape": list(_image_shape(cfg)),
        "reference_frame": cfg.reference_frame,
        "subset_size": cfg.dic.poi.subset_size,
        "step": cfg.dic.poi.step,
        "poi": _mask_info(cfg, "poi_mask"),
        "rbm": _mask_info(cfg, "rbm_mask"),
    }


@router.put("/masks")
def put_masks(request: Request, name: str, body: MasksPayload) -> dict[str, Any]:
    """Rasterize and save the drawn masks into the workspace, update ROI / POI grid settings."""
    ws = request.app.state.workspace
    cfg = ws.load(name)
    shape = _image_shape(cfg)
    data = ws.effective(cfg)
    try:
        validated = {kind: validate_shapes([s.model_dump() for s in getattr(body, kind)]) for kind in MASK_FIELDS}
        if body.roi is not None:
            ROI.from_list(body.roi)
    except ValueError as exc:
        raise WorkspaceError(str(exc), 422) from exc
    for kind, field in MASK_FIELDS.items():
        shapes = validated[kind]
        target = ws.dir(name) / "masks" / f"{kind}_mask.png"
        if shapes:
            mask = save_mask(target, shape, shapes)
            if not mask.any():
                raise WorkspaceError(f"the {kind} mask is empty", 422)
            data[field] = str(target)
        else:
            data[field] = None
    data["roi"] = body.roi
    if body.subset_size is not None:
        data["dic"]["poi"]["subset_size"] = body.subset_size
    if body.step is not None:
        data["dic"]["poi"]["step"] = body.step
    ws.save(name, data)
    return get_masks(request, name)


@router.post("/poi-preview")
def poi_preview(request: Request, name: str, body: PreviewPayload) -> dict[str, Any]:
    """POI grid for the given (possibly unsaved) ROI, mask shapes and grid settings."""
    cfg = request.app.state.workspace.load(name)
    shape = _image_shape(cfg)
    if body.subset_size % 2 == 0 or body.subset_size < 5 or body.step < 1:
        raise WorkspaceError("subset_size must be odd and >= 5, step >= 1", 422)
    mask = None
    try:
        if body.poi is not None:
            shapes = [s.model_dump() for s in body.poi]
            mask = rasterize(shape, shapes) if shapes else None
        elif cfg.poi_mask is not None:
            from stereodic.pipeline import _poi_mask

            mask = _poi_mask(cfg, shape)
        roi = ROI.from_list(body.roi)
        pois = make_poi_grid(shape, body.subset_size, body.step, roi=roi, mask=mask)
    except ValueError as exc:
        return {"n": 0, "points": [], "error": str(exc)}
    return {"n": len(pois), "points": pois.xy.astype(np.int32).ravel().tolist()}


def _round(a: np.ndarray, digits: int) -> list:
    a = np.asarray(a, dtype=np.float64)
    out = np.round(a, digits)
    return np.where(np.isfinite(out), out, None).ravel().tolist()


@router.get("/reference")
def reference_result(request: Request, name: str) -> dict[str, Any]:
    """Stereo result of the reference pair (POIs, status, ZNSSD, 3D shape, mesh)."""
    cfg = request.app.state.workspace.load(name)
    jobs = request.app.state.jobs
    r = jobs.running
    if r is not None and r.job["experiment"] == name and r.job["stage"] in ("stereo", "pipeline"):
        raise HTTPException(409, "stereo matching is running")
    if cfg.output_file is None:
        raise HTTPException(404, "no output_file configured")
    path = Path(cfg.results_path(cfg.output_file))
    if not path.exists():
        raise HTTPException(404, "no stereo result yet")
    with ResultStore.open(path, "r") as store:
        h5 = store.h5
        if "reference/points_3d" not in h5:
            raise HTTPException(404, "no stereo result yet")
        g = h5["reference"]
        st = g["stereo_status"][()]
        X = g["points_3d"][()]
        frames = g["local_frames"][()]
        meta = store.metadata()
        ok = st == Status.OK
        axes = {}
        if ok.any():
            for k, label in enumerate(("c", "s", "n")):
                v = np.nanmean(frames[ok, k, :], axis=0)
                axes[label] = (v / np.linalg.norm(v)).tolist()
        depth = X[ok, 2]
        return {
            "n": int(st.size),
            "frame_index": int(g["frame_index"][()]),
            "input": meta.get("input_manifest", {}),
            "points_left": _round(g["points_left_image"][()], 2),
            "points_right": _round(g["points_right_image"][()], 2),
            "status": st.astype(int).tolist(),
            "status_counts": status_counts(st),
            "znssd": _round(g["stereo_znssd"][()], 5),
            "reprojection_error": _round(g["reprojection_error"][()], 4),
            "points_3d": _round(X, 4),
            "connectivity": g["connectivity"][()].astype(np.int64).ravel().tolist(),
            "local_frames": _round(frames, 5),
            "mean_axes": axes,
            "depth_range": [float(depth.min()), float(depth.max())] if depth.size else None,
            "span_direction": cfg.recon.span_direction,
        }
