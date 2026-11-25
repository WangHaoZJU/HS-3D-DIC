"""Calibration page (U2): detections, report, per-view overlays, board image and 3D poses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response

from stereodic.calibration.detect import CharucoBoardSpec
from stereodic.calibration.report import detections_path, load_calibration, load_calibration_details, load_detections
from stereodic.io.sequence import open_sequence
from stereodic.pipeline import _calibration_status, _detection_hash

from ..workspace import WorkspaceError
from .frames import _window

router = APIRouter(prefix="/api/experiments/{name}/calibration", tags=["calibration"])
Camera = Literal["left", "right"]


def _cfg(request: Request, name: str):
    cfg = request.app.state.workspace.load(name)
    if cfg.calibration is None:
        raise WorkspaceError("this experiment has no `calibration` section", 404)
    if cfg.calibration_file is None:
        raise WorkspaceError("this experiment has no `calibration_file`", 404)
    return cfg


def _busy(request: Request, name: str) -> bool:
    r = request.app.state.jobs.running
    return r is not None and r.job["experiment"] == name and r.job["stage"] in ("calibrate", "calibrate_detect", "pipeline")


def _sequences(cfg):
    cc = cfg.calibration
    return (
        open_sequence(cfg.data_path(cc.left.path), cc.left.pattern, cc.left.frame_offset),
        open_sequence(cfg.data_path(cc.right.path), cc.right.pattern, cc.right.frame_offset),
    )


@router.get("")
def calibration_overview(request: Request, name: str) -> dict[str, Any]:
    cfg = _cfg(request, name)
    cc = cfg.calibration
    out: dict[str, Any] = {
        "busy": _busy(request, name),
        "board": cc.board.model_dump(),
        "board_size_mm": list(CharucoBoardSpec(**cc.board.model_dump()).size_mm),
        "n_corners_board": (cc.board.squares_x - 1) * (cc.board.squares_y - 1),
        "exclude_views": list(cc.exclude_views),
        "status": _calibration_status(cfg),
        "detections": None,
        "report": None,
        "pairs": [],
        "images": None,
    }
    try:
        seq_l, seq_r = _sequences(cfg)
        out["images"] = {"left": len(seq_l), "right": len(seq_r), "shape": list(seq_l.frame_shape)}
    except (FileNotFoundError, ValueError) as exc:
        out["images_error"] = str(exc)
        seq_l = seq_r = None
    if out["busy"]:
        return out
    cal_file = cfg.data_path(cfg.calibration_file)
    det_file = detections_path(cal_file)
    if det_file.exists():
        dl, dr, names, meta = load_detections(det_file)
        up_to_date = seq_l is not None and meta.get("hash") == _detection_hash(cfg, seq_l, seq_r)
        out["detections"] = {
            "up_to_date": up_to_date,
            "origins": meta.get("origins"),
            "image_shapes": meta.get("image_shapes"),
            "views": [
                {
                    "index": i,
                    "name_left": names[i][0],
                    "name_right": names[i][1],
                    "n_left": 0 if dl[i] is None else len(dl[i]),
                    "n_right": 0 if dr[i] is None else len(dr[i]),
                    "n_common": 0 if dl[i] is None or dr[i] is None else int(np.intersect1d(dl[i].ids, dr[i].ids).size),
                    "excluded": i in cc.exclude_views,
                }
                for i in range(len(names))
            ],
        }
    sidecar = cal_file.with_suffix(".json")
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        out["report"] = data.get("report")
    if cal_file.exists():
        d = load_calibration_details(cal_file)
        if d is not None:
            out["pairs"] = [
                {
                    "view": int(v),
                    "rms_left": float(d["view_rms_left"][k]),
                    "rms_right": float(d["view_rms_right"][k]),
                    "n": int(len(d["ids"][k])),
                    "distance_mm": float(np.linalg.norm(d["pose_tvec"][k])),
                }
                for k, v in enumerate(d["pairs"])
            ]
            out["residuals"] = {
                side: np.round(np.concatenate(d[f"residuals_{side}"]), 5).tolist() if d[f"residuals_{side}"] else []
                for side in ("left", "right")
            }
    return out


@router.get("/views/{i}")
def calibration_view(request: Request, name: str, i: int) -> dict[str, Any]:
    """Detected corners, reprojections and residuals of one view (image coordinates)."""
    cfg = _cfg(request, name)
    if _busy(request, name):
        raise HTTPException(409, "calibration is running")
    cal_file = cfg.data_path(cfg.calibration_file)
    det_file = detections_path(cal_file)
    if not det_file.exists():
        raise HTTPException(404, "no detections yet; run corner detection first")
    dl, dr, names, meta = load_detections(det_file)
    if not 0 <= i < len(names):
        raise HTTPException(404, f"view {i} out of range")
    origins = meta.get("origins", [[0, 0], [0, 0]])
    out: dict[str, Any] = {"index": i, "names": list(names[i]), "pair": None}
    for k, (side, dets) in enumerate((("left", dl), ("right", dr))):
        d = dets[i]
        o = np.asarray(origins[k])
        out[side] = None if d is None else {
            "ids": d.ids.tolist(),
            "points": np.round(d.image_points - o, 4).tolist(),
        }
    if cal_file.exists():
        det = load_calibration_details(cal_file)
        if det is not None and i in set(det["pairs"].tolist()):
            k = int(np.nonzero(det["pairs"] == i)[0][0])
            pair: dict[str, Any] = {"ids": det["ids"][k].tolist(), "rms_left": float(det["view_rms_left"][k]),
                                    "rms_right": float(det["view_rms_right"][k])}
            for j, side in enumerate(("left", "right")):
                o = np.asarray(origins[j])
                meas = det[f"points_{side}"][k]
                res = det[f"residuals_{side}"][k]
                pair[side] = {"points": np.round(meas - o, 4).tolist(), "residuals": np.round(res, 5).tolist()}
            out["pair"] = pair
    return out


@router.get("/views/{i}/{cam}.png")
def calibration_image(request: Request, name: str, i: int, cam: Camera, scale: float = 1.0) -> Response:
    cfg = _cfg(request, name)
    seq = _sequences(cfg)[0 if cam == "left" else 1]
    try:
        img = seq.read(i)
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc
    lo, hi = _window(img, None, None)
    g8 = np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    if 0 < scale < 1:
        g8 = cv2.resize(g8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", g8, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "private, max-age=300"})


@router.get("/board.png")
def board_image(
    request: Request,
    name: str,
    squares_x: int | None = None,
    squares_y: int | None = None,
    square_mm: float | None = None,
    marker_mm: float | None = None,
    dictionary: str | None = None,
    legacy_pattern: bool | None = None,
    px_per_mm: float = 12.0,
) -> Response:
    """Rendered board for the given (or configured) parameters, to check them against the print."""
    cfg = _cfg(request, name)
    b = cfg.calibration.board.model_dump()
    for k, v in dict(squares_x=squares_x, squares_y=squares_y, square_mm=square_mm, marker_mm=marker_mm,
                     dictionary=dictionary, legacy_pattern=legacy_pattern).items():
        if v is not None:
            b[k] = v
    try:
        spec = CharucoBoardSpec(**b)
    except ValueError as exc:
        raise WorkspaceError(str(exc), 422) from exc
    w_mm, h_mm = spec.size_mm
    scale = float(np.clip(px_per_mm, 2, 40))
    size = (int(round(w_mm * scale)), int(round(h_mm * scale)))
    if max(size) > 4000:
        scale *= 4000 / max(size)
        size = (int(round(w_mm * scale)), int(round(h_mm * scale)))
    img = spec.board().generateImage(size, marginSize=0, borderBits=1)
    ok, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.get("/poses")
def calibration_poses(request: Request, name: str) -> dict[str, Any]:
    """Cameras and board poses in the left-camera frame (mm), for the 3D view."""
    cfg = _cfg(request, name)
    if _busy(request, name):
        raise HTTPException(409, "calibration is running")
    cal_file = Path(cfg.data_path(cfg.calibration_file))
    if not cal_file.exists():
        raise HTTPException(404, "not calibrated yet")
    calib = load_calibration(cal_file)
    det = load_calibration_details(cal_file)
    spec = CharucoBoardSpec(**cfg.calibration.board.model_dump())
    w, h = spec.size_mm
    outline = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64)
    cams = []
    for label, cam, R, t in (
        ("left", calib.left, np.eye(3), np.zeros(3)),
        ("right", calib.right, calib.R_RL, calib.t_RL),
    ):
        center = -R.T @ t
        cams.append({"name": label, "center": center.tolist(), "R": R.tolist(), "K": cam.K.tolist(),
                     "image_size": list(cam.image_size)})
    boards = []
    if det is not None:
        for k, v in enumerate(det["pairs"]):
            Rb, _ = cv2.Rodrigues(det["pose_rvec"][k])
            corners = outline @ Rb.T + det["pose_tvec"][k]
            boards.append({"view": int(v), "corners": np.round(corners, 4).tolist(),
                           "rms": float(max(det["view_rms_left"][k], det["view_rms_right"][k]))})
    return {"cameras": cams, "boards": boards, "board_size_mm": [w, h], "baseline_mm": float(calib.baseline_mm)}
