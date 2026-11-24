"""Image sequences: probing, frame previews, per-frame checks, sync and sharpness (U1).

These endpoints read the exported image files only (never the HDF5 results), so they
are available while a job is running.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from stereodic.io.probe import describe, probe_sequence, sample_indices
from stereodic.io.sequence import ImageSequence, open_sequence
from stereodic.io.sync import sync_summary
from stereodic.pipeline import camera_timestamps
from stereodic.quality.image import frame_report, sharpness

from ..workspace import WorkspaceError

router = APIRouter(prefix="/api", tags=["frames"])
Camera = Literal["left", "right"]

_lock = threading.Lock()
_seq_cache: OrderedDict[tuple, ImageSequence] = OrderedDict()
_desc_cache: dict[tuple, dict] = {}
_frame_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
_sharp_cache: dict[tuple, dict] = {}
_SEQ_MAX, _FRAME_MAX = 16, 32


def _inside_roots(request: Request, path: Path) -> Path:
    p = Path(path).expanduser().resolve()
    for root in request.app.state.settings.roots().values():
        r = Path(root).expanduser().resolve()
        if p == r or r in p.parents:
            return p
    raise HTTPException(403, "path is outside the configured data/results roots")


def _dir_key(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _sequence(path: Path, pattern: str, offset: int) -> ImageSequence:
    key = (str(path), pattern, offset, _dir_key(path))
    with _lock:
        seq = _seq_cache.get(key)
        if seq is not None:
            _seq_cache.move_to_end(key)
            return seq
    seq = open_sequence(path, pattern, offset)
    with _lock:
        _seq_cache[key] = seq
        while len(_seq_cache) > _SEQ_MAX:
            _seq_cache.popitem(last=False)
    return seq


def _camera(request: Request, name: str, cam: str) -> tuple[Any, Any, ImageSequence]:
    cfg = request.app.state.workspace.load(name)
    src = getattr(cfg.cameras, cam)
    path = _inside_roots(request, cfg.data_path(src.path))
    try:
        seq = _sequence(path, src.pattern, src.frame_offset)
    except (FileNotFoundError, ValueError) as exc:
        raise WorkspaceError(f"{cam} camera: {exc}", 404) from exc
    return cfg, src, seq


def _describe(seq: ImageSequence, key: tuple) -> dict:
    with _lock:
        d = _desc_cache.get(key)
    if d is None:
        d = describe(seq, sample=5)
        with _lock:
            _desc_cache[key] = d
    return d


def _frame(seq: ImageSequence, key: tuple, k: int) -> np.ndarray:
    fk = (*key, k)
    with _lock:
        img = _frame_cache.get(fk)
        if img is not None:
            _frame_cache.move_to_end(fk)
            return img
    try:
        img = seq.read(k)
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc
    with _lock:
        _frame_cache[fk] = img
        while len(_frame_cache) > _FRAME_MAX:
            _frame_cache.popitem(last=False)
    return img


def _seq_key(cfg, src) -> tuple:
    p = cfg.data_path(src.path)
    return (str(p), src.pattern, src.frame_offset, _dir_key(p))


# ------------------------------------------------------------------ probing
class ProbeRequest(BaseModel):
    path: str
    pattern: str = "*.tif"


@router.post("/sequences/probe")
def probe(request: Request, body: ProbeRequest) -> dict[str, Any]:
    path = _inside_roots(request, Path(body.path))
    try:
        return probe_sequence(path, body.pattern)
    except (FileNotFoundError, ValueError) as exc:
        raise WorkspaceError(str(exc), 422) from exc


@router.get("/experiments/{name}/sequences")
def sequences(request: Request, name: str) -> dict[str, Any]:
    """Both measurement sequences as configured (offset applied) and their pairing."""
    cfg = request.app.state.workspace.load(name)
    out: dict[str, Any] = {"cameras": {}, "frame_rate_hz": cfg.cameras.frame_rate_hz}
    lengths = []
    for cam in ("left", "right"):
        src = getattr(cfg.cameras, cam)
        entry: dict[str, Any] = {"path": str(cfg.data_path(src.path)), "pattern": src.pattern,
                                 "frame_offset": src.frame_offset, "timestamp_path": src.timestamp_path and str(cfg.data_path(src.timestamp_path))}
        try:
            _, _, seq = _camera(request, name, cam)
            entry.update(_describe(seq, _seq_key(cfg, src)))
            entry["raw_frames"] = len(getattr(seq, "base", seq))
            lengths.append(len(seq))
        except (HTTPException, WorkspaceError) as exc:
            entry["error"] = getattr(exc, "detail", None) or str(exc)
        out["cameras"][cam] = entry
    n = min(lengths) if len(lengths) == 2 else 0
    out.update(
        n_frames=n,
        length_mismatch=len(lengths) == 2 and lengths[0] != lengths[1],
        reference_frame=cfg.reference_frame,
        frames=cfg.frames.model_dump(),
    )
    return out


# ------------------------------------------------------------------ frames
def _window(img: np.ndarray, lo: float | None, hi: float | None) -> tuple[float, float]:
    if lo is None or hi is None:
        sub = img[:: max(1, img.shape[0] // 256), :: max(1, img.shape[1] // 256)]
        plo, phi = np.percentile(sub, [0.5, 99.5])
        lo = float(plo) if lo is None else lo
        hi = float(phi) if hi is None else hi
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


@router.get("/experiments/{name}/frames/{cam}/{k}.png")
def frame_png(
    request: Request,
    name: str,
    cam: Camera,
    k: int,
    lo: float | None = None,
    hi: float | None = None,
    overlay: Literal["none", "saturation"] = "none",
    scale: float = 1.0,
) -> Response:
    cfg, src, seq = _camera(request, name, cam)
    img = _frame(seq, _seq_key(cfg, src), k)
    lo, hi = _window(img, lo, hi)
    g8 = np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    if overlay == "saturation":
        full = _describe(seq, _seq_key(cfg, src))["full_scale"]
        rgb = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
        rgb[img >= full] = (40, 40, 255)  # BGR red
        g8 = rgb
    if 0 < scale < 1:
        g8 = cv2.resize(g8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", g8, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise HTTPException(500, "PNG encoding failed")
    headers = {
        "Cache-Control": "private, max-age=60",
        "X-Frame-Name": seq.frame_name(k),
        "X-Window": f"{lo:g},{hi:g}",
        "Access-Control-Expose-Headers": "X-Frame-Name, X-Window",
    }
    return Response(buf.tobytes(), media_type="image/png", headers=headers)


@router.get("/experiments/{name}/frames/{cam}/{k}/report")
def frame_checks(request: Request, name: str, cam: Camera, k: int) -> dict[str, Any]:
    cfg, src, seq = _camera(request, name, cam)
    key = _seq_key(cfg, src)
    info = _describe(seq, key)
    img = _frame(seq, key, k)
    rep = frame_report(img, info["full_scale"])
    rep.update(frame=k, name=seq.frame_name(k), raw_index=getattr(seq, "raw_index", lambda i: i)(k))
    return rep


# ------------------------------------------------------------------ sync and sharpness
@router.get("/experiments/{name}/sync")
def sync(request: Request, name: str) -> dict[str, Any]:
    cfg = request.app.state.workspace.load(name)
    lens = {}
    for cam in ("left", "right"):
        _, _, seq = _camera(request, name, cam)
        lens[cam] = len(seq)
    try:
        ts_l = camera_timestamps(cfg, "left", lens["left"])
        ts_r = camera_timestamps(cfg, "right", lens["right"])
    except (ValueError, OSError) as exc:
        raise WorkspaceError(f"timestamps: {exc}", 422) from exc
    max_delta = None if cfg.cameras.max_sync_delta_us is None else cfg.cameras.max_sync_delta_us * 1e-6
    out = sync_summary(ts_l, ts_r, max_delta, cfg.cameras.frame_rate_hz)
    out["source"] = {
        cam: ("file" if getattr(cfg.cameras, cam).timestamp_path else "frame_rate") for cam in ("left", "right")
    }
    out["max_sync_delta_us"] = cfg.cameras.max_sync_delta_us
    return out


@router.get("/experiments/{name}/sharpness")
def sharpness_curve(request: Request, name: str, points: int = 120) -> dict[str, Any]:
    """Normalised gradient energy over the sequence (both cameras), for choosing the reference frame."""
    cfg = request.app.state.workspace.load(name)
    seqs = {cam: _camera(request, name, cam)[2] for cam in ("left", "right")}
    n = min(len(s) for s in seqs.values())
    points = int(np.clip(points, 2, 1000))
    key = (name, n, points, tuple(_seq_key(cfg, getattr(cfg.cameras, c)) for c in ("left", "right")))
    with _lock:
        cached = _sharp_cache.get(key)
    if cached is not None:
        return cached
    idx = sample_indices(n, points)
    out: dict[str, Any] = {"frames": idx, "n_frames": n}
    for cam, seq in seqs.items():
        out[cam] = [sharpness(seq.read(i), crop=384) for i in idx]
    with _lock:
        _sharp_cache[key] = out
    return out
