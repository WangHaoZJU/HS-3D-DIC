"""Temporal DIC, reconstruction and rigid-body results (U4): summaries, maps and slices.

Results files are only read when no job of this experiment is writing them (409
otherwise); while a job runs, the live per-frame summaries come over the WebSocket.
Large (T, N, C) arrays are never sent whole: one frame (N, 3), one point (T, 3), or
downsampled maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from fastapi import APIRouter, HTTPException, Request

from stereodic.core.status import Status, status_counts

router = APIRouter(prefix="/api/experiments/{name}", tags=["results"])

WRITERS = ("stereo", "temporal", "reconstruct", "rbm", "pipeline")
CHUNK_T = 64
KINDS = {
    "local": "reconstruction/displacement_local",
    "raw": "reconstruction/displacement_raw",
    "rbm_local": "reconstruction/displacement_rbm_local",
    "rbm": "reconstruction/displacement_rbm",
}


def _results(request: Request, name: str) -> Path:
    cfg = request.app.state.workspace.load(name)
    r = request.app.state.jobs.running
    if r is not None and r.job["experiment"] == name and r.job["stage"] in WRITERS:
        raise HTTPException(409, "a job is writing the results file; use the live summaries")
    if cfg.output_file is None:
        raise HTTPException(404, "no output_file configured")
    path = Path(cfg.results_path(cfg.output_file))
    if not path.exists():
        raise HTTPException(404, "no results yet")
    return path


def _open(path: Path) -> h5py.File:
    # chunk cache sized for per-point reads along time (review: 46 ms -> 1 ms per row)
    return h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=100_003)


def _clean(a: np.ndarray, digits: int = 6) -> list:
    a = np.round(np.asarray(a, dtype=np.float64), digits)
    return np.where(np.isfinite(a), a, None).tolist()


def _require(f: h5py.File, name: str) -> h5py.Dataset:
    if name not in f:
        raise HTTPException(404, f"{name} not in the results file (run the stage first)")
    return f[name]


def _times(f: h5py.File, n: int) -> np.ndarray:
    return f["time/timestamps"][()] if "time/timestamps" in f else np.arange(n, dtype=np.float64)


def _minmax_decimate(t: np.ndarray, y: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep min and max of each bucket so peaks survive decimation; y is (T, C)."""
    T = t.size
    if T <= max_samples:
        return t, y
    buckets = max(1, max_samples // 2)
    edges = np.linspace(0, T, buckets + 1).astype(int)
    idx: list[int] = []
    ref = np.nan_to_num(np.abs(y).max(axis=1) if y.ndim > 1 else np.abs(y), nan=-1.0)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = ref[a:b]
        i0, i1 = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        idx.extend(sorted({i0, i1}))
    idx_arr = np.asarray(idx)
    return t[idx_arr], y[idx_arr]


# ------------------------------------------------------------------ temporal DIC
@router.get("/dic/summary")
def dic_summary(request: Request, name: str) -> dict[str, Any]:
    with _open(_results(request, name)) as f:
        st_l = _require(f, "dic/left/status")
        T, N = st_l.shape
        done = f["progress/dic_frames_done"][()].all(axis=1) if "progress/dic_frames_done" in f else np.ones(T, bool)
        if "quality/frame_summary" in f:
            s = f["quality/frame_summary"][()]
            vl, vr, zm, it = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        else:  # results written before the summary existed: compute from the status arrays
            vl, vr, zm, it = (np.full(T, np.nan) for _ in range(4))
            for a in range(0, T, CHUNK_T):
                b = min(T, a + CHUNK_T)
                sl, sr = f["dic/left/status"][a:b], f["dic/right/status"][a:b]
                zl, zr = f["dic/left/znssd"][a:b], f["dic/right/znssd"][a:b]
                vl[a:b] = (sl == Status.OK).mean(axis=1)
                vr[a:b] = (sr == Status.OK).mean(axis=1)
                z = np.where(np.concatenate([sl, sr], 1) == Status.OK, np.concatenate([zl, zr], 1), np.nan)
                with np.errstate(all="ignore"):
                    zm[a:b] = np.nanmedian(z, axis=1)
        counts = {}
        for side in ("left", "right"):
            c: dict[str, int] = {}
            ds = f[f"dic/{side}/status"]
            for a in range(0, T, CHUNK_T):
                for k, v in status_counts(ds[a : a + CHUNK_T]).items():
                    c[k] = c.get(k, 0) + v
            counts[side] = c
        return {
            "n_frames": int(T),
            "n_points": int(N),
            "frames_done": int(done.sum()),
            "frame_ids": f["time/frame_ids"][()].tolist() if "time/frame_ids" in f else None,
            "timestamps": _clean(_times(f, T), 9),
            "valid_left": _clean(vl, 5),
            "valid_right": _clean(vr, 5),
            "znssd_median": _clean(zm, 6),
            "iterations_mean": _clean(it, 3),
            "status_counts": counts,
            "reference_frame": int(f["reference/frame_index"][()]) if "reference/frame_index" in f else None,
        }


@router.get("/dic/status-map")
def status_map(
    request: Request, name: str, side: Literal["left", "right", "recon"] = "left", t_bins: int = 240, n_bins: int = 200
) -> dict[str, Any]:
    """Valid fraction per (time bin, POI bin); POIs in grid (ID) order."""
    ds_name = "reconstruction/status" if side == "recon" else f"dic/{side}/status"
    with _open(_results(request, name)) as f:
        ds = _require(f, ds_name)
        T, N = ds.shape
        tb, nb = int(np.clip(t_bins, 1, min(T, 1000))), int(np.clip(n_bins, 1, min(N, 1000)))
        t_edges = np.linspace(0, T, tb + 1).astype(int)
        n_idx = np.minimum((np.arange(N) * nb) // max(N, 1), nb - 1)
        n_count = np.bincount(n_idx, minlength=nb).astype(np.float64)
        out = np.zeros((tb, nb))
        worst = np.zeros((tb, nb), dtype=np.int16)  # most frequent failure code per cell (0 = none)
        for i, (a, b) in enumerate(zip(t_edges[:-1], t_edges[1:])):
            if b <= a:
                continue
            st = ds[a:b]
            ok = (st == Status.OK).sum(axis=0)
            out[i] = np.bincount(n_idx, weights=ok, minlength=nb) / (n_count * (b - a))
            bad = st[st != Status.OK]
            if bad.size:
                worst[i, :] = 0
                codes, cnt = np.unique(bad, return_counts=True)
                worst[i, :] = np.where(out[i] < 1, codes[np.argmax(cnt)], 0)
        return {"valid": np.round(out, 4).tolist(), "worst": worst.tolist(), "t_edges": t_edges.tolist(),
                "n_frames": int(T), "n_points": int(N)}


@router.get("/dic/point/{i}")
def dic_point(request: Request, name: str, i: int) -> dict[str, Any]:
    """Image-plane trajectory of one POI in both cameras (u, v relative to the reference)."""
    with _open(_results(request, name)) as f:
        xy_l = _require(f, "dic/left/xy")
        T, N, _ = xy_l.shape
        if not 0 <= i < N:
            raise HTTPException(404, f"POI {i} out of range")
        out: dict[str, Any] = {"i": i, "timestamps": _clean(_times(f, T), 9)}
        for side in ("left", "right"):
            xy = f[f"dic/{side}/xy"][:, i, :]
            ref = f[f"reference/points_{side}_image"][i]
            out[side] = {
                "u": _clean(xy[:, 0] - ref[0], 5),
                "v": _clean(xy[:, 1] - ref[1], 5),
                "status": f[f"dic/{side}/status"][:, i].astype(int).tolist(),
                "znssd": _clean(f[f"dic/{side}/znssd"][:, i], 5),
            }
        return out


# ------------------------------------------------------------------ displacement
@router.get("/displacement/info")
def displacement_info(request: Request, name: str, samples: int = 40) -> dict[str, Any]:
    """Available kinds, time axis and robust colour ranges per component (sampled frames)."""
    with _open(_results(request, name)) as f:
        ds = _require(f, KINDS["local"])
        T, N, _ = ds.shape
        idx = np.unique(np.linspace(0, T - 1, min(samples, T)).astype(int))
        kinds = [k for k, v in KINDS.items() if v in f]
        ranges = {}
        for k in kinds:
            vals = f[KINDS[k]][idx.tolist()]
            mag = np.linalg.norm(vals, axis=-1)
            ranges[k] = {
                "abs": [float(np.nanpercentile(np.abs(vals[..., c]), 99)) if np.isfinite(vals[..., c]).any() else 0.0
                        for c in range(3)],
                "mag": float(np.nanpercentile(mag, 99)) if np.isfinite(mag).any() else 0.0,
            }
        valid = f["quality/valid_mask"][idx.tolist()].mean() if "quality/valid_mask" in f else None
        return {"n_frames": int(T), "n_points": int(N), "kinds": kinds, "timestamps": _clean(_times(f, T), 9),
                "ranges": ranges, "valid_fraction": None if valid is None else float(valid),
                "rbm": "reconstruction/rbm_t" in f}


@router.get("/displacement/frame/{t}")
def displacement_frame(request: Request, name: str, t: int, kind: str = "local") -> dict[str, Any]:
    if kind not in KINDS:
        raise HTTPException(400, f"kind must be one of {sorted(KINDS)}")
    with _open(_results(request, name)) as f:
        ds = _require(f, KINDS[kind])
        T = ds.shape[0]
        if not 0 <= t < T:
            raise HTTPException(404, f"frame {t} out of range")
        d = ds[t]
        st = f["reconstruction/status"][t] if "reconstruction/status" in f else np.ones(d.shape[0], np.uint8)
        return {"t": t, "kind": kind, "time": float(_times(f, T)[t]), "d": _clean(d, 7), "status": st.astype(int).tolist()}


@router.get("/displacement/point/{i}")
def displacement_point(request: Request, name: str, i: int, max_samples: int = 4000) -> dict[str, Any]:
    with _open(_results(request, name)) as f:
        ds = _require(f, KINDS["local"])
        T, N, _ = ds.shape
        if not 0 <= i < N:
            raise HTTPException(404, f"POI {i} out of range")
        t = _times(f, T)
        out: dict[str, Any] = {"i": i, "n_frames": int(T)}
        for k in ("local", "rbm_local"):
            if KINDS[k] in f:
                tt, y = _minmax_decimate(t, f[KINDS[k]][:, i, :], max_samples)
                out[k] = {"t": _clean(tt, 9), "c": _clean(y[:, 0], 7), "s": _clean(y[:, 1], 7), "n": _clean(y[:, 2], 7)}
        return out


@router.get("/rbm")
def rbm(request: Request, name: str, max_samples: int = 4000) -> dict[str, Any]:
    """Frame-wise rigid motion: translation (mm), rotation angles (deg, XYZ), fit residual."""
    with _open(_results(request, name)) as f:
        R = _require(f, "reconstruction/rbm_R")[()]
        tr = f["reconstruction/rbm_t"][()]
        res = f["reconstruction/rbm_residual_rms"][()].ravel()
        T = R.shape[0]
        # small-angle-safe XYZ Euler angles
        ang = np.stack(
            [np.arctan2(R[:, 2, 1], R[:, 2, 2]), -np.arcsin(np.clip(R[:, 2, 0], -1, 1)), np.arctan2(R[:, 1, 0], R[:, 0, 0])],
            axis=1,
        )
        t = _times(f, T)
        y = np.concatenate([tr, np.degrees(ang), res[:, None]], axis=1)
        tt, y = _minmax_decimate(t, y, max_samples)
        ref_mask = f["quality/rbm_reference_mask"][()] if "quality/rbm_reference_mask" in f else None
        return {
            "t": _clean(tt, 9),
            "tx": _clean(y[:, 0], 7), "ty": _clean(y[:, 1], 7), "tz": _clean(y[:, 2], 7),
            "rx": _clean(y[:, 3], 7), "ry": _clean(y[:, 4], 7), "rz": _clean(y[:, 5], 7),
            "residual_rms": _clean(y[:, 6], 7),
            "reference_points": None if ref_mask is None else int(ref_mask.sum()),
        }
