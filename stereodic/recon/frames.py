"""Surface normals and blade-local coordinate frames on the reference shape (docs §9.2).

frames[i] is a 3x3 matrix whose ROWS are (c_i, s_i, n_i): chord, span and normal unit
vectors in the left-camera frame, right-handed (c = s x n). A displacement U (left-camera
frame) in local components is frames[i] @ U. Frames are fixed at the reference time.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def estimate_normals(
    points: np.ndarray,
    valid: np.ndarray | None = None,
    radius: float | None = None,
    min_neighbors: int = 6,
    viewpoint: np.ndarray | None = None,
) -> np.ndarray:
    """Unit normals by PCA of the neighbours within `radius` (mm), oriented toward `viewpoint`.

    radius: default 2.5 x the median nearest-neighbour spacing.
    viewpoint: default the left camera centre (origin), so normals face the cameras.
    Points that are invalid or have too few neighbours get NaN.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(P)
    ok = np.all(np.isfinite(P), axis=1) if valid is None else (np.asarray(valid, bool) & np.all(np.isfinite(P), axis=1))
    normals = np.full((n, 3), np.nan)
    idx = np.nonzero(ok)[0]
    if idx.size < 3:
        return normals
    tree = cKDTree(P[idx])
    if radius is None:
        d, _ = tree.query(P[idx], k=2)
        radius = 2.5 * float(np.median(d[:, 1]))
    vp = np.zeros(3) if viewpoint is None else np.asarray(viewpoint, dtype=np.float64)
    for k, nb in enumerate(tree.query_ball_point(P[idx], radius)):
        if len(nb) < min_neighbors:
            continue
        Q = P[idx[nb]]
        C = np.cov((Q - Q.mean(axis=0)).T)
        w, V = np.linalg.eigh(C)
        nv = V[:, 0]
        if np.dot(nv, vp - P[idx[k]]) < 0:
            nv = -nv
        normals[idx[k]] = nv
    return normals


def local_frames(normals: np.ndarray, span_direction: np.ndarray) -> np.ndarray:
    """Frames (N, 3, 3) with rows (c, s, n); s = span direction projected onto the tangent plane."""
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    span = np.asarray(span_direction, dtype=np.float64).ravel()
    span = span / np.linalg.norm(span)
    s = span[None, :] - (nrm @ span)[:, None] * nrm
    with np.errstate(invalid="ignore", divide="ignore"):
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        c = np.cross(s, nrm)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
    frames = np.stack([c, s, nrm], axis=1)
    bad = ~np.all(np.isfinite(frames.reshape(len(nrm), -1)), axis=1)
    frames[bad] = np.nan
    return frames


def to_local(frames: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Express vectors (..., N, 3) in the local frames (N, 3, 3)."""
    return np.einsum("nij,...nj->...ni", frames, vectors)
