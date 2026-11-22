"""Reference 3D shape from the stereo correspondences of the reference pair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.status import Status
from ..core.types import StereoCalibration
from ..dic.mesh import grid_triangles
from .frames import estimate_normals, local_frames
from .triangulate import triangulate


@dataclass
class ReferenceShape:
    points_3d: np.ndarray  # (N, 3) mm, NaN where invalid
    status: np.ndarray  # (N,) uint8, DIC status merged with the geometric check
    reprojection_error: np.ndarray  # (N,) px
    normals: np.ndarray  # (N, 3)
    local_frames: np.ndarray  # (N, 3, 3) rows (c, s, n)
    connectivity: np.ndarray  # (M, 3) triangles of POI indices


def reference_shape(
    calib: StereoCalibration,
    xy_left: np.ndarray,
    xy_right: np.ndarray,
    stereo_status: np.ndarray,
    max_reprojection_px: float = 0.5,
    span_direction: np.ndarray = np.array([0.0, -1.0, 0.0]),
    normal_radius_steps: float = 2.5,
) -> ReferenceShape:
    """Triangulate, flag GEOMETRY_INCONSISTENT, and build normals, local frames and a mesh.

    xy_left / xy_right in full-sensor (calibration) coordinates.
    """
    status = np.asarray(stereo_status).copy()
    ok = status == Status.OK
    tri = triangulate(calib, np.where(ok[:, None], xy_left, np.nan), np.where(ok[:, None], xy_right, np.nan),
                      max_reprojection_px=max_reprojection_px)
    geo_bad = ok & (tri.status != Status.OK)
    status[geo_bad] = Status.GEOMETRY_INCONSISTENT
    valid = status == Status.OK
    X = np.where(valid[:, None], tri.X, np.nan)
    # spacing in mm from the 2D grid spacing and the local scale
    tris = grid_triangles(xy_left, valid)
    if len(tris):
        e = np.linalg.norm(X[tris[:, 0]] - X[tris[:, 1]], axis=1)
        spacing = float(np.nanmedian(e))
    else:
        spacing = None
    radius = None if spacing is None else normal_radius_steps * spacing
    normals = estimate_normals(X, valid, radius=radius)
    frames = local_frames(normals, span_direction)
    return ReferenceShape(X, status, tri.reprojection_error, normals, frames, tris)
