"""Synthetic stereo scenes (S4 / S6 ground truth): speckled surfaces seen by two cameras.

A target is a height field z = h(s, t) in its own frame, posed in the world (= left
camera) frame; the speckle texture lives in the (s, t) parameter plane, in mm, so a
material point keeps its texture coordinates under any motion. Images are rendered
with the exact inverse map pixel -> ray -> surface -> (s, t) (`render_pointwise`), so
perspective, lens distortion and curvature carry no approximation beyond the
Gauss–Legendre pixel integration.

Projection and undistortion use OpenCV directly (not `stereodic.calibration`), keeping
the ground truth independent of the code under test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from ..core.types import CameraModel, StereoCalibration
from .render import SensorModel, apply_sensor, render_pointwise
from .speckle import SpeckleTexture

def _distort(n: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brown–Conrady (k1, k2, p1, p2, k3) on normalized points (N, 2): distorted points and Jacobian (N, 2, 2).

    Written out here (and checked against cv2.projectPoints in tests) so the synthetic
    ground truth does not depend on stereodic.calibration.
    """
    k1, k2, p1, p2, k3 = (np.concatenate([np.ravel(dist), np.zeros(5)])[:5]).tolist()
    if np.ravel(dist).size > 5 and np.any(np.ravel(dist)[5:] != 0):
        raise ValueError("synthetic cameras support only k1, k2, p1, p2, k3")
    x, y = n[:, 0], n[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    drad = k1 + 2 * k2 * r2 + 3 * k3 * r2**2  # d radial / d r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    J = np.empty((n.shape[0], 2, 2))
    J[:, 0, 0] = radial + 2 * x * x * drad + 2 * p1 * y + 6 * p2 * x
    J[:, 0, 1] = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
    J[:, 1, 0] = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
    J[:, 1, 1] = radial + 2 * y * y * drad + 6 * p1 * y + 2 * p2 * x
    return np.column_stack([xd, yd]), J


def _undistort_normalized(xy: np.ndarray, K: np.ndarray, dist: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Pixels (N, 2) -> ideal normalized coordinates (Newton on the distortion model, to ~1e-15)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    fx, fy, s, cx, cy = K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2]
    yd = (xy[:, 1] - cy) / fy
    xd = (xy[:, 0] - cx - s * yd) / fx
    target = np.column_stack([xd, yd])
    n = target.copy()
    for _ in range(iterations):
        d, J = _distort(n, dist)
        step = np.linalg.solve(J, (d - target)[..., None])[..., 0]
        n -= step
        if np.abs(step).max() < 1e-16:
            break
    return n


HeightFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


def rotation(rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0) -> np.ndarray:
    """R = Rz @ Ry @ Rx from angles in degrees."""
    ax, ay, az = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


@dataclass(frozen=True)
class SynthCamera:
    """Pinhole + Brown–Conrady camera; X_cam = R @ X_world + t."""

    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: tuple[int, int]  # (width, height)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def shape(self) -> tuple[int, int]:
        return self.image_size[1], self.image_size[0]

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        rvec, _ = cv2.Rodrigues(self.R)
        uv, _ = cv2.projectPoints(X, rvec, np.asarray(self.t, dtype=np.float64), self.K, self.dist)
        return uv.reshape(-1, 2)

    def rays(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-frame ray origin (3,) and unit directions (N, 3) through pixels xy (N, 2)."""
        n = _undistort_normalized(xy, self.K, self.dist)
        d_cam = np.column_stack([n, np.ones(len(n))])
        d = d_cam @ self.R  # R^T d for each row
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return self.center, d

    def model(self) -> CameraModel:
        return CameraModel(K=self.K, dist=self.dist, image_size=self.image_size)


def _plane(s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.zeros_like(s)
    return z, z, z


def cylinder_height(radius_mm: float) -> HeightFn:
    """Cylindrical bump along s (curvature 1/radius), apex at s = 0, bending away from +z."""

    def h(s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = float(radius_mm)
        root = np.sqrt(np.maximum(r * r - s * s, 1e-12))
        z = root - r
        return z, -s / root, np.zeros_like(t)

    return h


@dataclass(frozen=True)
class HeightFieldTarget:
    """Surface X = R @ (s, t, h(s, t)) + t in the world frame."""

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    height: HeightFn = _plane
    extent: tuple[float, float, float, float] | None = None  # (s0, t0, s1, t1); None = unbounded

    def point(self, st: np.ndarray) -> np.ndarray:
        st = np.asarray(st, dtype=np.float64).reshape(-1, 2)
        z, _, _ = self.height(st[:, 0], st[:, 1])
        local = np.column_stack([st, z])
        return local @ np.asarray(self.R).T + np.asarray(self.t)

    def intersect(self, origin: np.ndarray, dirs: np.ndarray, iterations: int = 30) -> np.ndarray:
        """(s, t) of the first-order ray/surface intersection (Newton on the ray parameter); NaN on miss."""
        R = np.asarray(self.R, dtype=np.float64)
        o = R.T @ (np.asarray(origin, dtype=np.float64) - np.asarray(self.t))
        d = np.asarray(dirs, dtype=np.float64) @ R  # R^T d per row
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = -o[2] / d[:, 2]  # plane z = 0
            for _ in range(iterations):
                p = o + lam[:, None] * d
                z, hs, ht = self.height(p[:, 0], p[:, 1])
                f = p[:, 2] - z
                df = d[:, 2] - hs * d[:, 0] - ht * d[:, 1]
                step = f / df
                lam = lam - step
                if np.nanmax(np.abs(step)) < 1e-13:
                    break
            p = o + lam[:, None] * d
        st = p[:, :2].copy()
        bad = ~np.isfinite(lam) | (lam <= 0)
        if self.extent is not None:
            s0, t0, s1, t1 = self.extent
            bad |= (st[:, 0] < s0) | (st[:, 0] > s1) | (st[:, 1] < t0) | (st[:, 1] > t1)
        st[bad] = np.nan
        return st

    def moved(self, R_m: np.ndarray, t_m: np.ndarray) -> HeightFieldTarget:
        """Target after the rigid motion X -> R_m X + t_m."""
        R_m = np.asarray(R_m, dtype=np.float64)
        return replace(self, R=R_m @ np.asarray(self.R), t=R_m @ np.asarray(self.t) + np.asarray(t_m))


def pixel_to_texture(camera: SynthCamera, target: HeightFieldTarget, xy: np.ndarray) -> np.ndarray:
    origin, dirs = camera.rays(xy)
    return target.intersect(origin, dirs)


def render_view(
    texture: SpeckleTexture,
    target: HeightFieldTarget,
    camera: SynthCamera,
    oversample: int = 4,
    background: float = 1.0,
) -> np.ndarray:
    """Pixel-integrated transmission of the target seen by `camera`, shape (H, W)."""
    return render_pointwise(
        texture, camera.shape, lambda x: pixel_to_texture(camera, target, x), oversample, background
    )


# ------------------------------------------------------------------------ scene
@dataclass(frozen=True)
class StereoScene:
    left: SynthCamera
    right: SynthCamera
    target: HeightFieldTarget
    texture: SpeckleTexture
    mm_per_px: float

    def calibration(self) -> StereoCalibration:
        """Exact calibration (world frame = left camera)."""
        return StereoCalibration(left=self.left.model(), right=self.right.model(), R_RL=self.right.R, t_RL=self.right.t)

    def render(
        self,
        target: HeightFieldTarget | None = None,
        sensor: SensorModel | None = None,
        oversample: int = 4,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quantized (left, right) frames of `target` (default: the reference pose)."""
        target = target or self.target
        sensor = sensor or SensorModel()
        rng = np.random.default_rng(seed)
        frames = []
        for cam in (self.left, self.right):
            t = render_view(self.texture, target, cam, oversample)
            frames.append(apply_sensor(t, sensor, rng))
        return frames[0], frames[1]

    def truth_from_left(self, xy_left: np.ndarray, target: HeightFieldTarget | None = None) -> dict[str, np.ndarray]:
        """Ground truth for material points seen at xy_left in the *reference* left image.

        Returns st (texture coordinates), X0 (reference 3D), and for `target` (moved pose,
        default reference) X (3D), xy_left / xy_right (pixels in both cameras).
        """
        st = pixel_to_texture(self.left, self.target, xy_left)
        X0 = self.target.point(st)
        tgt = target or self.target
        X = tgt.point(st)
        return {"st": st, "X0": X0, "X": X, "xy_left": self.left.project(X), "xy_right": self.right.project(X)}


def make_stereo_scene(
    image_size: tuple[int, int] = (512, 416),
    distance_mm: float = 250.0,
    mm_per_px: float = 0.1,
    stereo_angle_deg: float = 20.0,
    dist: tuple[float, ...] = (-0.12, 0.3, 2e-4, -1e-4, 0.0),
    speckle_px: float = 4.0,
    target_tilt_deg: tuple[float, float] = (6.0, 0.0),
    height: HeightFn | None = None,
    seed: int = 0,
) -> StereoScene:
    """Symmetric stereo rig looking at a speckled target about `distance_mm` away.

    World frame = left camera. The right camera sits at +x and is turned toward the
    target centre so that the angle between the optical axes is `stereo_angle_deg`.
    The target faces the bisector of the two cameras, plus `target_tilt_deg` (about x, y).
    """
    w, h = image_size
    f = distance_mm / mm_per_px
    K = np.array([[f, 0.0, (w - 1) / 2 + 3.2], [0.0, f * 1.0002, (h - 1) / 2 - 2.7], [0.0, 0.0, 1.0]])
    d = np.asarray(dist, dtype=np.float64)
    left = SynthCamera(K=K, dist=d, R=np.eye(3), t=np.zeros(3), image_size=(w, h))

    a = np.deg2rad(stereo_angle_deg)
    T = np.array([0.0, 0.0, distance_mm])
    C_R = np.array([distance_mm * np.sin(a), 0.0, distance_mm * (1 - np.cos(a))])  # on the circle around T
    z = (T - C_R) / np.linalg.norm(T - C_R)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_R = np.vstack([x, y, z])
    K_R = K.copy()
    K_R[0, 0] *= 0.998
    K_R[1, 1] *= 0.998
    K_R[0, 2] -= 4.1
    K_R[1, 2] += 1.3
    d_R = d * np.array([0.9, 1.1, -0.5, 0.7, 1.0])
    right = SynthCamera(K=K_R, dist=d_R, R=R_R, t=-R_R @ C_R, image_size=(w, h))

    # target faces the bisector (normal toward the cameras = -z after rotation by half the angle)
    R_T = rotation(target_tilt_deg[0], -np.rad2deg(a) / 2 + target_tilt_deg[1], 0.0)
    target = HeightFieldTarget(R=R_T, t=T, height=height or _plane)

    # texture extent: what both cameras can see, plus margin
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [(w - 1) / 2, (h - 1) / 2]], dtype=float)
    st = np.vstack([pixel_to_texture(c, target, corners) for c in (left, right)])
    margin = 0.15 * (np.nanmax(st) - np.nanmin(st)) + 10 * speckle_px * mm_per_px
    lo = np.nanmin(st, axis=0) - margin
    hi = np.nanmax(st, axis=0) + margin
    texture = SpeckleTexture.random((lo[0], lo[1], hi[0], hi[1]), diameter_px=speckle_px * mm_per_px, seed=seed)
    return StereoScene(left=left, right=right, target=target, texture=texture, mm_per_px=mm_per_px)
