"""Camera projection model (OpenCV-compatible), with analytic Jacobians.

Supported distortion vectors (OpenCV ordering):
    4:  k1, k2, p1, p2
    5:  k1, k2, p1, p2, k3                      (Brown–Conrady, default)
    8:  k1, k2, p1, p2, k3, k4, k5, k6          (rational radial model)
Thin-prism and tilt terms (12/14 coefficients) are rejected unless they are zero.

Pixel coordinates follow OpenCV: integer = pixel centre, zero-based.
"""

from __future__ import annotations

import numpy as np

from ..core.types import CameraModel, StereoCalibration


def _coeffs(dist: np.ndarray) -> np.ndarray:
    d = np.asarray(dist, dtype=np.float64).ravel()
    if d.size > 8 and np.any(d[8:] != 0):
        raise NotImplementedError("thin-prism / tilted-sensor distortion terms are not supported")
    out = np.zeros(8)
    out[: min(d.size, 8)] = d[:8]
    return out


def distort(n: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply lens distortion to ideal normalized points n (N, 2).

    Returns distorted normalized points (N, 2) and the Jacobian d(distorted)/d(n), (N, 2, 2).
    """
    k1, k2, p1, p2, k3, k4, k5, k6 = _coeffs(dist)
    n = np.asarray(n, dtype=np.float64).reshape(-1, 2)
    x, y = n[:, 0], n[:, 1]
    r2 = x * x + y * y
    num = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    den = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    dnum = k1 + 2 * k2 * r2 + 3 * k3 * r2**2
    dden = k4 + 2 * k5 * r2 + 3 * k6 * r2**2
    radial = num / den
    drad = (dnum * den - num * dden) / (den * den)  # d radial / d r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    J = np.empty((n.shape[0], 2, 2))
    J[:, 0, 0] = radial + 2 * x * x * drad + 2 * p1 * y + 6 * p2 * x
    J[:, 0, 1] = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
    J[:, 1, 0] = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
    J[:, 1, 1] = radial + 2 * y * y * drad + 6 * p1 * y + 2 * p2 * x
    return np.column_stack([xd, yd]), J


def project(cam: CameraModel, X_cam: np.ndarray, with_jacobian: bool = False):
    """Project camera-frame points (N, 3) to pixels (N, 2); optionally d(pixel)/d(X_cam), (N, 2, 3)."""
    X = np.asarray(X_cam, dtype=np.float64).reshape(-1, 3)
    z = X[:, 2]
    n = X[:, :2] / z[:, None]
    d, Jd = distort(n, cam.dist)
    K = cam.K
    uv = np.column_stack(
        [K[0, 0] * d[:, 0] + K[0, 1] * d[:, 1] + K[0, 2], K[1, 1] * d[:, 1] + K[1, 2]]
    )
    if not with_jacobian:
        return uv
    Kd = np.array([[K[0, 0], K[0, 1]], [0.0, K[1, 1]]])
    Jn = np.zeros((X.shape[0], 2, 3))  # d n / d X
    Jn[:, 0, 0] = 1 / z
    Jn[:, 1, 1] = 1 / z
    Jn[:, 0, 2] = -X[:, 0] / z**2
    Jn[:, 1, 2] = -X[:, 1] / z**2
    J = Kd[None] @ Jd @ Jn
    return uv, J


def pixel_to_normalized(cam: CameraModel, xy: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Remove intrinsics and distortion: pixels (N, 2) -> ideal normalized coordinates (N, 2).

    Newton iterations on the distortion model (converges to ~1e-15 for usual lenses).
    Points where the iteration does not converge are returned as NaN.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    K = cam.K
    yd = (xy[:, 1] - K[1, 2]) / K[1, 1]
    xd = (xy[:, 0] - K[0, 2] - K[0, 1] * yd) / K[0, 0]
    target = np.column_stack([xd, yd])
    n = target.copy()
    step = np.zeros_like(n)
    for _ in range(iterations):
        d, J = distort(n, cam.dist)
        step = np.linalg.solve(J, (d - target)[..., None])[..., 0]
        n = n - step
        if not np.any(np.abs(step) > 1e-15):
            break
    bad = ~np.all(np.isfinite(n), axis=1) | np.any(np.abs(step) > 1e-9, axis=1)
    n[bad] = np.nan
    return n


def to_right(calib: StereoCalibration, X_left: np.ndarray) -> np.ndarray:
    """Left-camera (world) frame -> right-camera frame."""
    return np.asarray(X_left, dtype=np.float64).reshape(-1, 3) @ calib.R_RL.T + calib.t_RL


def project_stereo(calib: StereoCalibration, X_left: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixels of world points (left-camera frame) in the left and right images."""
    X = np.asarray(X_left, dtype=np.float64).reshape(-1, 3)
    return project(calib.left, X), project(calib.right, to_right(calib, X))


def fundamental_matrix(calib: StereoCalibration) -> np.ndarray:
    """F for *undistorted* pixels: x_R^T F x_L = 0 (ideal pinhole part only)."""
    t = calib.t_RL
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ calib.R_RL
    return np.linalg.inv(calib.right.K).T @ E @ np.linalg.inv(calib.left.K)


def optical_axis_angle_deg(calib: StereoCalibration) -> float:
    """Angle between the two optical axes."""
    z_right_in_left = calib.R_RL.T @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(z_right_in_left[2], -1.0, 1.0))))
