"""Mono and stereo calibration with OpenCV (Zhang's method + stereo bundle adjustment).

Workflow (docs §3.5):
  1. each camera: calibrateCamera on its own detections; views whose RMS exceeds
     `outlier_factor` x median are dropped and the camera is recalibrated;
  2. stereoCalibrate on the views kept by both cameras, using the corners seen in both
     images: first with fixed intrinsics (R, T only), then optionally a joint refinement
     with the mono intrinsics as the initial guess;
  3. residuals and per-view errors are recomputed here from the final parameters, so the
     report does not depend on OpenCV's internal bookkeeping.

The reprojection RMS is a diagnostic only; 3D accuracy is verified with E2 (translation stage).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..core.types import CameraModel, StereoCalibration
from .camera_model import optical_axis_angle_deg, project
from .detect import BoardDetection

_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)


@dataclass(frozen=True)
class CalibrationOptions:
    distortion: str = "k1k2p1p2"  # "k1k2p1p2" | "k1k2p1p2k3" | "rational"
    fix_aspect_ratio: bool = False
    zero_tangent: bool = False
    refine_stereo_intrinsics: bool = True
    outlier_factor: float = 3.0  # drop views with RMS > factor * median RMS
    min_views: int = 6
    min_common_corners: int = 8

    def flags(self) -> int:
        f = 0
        if self.distortion == "k1k2p1p2":
            f |= cv2.CALIB_FIX_K3
        elif self.distortion == "rational":
            f |= cv2.CALIB_RATIONAL_MODEL
        elif self.distortion != "k1k2p1p2k3":
            raise ValueError(f"unknown distortion model {self.distortion}")
        if self.fix_aspect_ratio:
            f |= cv2.CALIB_FIX_ASPECT_RATIO
        if self.zero_tangent:
            f |= cv2.CALIB_ZERO_TANGENT_DIST
        return f

    @property
    def n_dist(self) -> int:
        return 8 if self.distortion == "rational" else 5


@dataclass
class MonoResult:
    camera: CameraModel
    rms: float
    used: np.ndarray  # (V,) bool over the input views
    view_rms: np.ndarray  # (V,) RMS per view with the final model, NaN if not detected
    rvecs: dict[int, np.ndarray] = field(default_factory=dict)
    tvecs: dict[int, np.ndarray] = field(default_factory=dict)
    std_intrinsics: np.ndarray | None = None


@dataclass
class StereoResult:
    calibration: StereoCalibration
    rms: float
    left: MonoResult
    right: MonoResult
    pairs: np.ndarray  # view indices used for the stereo step
    view_rms_left: np.ndarray  # per stereo pair
    view_rms_right: np.ndarray
    residuals_left: list[np.ndarray]  # (K, 2) per pair, pixels
    residuals_right: list[np.ndarray]
    pair_ids: list[np.ndarray] = field(default_factory=list)  # (K,) corner ids per pair (common to both cameras)
    pair_points_left: list[np.ndarray] = field(default_factory=list)  # (K, 2) measured corners per pair
    pair_points_right: list[np.ndarray] = field(default_factory=list)
    poses: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # board pose per pair (left frame)

    def report(self) -> dict:
        cal = self.calibration
        all_l = np.vstack(self.residuals_left) if self.residuals_left else np.zeros((0, 2))
        all_r = np.vstack(self.residuals_right) if self.residuals_right else np.zeros((0, 2))
        return {
            "stereo_rms_px": self.rms,
            "left": _camera_report(self.left),
            "right": _camera_report(self.right),
            "baseline_mm": cal.baseline_mm,
            "optical_axis_angle_deg": optical_axis_angle_deg(cal),
            "R_RL": cal.R_RL.tolist(),
            "t_RL": cal.t_RL.tolist(),
            "pairs": self.pairs.tolist(),
            "pair_rms_left_px": self.view_rms_left.tolist(),
            "pair_rms_right_px": self.view_rms_right.tolist(),
            "residual_left_rms_px": float(np.sqrt((all_l**2).sum(1).mean())) if len(all_l) else None,
            "residual_right_rms_px": float(np.sqrt((all_r**2).sum(1).mean())) if len(all_r) else None,
            "residual_left_max_px": float(np.linalg.norm(all_l, axis=1).max()) if len(all_l) else None,
            "residual_right_max_px": float(np.linalg.norm(all_r, axis=1).max()) if len(all_r) else None,
            "n_corners": int(len(all_l)),
        }


def _camera_report(m: MonoResult) -> dict:
    out = {
        "K": m.camera.K.tolist(),
        "dist": m.camera.dist.tolist(),
        "image_size": list(m.camera.image_size),
        "mono_rms_px": m.rms,
        "views_used": int(m.used.sum()),
        "views_total": int(m.used.size),
        "view_rms_px": [None if not np.isfinite(v) else float(v) for v in m.view_rms],
    }
    if m.std_intrinsics is not None:
        s = m.std_intrinsics.ravel()
        out["std_fx_fy_cx_cy"] = s[:4].tolist()
    return out


def _view_rms(cam: CameraModel, det: BoardDetection, rvec: np.ndarray, tvec: np.ndarray) -> tuple[float, np.ndarray]:
    R, _ = cv2.Rodrigues(rvec)
    X = det.object_points @ R.T + tvec.ravel()
    r = project(cam, X) - det.image_points
    return float(np.sqrt((r**2).sum(1).mean())), r


def calibrate_mono(
    detections: list[BoardDetection | None],
    image_size: tuple[int, int],
    options: CalibrationOptions = CalibrationOptions(),
) -> MonoResult:
    """Single-camera calibration with iterative rejection of outlier views."""
    n_views = len(detections)
    used = np.array([d is not None and len(d) >= options.min_common_corners for d in detections])
    if used.sum() < options.min_views:
        raise ValueError(f"only {used.sum()} usable views, need >= {options.min_views}")
    flags = options.flags()
    for _ in range(5):
        idx = np.nonzero(used)[0]
        obj = [detections[i].object_points.astype(np.float32) for i in idx]
        img = [detections[i].image_points.astype(np.float32).reshape(-1, 1, 2) for i in idx]
        rms, K, D, rvecs, tvecs, std_in, _, per_view = cv2.calibrateCameraExtended(
            obj, img, tuple(image_size), None, None, flags=flags, criteria=_CRITERIA
        )
        per_view = per_view.ravel()
        med = float(np.median(per_view))
        bad = per_view > options.outlier_factor * max(med, 1e-6)
        if not bad.any() or used.sum() - bad.sum() < options.min_views:
            break
        used[idx[bad]] = False
    used = np.zeros(n_views, dtype=bool)
    used[idx] = True  # exactly the views of the final calibration
    cam = CameraModel(K=K, dist=np.asarray(D).ravel()[: options.n_dist], image_size=tuple(image_size))
    view_rms = np.full(n_views, np.nan)
    rv, tv = {}, {}
    for k, i in enumerate(idx):
        rv[int(i)], tv[int(i)] = rvecs[k].ravel(), tvecs[k].ravel()
        view_rms[i] = _view_rms(cam, detections[i], rv[int(i)], tv[int(i)])[0]
    return MonoResult(cam, float(rms), used, view_rms, rv, tv, np.asarray(std_in))


def _common(dl: BoardDetection, dr: BoardDetection) -> tuple[BoardDetection, BoardDetection]:
    ids = np.intersect1d(dl.ids, dr.ids)
    return dl.subset(ids), dr.subset(ids)


def calibrate_stereo(
    left: list[BoardDetection | None],
    right: list[BoardDetection | None],
    image_size_left: tuple[int, int],
    image_size_right: tuple[int, int] | None = None,
    options: CalibrationOptions = CalibrationOptions(),
) -> StereoResult:
    """Full stereo calibration from synchronized board views (same index = same instant)."""
    if len(left) != len(right):
        raise ValueError("left and right view lists must have the same length")
    image_size_right = image_size_right or image_size_left
    ml = calibrate_mono(left, image_size_left, options)
    mr = calibrate_mono(right, image_size_right, options)

    pairs, obj, img_l, img_r, pair_ids, pts_l, pts_r = [], [], [], [], [], [], []
    for i in range(len(left)):
        if not (ml.used[i] and mr.used[i]):
            continue
        cl, cr = _common(left[i], right[i])
        if len(cl) < options.min_common_corners:
            continue
        pairs.append(i)
        pair_ids.append(cl.ids.copy())
        pts_l.append(cl.image_points.copy())
        pts_r.append(cr.image_points.copy())
        obj.append(cl.object_points.astype(np.float32))
        img_l.append(cl.image_points.astype(np.float32).reshape(-1, 1, 2))
        img_r.append(cr.image_points.astype(np.float32).reshape(-1, 1, 2))
    if len(pairs) < options.min_views:
        raise ValueError(f"only {len(pairs)} stereo pairs, need >= {options.min_views}")

    flags = options.flags()
    K1, D1, K2, D2 = ml.camera.K.copy(), _full_dist(ml.camera), mr.camera.K.copy(), _full_dist(mr.camera)
    args = (obj, img_l, img_r, K1, D1, K2, D2, tuple(image_size_left), None, None)
    rms, K1, D1, K2, D2, R, T, *_ = cv2.stereoCalibrateExtended(
        *args, flags=flags | cv2.CALIB_FIX_INTRINSIC, criteria=_CRITERIA
    )
    if options.refine_stereo_intrinsics:
        rms, K1, D1, K2, D2, R, T, *_ = cv2.stereoCalibrateExtended(
            obj, img_l, img_r, K1, D1, K2, D2, tuple(image_size_left), R, T,
            flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS, criteria=_CRITERIA,
        )
    cam_l = CameraModel(K=K1, dist=np.asarray(D1).ravel()[: options.n_dist], image_size=tuple(image_size_left))
    cam_r = CameraModel(K=K2, dist=np.asarray(D2).ravel()[: options.n_dist], image_size=tuple(image_size_right))
    calib = StereoCalibration(left=cam_l, right=cam_r, R_RL=_orthonormalize(R), t_RL=np.asarray(T).ravel())

    # residuals with the final model: board pose per pair from the left camera (solvePnP),
    # right camera through the stereo extrinsics
    vr_l, vr_r, res_l, res_r, poses = [], [], [], [], []
    for o, il, ir in zip(obj, img_l, img_r):
        ok, rvec, tvec = cv2.solvePnP(o, il, K1, D1, flags=cv2.SOLVEPNP_ITERATIVE)
        rvec, tvec = _refine_pose_stereo(calib, o.astype(np.float64), il.reshape(-1, 2), ir.reshape(-1, 2), rvec, tvec)
        poses.append((np.asarray(rvec, dtype=np.float64).ravel(), np.asarray(tvec, dtype=np.float64).ravel()))
        Rb, _ = cv2.Rodrigues(rvec)
        Xl = o.astype(np.float64) @ Rb.T + tvec.ravel()
        rl = project(cam_l, Xl) - il.reshape(-1, 2)
        rr = project(cam_r, Xl @ calib.R_RL.T + calib.t_RL) - ir.reshape(-1, 2)
        res_l.append(rl)
        res_r.append(rr)
        vr_l.append(float(np.sqrt((rl**2).sum(1).mean())))
        vr_r.append(float(np.sqrt((rr**2).sum(1).mean())))
    return StereoResult(
        calibration=calib, rms=float(rms), left=ml, right=mr, pairs=np.asarray(pairs),
        view_rms_left=np.asarray(vr_l), view_rms_right=np.asarray(vr_r), residuals_left=res_l, residuals_right=res_r,
        pair_ids=pair_ids,
        pair_points_left=pts_l,
        pair_points_right=pts_r,
        poses=poses,
    )


def _full_dist(cam: CameraModel) -> np.ndarray:
    d = np.zeros(max(5, cam.dist.size))
    d[: cam.dist.size] = cam.dist
    return d


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def _refine_pose_stereo(calib, obj, pl, pr, rvec, tvec, iterations: int = 10):
    """Board pose minimizing the reprojection error in both cameras (Gauss–Newton, numeric Jacobian)."""
    x = np.concatenate([np.ravel(rvec), np.ravel(tvec)]).astype(np.float64)

    def resid(x):
        Rb, _ = cv2.Rodrigues(x[:3])
        Xl = obj @ Rb.T + x[3:]
        rl = project(calib.left, Xl) - pl
        rr = project(calib.right, Xl @ calib.R_RL.T + calib.t_RL) - pr
        return np.concatenate([rl.ravel(), rr.ravel()])

    for _ in range(iterations):
        r = resid(x)
        J = np.empty((r.size, 6))
        for k in range(6):
            h = 1e-7 if k < 3 else 1e-6
            e = np.zeros(6)
            e[k] = h
            J[:, k] = (resid(x + e) - resid(x - e)) / (2 * h)
        dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        x += dx
        if np.abs(dx).max() < 1e-12:
            break
    return x[:3], x[3:]
