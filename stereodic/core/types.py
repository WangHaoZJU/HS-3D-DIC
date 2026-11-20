"""Core data types shared across modules.

Conventions (see docs §6.1):
- image coordinates (x, y), x right, y down, zero-based, integers at pixel centres;
- NumPy images are indexed image[y, x];
- all coordinates are expressed in the full-sensor frame used for calibration;
  ROI crops only change how pixels are read, conversions go through `ROI`;
- stereo extrinsics: X_right = R_RL @ X_left + t_RL; lengths in mm;
- geometry and parameters are float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .status import STATUS_DTYPE, Status


def _as_f64(a: Any, shape: tuple[int, ...] | None = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if shape is not None:
        if len(arr.shape) != len(shape) or any(s != -1 and s != d for s, d in zip(shape, arr.shape)):
            raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
    return arr


@dataclass(frozen=True)
class ROI:
    """Rectangular crop in full-sensor pixel coordinates."""

    x0: int
    y0: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x0 < 0 or self.y0 < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError(f"invalid ROI {self}")

    @classmethod
    def from_list(cls, values: list[int] | tuple[int, int, int, int] | None) -> ROI | None:
        if values is None:
            return None
        x0, y0, w, h = (int(v) for v in values)
        return cls(x0, y0, w, h)

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.x0, self.y0], dtype=np.float64)

    @property
    def slices(self) -> tuple[slice, slice]:
        """(row slice, column slice) for cropping a full-frame array."""
        return slice(self.y0, self.y0 + self.height), slice(self.x0, self.x0 + self.width)

    def to_full(self, xy_roi: np.ndarray) -> np.ndarray:
        """ROI-local pixel coordinates -> full-sensor coordinates."""
        return np.asarray(xy_roi, dtype=np.float64) + self.origin

    def to_roi(self, xy_full: np.ndarray) -> np.ndarray:
        """Full-sensor pixel coordinates -> ROI-local coordinates."""
        return np.asarray(xy_full, dtype=np.float64) - self.origin

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.width, self.height]


@dataclass
class CameraModel:
    """Pinhole camera with Brown–Conrady distortion (OpenCV ordering k1, k2, p1, p2, k3, ...)."""

    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]  # (width, height) of the calibrated full frame

    def __post_init__(self) -> None:
        self.K = _as_f64(self.K, (3, 3), "K")
        self.dist = _as_f64(self.dist, name="dist").ravel()
        if self.dist.size not in (4, 5, 8, 12, 14):
            raise ValueError(f"dist: unsupported number of coefficients {self.dist.size}")
        w, h = (int(v) for v in self.image_size)
        self.image_size = (w, h)

    def to_dict(self) -> dict[str, Any]:
        return {"K": self.K.tolist(), "dist": self.dist.tolist(), "image_size": list(self.image_size)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraModel:
        return cls(K=np.asarray(d["K"]), dist=np.asarray(d["dist"]), image_size=tuple(d["image_size"]))


@dataclass
class StereoCalibration:
    left: CameraModel
    right: CameraModel
    R_RL: np.ndarray
    t_RL: np.ndarray  # mm

    def __post_init__(self) -> None:
        self.R_RL = _as_f64(self.R_RL, (3, 3), "R_RL")
        self.t_RL = _as_f64(self.t_RL, name="t_RL").ravel()
        if self.t_RL.shape != (3,):
            raise ValueError(f"t_RL: expected 3 elements, got {self.t_RL.shape}")
        if not np.allclose(self.R_RL @ self.R_RL.T, np.eye(3), atol=1e-9):
            raise ValueError("R_RL is not orthonormal")
        if np.linalg.det(self.R_RL) < 0:
            raise ValueError("R_RL is a reflection (det < 0)")

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.t_RL))

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "R_RL": self.R_RL.tolist(),
            "t_RL": self.t_RL.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StereoCalibration:
        return cls(
            left=CameraModel.from_dict(d["left"]),
            right=CameraModel.from_dict(d["right"]),
            R_RL=np.asarray(d["R_RL"]),
            t_RL=np.asarray(d["t_RL"]),
        )


@dataclass
class POISet:
    """Points of interest on the left reference image, with stable IDs.

    IDs are never reordered or removed; invalid points are flagged via status.
    """

    point_ids: np.ndarray  # (N,) int64
    xy: np.ndarray  # (N, 2) float64, full-sensor coordinates

    def __post_init__(self) -> None:
        self.point_ids = np.asarray(self.point_ids, dtype=np.int64).ravel()
        self.xy = _as_f64(self.xy, (-1, 2), "xy")
        if self.point_ids.shape[0] != self.xy.shape[0]:
            raise ValueError("point_ids and xy length mismatch")
        if np.unique(self.point_ids).size != self.point_ids.size:
            raise ValueError("point_ids must be unique")

    def __len__(self) -> int:
        return int(self.point_ids.size)


@dataclass
class DICBatchResult:
    """Array-oriented result of matching N points in one target image."""

    xy: np.ndarray  # (N, 2) float64, matched position (NaN if invalid)
    znssd: np.ndarray  # (N,) float32
    iterations: np.ndarray  # (N,) uint8
    status: np.ndarray  # (N,) uint8
    params: np.ndarray | None = field(default=None)  # (N, P) float64, optional

    @classmethod
    def empty(cls, n: int, n_params: int | None = None) -> DICBatchResult:
        return cls(
            xy=np.full((n, 2), np.nan, dtype=np.float64),
            znssd=np.full(n, np.nan, dtype=np.float32),
            iterations=np.zeros(n, dtype=np.uint8),
            status=np.full(n, Status.NOT_COMPUTED, dtype=STATUS_DTYPE),
            params=None if n_params is None else np.full((n, n_params), np.nan, dtype=np.float64),
        )

    def __len__(self) -> int:
        return int(self.status.size)

    @property
    def valid(self) -> np.ndarray:
        return self.status == Status.OK
