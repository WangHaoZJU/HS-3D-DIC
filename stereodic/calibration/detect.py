"""ChArUco board detection with sub-pixel saddle-point refinement.

ArUco markers only identify the chessboard corners; the measured points are the
chessboard saddle points, refined with `cv2.cornerSubPix` on the full bit-depth
image (the marker detector itself needs 8 bit). Saddle points are projectively
invariant, unlike circle centres (no eccentricity bias under tilt).

Board coordinates: OpenCV convention, origin at the board's top-left corner, x along
the squares_x direction, y along squares_y, z = 0 on the board; units mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.imaging import to_uint8


@dataclass(frozen=True)
class CharucoBoardSpec:
    squares_x: int
    squares_y: int
    square_mm: float
    marker_mm: float
    dictionary: str = "DICT_5X5_100"
    legacy_pattern: bool = False  # boards generated with OpenCV < 4.6 (and some vendors) with even rows

    def __post_init__(self) -> None:
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("ChArUco board needs at least 3 x 3 squares")
        if not 0 < self.marker_mm < self.square_mm:
            raise ValueError("marker_mm must be positive and smaller than square_mm")
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"unknown ArUco dictionary {self.dictionary}")

    def board(self) -> cv2.aruco.CharucoBoard:
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        b = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), float(self.square_mm), float(self.marker_mm), d)
        b.setLegacyPattern(self.legacy_pattern)
        return b

    @property
    def n_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def corner_points(self) -> np.ndarray:
        """Board coordinates of all chessboard corners (n_corners, 3), indexed by corner id."""
        return np.asarray(self.board().getChessboardCorners(), dtype=np.float64).reshape(-1, 3)

    @property
    def size_mm(self) -> tuple[float, float]:
        return self.squares_x * self.square_mm, self.squares_y * self.square_mm


@dataclass
class BoardDetection:
    ids: np.ndarray  # (K,) int64 corner ids
    image_points: np.ndarray  # (K, 2) float64, pixels
    object_points: np.ndarray  # (K, 3) float64, mm

    def __len__(self) -> int:
        return int(self.ids.size)

    def subset(self, ids: np.ndarray) -> BoardDetection:
        """Rows for the given corner ids, in that order (all must be present)."""
        pos = {int(k): i for i, k in enumerate(self.ids)}
        idx = np.array([pos[int(k)] for k in ids], dtype=np.int64)
        return BoardDetection(self.ids[idx], self.image_points[idx], self.object_points[idx])


def _square_px(det_ids: np.ndarray, pts: np.ndarray, spec: CharucoBoardSpec) -> float:
    """Median image distance between horizontally adjacent corners (= square size in px)."""
    per_row = spec.squares_x - 1
    pos = {int(k): i for i, k in enumerate(det_ids)}
    d = [
        np.linalg.norm(pts[pos[k + 1]] - pts[i])
        for k, i in pos.items()
        if (k + 1) in pos and (k + 1) // per_row == k // per_row
    ]
    if not d:
        d = [np.median(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) / 2]
    return float(np.median(d))


def detect_charuco(
    image: np.ndarray,
    spec: CharucoBoardSpec,
    min_corners: int = 8,
    refine_window: int | None = None,
) -> BoardDetection | None:
    """Detect ChArUco corners; returns None if fewer than `min_corners` or collinear.

    refine_window: half-size of the cornerSubPix window in px. Default: 80 % of the white
        rim between a marker and its square edge, so the window sees only the four
        chessboard squares. A window reaching the marker edges biases the corner by
        up to ~1 px (measured on synthetic boards with marker/square = 0.75).
    """
    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    board = spec.board()
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(to_uint8(img))
    if ids is None or len(ids) < min_corners:
        return None
    ids = ids.ravel().astype(np.int64)
    pts = corners.reshape(-1, 2).astype(np.float64)
    if board.checkCharucoCornersCollinear(ids.astype(np.int32)):
        return None
    win = refine_window
    if win is None:
        rim_px = 0.5 * (spec.square_mm - spec.marker_mm) / spec.square_mm * _square_px(ids, pts, spec)
        win = int(np.clip(np.floor(0.8 * rim_px), 2, 20))
    gray = img.astype(np.float32)
    refined = cv2.cornerSubPix(
        gray,
        pts.astype(np.float32).reshape(-1, 1, 2),
        (win, win),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
    ).reshape(-1, 2)
    obj = spec.corner_points()[ids]
    order = np.argsort(ids)
    return BoardDetection(ids[order], refined.astype(np.float64)[order], obj[order])
