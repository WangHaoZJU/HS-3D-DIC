"""Synthetic ChArUco calibration views (ground truth for the calibration tests).

The board raster comes from OpenCV's `generateImage` at high resolution and is
rendered through the exact inverse map pixel -> ray -> board plane with
Gauss–Legendre pixel integration, optional optical blur and sensor noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .render import SensorModel, apply_sensor, render_function
from .stereo import HeightFieldTarget, SynthCamera, pixel_to_texture


@dataclass(frozen=True)
class BoardRaster:
    image: np.ndarray  # float64 transmission in [0, 1], white = 1
    px_per_mm: float
    margin_mm: float

    def sample(self, X: np.ndarray) -> np.ndarray:
        """Transmission at board coordinates X (K, 2) in mm (white outside the raster)."""
        # raster pixel centre i <-> board coordinate (i + 0.5) / px_per_mm - margin
        c = (X + self.margin_mm) * self.px_per_mm - 0.5
        return map_coordinates(self.image, [c[:, 1], c[:, 0]], order=1, mode="constant", cval=1.0)


def board_raster(
    board: cv2.aruco.CharucoBoard, px_per_mm: float = 100.0, margin_mm: float = 5.0, blur_mm: float = 0.02
) -> BoardRaster:
    """High-resolution board pattern, blurred by a Gaussian of `blur_mm` (on the board).

    The blur stands for the optics and makes the integrand smooth: a hard step would make
    the Gauss–Legendre pixel integration converge slowly and alias the corner positions
    by up to ~0.1 px. A symmetric blur leaves the saddle points where they are.
    """
    sx, sy = board.getChessboardSize()
    sq = board.getSquareLength()
    w = int(round((sx * sq + 2 * margin_mm) * px_per_mm))
    h = int(round((sy * sq + 2 * margin_mm) * px_per_mm))
    m = int(round(margin_mm * px_per_mm))
    img = board.generateImage((w, h), marginSize=m, borderBits=1).astype(np.float64) / 255.0
    if blur_mm > 0:
        img = gaussian_filter(img, blur_mm * px_per_mm, mode="nearest")
    return BoardRaster(img, px_per_mm, m / px_per_mm)


def render_board_view(
    raster: BoardRaster,
    pose: HeightFieldTarget,
    camera: SynthCamera,
    sensor: SensorModel | None = None,
    oversample: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Quantized camera image of a flat board at `pose` (board (x, y) mm = target (s, t))."""
    t = render_function(camera.shape, lambda x: pixel_to_texture(camera, pose, x), raster.sample, oversample, 1.0)
    return apply_sensor(t, sensor or SensorModel(noise_std=4.0), rng)
