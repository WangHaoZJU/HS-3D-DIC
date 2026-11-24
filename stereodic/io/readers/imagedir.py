"""Directory of single-frame PNG / BMP files (OpenCV, native bit depth).

Lossy formats (JPEG, video) are deliberately not supported: compression artefacts
bias sub-pixel matching (docs §3.1).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..sequence import ImageSequence
from .tiff import natural_key

IMAGE_SUFFIXES = (".png", ".bmp")


def is_image_dir(path: Path, pattern: str) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    return any(p.suffix.lower() in IMAGE_SUFFIXES for p in path.glob(pattern))


class ImageDirSequence(ImageSequence):
    def __init__(self, directory: str | Path, pattern: str = "*.png") -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"not a directory: {self.directory}")
        self.files = sorted(
            (p for p in self.directory.glob(pattern) if p.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda p: natural_key(p.name),
        )
        if not self.files:
            raise FileNotFoundError(f"no PNG/BMP files matching {pattern!r} in {self.directory}")
        first = self._load(self.files[0])
        self._shape = first.shape
        self._dtype = first.dtype

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"cannot read {path}")
        if img.ndim == 3:
            if not (np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2])):
                raise ValueError(f"{path}: colour image; export grayscale frames")
            img = img[..., 0]
        return np.ascontiguousarray(img)

    def __len__(self) -> int:
        return len(self.files)

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self._shape  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def frame_name(self, index: int) -> str:
        return self.files[index].name

    def _read_full(self, index: int) -> np.ndarray:
        img = self._load(self.files[index])
        if img.dtype != self._dtype:
            raise ValueError(f"{self.files[index]}: dtype {img.dtype} != sequence dtype {self._dtype}")
        return img
