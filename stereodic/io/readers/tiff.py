"""Directory of single-frame TIFF files (typical high-speed camera export)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

from ..sequence import ImageSequence


def natural_key(name: str) -> list[object]:
    """Sort key so that frame_2 < frame_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


class TiffSequence(ImageSequence):
    def __init__(self, directory: str | Path, pattern: str = "*.tif") -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"not a directory: {self.directory}")
        self.files = sorted(self.directory.glob(pattern), key=lambda p: natural_key(p.name))
        if not self.files:
            raise FileNotFoundError(f"no files matching {pattern!r} in {self.directory}")
        with tifffile.TiffFile(self.files[0]) as tf:
            page = tf.pages[0]
            self._shape = tuple(int(s) for s in page.shape)
            self._dtype = np.dtype(page.dtype)
        if len(self._shape) != 2:
            raise ValueError(f"{self.files[0]}: expected grayscale frames, got shape {self._shape}")

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
        img = tifffile.imread(self.files[index])
        if img.dtype != self._dtype:
            raise ValueError(f"{self.files[index]}: dtype {img.dtype} != sequence dtype {self._dtype}")
        return img
