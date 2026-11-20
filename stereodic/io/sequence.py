"""Lazy image sequences with ROI cropping and threaded prefetch.

Frames are returned in their native dtype (bit depth preserved); conversion to
float happens in the DIC code. ROIs are given in full-sensor coordinates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..core.types import ROI


class ImageSequence(ABC):
    """Random-access sequence of grayscale frames."""

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """(height, width) of a full frame."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype: ...

    @abstractmethod
    def _read_full(self, index: int) -> np.ndarray: ...

    def frame_name(self, index: int) -> str:
        return str(index)

    def read(self, index: int, roi: ROI | None = None) -> np.ndarray:
        n = len(self)
        if not -n <= index < n:
            raise IndexError(f"frame {index} out of range [0, {n})")
        img = self._read_full(index % n)
        if img.ndim != 2:
            raise ValueError(f"frame {index}: expected 2D grayscale image, got shape {img.shape}")
        if img.shape != self.frame_shape:
            raise ValueError(f"frame {index}: shape {img.shape} != sequence shape {self.frame_shape}")
        if roi is None:
            return img
        h, w = self.frame_shape
        if roi.x0 + roi.width > w or roi.y0 + roi.height > h:
            raise ValueError(f"ROI {roi} exceeds frame size {w}x{h}")
        return img[roi.slices]

    def __getitem__(self, index: int) -> np.ndarray:
        return self.read(index)

    def iter_frames(
        self,
        indices: Iterable[int],
        roi: ROI | None = None,
        workers: int = 4,
        prefetch: int = 8,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, frame) in the given order, reading ahead with a thread pool."""
        indices = list(indices)
        if workers <= 1:
            for i in indices:
                yield i, self.read(i, roi)
            return
        prefetch = max(prefetch, 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(self.read, i, roi) for i in indices[:prefetch]]
            next_submit = len(pending)
            for k, i in enumerate(indices):
                frame = pending[k].result()
                pending[k] = None  # release memory
                if next_submit < len(indices):
                    pending.append(pool.submit(self.read, indices[next_submit], roi))
                    next_submit += 1
                yield i, frame


def frame_indices(n_frames: int, start: int = 0, stop: int | None = None, step: int = 1) -> range:
    stop = n_frames if stop is None else min(stop, n_frames)
    if start >= stop:
        raise ValueError(f"empty frame range start={start} stop={stop} (sequence has {n_frames})")
    return range(start, stop, step)


def open_sequence(path: str | Path, pattern: str = "*.tif") -> ImageSequence:
    """Open a sequence by path. Currently: a directory of single-frame TIFF files."""
    from .readers.tiff import TiffSequence

    path = Path(path)
    if path.is_dir():
        return TiffSequence(path, pattern)
    raise ValueError(f"unsupported sequence source: {path} (expected a directory of TIFF frames)")
