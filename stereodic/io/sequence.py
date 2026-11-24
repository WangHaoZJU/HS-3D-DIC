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


class OffsetSequence(ImageSequence):
    """View of `base` that skips its first `offset` frames (per-camera trigger alignment).

    Frame k of the view is raw frame k + offset of the base sequence; `raw_index`
    and `frame_name` keep the link to the original files.
    """

    def __init__(self, base: ImageSequence, offset: int = 0) -> None:
        if offset < 0:
            raise ValueError("frame offset must be >= 0")
        if offset >= len(base):
            raise ValueError(f"frame offset {offset} leaves no frames ({len(base)} in the sequence)")
        self.base = base
        self.offset = int(offset)

    def __len__(self) -> int:
        return len(self.base) - self.offset

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.base.frame_shape

    @property
    def dtype(self) -> np.dtype:
        return self.base.dtype

    def raw_index(self, index: int) -> int:
        n = len(self)
        if not -n <= index < n:
            raise IndexError(f"frame {index} out of range [0, {n})")
        return index % n + self.offset

    def frame_name(self, index: int) -> str:
        return self.base.frame_name(self.raw_index(index))

    def _read_full(self, index: int) -> np.ndarray:
        return self.base._read_full(self.raw_index(index))


def open_sequence(path: str | Path, pattern: str = "*.tif", frame_offset: int = 0) -> ImageSequence:
    """Open a sequence with the first registered reader that accepts `path` (readers.Reader).

    `frame_offset` skips that many leading frames (see `OffsetSequence`).
    """
    from .readers import find_reader, readers

    path = Path(path)
    reader = find_reader(path, pattern)
    if reader is None:
        kinds = "; ".join(f"{r.name}: {r.description}" for r in readers())
        raise ValueError(f"unsupported sequence source: {path} (pattern {pattern!r}). Supported: {kinds}")
    seq = reader.open(path, pattern)
    return OffsetSequence(seq, frame_offset) if frame_offset else seq
