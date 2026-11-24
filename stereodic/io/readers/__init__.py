"""Format-specific frame readers and the reader registry.

`open_sequence` asks every registered reader, in order, whether it can open a
source (a directory of frames, or a single container file such as a camera's
native format) and uses the first match. New formats (e.g. the vendor format of
the final high-speed camera) register a `Reader` here without touching callers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..sequence import ImageSequence


@dataclass(frozen=True)
class Reader:
    name: str
    description: str
    can_open: Callable[[Path, str], bool]
    open: Callable[[Path, str], ImageSequence]


_READERS: list[Reader] = []


def register_reader(reader: Reader, first: bool = False) -> None:
    if any(r.name == reader.name for r in _READERS):
        raise ValueError(f"reader {reader.name!r} already registered")
    _READERS.insert(0, reader) if first else _READERS.append(reader)


def readers() -> list[Reader]:
    _ensure_builtin()
    return list(_READERS)


def find_reader(path: Path, pattern: str) -> Reader | None:
    for r in readers():
        if r.can_open(path, pattern):
            return r
    return None


_builtin_done = False


def _ensure_builtin() -> None:
    global _builtin_done
    if _builtin_done:
        return
    _builtin_done = True
    from .imagedir import ImageDirSequence, is_image_dir
    from .tiff import TiffSequence, is_tiff_dir

    register_reader(Reader("tiff", "directory of single-frame TIFF files", is_tiff_dir, TiffSequence))
    register_reader(Reader("imagedir", "directory of PNG / BMP frames (lossless only)", is_image_dir, ImageDirSequence))
