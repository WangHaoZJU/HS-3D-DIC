"""Inspect an exported image sequence before using it (import step, docs/UI_Design.md §7.4)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ..quality.image import effective_bits, trailing_zero_bits
from .sequence import ImageSequence, open_sequence

_TIMESTAMP_HINT = re.compile(r"(time|stamp|ts)", re.IGNORECASE)
_TIMESTAMP_SUFFIXES = (".csv", ".txt", ".npy")


def _frame_numbers(names: list[str]) -> list[int] | None:
    """Last integer in each file name, if every name has one."""
    out = []
    for n in names:
        m = re.findall(r"\d+", Path(n).stem)
        if not m:
            return None
        out.append(int(m[-1]))
    return out


def numbering_gaps(names: list[str], limit: int = 20) -> dict[str, Any]:
    """Missing frame numbers in file names (dropped frames or partial export)."""
    nums = _frame_numbers(names)
    if nums is None or len(nums) < 2:
        return {"numbered": nums is not None, "first": nums[0] if nums else None, "last": nums[-1] if nums else None,
                "gaps": [], "n_missing": 0, "duplicates": 0}
    arr = np.asarray(nums)
    d = np.diff(arr)
    gaps = [{"after": int(arr[i]), "missing": int(d[i] - 1)} for i in np.nonzero(d > 1)[0][:limit]]
    return {
        "numbered": True,
        "first": int(arr[0]),
        "last": int(arr[-1]),
        "gaps": gaps,
        "n_missing": int(np.sum(np.clip(d - 1, 0, None))),
        "duplicates": int(np.sum(d <= 0)),
    }


def timestamp_candidates(directory: Path) -> list[str]:
    """Files next to (or one level above) the frames that look like timestamp tables."""
    out: list[str] = []
    for d in (directory, directory.parent):
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in _TIMESTAMP_SUFFIXES and _TIMESTAMP_HINT.search(p.stem):
                out.append(str(p))
    return out


def sample_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def describe(seq: ImageSequence, sample: int = 5) -> dict[str, Any]:
    """Shape, dtype, effective bit depth and value range from a few sampled frames."""
    n = len(seq)
    idx = sample_indices(n, sample)
    frames = [seq.read(i) for i in idx]
    vmax = max(int(np.max(f)) for f in frames)
    vmin = min(int(np.min(f)) for f in frames)
    shift = trailing_zero_bits(frames)
    bits = effective_bits(vmax)
    container = int(np.dtype(seq.dtype).itemsize * 8)
    data_bits = bits - shift
    return {
        "n_frames": n,
        "shape": list(seq.frame_shape),
        "dtype": str(seq.dtype),
        "container_bits": container,
        "effective_bits": bits,
        "msb_aligned_shift": shift,  # >0: low bits always zero (e.g. 12-bit data << 4)
        "data_bits": data_bits,
        "value_min": vmin,
        "value_max": vmax,
        "full_scale": (1 << bits) - 1,
        "sampled": idx,
        "first_name": seq.frame_name(0),
        "last_name": seq.frame_name(n - 1),
    }


def probe_sequence(path: str | Path, pattern: str = "*.tif", sample: int = 5) -> dict[str, Any]:
    """Everything the import page needs to know about one exported sequence."""
    from .readers import find_reader

    path = Path(path)
    reader = find_reader(path, pattern)
    seq = open_sequence(path, pattern)
    info = describe(seq, sample)
    names = [seq.frame_name(i) for i in range(len(seq))]
    files = getattr(seq, "files", None)
    total = sum(p.stat().st_size for p in files) if files else None
    warnings = []
    if info["data_bits"] <= 8 and info["container_bits"] <= 8:
        warnings.append("8-bit frames: export the camera's full bit depth (16-bit TIFF) if it has one")
    if info["msb_aligned_shift"] > 0:
        warnings.append(
            f"low {info['msb_aligned_shift']} bits are always zero: {info['data_bits']}-bit data stored MSB-aligned"
        )
    gaps = numbering_gaps(names)
    if gaps["n_missing"]:
        warnings.append(f"{gaps['n_missing']} frame numbers missing in the file names (dropped or not exported)")
    return {
        "path": str(path),
        "pattern": pattern,
        "reader": reader.name if reader else None,
        **info,
        "total_bytes": total,
        "numbering": gaps,
        "timestamp_candidates": timestamp_candidates(path if path.is_dir() else path.parent),
        "warnings": warnings,
    }
