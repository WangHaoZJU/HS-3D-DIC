"""Per-point measurement status codes (stored as uint8 in HDF5).

Values are part of the on-disk format: never renumber existing members,
only append new ones. Converged is not the same as valid.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

STATUS_DTYPE = np.uint8


class Status(IntEnum):
    NOT_COMPUTED = 0
    OK = 1
    OUT_OF_BOUNDS = 2  # subset leaves the image or ROI
    LOW_TEXTURE = 3  # gray-level variance / SSSIG too low
    ILL_CONDITIONED = 4  # reference Hessian ill-conditioned
    NOT_CONVERGED = 5  # max iterations reached
    HIGH_RESIDUAL = 6  # ZNSSD above threshold
    GEOMETRY_INCONSISTENT = 7  # epipolar / reprojection residual too large
    TEMPORAL_JUMP = 8  # outlier jump between consecutive frames
    MASKED = 9  # occluded or inside a manual mask


def status_counts(status: np.ndarray) -> dict[str, int]:
    """Count occurrences of each status code (unknown codes reported as 'UNKNOWN_<n>')."""
    values, counts = np.unique(np.asarray(status), return_counts=True)
    out: dict[str, int] = {}
    for v, c in zip(values.tolist(), counts.tolist()):
        try:
            name = Status(v).name
        except ValueError:
            name = f"UNKNOWN_{v}"
        out[name] = int(c)
    return out
