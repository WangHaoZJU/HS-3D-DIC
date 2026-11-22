"""Fixed-reference temporal DIC.

The reference subsets are prepared once. Each target frame is solved against
that fixed reference, while the previous successful solution is used only as an
initial guess. This preserves stable material-point IDs and avoids cumulative
drift from incrementally adding frame-to-frame displacements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..core.config import DICConfig
from ..core.status import STATUS_DTYPE, Status
from ..core.types import DICBatchResult
from . import shape_functions as sf
from .icgn import ICGNOptions, ReferenceSubsets, ICGNSolver
from .interp import BSplineImage


@dataclass(frozen=True)
class TemporalMatchOptions:
    """Options for one camera's fixed-reference temporal pass."""

    icgn: ICGNOptions
    interp_order: int = 5
    max_znssd: float = 0.4
    initialization: str = "previous_solution"
    max_jump_px: float | None = None

    @classmethod
    def from_config(cls, cfg: DICConfig) -> TemporalMatchOptions:
        return cls(
            icgn=ICGNOptions.from_config(cfg, stage="temporal"),
            interp_order=cfg.interpolation.order,
            max_znssd=cfg.temporal.max_znssd,
            initialization=cfg.temporal.initialization,
            max_jump_px=cfg.temporal.max_jump_px,
        )


@dataclass
class TemporalResult:
    """Array-oriented output with axes ``(T, N, ...)``."""

    frame_indices: np.ndarray
    xy: np.ndarray
    params: np.ndarray
    znssd: np.ndarray
    iterations: np.ndarray
    status: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return self.status == Status.OK


def _empty_frame_result(n: int, n_params: int) -> DICBatchResult:
    return DICBatchResult.empty(n, n_params=n_params)


class TemporalTracker:
    """Reusable fixed-reference solver for one camera."""

    def __init__(
        self,
        reference_image: np.ndarray,
        centers: np.ndarray,
        options: TemporalMatchOptions,
        backend: str = "auto",
    ) -> None:
        self.options = options
        self.reference = BSplineImage(reference_image, options.interp_order)
        self.centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
        self.solver = ICGNSolver(options.icgn, backend)
        self.refs: ReferenceSubsets = self.solver.prepare_reference(self.reference, self.centers)
        self.n_params = sf.n_params(options.icgn.shape_order)
        self._zero = np.zeros((len(self.centers), self.n_params), dtype=np.float64)

    def reference_result(self, frame_index: int) -> DICBatchResult:
        """Return the exact fixed-reference result for the reference frame."""
        out = _empty_frame_result(len(self.centers), self.n_params)
        out.status[:] = self.refs.status
        ok = out.status == Status.OK
        out.xy[ok] = self.centers[ok]
        out.params[ok] = 0.0
        out.znssd[ok] = 0.0
        return out

    def solve_frame(self, image: np.ndarray, p0: np.ndarray | None = None) -> DICBatchResult:
        target = BSplineImage(image, self.options.interp_order)
        if p0 is None:
            p0 = self._zero
        result = self.solver.solve(self.refs, target, p0)

        high = (result.status == Status.OK) & (result.znssd > self.options.max_znssd)
        result.status[high] = Status.HIGH_RESIDUAL
        result.xy[result.status != Status.OK] = np.nan
        result.params[result.status != Status.OK] = np.nan
        return result

    def check_jump(self, result: DICBatchResult, previous_xy: np.ndarray | None) -> None:
        """Mark implausible frame-to-frame jumps when a threshold is configured."""
        if self.options.max_jump_px is None or previous_xy is None:
            return
        good = (result.status == Status.OK) & np.all(np.isfinite(previous_xy), axis=1)
        jump = np.linalg.norm(result.xy - previous_xy, axis=1)
        bad = good & (jump > self.options.max_jump_px)
        result.status[bad] = Status.TEMPORAL_JUMP
        result.xy[bad] = np.nan
        result.params[bad] = np.nan


def track_sequence(
    reference_image: np.ndarray,
    target_frames: Iterable[tuple[int, np.ndarray]],
    centers: np.ndarray,
    options: TemporalMatchOptions,
    backend: str = "auto",
    reference_frame: int = 0,
) -> TemporalResult:
    """Track a finite sequence of ``(frame_index, image)`` pairs.

    The reference frame is inserted by the caller when it is part of the
    requested output. If it is absent, all provided frames are solved against
    the supplied reference image.
    """
    frames = list(target_frames)
    indices = np.asarray([int(i) for i, _ in frames], dtype=np.int64)
    if indices.size == 0:
        raise ValueError("target_frames is empty")
    if np.unique(indices).size != indices.size:
        raise ValueError("target frame indices must be unique")
    order = np.argsort(indices)
    indices = indices[order]
    images = [frames[int(k)][1] for k in order]

    tracker = TemporalTracker(reference_image, centers, options, backend)
    n_frames, n_points = indices.size, len(tracker.centers)
    xy = np.full((n_frames, n_points, 2), np.nan, dtype=np.float64)
    params = np.full((n_frames, n_points, tracker.n_params), np.nan, dtype=np.float64)
    znssd = np.full((n_frames, n_points), np.nan, dtype=np.float32)
    iterations = np.zeros((n_frames, n_points), dtype=np.uint8)
    status = np.full((n_frames, n_points), Status.NOT_COMPUTED, dtype=STATUS_DTYPE)

    previous_params = np.zeros((n_points, tracker.n_params), dtype=np.float64)
    previous_xy: np.ndarray | None = None
    for k, (frame_index, image) in enumerate(zip(indices.tolist(), images)):
        if frame_index == reference_frame:
            result = tracker.reference_result(frame_index)
        else:
            p0 = previous_params if options.initialization == "previous_solution" else None
            result = tracker.solve_frame(image, p0)
            tracker.check_jump(result, previous_xy)
        xy[k] = result.xy
        params[k] = result.params
        znssd[k] = result.znssd
        iterations[k] = result.iterations
        status[k] = result.status
        good = result.status == Status.OK
        if np.any(good):
            previous_params[good] = result.params[good]
        previous_xy = result.xy.copy()

    return TemporalResult(indices, xy, params, znssd, iterations, status)
