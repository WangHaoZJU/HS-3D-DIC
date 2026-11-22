"""Reliability-guided propagation of initial guesses (RG-DIC, Pan 2009).

Solved points are kept in a priority queue ordered by ZNSSD (most reliable first).
Each popped point hands its warp, re-centred on the neighbour (`shape_functions.shift_origin`,
exact for polynomial warps), to unsolved neighbours as their initial guess. Every point is
still solved independently by IC-GN; propagation only supplies starting values.

A neighbour that fails (or exceeds `max_znssd`) may be retried from another solved
neighbour, up to the number of neighbours it has.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from ..core.status import Status
from ..core.types import DICBatchResult
from . import shape_functions as sf
from .icgn import ICGNSolver, ReferenceSubsets
from .interp import BSplineImage


@dataclass
class PropagationStats:
    n_seeds: int = 0
    n_solved: int = 0
    n_attempts: int = 0


def propagate(
    solver: ICGNSolver,
    refs: ReferenceSubsets,
    target: BSplineImage,
    neighbors: list[np.ndarray],
    result: DICBatchResult,
    seeds: np.ndarray,
    max_znssd: float,
    stats: PropagationStats | None = None,
) -> PropagationStats:
    """Grow the solution in `result` (modified in place) from already solved `seeds`.

    result: DICBatchResult for all POIs with params; seed rows must be solved (status OK).
    Points already OK in `result` are never re-solved.
    """
    stats = stats or PropagationStats()
    order = refs.options.shape_order
    centers = refs.centers
    solved = (result.status == Status.OK) & (result.znssd <= max_znssd)
    attempts = np.zeros(len(refs), dtype=np.int64)
    heap: list[tuple[float, int]] = []
    for s in np.asarray(seeds, dtype=np.int64):
        if solved[s]:
            heapq.heappush(heap, (float(result.znssd[s]), int(s)))
            stats.n_seeds += 1
    while heap:
        _, i = heapq.heappop(heap)
        for j in neighbors[i]:
            if solved[j] or refs.status[j] != Status.OK or attempts[j] >= max(1, len(neighbors[j])):
                continue
            attempts[j] += 1
            stats.n_attempts += 1
            p0 = sf.shift_origin(result.params[i], centers[j] - centers[i], order)
            r = solver.solve(refs, target, p0[None, :], indices=np.array([j]))
            ok = r.status[0] == Status.OK and r.znssd[0] <= max_znssd
            if ok or result.status[j] != Status.OK or r.znssd[0] < result.znssd[j]:
                result.xy[j] = r.xy[0]
                result.znssd[j] = r.znssd[0]
                result.iterations[j] = r.iterations[0]
                result.status[j] = r.status[0]
                result.params[j] = r.params[0]
            if ok:
                solved[j] = True
                stats.n_solved += 1
                heapq.heappush(heap, (float(r.znssd[0]), int(j)))
    return stats
