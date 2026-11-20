"""Shared helpers for synthetic DIC benchmarks."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stereodic.dic import shape_functions as sf
from stereodic.dic.icgn import ICGNOptions, ICGNSolver
from stereodic.dic.interp import BSplineImage
from stereodic.synth import SensorModel, SpeckleTexture
from stereodic.synth.render import apply_sensor

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "results" / "benchmarks"


@dataclass
class MatchStats:
    err: np.ndarray  # (N, 2) measured - truth, valid points only
    n_total: int
    seconds: float

    @property
    def n_valid(self) -> int:
        return int(self.err.shape[0])


def make_texture(shape: tuple[int, int], diameter_px: float, seed: int, margin: float = 40.0) -> SpeckleTexture:
    h, w = shape
    return SpeckleTexture.random((-margin, -margin, w + margin, h + margin), diameter_px=diameter_px, seed=seed)


def to_frame(transmission: np.ndarray, noise_std: float, seed: int, bits: int = 12) -> np.ndarray:
    return apply_sensor(transmission, SensorModel(bits=bits, noise_std=noise_std), np.random.default_rng(seed))


def match(
    ref_frame: np.ndarray,
    tgt_frame: np.ndarray,
    centers: np.ndarray,
    truth_uv: np.ndarray,
    options: ICGNOptions,
    interp_order: int,
    backend: str = "numba",
) -> MatchStats:
    """Run IC-GN from a zero initial guess and return errors of converged points."""
    ref = BSplineImage(ref_frame, interp_order)
    tgt = BSplineImage(tgt_frame, interp_order)
    solver = ICGNSolver(options, backend)
    refs = solver.prepare_reference(ref, centers)
    p0 = np.zeros((len(centers), sf.n_params(options.shape_order)))
    t0 = time.perf_counter()
    res = solver.solve(refs, tgt, p0)
    dt = time.perf_counter() - t0
    ok = res.valid
    return MatchStats(err=res.xy[ok] - centers[ok] - truth_uv, n_total=len(centers), seconds=dt)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
