"""BK — temporal IC-GN backends: throughput and agreement (docs §8.4, §14.1).

Synthetic 12-bit speckle frame pair(s) at camera resolution, a full POI grid, and the
production temporal path (`TemporalTracker.solve_frame`, including per-frame
B-spline prefiltering). Reports ms per frame, solves/s and the position difference
of every backend to the first one.

    uv run python -m benchmarks.bk_backends [--quick] [--backends numba cuda]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from benchmarks.common import DEFAULT_OUT, write_csv
from stereodic.dic.backends import available_backends
from stereodic.dic.icgn import ICGNOptions
from stereodic.dic.mesh import make_poi_grid
from stereodic.dic.temporal import TemporalMatchOptions, TemporalTracker
from stereodic.synth import Affine, SensorModel, SpeckleTexture, render_frame


def _sync(backend: str) -> None:
    if backend == "cuda":
        import cupy as cp

        cp.cuda.Device().synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="512 x 416 image instead of 1280 x 1024")
    ap.add_argument("--backends", nargs="+", default=None, help="default: all available except numpy")
    ap.add_argument("--subset", type=int, default=31)
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--interp", type=int, default=5, choices=[3, 5])
    ap.add_argument("--order", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--frames", type=int, default=3, help="distinct target frames")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    h, w = (416, 512) if args.quick else (1024, 1280)
    backends = args.backends or [b for b in available_backends() if b != "numpy"]
    tex = SpeckleTexture.random((-40, -40, w + 40, h + 40), diameter_px=4.0, seed=5)
    sensor = SensorModel(bits=12, noise_std=6.0)
    ref = render_frame(tex, (h, w), None, sensor, oversample=2, seed=1)
    centre = np.array([w / 2, h / 2])
    targets = [
        render_frame(
            tex,
            (h, w),
            Affine(A=np.eye(2) + 1e-3 * k * np.array([[1.0, 0.2], [-0.1, 0.5]]), t=np.array([0.13 * k, -0.07 * k]), center=centre),
            sensor,
            oversample=2,
            seed=2 + k,
        )
        for k in range(1, args.frames + 1)
    ]
    centers = make_poi_grid((h, w), args.subset, args.step, margin=16).xy
    opts = TemporalMatchOptions(icgn=ICGNOptions(subset_size=args.subset, shape_order=args.order), interp_order=args.interp)
    print(f"BK backends: {w}x{h}, {len(centers)} POIs, subset {args.subset}, order {args.order}, quintic={args.interp == 5}")
    print(f"{'backend':>8} {'prepare s':>10} {'ms/frame':>9} {'solves/s':>11} {'OK':>7} {'iters':>6} {'P99 diff px':>12}")

    rows, first = [], None
    for backend in backends:
        t0 = time.perf_counter()
        tracker = TemporalTracker(ref, centers, opts, backend)
        prepare = time.perf_counter() - t0
        tracker.solve_frame(targets[0])  # warm-up: JIT / kernel compilation, uploads
        _sync(backend)
        reps = 10 if backend == "cuda" else 1  # GPU frames are short: repeat for a stable timing
        results = []
        t0 = time.perf_counter()
        for _ in range(reps):
            results = [tracker.solve_frame(im) for im in targets]
        _sync(backend)
        n = reps * len(targets)
        dt = (time.perf_counter() - t0) / n
        xy = np.stack([r.xy for r in results])
        ok = np.stack([r.status == 1 for r in results])
        if first is None:
            first, diff = xy, float("nan")
        else:
            d = np.linalg.norm(xy - first, axis=-1)
            diff = float(np.nanpercentile(d, 99))
        it = np.concatenate([r.iterations[r.status == 1] for r in results]).mean()
        print(f"{backend:>8} {prepare:10.1f} {dt * 1e3:9.1f} {len(centers) / dt:11,.0f} {ok.mean():7.4f} {it:6.2f} {diff:12.2e}")
        rows.append({"backend": backend, "width": w, "height": h, "pois": len(centers), "prepare_s": prepare,
                     "ms_per_frame": dt * 1e3, "solves_per_s": len(centers) / dt, "valid": ok.mean(),
                     "iterations": it, "p99_diff_px": diff})
    write_csv(args.out / "bk_backends.csv", rows)
    print(f"written: {args.out / 'bk_backends.csv'}")


if __name__ == "__main__":
    main()
