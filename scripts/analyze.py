"""Vibration analysis of reconstructed displacement (spectrum, peaks, damping, harmonic / ODS, FDD, FRF).

Thin wrapper around `stereodic.analysis.run_analyze`; the web UI runs the same code.

    uv run python scripts/analyze.py --config <exp.yaml> --direction 2 [--use-rbm] [--frequency 1000] [--fdd]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stereodic.analysis import AnalyzeOptions, run_analyze  # noqa: E402
from stereodic.core.config import load_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--direction", type=int, choices=[0, 1, 2], default=2, help="local component: 0 c, 1 s, 2 n")
    ap.add_argument("--min-frequency", type=float, default=0.0)
    ap.add_argument("--max-frequency", type=float, default=None)
    ap.add_argument("--max-peaks", type=int, default=10)
    ap.add_argument("--frequency", type=float, action="append", default=[], help="harmonic fit frequency (repeatable)")
    ap.add_argument("--fit-peaks", type=int, default=3, help="without --frequency: fit the N strongest peaks")
    ap.add_argument("--no-refine", action="store_true", help="use the given frequencies exactly")
    ap.add_argument("--n-harmonics", type=int, default=1)
    ap.add_argument("--fdd", action="store_true", help="frequency-domain decomposition")
    ap.add_argument("--n-modes", type=int, default=3)
    ap.add_argument("--nperseg", type=int, default=None)
    ap.add_argument("--force-file", default=None, help="force per frame, or (time, force) columns")
    ap.add_argument("--use-rbm", action="store_true", help="analyze rigid-body-compensated displacement")
    args = ap.parse_args()

    opt = AnalyzeOptions(
        direction=args.direction, use_rbm=args.use_rbm, min_frequency=args.min_frequency,
        max_frequency=args.max_frequency, max_peaks=args.max_peaks, nperseg=args.nperseg,
        frequencies=args.frequency, fit_peaks=args.fit_peaks, refine=not args.no_refine,
        n_harmonics=args.n_harmonics, fdd=args.fdd, n_modes=args.n_modes, force_file=args.force_file,
    )
    summary = run_analyze(load_experiment(args.config), opt)
    print(json.dumps({k: v for k, v in summary.items() if k != "options"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
