"""First-order and optional Monte Carlo stereo uncertainty (wrapper of `stereodic.analysis.run_uncertainty`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stereodic.analysis import run_uncertainty  # noqa: E402
from stereodic.core.config import load_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pixel-sigma", type=float, required=True, help="one-dimensional pixel standard deviation")
    ap.add_argument("--frame-index", type=int, default=None, help="optional temporal row to analyze")
    ap.add_argument("--monte-carlo", type=int, default=0, help="number of Monte Carlo samples, zero disables")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = run_uncertainty(load_experiment(args.config), args.pixel_sigma, args.monte_carlo, args.seed, args.frame_index)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
