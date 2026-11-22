"""Run DIC pipeline stages for an experiment.

    uv run python scripts/run_dic.py --config configs/experiments/exp001.yaml --stage stereo

Stages implemented so far: stereo (POI grid, L0 <-> R0 matching, reference 3D shape).
temporal and reconstruct follow in M3.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from stereodic.core.config import load_experiment
from stereodic.core.status import status_counts
from stereodic.io.storage import ResultStore
from stereodic.pipeline import run_stereo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["stereo"])
    ap.add_argument("--force", action="store_true", help="recompute even if the stage is up to date")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    cfg = load_experiment(args.config)
    out = run_stereo(cfg, force=args.force)
    with ResultStore.open(out, "r") as store:
        st = store.read_array("reference/stereo_status")
        rep = store.read_array("reference/reprojection_error")
    print(f"written: {out}")
    print(f"POIs: {st.size}, status: {status_counts(st)}")
    ok = st == 1
    if ok.any():
        print(f"reprojection error [px]: median {np.median(rep[ok]):.4f}, P99 {np.percentile(rep[ok], 99):.4f}")


if __name__ == "__main__":
    main()
