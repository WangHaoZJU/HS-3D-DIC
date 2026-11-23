"""Run DIC pipeline stages for an experiment.

    uv run python scripts/run_dic.py --config configs/experiments/exp001.yaml --stage stereo

Stages: stereo, temporal fixed-reference DIC, and reconstruct.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from stereodic.core.config import load_experiment
from stereodic.core.status import status_counts
from stereodic.io.storage import ResultStore
from stereodic.pipeline import run_rbm, run_reconstruct, run_stereo, run_temporal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["stereo", "temporal", "reconstruct", "rbm"])
    ap.add_argument("--force", action="store_true", help="recompute even if the stage is up to date")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    cfg = load_experiment(args.config)
    if args.stage == "stereo":
        out = run_stereo(cfg, force=args.force)
    elif args.stage == "temporal":
        out = run_temporal(cfg, force=args.force)
    elif args.stage == "reconstruct":
        out = run_reconstruct(cfg, force=args.force)
    else:
        out = run_rbm(cfg, force=args.force)
    with ResultStore.open(out, "r") as store:
        if args.stage == "stereo":
            st = store.read_array("reference/stereo_status")
            rep = store.read_array("reference/reprojection_error")
        elif args.stage == "temporal":
            st = store.read_array("dic/left/status")
            rep = None
        elif args.stage == "reconstruct":
            st = store.read_array("reconstruction/status")
            rep = store.read_array("reconstruction/reprojection_error")
        else:
            st = store.read_array("quality/rbm_valid")
            rep = store.read_array("reconstruction/rbm_residual_rms")
    print(f"written: {out}")
    print(f"status: {status_counts(st)}")
    if rep is not None:
        ok = st == 1
        if ok.any():
            print(f"reprojection error [px]: median {np.nanmedian(rep[ok]):.4f}, P99 {np.nanpercentile(rep[ok], 99):.4f}")


if __name__ == "__main__":
    main()
