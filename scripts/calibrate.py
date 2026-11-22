"""Stereo calibration from ChArUco images (the `calibration` section of an experiment config).

    uv run python scripts/calibrate.py --config configs/experiments/exp001.yaml

Writes `calibration_file` (HDF5 + .json sidecar + residual plot .png) and prints the report.
"""

from __future__ import annotations

import argparse
import json
import logging

from stereodic.core.config import load_experiment
from stereodic.pipeline import run_calibration


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    cfg = load_experiment(args.config)
    result = run_calibration(cfg, plot=not args.no_plot)
    rep = result.report()
    summary = {k: rep[k] for k in ("stereo_rms_px", "baseline_mm", "optical_axis_angle_deg", "n_corners")}
    summary["left_views"] = f"{rep['left']['views_used']}/{rep['left']['views_total']}"
    summary["right_views"] = f"{rep['right']['views_used']}/{rep['right']['views_total']}"
    summary["written"] = str(cfg.data_path(cfg.calibration_file))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
