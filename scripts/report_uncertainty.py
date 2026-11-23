"""Write first-order and optional Monte Carlo stereo uncertainty to HDF5."""

from __future__ import annotations

import argparse

import numpy as np

from stereodic.calibration.report import load_calibration
from stereodic.core.config import config_hash, load_experiment
from stereodic.io.storage import ResultStore
from stereodic.uncertainty.monte_carlo import monte_carlo_triangulation
from stereodic.uncertainty.propagation import triangulation_uncertainty


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pixel-sigma", type=float, required=True, help="one-dimensional pixel standard deviation")
    ap.add_argument("--frame-index", type=int, default=None, help="optional temporal row to analyze")
    ap.add_argument("--monte-carlo", type=int, default=0, help="number of Monte Carlo samples, zero disables")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.pixel_sigma <= 0:
        raise ValueError("pixel-sigma must be positive")
    cfg = load_experiment(args.config)
    if cfg.output_file is None or cfg.calibration_file is None:
        raise ValueError("experiment config needs output_file and calibration_file")
    out = cfg.results_path(cfg.output_file)
    calibration = load_calibration(cfg.data_path(cfg.calibration_file))
    pixel_cov = np.eye(2) * args.pixel_sigma**2
    with ResultStore.open(out, "r+") as store:
        if args.frame_index is None:
            left = store.read_array("reference/points_left")
            right = store.read_array("reference/points_right")
            key = "reference"
        else:
            left = store.read_array("dic/left/xy")[args.frame_index] + np.asarray(cfg.cameras.left.sensor_origin)
            right = store.read_array("dic/right/xy")[args.frame_index] + np.asarray(cfg.cameras.right.sensor_origin)
            key = f"frame_{args.frame_index}"
        result = triangulation_uncertainty(calibration, left, right, pixel_cov)
        base = f"uncertainty/{key}"
        store.write_array(base + "/covariance_xyz", result.covariance, units="mm^2")
        store.write_array(base + "/std_xyz", result.standard_deviation, units="mm")
        store.write_array(base + "/jacobian", result.jacobian, units="mm/px")
        store.write_array(base + "/valid", result.valid)
        if args.monte_carlo > 0:
            mc = monte_carlo_triangulation(
                calibration,
                left,
                right,
                pixel_cov,
                samples=args.monte_carlo,
                seed=args.seed,
            )
            store.write_array(base + "/monte_carlo_mean", mc.mean, units="mm")
            store.write_array(base + "/monte_carlo_covariance", mc.covariance, units="mm^2")
            store.write_array(base + "/monte_carlo_std", mc.standard_deviation, units="mm")
            store.write_array(base + "/monte_carlo_valid_samples", mc.valid_samples)
        stage_hash = config_hash(
            {
                "pixel_sigma": args.pixel_sigma,
                "frame_index": args.frame_index,
                "monte_carlo": args.monte_carlo,
                "seed": args.seed,
            }
        )
        store.mark_stage_done("uncertainty", stage_hash)
        print(f"written: {out}")
        print(f"valid points: {int(result.valid.sum())}/{len(result.valid)}")
        print(f"median std [mm]: {np.nanmedian(result.standard_deviation, axis=0)}")


if __name__ == "__main__":
    main()
