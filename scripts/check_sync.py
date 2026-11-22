"""Compare left/right timestamp files or a configured frame-rate fallback."""

from __future__ import annotations

import argparse

from stereodic.io.sync import load_timestamps, pair_timestamps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", required=True, help="left timestamp file (.npy, .csv, or text)")
    ap.add_argument("--right", required=True, help="right timestamp file (.npy, .csv, or text)")
    ap.add_argument("--max-delta-us", type=float, default=None)
    args = ap.parse_args()
    left = load_timestamps(args.left)
    right = load_timestamps(args.right)
    limit = None if args.max_delta_us is None else args.max_delta_us * 1e-6
    report = pair_timestamps(left, right, limit)
    print(f"paired frames: {len(report.left_indices)}")
    print(f"mean absolute delta: {report.mean_abs_delta_seconds * 1e6:.3f} us")
    print(f"maximum absolute delta: {report.max_abs_delta_seconds * 1e6:.3f} us")
    if len(report.left_indices) < min(len(left), len(right)):
        print(f"unpaired frames: left={len(left) - len(report.left_indices)}, right={len(right) - len(report.right_indices)}")


if __name__ == "__main__":
    main()
