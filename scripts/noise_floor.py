"""Static noise floor (E1) of an acquisition of the blade at rest (wrapper of `stereodic.analysis.run_noise_floor`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stereodic.analysis import run_noise_floor  # noqa: E402
from stereodic.core.config import load_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--use-rbm", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_noise_floor(load_experiment(args.config), args.use_rbm), indent=2))


if __name__ == "__main__":
    main()
