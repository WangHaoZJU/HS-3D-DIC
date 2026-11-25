"""Render a synthetic demo dataset (calibration views + vibrating curved blade).

Example::

    uv run python scripts/make_demo_dataset.py --data-root ~/Data/hs3ddic --frames 160

By default the experiment workspace is ``<results_root>/experiments/<name>/`` (from
configs/local.yaml), which is where the web UI looks for experiments.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stereodic.core.config import load_local_config  # noqa: E402
from stereodic.synth.demo import DemoSpec, make_demo_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="demo_s6")
    ap.add_argument("--data-root", default=None, help="default: data_root from configs/local.yaml")
    ap.add_argument("--workspace", default=None, help="default: <results_root>/experiments/<name>")
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--fps", type=float, default=2000.0)
    ap.add_argument("--views", type=int, default=12, help="ChArUco calibration views")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=416)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--f-torsion", type=float, default=137.0, help="torsion mode frequency, Hz")
    ap.add_argument("--torsion-deg", type=float, default=0.5, help="torsion amplitude, deg (0: off)")
    ap.add_argument("--f-bending", type=float, default=412.0, help="out-of-plane mode frequency, Hz")
    ap.add_argument("--bending-mm", type=float, default=0.02, help="out-of-plane amplitude, mm (0: off)")
    ap.add_argument("--exposure-us", type=float, default=0.0, help="exposure time for motion blur, µs (0: none)")
    ap.add_argument("--exact-calibration", action="store_true", help="also write the true calibration")
    args = ap.parse_args()

    local = load_local_config()
    data_root = Path(args.data_root).expanduser() if args.data_root else local.data_root
    if data_root is None:
        ap.error("no --data-root and no data_root in configs/local.yaml")
    if args.workspace:
        workspace = Path(args.workspace).expanduser()
    elif local.results_root is not None:
        workspace = Path(local.results_root).expanduser() / "experiments" / args.name
    else:
        workspace = None

    spec = DemoSpec(
        name=args.name,
        image_size=(args.width, args.height),
        n_frames=args.frames,
        frame_rate_hz=args.fps,
        n_calib_views=args.views,
        f_torsion_hz=args.f_torsion,
        torsion_deg=args.torsion_deg,
        f_bending_hz=args.f_bending,
        bending_mm=args.bending_mm,
        exposure_us=args.exposure_us,
    )
    t0 = time.time()
    out = make_demo_dataset(data_root, spec, workspace, args.workers, args.exact_calibration)
    print(f"data:       {out}")
    print(f"experiment: {(workspace or out) / 'experiment.yaml'}")
    print(f"elapsed:    {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
