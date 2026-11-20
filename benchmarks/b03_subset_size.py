"""B03 — random/systematic error versus subset size (S1 synthetic translation with noise).

Random sub-pixel shifts, 12-bit images with Gaussian noise. Larger subsets lower
the random error but reduce spatial resolution (resolution itself is assessed with
non-uniform deformation in M2). Also reports Numba IC-GN throughput.

    uv run python -m benchmarks.b03_subset_size [--quick] [--noise 10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks.common import DEFAULT_OUT, make_texture, match, to_frame, write_csv
from stereodic.dic.icgn import ICGNOptions
from stereodic.dic.mesh import make_poi_grid
from stereodic.synth import Translation, render_transmission


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--noise", type=float, default=10.0, help="Gaussian noise std, gray levels (12-bit)")
    ap.add_argument("--diameter", type=float, default=4.0)
    ap.add_argument("--interp", type=int, default=3, choices=[3, 5])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    shape = (192, 192) if args.quick else (256, 256)
    subsets = [21, 31, 51] if args.quick else [21, 25, 31, 41, 51]
    rng = np.random.default_rng(7)
    n_shifts = 3 if args.quick else 8
    shifts = rng.uniform(-1.0, 1.0, size=(n_shifts, 2))
    seeds = [0] if args.quick else [0, 1]

    pairs = []
    for seed in seeds:
        tex = make_texture(shape, args.diameter, seed)
        t_ref = render_transmission(tex, shape, None, oversample=6)
        ref = to_frame(t_ref, args.noise, seed=10 * seed + 1)
        for k, s in enumerate(shifts):
            tgt = to_frame(render_transmission(tex, shape, Translation(*s), oversample=6), args.noise, seed=10 * seed + 2 + k)
            pairs.append((ref, tgt, s))

    rows = []
    print(f"B03 subset size, D={args.diameter:g} px, noise {args.noise:g} GL (12-bit), cubic={args.interp == 3}")
    print(f"{'subset':>6} {'bias':>9} {'std':>8} {'RMSE':>8} {'valid':>7} {'solves/s':>10}")
    max_subset = max(subsets)
    for S in subsets:
        # same POI positions for all subset sizes (fits the largest subset)
        centers = make_poi_grid(shape, max_subset, 8, margin=12).xy
        errs, n_valid, n_total, secs = [], 0, 0, 0.0
        for ref, tgt, s in pairs:
            st = match(ref, tgt, centers, s, ICGNOptions(subset_size=S, shape_order=1), args.interp)
            errs.append(st.err.ravel())
            n_valid += st.n_valid
            n_total += st.n_total
            secs += st.seconds
        e = np.concatenate(errs)
        row = dict(
            subset=S, noise=args.noise, diameter=args.diameter, interp=args.interp,
            bias=float(e.mean()), std=float(e.std()), rmse=float(np.sqrt((e**2).mean())),
            valid=n_valid / n_total, solves_per_s=n_total / secs,
        )
        rows.append(row)
        print(f"{S:6d} {row['bias']:9.5f} {row['std']:8.5f} {row['rmse']:8.5f} {row['valid']:7.1%} {row['solves_per_s']:10.0f}")
    tag = f"b03_D{args.diameter:g}_N{args.noise:g}" + ("_quick" if args.quick else "")
    write_csv(args.out / f"{tag}.csv", rows)
    print(f"\nwritten: {args.out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
