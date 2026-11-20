"""B01 — sub-pixel bias of IC-GN versus fractional shift (S1 synthetic translation).

For each speckle seed, the reference and shifted images are rendered independently
(analytic texture, Gauss–Legendre pixel integration, 12-bit quantization, optional
Gaussian noise). Shift s is applied in both x and y; u and v errors are pooled.

    uv run python -m benchmarks.b01_subpixel_bias            # full sweep
    uv run python -m benchmarks.b01_subpixel_bias --quick    # smoke run

Outputs CSV + PNG in results/benchmarks/ (git-ignored) and prints a summary.
M1 exit criterion: ideal (noise 0) RMSE <= 0.005 px.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.common import DEFAULT_OUT, make_texture, match, to_frame, write_csv
from stereodic.dic.icgn import ICGNOptions
from stereodic.dic.mesh import make_poi_grid
from stereodic.synth import Translation, render_transmission


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--diameter", type=float, default=4.0, help="speckle diameter, px")
    ap.add_argument("--subset", type=int, default=31)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    shape = (160, 160) if args.quick else (224, 224)
    shifts = np.round(np.arange(0.0, 1.0001, 0.25 if args.quick else 0.05), 4)
    seeds = [0] if args.quick else [0, 1, 2]
    noises = [0.0, 10.0]
    interp_orders = [3, 5]
    options = ICGNOptions(subset_size=args.subset, shape_order=1)
    centers = make_poi_grid(shape, args.subset, 12, margin=12).xy

    rows: list[dict] = []
    for seed in seeds:
        tex = make_texture(shape, args.diameter, seed)
        t_ref = render_transmission(tex, shape, None, oversample=6)
        for s in shifts:
            t_tgt = render_transmission(tex, shape, Translation(s, s), oversample=6)
            for noise in noises:
                ref = to_frame(t_ref, noise, seed=1000 * seed + 1)
                tgt = to_frame(t_tgt, noise, seed=1000 * seed + 2 + int(s * 1000))
                for io in interp_orders:
                    st = match(ref, tgt, centers, np.array([s, s]), options, io)
                    e = st.err.ravel()  # pool u and v
                    rows.append(
                        dict(
                            seed=seed, shift=float(s), noise=noise, interp=io, diameter=args.diameter,
                            n_valid=st.n_valid, n_total=st.n_total,
                            bias=float(e.mean()), std=float(e.std()), rmse=float(np.sqrt((e**2).mean())),
                            sq_sum=float((e**2).sum()), sum=float(e.sum()), count=int(e.size),
                        )
                    )
        print(f"seed {seed} done")

    tag = f"b01_D{args.diameter:g}_S{args.subset}" + ("_quick" if args.quick else "")
    write_csv(args.out / f"{tag}.csv", rows)

    # aggregate over seeds per (noise, interp, shift)
    print(f"\nB01 sub-pixel bias, speckle D={args.diameter:g} px, subset {args.subset}, 1st-order IC-GN")
    print(f"{'noise':>6} {'interp':>6} {'max|bias|':>10} {'mean std':>9} {'RMSE':>8} {'valid':>7}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for noise in noises:
        for io in interp_orders:
            sel = [r for r in rows if r["noise"] == noise and r["interp"] == io]
            bias_curve, std_curve = [], []
            for s in shifts:
                rs = [r for r in sel if r["shift"] == float(s)]
                n = sum(r["count"] for r in rs)
                mean = sum(r["sum"] for r in rs) / n
                var = sum(r["sq_sum"] for r in rs) / n - mean**2
                bias_curve.append(mean)
                std_curve.append(np.sqrt(max(var, 0.0)))
            n_all = sum(r["count"] for r in sel)
            rmse = np.sqrt(sum(r["sq_sum"] for r in sel) / n_all)
            valid = sum(r["n_valid"] for r in sel) / sum(r["n_total"] for r in sel)
            print(f"{noise:6.1f} {io:6d} {np.max(np.abs(bias_curve)):10.5f} {np.mean(std_curve):9.5f} {rmse:8.5f} {valid:7.1%}")
            axes[0].plot(shifts, bias_curve, marker="o", ms=3, label=f"noise {noise:g}, order {io}")
            axes[1].plot(shifts, std_curve, marker="o", ms=3, label=f"noise {noise:g}, order {io}")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set(xlabel="sub-pixel shift [px]", ylabel="bias [px]", title="Mean error")
    axes[1].set(xlabel="sub-pixel shift [px]", ylabel="std [px]", title="Random error")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"B01  D={args.diameter:g} px, subset {args.subset}")
    fig.tight_layout()
    fig.savefig(args.out / f"{tag}.png", dpi=130)
    print(f"\nwritten: {args.out / (tag + '.csv')} and .png")


if __name__ == "__main__":
    main()
