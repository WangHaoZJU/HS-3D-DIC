"""B04 — S4 stereo benchmark: reference shape and 3D rigid displacement on synthetic scenes.

Scene: two cameras (20 deg stereo angle, f = 2500 px, Brown distortion), speckled planar or
cylindrical target at ~250 mm, 0.1 mm/px. Images are rendered with the exact inverse map
(no local-affine approximation), Gauss–Legendre pixel integration, 12-bit quantization.

Per scene and noise level:
  stereo:   L0 -> R0 matching error (px) for first- and second-order warps;
  shape:    3D error of the triangulated reference points (mm);
  motion:   known rigid motion of the target; temporal IC-GN (first order) L0 -> L1 and
            R0 -> R1 from the reference positions, triangulation, displacement error (mm).

    uv run python -m benchmarks.b04_stereo_s4 [--quick]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from benchmarks.common import DEFAULT_OUT, write_csv
from stereodic.core.status import Status
from stereodic.dic import shape_functions as sf
from stereodic.dic.icgn import ICGNOptions, ICGNSolver
from stereodic.dic.interp import BSplineImage
from stereodic.dic.mesh import make_poi_grid
from stereodic.dic.stereo_match import StereoMatchOptions, stereo_match
from stereodic.recon.shape import reference_shape
from stereodic.recon.triangulate import triangulate
from stereodic.synth import SensorModel
from stereodic.synth.stereo import cylinder_height, make_stereo_scene, rotation

MOTION_T = np.array([0.05, -0.03, 0.10])  # mm
MOTION_R = rotation(0.03, -0.05, 0.02)  # deg -> about 0.01 mm extra at the field edge


def _temporal(ref_img, tgt_img, centers, order, interp):
    solver = ICGNSolver(ICGNOptions(31, order), "numba")
    refs = solver.prepare_reference(BSplineImage(ref_img, interp), centers)
    return solver.solve(refs, BSplineImage(tgt_img, interp), np.zeros((len(centers), sf.n_params(order))))


def run_case(name, height, noise, quick, interp=5):
    size = (320, 256) if quick else (512, 416)
    scene = make_stereo_scene(image_size=size, height=height, seed=11)
    sensor = SensorModel(noise_std=noise)
    os_ = 3 if quick else 4
    L0, R0 = scene.render(sensor=sensor, oversample=os_, seed=1)
    # rigid motion about the target centre
    c = scene.target.t
    moved = scene.target.moved(MOTION_R, c - MOTION_R @ c + MOTION_T)
    L1, R1 = scene.render(moved, sensor=sensor, oversample=os_, seed=2)
    cal = scene.calibration()
    pois = make_poi_grid(L0.shape, 31, 8 if quick else 6, margin=12)
    truth = scene.truth_from_left(pois.xy, moved)  # X0: reference, X: moved
    right0 = scene.truth_from_left(pois.xy)["xy_right"]  # reference right positions
    rows = []
    for order in (1, 2):
        t0 = time.perf_counter()
        sm = stereo_match(L0, R0, pois, cal, StereoMatchOptions(icgn=ICGNOptions(31, order), interp_order=interp))
        t_stereo = time.perf_counter() - t0
        shape = reference_shape(cal, pois.xy, sm.xy_right, sm.result.status)
        ok_s = shape.status == Status.OK
        e_px = np.linalg.norm(sm.xy_right[ok_s] - right0[ok_s], axis=1)
        e_3d = shape.points_3d[ok_s] - truth["X0"][ok_s]

        # temporal (first order) from the reference positions in each camera
        rl = _temporal(L0, L1, pois.xy, 1, interp)
        rr = _temporal(R0, R1, np.where(ok_s[:, None], sm.xy_right, -1e6), 1, interp)
        ok = ok_s & (rl.status == Status.OK) & (rr.status == Status.OK)
        t1 = triangulate(cal, rl.xy[ok], rr.xy[ok])
        U = t1.X - shape.points_3d[ok]
        U_true = truth["X"][ok] - truth["X0"][ok]
        e_u = U - U_true
        rows.append(
            dict(
                scene=name, noise=noise, stereo_order=order, n_poi=len(pois), valid=float(ok.mean()),
                stereo_rms_px=float(np.sqrt((e_px**2).mean())), stereo_max_px=float(e_px.max()),
                shape_rms_x_mm=float(np.sqrt((e_3d[:, 0] ** 2).mean())),
                shape_rms_y_mm=float(np.sqrt((e_3d[:, 1] ** 2).mean())),
                shape_rms_z_mm=float(np.sqrt((e_3d[:, 2] ** 2).mean())),
                disp_bias_x_mm=float(e_u[:, 0].mean()), disp_bias_y_mm=float(e_u[:, 1].mean()),
                disp_bias_z_mm=float(e_u[:, 2].mean()),
                disp_std_x_mm=float(e_u[:, 0].std()), disp_std_y_mm=float(e_u[:, 1].std()),
                disp_std_z_mm=float(e_u[:, 2].std()),
                reproj_median_px=float(np.median(t1.reprojection_error)),
                stereo_seconds=t_stereo,
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    cases = [("plane", None), ("cylinder_R80", cylinder_height(80.0))]
    noises = [0.0, 10.0]
    rows = []
    for name, h in cases:
        for noise in noises:
            rows.extend(run_case(name, h, noise, args.quick))
            print(f"{name}, noise {noise:g} done")
    tag = "b04_s4" + ("_quick" if args.quick else "")
    write_csv(args.out / f"{tag}.csv", rows)
    um = 1e3
    print(f"\nB04 S4 stereo (motion t = {MOTION_T.tolist()} mm + small rotation); 3D errors in um")
    print(f"{'scene':>13} {'noise':>5} {'ord':>3} {'stereo px':>10} {'shape z':>8} {'shape xy':>8} "
          f"{'U bias xyz':>22} {'U std xyz':>20} {'valid':>6}")
    for r in rows:
        bias = f"{r['disp_bias_x_mm']*um:6.2f} {r['disp_bias_y_mm']*um:6.2f} {r['disp_bias_z_mm']*um:6.2f}"
        std = f"{r['disp_std_x_mm']*um:5.2f} {r['disp_std_y_mm']*um:5.2f} {r['disp_std_z_mm']*um:5.2f}"
        sxy = np.hypot(r["shape_rms_x_mm"], r["shape_rms_y_mm"]) * um
        print(f"{r['scene']:>13} {r['noise']:5.0f} {r['stereo_order']:3d} {r['stereo_rms_px']:10.5f} "
              f"{r['shape_rms_z_mm']*um:8.2f} {sxy:8.2f} {bias:>22} {std:>20} {r['valid']:6.1%}")
    print(f"\nwritten: {args.out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
