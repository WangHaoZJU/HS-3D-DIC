"""Identify dominant frequencies from reconstructed displacement."""

from __future__ import annotations

import argparse
import numpy as np

from stereodic.core.config import config_hash, load_experiment
from stereodic.io.storage import ResultStore
from stereodic.vibration.damping import half_power_damping
from stereodic.vibration.fdd import frequency_domain_decomposition
from stereodic.vibration.frf import estimate_frf
from stereodic.vibration.ods import extract_ods
from stereodic.vibration.spectrum import dominant_peaks, one_sided_spectrum


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--direction", type=int, choices=[0, 1, 2], default=2, help="local displacement component")
    ap.add_argument("--min-frequency", type=float, default=0.0)
    ap.add_argument("--max-frequency", type=float, default=None)
    ap.add_argument("--max-peaks", type=int, default=10)
    ap.add_argument("--frequency", type=float, default=None, help="fit harmonic amplitude/phase at this frequency")
    ap.add_argument("--n-harmonics", type=int, default=1)
    ap.add_argument("--fdd", action="store_true", help="write frequency-domain decomposition results")
    ap.add_argument("--n-modes", type=int, default=3)
    ap.add_argument("--nperseg", type=int, default=None)
    ap.add_argument("--force-file", default=None, help="optional force samples for FRF estimation")
    ap.add_argument("--use-rbm", action="store_true", help="analyze rigid-body-compensated displacement")
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    if cfg.output_file is None:
        raise ValueError("experiment config has no output_file")
    out = cfg.results_path(cfg.output_file)
    with ResultStore.open(out, "r+") as store:
        if "reconstruction/displacement_local" not in store.h5:
            raise ValueError("reconstruction data is missing; run --stage reconstruct first")
        timestamps = store.read_array("time/timestamps")
        source_name = "reconstruction/displacement_local"
        if args.use_rbm:
            if "reconstruction/displacement_rbm" not in store.h5:
                raise ValueError("RBM displacement is missing; run --stage rbm first")
            source_name = "reconstruction/displacement_rbm"
        displacement_all = store.read_array(source_name)
        displacement = displacement_all[:, :, args.direction]
        # A robust aggregate trace keeps the frequency stage independent of a
        # selected single POI while preserving the per-point data in HDF5.
        trace = np.nanmedian(displacement, axis=1)
        spec = one_sided_spectrum(trace, timestamps)
        frequencies, amplitudes = dominant_peaks(
            spec, min_frequency_hz=args.min_frequency, max_frequency_hz=args.max_frequency, max_peaks=args.max_peaks
        )
        store.write_array("vibration/frequencies", frequencies, units="Hz")
        store.write_array("vibration/amplitudes", amplitudes, units="mm")
        store.write_array("vibration/aggregate_trace", trace, units="mm", direction=args.direction)
        store.write_array("vibration/spectrum_frequencies", spec.frequencies, units="Hz")
        store.write_array("vibration/spectrum_amplitude", spec.amplitude, units="mm")
        selected_frequency = args.frequency
        if selected_frequency is None and frequencies.size:
            selected_frequency = float(frequencies[0])
        if selected_frequency is not None:
            ods = extract_ods(displacement_all, timestamps, selected_frequency, n_harmonics=args.n_harmonics)
            key = f"{selected_frequency:.9g}".replace("-", "m").replace(".", "p")
            base = f"vibration/harmonic/{key}"
            store.write_array(base + "/frequency_hz", np.array(selected_frequency), units="Hz")
            store.write_array(base + "/response", ods.response)
            store.write_array(base + "/amplitude", ods.amplitude, units="mm")
            store.write_array(base + "/phase", ods.phase, units="rad")
            store.write_array(base + "/residual_rms", ods.residual_rms, units="mm")
            print(f"harmonic frequency: {selected_frequency:.6g} Hz")
        if args.fdd:
            fdd = frequency_domain_decomposition(
                displacement_all,
                timestamps,
                nperseg=args.nperseg,
                n_modes=args.n_modes,
            )
            store.write_array("vibration/fdd/frequencies", fdd.frequencies, units="Hz")
            store.write_array("vibration/fdd/singular_values", fdd.singular_values)
            store.write_array("vibration/fdd/mode_shapes", fdd.mode_shapes)
            print(f"FDD modes: {args.n_modes}")
        if args.force_file is not None:
            raw_force = np.loadtxt(args.force_file, delimiter=",", ndmin=1)
            force = raw_force[:, -1] if raw_force.ndim > 1 else raw_force
            frf = estimate_frf(displacement_all, force, timestamps, nperseg=args.nperseg)
            store.write_array("vibration/frf/frequencies", frf.frequencies, units="Hz")
            store.write_array("vibration/frf/H1", frf.H1)
            store.write_array("vibration/frf/coherence", frf.coherence)
            print("FRF written")
        if frequencies.size:
            try:
                damping = half_power_damping(spec.frequencies, spec.amplitude)
                store.write_array("vibration/damping/peak_frequency_hz", np.array(damping.peak_frequency_hz), units="Hz")
                store.write_array("vibration/damping/left_frequency_hz", np.array(damping.left_frequency_hz), units="Hz")
                store.write_array("vibration/damping/right_frequency_hz", np.array(damping.right_frequency_hz), units="Hz")
                store.write_array("vibration/damping/ratio", np.array(damping.damping_ratio))
            except (ValueError, IndexError):
                pass
        analysis_hash = config_hash(
            {
                "direction": args.direction,
                "min_frequency": args.min_frequency,
                "max_frequency": args.max_frequency,
                "max_peaks": args.max_peaks,
                "frequency": selected_frequency,
                "n_harmonics": args.n_harmonics,
                "fdd": args.fdd,
                "n_modes": args.n_modes,
                "nperseg": args.nperseg,
                "force_file": args.force_file,
                "use_rbm": args.use_rbm,
            }
        )
        store.mark_stage_done("analyze", analysis_hash)
        print(f"written: {out}")
        for f, a in zip(frequencies, amplitudes):
            print(f"{f:.6g} Hz  amplitude {a:.6g} mm")


if __name__ == "__main__":
    main()
