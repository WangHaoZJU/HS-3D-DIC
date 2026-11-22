"""Identify dominant frequencies from reconstructed displacement."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stereodic.core.config import load_experiment
from stereodic.io.storage import ResultStore
from stereodic.vibration.spectrum import dominant_peaks, one_sided_spectrum


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--direction", type=int, choices=[0, 1, 2], default=2, help="local displacement component")
    ap.add_argument("--min-frequency", type=float, default=0.0)
    ap.add_argument("--max-frequency", type=float, default=None)
    ap.add_argument("--max-peaks", type=int, default=10)
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    if cfg.output_file is None:
        raise ValueError("experiment config has no output_file")
    out = cfg.results_path(cfg.output_file)
    with ResultStore.open(out, "r+") as store:
        if "reconstruction/displacement_local" not in store.h5:
            raise ValueError("reconstruction data is missing; run --stage reconstruct first")
        timestamps = store.read_array("time/timestamps")
        displacement = store.read_array("reconstruction/displacement_local")[:, :, args.direction]
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
        print(f"written: {out}")
        for f, a in zip(frequencies, amplitudes):
            print(f"{f:.6g} Hz  amplitude {a:.6g} mm")


if __name__ == "__main__":
    main()
