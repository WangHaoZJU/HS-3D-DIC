"""Analysis stages on reconstructed displacement (B2/B3): vibration, noise floor, uncertainty.

Moved out of scripts/analyze.py and scripts/report_uncertainty.py so the web UI and the
command line run the same code. All results go into the experiment results file:

    vibration/          spectra (all 3 local components), peaks, damping per peak,
                        harmonic/<key>/ (one group per fitted frequency, kept across runs),
                        fdd/ (singular values; first singular vector at each peak), frf/
    quality/noise_floor/  per-point static noise (std, mean, rms) and its spectrum
    uncertainty/<key>/    first-order (and Monte Carlo) triangulation uncertainty
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .core.config import ExperimentConfig, config_hash
from .core.progress import ProgressReporter, ensure_progress

log = logging.getLogger(__name__)


class AnalyzeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: Literal[0, 1, 2] = 2  # local component for spectrum / peaks / FDD: 0 c, 1 s, 2 n
    use_rbm: bool = False  # analyse rigid-body-compensated displacement
    min_frequency: float = Field(0.0, ge=0)
    max_frequency: float | None = Field(None, gt=0)
    max_peaks: int = Field(10, ge=1, le=50)
    nperseg: int | None = Field(None, ge=8)
    frequencies: list[float] = Field(default_factory=list)  # harmonic fit frequencies (e.g. the excitation)
    fit_peaks: int = Field(3, ge=0, le=20)  # without `frequencies`: fit the strongest spectral peaks
    refine: bool = True  # refine each fit frequency by maximizing the field harmonic energy
    n_harmonics: int = Field(1, ge=1, le=5)
    fdd: bool = True
    n_modes: int = Field(3, ge=1, le=10)
    force_file: str | None = None  # CSV: one force column per frame, or (time, force) resampled to the frames


def _key(f: float) -> str:
    return f"{f:.9g}".replace("-", "m").replace(".", "p")


def _results_path(cfg: ExperimentConfig) -> Path:
    if cfg.output_file is None:
        raise ValueError("experiment config has no output_file")
    out = cfg.results_path(cfg.output_file)
    if not out.exists():
        raise FileNotFoundError(f"results file does not exist: {out}")
    return out


def _displacement(store, use_rbm: bool) -> np.ndarray:
    h5 = store.h5
    if use_rbm:
        if "reconstruction/displacement_rbm_local" in h5:
            return store.read_array("reconstruction/displacement_rbm_local")
        raise ValueError("rigid-body-compensated displacement is missing; run the rbm stage first")
    if "reconstruction/displacement_local" not in h5:
        raise ValueError("reconstruction data is missing; run the reconstruct stage first")
    return store.read_array("reconstruction/displacement_local")


def load_force(path: str | Path, timestamps: np.ndarray) -> np.ndarray:
    """Force samples on the camera time axis.

    One column: one value per analysed frame. Two or more columns: first = time (s),
    last = force, linearly resampled to the frame timestamps (review: the force signal
    is usually recorded at a different rate than the cameras).
    """
    raw = np.loadtxt(path, delimiter="," if str(path).lower().endswith(".csv") else None, ndmin=2)
    t = np.asarray(timestamps, dtype=np.float64)
    if raw.shape[1] == 1:
        f = raw[:, 0]
        if f.size != t.size:
            raise ValueError(f"force file has {f.size} samples, expected {t.size} (one per frame) or (time, force) columns")
        return f
    tf, f = raw[:, 0], raw[:, -1]
    order = np.argsort(tf)
    tf, f = tf[order], f[order]
    if t[0] < tf[0] or t[-1] > tf[-1]:
        raise ValueError(f"force time axis [{tf[0]:g}, {tf[-1]:g}] s does not cover the frames [{t[0]:g}, {t[-1]:g}] s")
    return np.interp(t, tf, f)


def run_analyze(cfg: ExperimentConfig, options: AnalyzeOptions | None = None, progress: ProgressReporter | None = None) -> dict:
    """Spectrum, peaks, damping, harmonic fits / ODS, FDD and FRF (docs §11)."""
    from .io.storage import ResultStore
    from .vibration.damping import half_power_damping
    from .vibration.fdd import frequency_domain_decomposition
    from .vibration.frf import estimate_frf
    from .vibration.harmonic import refine_frequency
    from .vibration.ods import extract_ods
    from .vibration.spectrum import dominant_peaks, field_average_psd

    opt = options or AnalyzeOptions()
    progress = ensure_progress(progress)
    out = _results_path(cfg)
    summary: dict = {"options": opt.model_dump()}
    with ResultStore.open(out, "r+") as store:
        progress.phase("loading")
        t = store.read_array("time/timestamps")
        disp = _displacement(store, opt.use_rbm)  # (T, N, 3)
        d = opt.direction
        T = t.size
        progress.check_cancel()

        progress.phase("spectrum")
        spectra = [field_average_psd(disp[:, :, c], t, nperseg=opt.nperseg) for c in range(3)]
        spec = spectra[d]
        freqs, amps = dominant_peaks(
            spec, min_frequency_hz=opt.min_frequency, max_frequency_hz=opt.max_frequency, max_peaks=opt.max_peaks
        )
        g = "vibration/"
        if "vibration" in store.h5:
            for name in list(store.h5["vibration"].keys()):
                if name != "harmonic":
                    del store.h5["vibration"][name]
        store.write_array(g + "spectrum_frequencies", spec.frequencies, units="Hz")
        store.write_array(g + "spectrum_amplitude", spec.amplitude, units="mm", direction=d)
        store.write_array(g + "spectrum_power", spec.power, units="mm^2/Hz", direction=d)
        store.write_array(g + "spectrum_amplitude_all", np.stack([s.amplitude for s in spectra], axis=1), units="mm",
                          columns="c, s, n")
        store.write_array(g + "frequencies", freqs, units="Hz")
        store.write_array(g + "amplitudes", amps, units="mm")
        store.h5["vibration"].attrs["options"] = json.dumps(opt.model_dump(), sort_keys=True)
        store.h5["vibration"].attrs["sample_rate_hz"] = spec.sample_rate_hz
        store.h5["vibration"].attrs["nperseg"] = int(min(opt.nperseg or 1024, T))
        summary["peaks"] = [{"frequency_hz": float(f), "amplitude_mm": float(a)} for f, a in zip(freqs, amps)]
        progress.check_cancel()

        # damping from the full-length (finest resolution) field spectrum, flagged when resolution-limited
        progress.phase("damping")
        hi = field_average_psd(disp[:, :, d], t, nperseg=T)
        damp = []
        for f0 in freqs:
            k = int(np.argmin(np.abs(hi.frequencies - f0)))
            lo_k, hi_k = max(k - 3, 1), min(k + 4, hi.frequencies.size - 1)
            k = lo_k + int(np.argmax(hi.amplitude[lo_k:hi_k]))
            try:
                r = half_power_damping(hi.frequencies, hi.amplitude, peak_index=k)
                damp.append([r.peak_frequency_hz, r.left_frequency_hz, r.right_frequency_hz, r.damping_ratio,
                             float(r.resolution_limited)])
            except (ValueError, IndexError):
                damp.append([float(f0), np.nan, np.nan, np.nan, 1.0])
        store.write_array(g + "damping", np.asarray(damp).reshape(-1, 5),
                          columns="peak_hz, left_hz, right_hz, zeta, resolution_limited",
                          resolution_hz=float(hi.frequencies[1] - hi.frequencies[0]) if hi.frequencies.size > 1 else np.nan)
        progress.check_cancel()

        # harmonic fits / ODS at the excitation or the strongest peaks
        targets = list(opt.frequencies) or [float(f) for f in freqs[: opt.fit_peaks]]
        summary["harmonic"] = []
        for m, f0 in enumerate(targets):
            progress.phase(f"harmonic {f0:.6g} Hz")
            progress.update(m, len(targets))
            f_fit = refine_frequency(disp[:, :, d], t, f0) if opt.refine else f0
            ods = extract_ods(disp, t, f_fit, n_harmonics=opt.n_harmonics)
            base = f"{g}harmonic/{_key(f0)}"
            if base in store.h5:
                del store.h5[base]
            store.write_array(base + "/frequency_hz", np.array(f_fit), units="Hz")
            store.write_array(base + "/requested_frequency_hz", np.array(f0), units="Hz")
            store.write_array(base + "/response", ods.response)
            store.write_array(base + "/amplitude", ods.amplitude, units="mm")
            store.write_array(base + "/phase", ods.phase, units="rad")
            store.write_array(base + "/residual_rms", ods.residual_rms, units="mm")
            store.h5[base].attrs["direction"] = d
            store.h5[base].attrs["use_rbm"] = opt.use_rbm
            amp_d = ods.amplitude[0, :, d]
            summary["harmonic"].append({"requested_hz": f0, "frequency_hz": f_fit, "key": _key(f0),
                                        "max_amplitude_mm": float(np.nanmax(amp_d)) if np.isfinite(amp_d).any() else None})
            progress.check_cancel()

        if opt.fdd:
            progress.phase("fdd")
            fdd = frequency_domain_decomposition(disp[:, :, d], t, nperseg=opt.nperseg, n_modes=opt.n_modes)
            store.write_array(g + "fdd/frequencies", fdd.frequencies, units="Hz")
            store.write_array(g + "fdd/singular_values", fdd.singular_values)
            # first singular vector at each spectral peak only: the full (F, modes, N) array is far too large
            peak_bins = [int(np.argmin(np.abs(fdd.frequencies - f))) for f in freqs]
            store.write_array(g + "fdd/peak_frequencies", fdd.frequencies[peak_bins] if peak_bins else np.empty(0), units="Hz")
            store.write_array(g + "fdd/peak_mode_shapes",
                              fdd.mode_shapes[peak_bins, 0, :] if peak_bins else np.empty((0, disp.shape[1]), complex))
            progress.check_cancel()

        if opt.force_file:
            progress.phase("frf")
            force = load_force(cfg.data_path(Path(opt.force_file)), t)
            frf = estimate_frf(disp[:, :, d], force, t, nperseg=opt.nperseg)
            store.write_array(g + "frf/frequencies", frf.frequencies, units="Hz")
            store.write_array(g + "frf/H1_mean_abs", np.nanmean(np.abs(frf.H1), axis=1), units="mm/N")
            store.write_array(g + "frf/coherence_mean", np.nanmean(frf.coherence, axis=1))
            bins = [int(np.argmin(np.abs(frf.frequencies - f))) for f in freqs]
            store.write_array(g + "frf/peak_H1", frf.H1[bins] if bins else np.empty((0, disp.shape[1]), complex))

        store.mark_stage_done("analyze", config_hash(opt.model_dump()))
    return summary


def run_noise_floor(cfg: ExperimentConfig, use_rbm: bool = False, progress: ProgressReporter | None = None) -> dict:
    """Static noise floor (E1, docs §10.2): run on an acquisition of the blade at rest."""
    from .io.storage import ResultStore
    from .quality.noise_floor import estimate_noise_floor
    from .vibration.spectrum import field_average_psd

    progress = ensure_progress(progress)
    out = _results_path(cfg)
    with ResultStore.open(out, "r+") as store:
        progress.phase("loading")
        t = store.read_array("time/timestamps")
        disp = _displacement(store, use_rbm)
        valid = store.read_array("quality/valid_mask") if "quality/valid_mask" in store.h5 else None
        progress.phase("statistics")
        nf = estimate_noise_floor(disp, valid)
        g = "quality/noise_floor/"
        if "quality/noise_floor" in store.h5:
            del store.h5["quality/noise_floor"]
        store.write_array(g + "std", nf.standard_deviation, units="mm", columns="c, s, n")
        store.write_array(g + "mean", nf.mean, units="mm")
        store.write_array(g + "rms", nf.rms, units="mm")
        store.write_array(g + "valid_samples", nf.valid_samples)
        progress.phase("spectrum")
        spectra = [field_average_psd(disp[:, :, c], t) for c in range(3)]
        store.write_array(g + "spectrum_frequencies", spectra[0].frequencies, units="Hz")
        store.write_array(g + "spectrum_amplitude", np.stack([s.amplitude for s in spectra], axis=1), units="mm")
        med = np.nanmedian(nf.standard_deviation, axis=0)
        store.h5["quality/noise_floor"].attrs["use_rbm"] = use_rbm
        store.h5["quality/noise_floor"].attrs["n_frames"] = int(t.size)
        store.h5["quality/noise_floor"].attrs["median_std_mm"] = json.dumps(med.tolist())
        store.mark_stage_done("noise_floor", config_hash({"use_rbm": use_rbm}))
    return {"median_std_mm": med.tolist(), "n_frames": int(t.size)}


def run_uncertainty(
    cfg: ExperimentConfig,
    pixel_sigma: float = 0.01,
    monte_carlo: int = 0,
    seed: int = 0,
    frame_index: int | None = None,
    progress: ProgressReporter | None = None,
) -> dict:
    """First-order (and optional Monte Carlo) triangulation uncertainty for a pixel noise level."""
    from .calibration.report import load_calibration
    from .io.storage import ResultStore
    from .uncertainty.monte_carlo import monte_carlo_triangulation
    from .uncertainty.propagation import triangulation_uncertainty

    if pixel_sigma <= 0:
        raise ValueError("pixel_sigma must be positive")
    if cfg.calibration_file is None:
        raise ValueError("experiment config needs calibration_file")
    progress = ensure_progress(progress)
    out = _results_path(cfg)
    calibration = load_calibration(cfg.data_path(cfg.calibration_file))
    cov = np.eye(2) * pixel_sigma**2
    with ResultStore.open(out, "r+") as store:
        if frame_index is None:
            left = store.read_array("reference/points_left")
            right = store.read_array("reference/points_right")
            key = "reference"
        else:
            left = store.h5["dic/left/xy"][frame_index] + np.asarray(cfg.cameras.left.sensor_origin)
            right = store.h5["dic/right/xy"][frame_index] + np.asarray(cfg.cameras.right.sensor_origin)
            key = f"frame_{frame_index}"
        progress.phase("propagation")
        res = triangulation_uncertainty(calibration, left, right, cov)
        base = f"uncertainty/{key}"
        if base in store.h5:
            del store.h5[base]
        store.write_array(base + "/covariance_xyz", res.covariance, units="mm^2")
        store.write_array(base + "/std_xyz", res.standard_deviation, units="mm")
        store.write_array(base + "/jacobian", res.jacobian, units="mm/px")
        store.write_array(base + "/valid", res.valid)
        store.h5[base].attrs["pixel_sigma"] = pixel_sigma
        out_summary = {"key": key, "median_std_mm": np.nanmedian(res.standard_deviation, axis=0).tolist()}
        if monte_carlo > 0:
            progress.phase("monte carlo")
            mc = monte_carlo_triangulation(calibration, left, right, cov, samples=monte_carlo, seed=seed)
            store.write_array(base + "/monte_carlo_mean", mc.mean, units="mm")
            store.write_array(base + "/monte_carlo_covariance", mc.covariance, units="mm^2")
            store.write_array(base + "/monte_carlo_std", mc.standard_deviation, units="mm")
            store.write_array(base + "/monte_carlo_valid_samples", mc.valid_samples)
            store.h5[base].attrs["monte_carlo"] = monte_carlo
            out_summary["monte_carlo_median_std_mm"] = np.nanmedian(mc.standard_deviation, axis=0).tolist()
        store.mark_stage_done(
            "uncertainty", config_hash({"pixel_sigma": pixel_sigma, "frame_index": frame_index, "monte_carlo": monte_carlo, "seed": seed})
        )
    return out_summary
