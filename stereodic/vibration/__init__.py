"""Frequency-domain utilities for StereoDIC displacement sequences."""

from .spectrum import Spectrum, dominant_peaks, one_sided_spectrum, welch_spectrum
from .damping import HalfPowerResult, half_power_damping
from .fdd import FDDResult, frequency_domain_decomposition
from .frf import FRFResult, estimate_frf
from .harmonic import HarmonicFit, fit_harmonic
from .mac import mac_matrix, modal_assurance
from .ods import ODSResult, extract_ods

__all__ = [
    "Spectrum",
    "dominant_peaks",
    "one_sided_spectrum",
    "welch_spectrum",
    "HarmonicFit",
    "fit_harmonic",
    "ODSResult",
    "extract_ods",
    "FDDResult",
    "frequency_domain_decomposition",
    "FRFResult",
    "estimate_frf",
    "HalfPowerResult",
    "half_power_damping",
    "modal_assurance",
    "mac_matrix",
]
