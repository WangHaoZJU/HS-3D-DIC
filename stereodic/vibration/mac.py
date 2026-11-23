"""Modal assurance criterion utilities."""

from __future__ import annotations

import numpy as np


def modal_assurance(phi_a: np.ndarray, phi_b: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute MAC between two mode vectors, allowing complex values."""
    a = np.asarray(phi_a, dtype=np.complex128)
    b = np.asarray(phi_b, dtype=np.complex128)
    numerator = np.abs(np.sum(np.conj(a) * b, axis=axis)) ** 2
    denominator = np.real(np.sum(np.conj(a) * a, axis=axis) * np.sum(np.conj(b) * b, axis=axis))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    return np.where(denominator > 0, out, np.nan)


def mac_matrix(modes_a: np.ndarray, modes_b: np.ndarray) -> np.ndarray:
    """Pairwise MAC matrix for arrays shaped ``(M, DOF)`` and ``(K, DOF)``."""
    a = np.asarray(modes_a, dtype=np.complex128)
    b = np.asarray(modes_b, dtype=np.complex128)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("mode arrays must have shape (n_modes, n_dof) with matching n_dof")
    cross = np.abs(a.conj() @ b.T) ** 2
    norms = np.sum(np.abs(a) ** 2, axis=1)[:, None] * np.sum(np.abs(b) ** 2, axis=1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = cross / norms
    return np.where(norms > 0, out, np.nan)
