"""Continuous speckle texture: a sum of anisotropic Gaussian dark blobs.

Optical density at material point X:
    S(X) = sum_k a_k * exp(-0.5 (X - c_k)^T Sigma_k^{-1} (X - c_k))
Transmission (relative brightness) t = exp(-S) in (0, 1]. A smooth deformation
maps each blob to a new Gaussian (centre -> phi(c_k), Sigma -> J Sigma J^T), which
is exact for affine maps and second-order accurate otherwise (blobs are small).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeckleTexture:
    centers: np.ndarray  # (M, 2) material coordinates [px of the reference image]
    covariances: np.ndarray  # (M, 2, 2)
    amplitudes: np.ndarray  # (M,) optical density at blob centre

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=np.float64).reshape(-1, 2)
        self.covariances = np.asarray(self.covariances, dtype=np.float64).reshape(-1, 2, 2)
        self.amplitudes = np.asarray(self.amplitudes, dtype=np.float64).ravel()
        m = self.centers.shape[0]
        if self.covariances.shape[0] != m or self.amplitudes.shape[0] != m:
            raise ValueError("centers, covariances and amplitudes must have the same length")

    def __len__(self) -> int:
        return int(self.centers.shape[0])

    @classmethod
    def random(
        cls,
        extent: tuple[float, float, float, float],
        diameter_px: float = 4.0,
        coverage: float = 0.5,
        amplitude: float = 3.0,
        size_jitter: float = 0.2,
        anisotropy: float = 0.15,
        seed: int | None = None,
    ) -> SpeckleTexture:
        """Poisson-distributed speckles over extent = (x0, y0, x1, y1).

        diameter_px: nominal speckle diameter (2r, where the blob falls to e^-1 at r).
        coverage: target fraction of the area covered by blobs, via 1 - exp(-lambda*pi*r^2).
        size_jitter: relative std of the radius; anisotropy: max relative axis difference.
        """
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1)")
        rng = np.random.default_rng(seed)
        x0, y0, x1, y1 = extent
        area = (x1 - x0) * (y1 - y0)
        r = diameter_px / 2.0
        density = -np.log(1.0 - coverage) / (np.pi * r * r)
        m = rng.poisson(density * area)
        centers = np.column_stack([rng.uniform(x0, x1, m), rng.uniform(y0, y1, m)])
        radii = r * np.clip(1.0 + size_jitter * rng.standard_normal(m), 0.4, 2.0)
        aspect = 1.0 + anisotropy * rng.uniform(-1.0, 1.0, m)
        theta = rng.uniform(0.0, np.pi, m)
        # exp(-|d|^2/r^2) == exp(-0.5 d^T Sigma^-1 d) with Sigma = r^2/2 I
        s1 = (radii * aspect) ** 2 / 2.0
        s2 = (radii / aspect) ** 2 / 2.0
        c, s = np.cos(theta), np.sin(theta)
        cov = np.empty((m, 2, 2))
        cov[:, 0, 0] = c * c * s1 + s * s * s2
        cov[:, 1, 1] = s * s * s1 + c * c * s2
        cov[:, 0, 1] = cov[:, 1, 0] = c * s * (s1 - s2)
        amps = amplitude * np.clip(1.0 + 0.1 * rng.standard_normal(m), 0.5, 1.5)
        return cls(centers=centers, covariances=cov, amplitudes=amps)

    def density_at(self, points: np.ndarray) -> np.ndarray:
        """Exact optical density S at material points (N, 2). O(N*M); for tests only."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        inv = np.linalg.inv(self.covariances)
        d = p[:, None, :] - self.centers[None, :, :]
        q = np.einsum("nmi,mij,nmj->nm", d, inv, d)
        return (self.amplitudes[None, :] * np.exp(-0.5 * q)).sum(axis=1)
