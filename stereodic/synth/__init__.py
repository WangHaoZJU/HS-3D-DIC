"""Independent synthetic speckle images with exact ground truth.

The renderer uses an analytic Gaussian-blob texture with supersampled pixel
integration. It deliberately shares nothing with the DIC interpolation code
(no B-splines), so synthetic benchmarks are not an "inverse crime".
"""

from .render import Affine, SensorModel, Translation, render_frame, render_transmission, s1_translation
from .speckle import SpeckleTexture

__all__ = [
    "Affine",
    "SensorModel",
    "SpeckleTexture",
    "Translation",
    "render_frame",
    "render_transmission",
    "s1_translation",
]
