from __future__ import annotations

import numpy as np

from stereodic.dic.icgn import ICGNOptions
from stereodic.dic.mesh import make_poi_grid
from stereodic.dic.temporal import TemporalMatchOptions, track_sequence
from stereodic.synth import SensorModel, SpeckleTexture, Translation, render_frame


def test_fixed_reference_temporal_tracks_known_shifts():
    shape = (100, 120)
    texture = SpeckleTexture.random((-20, -20, 140, 120), seed=11)
    sensor = SensorModel(bits=12)
    reference = render_frame(texture, shape, None, sensor, oversample=4, seed=1)
    shifts = np.array([0.0, 0.31, -0.22, 0.48])
    frames = [
        (k, render_frame(texture, shape, Translation(u, 0.0), sensor, oversample=4, seed=2 + k))
        for k, u in enumerate(shifts)
    ]
    pois = make_poi_grid(shape, 31, 15, margin=10)
    result = track_sequence(
        reference,
        frames,
        pois.xy,
        TemporalMatchOptions(ICGNOptions(subset_size=31, shape_order=1), max_znssd=0.8),
        backend="numba",
        reference_frame=0,
    )
    assert result.xy.shape == (4, len(pois), 2)
    assert result.valid.all()
    truth = pois.xy[None, :, :] + shifts[:, None, None] * np.array([1.0, 0.0])
    np.testing.assert_allclose(result.xy, truth, atol=0.01)
