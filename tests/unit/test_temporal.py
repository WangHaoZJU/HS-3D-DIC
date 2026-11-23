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


def test_temporal_fallback_recovers_large_jump_via_zncc():
    """A sudden jump well outside IC-GN's basin from p0=0 is recovered via ZNCC fallback."""
    shape = (120, 140)
    shift = (8.35, -5.2)
    texture = SpeckleTexture.random((-30, -30, 170, 150), seed=21)
    sensor = SensorModel(bits=12)
    reference = render_frame(texture, shape, None, sensor, oversample=4, seed=1)
    jumped = render_frame(texture, shape, Translation(*shift), sensor, oversample=4, seed=2)
    pois = make_poi_grid(shape, 31, 15, margin=18)

    opt_no_retry = TemporalMatchOptions(
        ICGNOptions(subset_size=31, shape_order=1), max_znssd=0.4, initialization="zero", retry_search_radius=0
    )
    res_no_retry = track_sequence(reference, [(0, reference), (1, jumped)], pois.xy, opt_no_retry, backend="numba")
    assert not res_no_retry.valid[1].all()

    opt_retry = TemporalMatchOptions(
        ICGNOptions(subset_size=31, shape_order=1), max_znssd=0.4, initialization="zero", retry_search_radius=16
    )
    res_retry = track_sequence(reference, [(0, reference), (1, jumped)], pois.xy, opt_retry, backend="numba")
    assert res_retry.valid[1].all()
    truth = pois.xy + np.asarray(shift)
    np.testing.assert_allclose(res_retry.xy[1], truth, atol=0.01)
