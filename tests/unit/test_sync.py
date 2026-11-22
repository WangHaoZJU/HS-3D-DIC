from __future__ import annotations

import numpy as np
import pytest

from stereodic.io.sync import pair_timestamps, regular_timestamps, validate_timestamps


def test_pair_timestamps_and_regular_fallback():
    left = regular_timestamps(4, 1000.0)
    right = left + 2e-6
    report = pair_timestamps(left, right, max_delta_seconds=5e-6)
    np.testing.assert_array_equal(report.left_indices, np.arange(4))
    np.testing.assert_array_equal(report.right_indices, np.arange(4))
    np.testing.assert_allclose(report.delta_seconds, 2e-6)


def test_timestamps_must_increase():
    with pytest.raises(ValueError):
        validate_timestamps(np.array([0.0, 0.1, 0.1]))
