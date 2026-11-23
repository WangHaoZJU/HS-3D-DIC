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


def test_pair_timestamps_handles_independent_drops_with_equal_length():
    left = np.array([0.0, 0.2, 0.3, 0.4])
    right = np.array([0.0, 0.1, 0.2, 0.4])
    report = pair_timestamps(left, right, max_delta_seconds=0.01)
    np.testing.assert_array_equal(report.left_indices, [0, 1, 3])
    np.testing.assert_array_equal(report.right_indices, [0, 2, 3])
    np.testing.assert_allclose(report.delta_seconds, 0.0, atol=1e-15)


def test_pair_timestamps_avoids_greedy_shift_with_leading_extra_frame():
    left = np.array([0.0, 0.1, 0.2])
    right = np.array([0.1, 0.2])
    report = pair_timestamps(left, right)
    np.testing.assert_array_equal(report.left_indices, [1, 2])
    np.testing.assert_array_equal(report.right_indices, [0, 1])
    np.testing.assert_allclose(report.delta_seconds, 0.0, atol=1e-15)
