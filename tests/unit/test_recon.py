"""Triangulation."""

from __future__ import annotations

import numpy as np

from stereodic.calibration.camera_model import project_stereo
from stereodic.core.status import Status
from stereodic.recon.triangulate import triangulate
from stereodic.synth.stereo import make_stereo_scene


def test_triangulation_exact_and_noisy():
    cal = make_stereo_scene().calibration()
    X = np.random.default_rng(0).uniform([-15, -12, 230], [15, 12, 270], (500, 3))
    pl, pr = project_stereo(cal, X)
    tri = triangulate(cal, pl, pr)
    assert (tri.status == Status.OK).all()
    np.testing.assert_allclose(tri.X, X, atol=1e-9)
    assert tri.reprojection_error.max() < 1e-8
    assert np.all((tri.angle_deg > 15) & (tri.angle_deg < 25))
    # epipolar inconsistency shows up in the reprojection error
    pr_bad = pr.copy()
    pr_bad[:10, 1] += 2.0  # vertical shift ~ across the (near-horizontal) epipolar lines
    tri = triangulate(cal, pl, pr_bad, max_reprojection_px=0.5)
    assert (tri.status[:10] == Status.GEOMETRY_INCONSISTENT).all() and (tri.status[10:] == Status.OK).all()
    assert np.all(tri.reprojection_error[:10] > 0.6)


def test_triangulation_handles_nan():
    cal = make_stereo_scene().calibration()
    X = np.array([[0.0, 0.0, 250.0], [1.0, 2.0, 252.0]])
    pl, pr = project_stereo(cal, X)
    pl[1] = np.nan
    tri = triangulate(cal, pl, pr)
    assert tri.status.tolist() == [Status.OK, Status.NOT_COMPUTED]
    assert np.isnan(tri.X[1]).all()
