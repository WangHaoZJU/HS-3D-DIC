from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile

from stereodic.core.status import Status
from stereodic.core.types import ROI, CameraModel, StereoCalibration
from stereodic.io.readers.tiff import TiffSequence, natural_key
from stereodic.io.sequence import frame_indices, open_sequence
from stereodic.io.storage import SCHEMA_VERSION, ResultStore


def _write_frames(directory: Path, n: int, shape=(12, 16), dtype=np.uint16) -> list[np.ndarray]:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):
        img = rng.integers(0, 4096, size=shape).astype(dtype)
        tifffile.imwrite(directory / f"frame_{i}.tif", img)
        frames.append(img)
    return frames


def test_natural_sort():
    names = ["f10.tif", "f2.tif", "f1.tif"]
    assert sorted(names, key=natural_key) == ["f1.tif", "f2.tif", "f10.tif"]


def test_tiff_sequence_read(tmp_path: Path):
    frames = _write_frames(tmp_path / "seq", 12)
    seq = open_sequence(tmp_path / "seq")
    assert isinstance(seq, TiffSequence)
    assert len(seq) == 12 and seq.frame_shape == (12, 16) and seq.dtype == np.uint16
    assert seq.frame_name(10) == "frame_10.tif"  # natural order, not lexicographic
    np.testing.assert_array_equal(seq[10], frames[10])
    np.testing.assert_array_equal(seq[-1], frames[11])
    roi = ROI(2, 3, 5, 4)
    np.testing.assert_array_equal(seq.read(4, roi), frames[4][3:7, 2:7])
    with pytest.raises(ValueError):
        seq.read(0, ROI(10, 0, 10, 4))
    with pytest.raises(IndexError):
        seq.read(12)


@pytest.mark.parametrize("workers", [1, 3])
def test_iter_frames_order(tmp_path: Path, workers: int):
    frames = _write_frames(tmp_path / "seq", 9)
    seq = TiffSequence(tmp_path / "seq")
    idx = [5, 0, 8, 3]
    got = list(seq.iter_frames(idx, workers=workers, prefetch=2))
    assert [i for i, _ in got] == idx
    for i, img in got:
        np.testing.assert_array_equal(img, frames[i])


def test_sequence_errors(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        TiffSequence(tmp_path / "empty")
    with pytest.raises(ValueError):
        open_sequence(tmp_path / "nope.cine")
    assert list(frame_indices(10, 2, None, 3)) == [2, 5, 8]
    assert list(frame_indices(10, 0, 100)) == list(range(10))
    with pytest.raises(ValueError):
        frame_indices(10, 10)


# ------------------------------------------------------------------------ storage
def test_store_metadata_and_timeseries(tmp_path: Path):
    path = tmp_path / "out" / "r.h5"
    snap = {"name": "t", "dic": {"poi": {"step": 5}}}
    with ResultStore.create(path, config_snapshot=snap, input_manifest={"left": "a"}) as st:
        xy = st.create_timeseries("dic/left/xy", 100, 5000, 2)
        status = st.create_timeseries("dic/left/status", 100, 5000, dtype=np.uint8)
        xy[3] = np.ones((5000, 2))
        assert xy.chunks == (64, 2048, 2)
        assert xy.compression == "lzf"
        with pytest.raises(ValueError):
            st.create_timeseries("dic/left/xy", 1, 1, 2)
    with ResultStore.open(path, "r") as st:
        meta = st.metadata()
        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["config_snapshot"] == snap
        assert meta["input_manifest"] == {"left": "a"}
        assert meta["units"]["length"] == "mm"
        data = st.read_array("dic/left/xy")
        assert np.isnan(data[0]).all() and (data[3] == 1).all()
        assert (st.read_array("dic/left/status") == Status.NOT_COMPUTED).all()
    with pytest.raises(FileExistsError):
        ResultStore.create(path)


def test_store_stage_progress(tmp_path: Path):
    with ResultStore.create(tmp_path / "r.h5") as st:
        assert not st.stage_done("temporal", "abc")
        st.mark_stage_done("temporal", "abc")
        assert st.stage_done("temporal", "abc")
        assert not st.stage_done("temporal", "def")  # different config -> recompute
        st.invalidate_stage("temporal")
        assert not st.stage_done("temporal", "abc")


def test_store_calibration_roundtrip(tmp_path: Path):
    K = np.array([[1800.0, 0, 500], [0, 1801.0, 400], [0, 0, 1]])
    cam_l = CameraModel(K=K, dist=[0.1, -0.2, 1e-3, 2e-3, 0.05], image_size=(1024, 800))
    cam_r = CameraModel(K=K * [[1.01], [1], [1]], dist=np.zeros(5), image_size=(1024, 800))
    c, s = np.cos(0.4), np.sin(0.4)
    calib = StereoCalibration(cam_l, cam_r, R_RL=[[c, 0, -s], [0, 1, 0], [s, 0, c]], t_RL=[-120.0, 1.0, 30.0])
    with ResultStore.create(tmp_path / "r.h5") as st:
        st.write_calibration(calib, roi_origin=(0, 0))
        st.write_calibration(calib)  # overwrite allowed
    with ResultStore.open(tmp_path / "r.h5", "r") as st:
        back = st.read_calibration()
    np.testing.assert_array_equal(back.left.dist, cam_l.dist)
    np.testing.assert_array_equal(back.right.K, cam_r.K)
    np.testing.assert_array_equal(back.R_RL, calib.R_RL)
    assert back.left.image_size == (1024, 800)


def test_store_rejects_other_schema(tmp_path: Path):
    p = tmp_path / "r.h5"
    with h5py.File(p, "w") as f:
        f.require_group("metadata").attrs["schema_version"] = "999"
    with pytest.raises(ValueError, match="schema_version"):
        ResultStore.open(p)
