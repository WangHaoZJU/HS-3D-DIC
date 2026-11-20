from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from stereodic.core.config import (
    DEFAULTS_DIR,
    REPO_ROOT,
    DICConfig,
    config_hash,
    deep_merge,
    load_experiment,
    load_local_config,
)
from stereodic.core.status import Status, status_counts
from stereodic.core.types import ROI, CameraModel, DICBatchResult, POISet, StereoCalibration

EXAMPLE = REPO_ROOT / "configs" / "experiments" / "example.yaml"


# ------------------------------------------------------------------------- status
def test_status_codes_are_frozen():
    # On-disk format: these values must never change.
    expected = {
        "NOT_COMPUTED": 0, "OK": 1, "OUT_OF_BOUNDS": 2, "LOW_TEXTURE": 3, "ILL_CONDITIONED": 4,
        "NOT_CONVERGED": 5, "HIGH_RESIDUAL": 6, "GEOMETRY_INCONSISTENT": 7, "TEMPORAL_JUMP": 8, "MASKED": 9,
    }
    assert {s.name: int(s) for s in Status} == expected


def test_status_counts():
    counts = status_counts(np.array([1, 1, 5, 200], dtype=np.uint8))
    assert counts == {"OK": 2, "NOT_CONVERGED": 1, "UNKNOWN_200": 1}


# -------------------------------------------------------------------------- types
def test_roi_roundtrip_and_slices():
    roi = ROI(10, 20, 30, 40)
    xy = np.array([[0.5, 1.25], [29.0, 39.0]])
    np.testing.assert_array_equal(roi.to_roi(roi.to_full(xy)), xy)
    img = np.arange(100 * 100).reshape(100, 100)
    crop = img[roi.slices]
    assert crop.shape == (40, 30)
    assert crop[0, 0] == img[20, 10]
    assert ROI.from_list(None) is None
    with pytest.raises(ValueError):
        ROI(0, 0, 0, 5)


def _calib() -> StereoCalibration:
    K = np.array([[2000.0, 0, 512], [0, 2000.0, 512], [0, 0, 1]])
    cam = CameraModel(K=K, dist=np.zeros(5), image_size=(1024, 1024))
    a = np.deg2rad(30)
    R = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    return StereoCalibration(left=cam, right=cam, R_RL=R, t_RL=np.array([-150.0, 0, 40.0]))


def test_stereo_calibration_dict_roundtrip():
    c = _calib()
    c2 = StereoCalibration.from_dict(c.to_dict())
    np.testing.assert_array_equal(c2.R_RL, c.R_RL)
    np.testing.assert_array_equal(c2.left.K, c.left.K)
    assert c2.right.image_size == (1024, 1024)
    assert c.baseline_mm == pytest.approx(np.hypot(150, 40))


def test_calibration_validation():
    with pytest.raises(ValueError):
        CameraModel(K=np.eye(2), dist=np.zeros(5), image_size=(10, 10))
    with pytest.raises(ValueError):
        CameraModel(K=np.eye(3), dist=np.zeros(3), image_size=(10, 10))
    cam = _calib().left
    with pytest.raises(ValueError):
        StereoCalibration(left=cam, right=cam, R_RL=2 * np.eye(3), t_RL=np.zeros(3))
    with pytest.raises(ValueError):
        StereoCalibration(left=cam, right=cam, R_RL=np.diag([1.0, 1.0, -1.0]), t_RL=np.zeros(3))


def test_poiset_and_batch_result():
    pois = POISet(point_ids=[0, 1, 2], xy=[[1, 2], [3, 4], [5, 6]])
    assert len(pois) == 3 and pois.xy.dtype == np.float64
    with pytest.raises(ValueError):
        POISet(point_ids=[0, 0], xy=[[1, 2], [3, 4]])
    res = DICBatchResult.empty(4, n_params=6)
    assert res.xy.shape == (4, 2) and np.isnan(res.xy).all()
    assert res.status.dtype == np.uint8 and (res.status == Status.NOT_COMPUTED).all()
    assert res.params.shape == (4, 6)
    assert not res.valid.any()


# ------------------------------------------------------------------------- config
def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 10}})
    assert out == {"a": {"b": 10, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


def test_defaults_file_matches_model():
    # configs/defaults/dic.yaml must validate and equal the model defaults
    from stereodic.core.config import load_yaml

    assert DICConfig.model_validate(load_yaml(DEFAULTS_DIR / "dic.yaml")) == DICConfig()


def test_load_example_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("STEREODIC_DATA_ROOT", raising=False)
    monkeypatch.delenv("STEREODIC_RESULTS_ROOT", raising=False)
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({"data_root": str(tmp_path / "data"), "results_root": str(tmp_path / "res")}))
    cfg = load_experiment(EXAMPLE, local_path=local)
    assert cfg.name == "example"
    assert cfg.dic.poi.subset_size == 31
    assert cfg.dic.backend == "auto"
    assert cfg.data_path(cfg.cameras.left.path) == tmp_path / "data" / "example" / "left"
    assert cfg.results_path(cfg.output_file) == tmp_path / "res" / "example.h5"
    # snapshot is JSON-serialisable and reloads to an equal config
    assert type(cfg).model_validate(cfg.snapshot()) == cfg


def test_experiment_overrides_and_validation(tmp_path: Path):
    base = yaml.safe_load(EXAMPLE.read_text())
    base["dic"] = {"poi": {"step": 7}, "backend": "numba"}
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(base))
    cfg = load_experiment(p, local_path=tmp_path / "missing.yaml")
    assert cfg.dic.poi.step == 7
    assert cfg.dic.poi.subset_size == 31  # from defaults
    assert cfg.dic.backend == "numba"

    for bad in ({"poi": {"subset_size": 30}}, {"backend": "mps"}, {"unknown_key": 1}):
        base["dic"] = bad
        p.write_text(yaml.safe_dump(base))
        with pytest.raises(ValidationError):
            load_experiment(p, local_path=tmp_path / "missing.yaml")


def test_relative_path_needs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("STEREODIC_DATA_ROOT", raising=False)
    cfg = load_experiment(EXAMPLE, local_path=tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="data_root"):
        cfg.data_path(Path("x"))
    assert cfg.data_path(Path("/abs/x")) == Path("/abs/x")


def test_env_overrides_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({"data_root": "/from/file"}))
    monkeypatch.setenv("STEREODIC_DATA_ROOT", "/from/env")
    assert load_local_config(local).data_root == Path("/from/env")


def test_config_hash_stable_and_sensitive():
    a, b = DICConfig(), DICConfig()
    assert config_hash(a) == config_hash(b)
    b.poi.step = 6
    assert config_hash(a) != config_hash(b)
