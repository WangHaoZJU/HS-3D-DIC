"""Progress reporting, cooperative cancel + resume, and stage_status (UI support, B1/B4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stereodic.core.config import load_experiment
from stereodic.core.progress import JobCancelled
from stereodic.core.status import Status
from stereodic.io.storage import ResultStore
from stereodic.pipeline import run_reconstruct, run_stereo, run_temporal, stage_status
from stereodic.synth.demo import DemoSpec, make_demo_dataset


class Recorder:
    def __init__(self, cancel_after: int | None = None) -> None:
        self.updates: list[tuple[int, int, str]] = []
        self.phases: list[str] = []
        self.summaries: list[dict] = []
        self.cancel_after = cancel_after

    def update(self, done: int, total: int, message: str = "") -> None:
        self.updates.append((done, total, message))

    def phase(self, name: str) -> None:
        self.phases.append(name)

    def summary(self, data: dict) -> None:
        self.summaries.append(data)

    def check_cancel(self) -> None:
        if self.cancel_after is not None and len(self.updates) >= self.cancel_after:
            raise JobCancelled()


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("demo")
    spec = DemoSpec(name="tiny", image_size=(320, 256), n_frames=6, n_calib_views=0, poi_step=24, oversample=2)
    make_demo_dataset(root / "data", spec, workspace=root / "ws", workers=1, exact_calibration=True)
    return root / "ws" / "experiment.yaml"


def test_cancel_resume_and_stage_status(demo: Path):
    cfg = load_experiment(demo, local_path=demo.parent / "no-local.yaml")
    st = stage_status(cfg)
    assert st["calibrate"]["state"] == "done"
    assert st["stereo"]["state"] == "missing"

    rec = Recorder()
    out = run_stereo(cfg, progress=rec)
    assert rec.phases[:2] == ["loading", "matching"]
    st = stage_status(cfg)
    assert st["stereo"]["state"] == "done"
    assert st["temporal"]["state"] == "missing"

    # cancel after 3 frames: those rows are flushed and the run is resumable
    rec = Recorder(cancel_after=3)
    with pytest.raises(JobCancelled):
        run_temporal(cfg, progress=rec)
    st = stage_status(cfg)
    assert st["temporal"]["state"] == "partial"
    assert st["temporal"]["fraction"] == pytest.approx(3 / 6)

    rec = Recorder()
    run_temporal(cfg, progress=rec)
    assert rec.updates[-1][:2] == (6, 6)
    assert {"valid_left", "valid_right", "znssd_median"} <= set(rec.summaries[-1])
    assert len(rec.summaries) == 3  # only the frames solved in this run
    with ResultStore.open(out, "r") as store:
        assert store.read_array("progress/dic_frames_done").all()
        assert np.all(store.read_array("dic/left/status") == Status.OK)

    run_reconstruct(cfg, progress=Recorder())
    st = stage_status(cfg)
    assert [st[s]["state"] for s in ("stereo", "temporal", "reconstruct")] == ["done"] * 3
    assert st["rbm"]["state"] == "missing"

    # a DIC setting change makes stereo and everything downstream stale
    cfg.dic.poi.step = 20
    st = stage_status(cfg)
    assert [st[s]["state"] for s in ("stereo", "temporal", "reconstruct")] == ["stale"] * 3
