"""Web server API: workspace, configuration, file browser, jobs (U0)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from stereodic.synth.demo import DemoSpec, make_demo_dataset  # noqa: E402
from stereodic_server.app import create_app  # noqa: E402
from stereodic_server.settings import Settings  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path):
    data, results = tmp_path / "data", tmp_path / "results"
    data.mkdir()
    (data / "seq" / "left").mkdir(parents=True)
    (data / "seq" / "left" / "a.tif").write_bytes(b"")
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({"data_root": str(data), "results_root": str(results)}))
    settings = Settings.from_env(local)
    settings.state_dir = tmp_path / "state"
    with TestClient(create_app(settings)) as c:
        yield c


def _wait(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise TimeoutError(job)


def test_experiment_crud_and_validation(client: TestClient):
    assert client.get("/api/experiments").json() == []
    r = client.post("/api/experiments", json={"name": "exp1", "description": "first"})
    assert r.status_code == 201, r.text
    detail = r.json()
    assert detail["config"]["dic"]["poi"]["subset_size"] == 31  # defaults merged
    assert detail["config"]["output_file"].endswith("exp1/results.h5")
    assert client.post("/api/experiments", json={"name": "exp1"}).status_code == 409
    assert client.post("/api/experiments", json={"name": "../x"}).status_code == 400

    bad = dict(detail["config"])
    bad["dic"] = {**bad["dic"], "poi": {"subset_size": 30, "step": 5}}
    v = client.post("/api/experiments/exp1/validate", json={"config": bad}).json()
    assert not v["valid"] and v["errors"][0]["loc"][-1] == "subset_size"
    assert client.put("/api/experiments/exp1/config", json={"config": bad}).status_code == 422

    good = dict(detail["config"])
    good["dic"] = {**good["dic"], "poi": {"subset_size": 41, "step": 7}}
    assert client.post("/api/experiments/exp1/validate", json={"config": good}).json()["valid"]
    saved = client.put("/api/experiments/exp1/config", json={"config": good}).json()
    assert saved["config"]["dic"]["poi"] == {"subset_size": 41, "step": 7}
    assert "subset_size: 41" in client.get("/api/experiments/exp1/export.yaml").text

    y = client.get("/api/experiments/exp1").json()["yaml"].replace("subset_size: 41", "subset_size: 51")
    assert client.put("/api/experiments/exp1/config", json={"yaml": y}).json()["config"]["dic"]["poi"]["subset_size"] == 51
    bad_yaml = client.post("/api/experiments/exp1/validate", json={"yaml": "a: [1"}).json()
    assert not bad_yaml["valid"]

    copy = client.post("/api/experiments", json={"name": "exp2", "copy_from": "exp1"}).json()
    assert copy["config"]["output_file"].endswith("exp2/results.h5")
    assert copy["config"]["calibration_file"] == saved["config"]["calibration_file"]  # shared

    st = client.get("/api/experiments/exp1/status").json()
    assert st["stages"]["stereo"]["state"] == "missing" and not st["busy"]
    assert [e["name"] for e in client.get("/api/experiments").json()] == ["exp1", "exp2"]
    assert client.delete("/api/experiments/exp2").status_code == 200
    assert [e["name"] for e in client.get("/api/experiments").json()] == ["exp1"]
    assert "properties" in client.get("/api/schema/experiment").json()


def test_file_browser_is_sandboxed(client: TestClient):
    r = client.get("/api/fs", params={"root": "data", "path": "seq/left"}).json()
    assert r["n_images"] == 1 and r["parent"] == "seq"
    assert client.get("/api/fs", params={"root": "data", "path": "../"}).status_code == 403
    assert client.get("/api/fs", params={"root": "nope"}).status_code == 400
    assert client.get("/api/system").json()["roots"]["data"]


def test_noop_job_progress_cancel_and_events(client: TestClient):
    with client.websocket_connect("/api/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        job = client.post("/api/jobs", json={"stage": "noop", "params": {"seconds": 0.5, "steps": 5}}).json()
        types = set()
        while True:
            msg = ws.receive_json()
            types.add(msg["type"])
            if msg["type"] == "job" and msg["job"]["id"] == job["id"] and msg["job"]["state"] == "succeeded":
                break
        assert {"job", "progress", "summary"} <= types
    done = _wait(client, job["id"])
    assert done["done"] == done["total"] == 5
    assert "stage noop finished" in client.get(f"/api/jobs/{job['id']}/log").json()["text"]

    slow = client.post("/api/jobs", json={"stage": "noop", "params": {"seconds": 30, "steps": 300}}).json()
    queued = client.post("/api/jobs", json={"stage": "noop", "params": {"seconds": 1}}).json()
    assert client.post(f"/api/jobs/{queued['id']}/cancel").json()["state"] == "cancelled"
    t0 = time.time()
    while client.get(f"/api/jobs/{slow['id']}").json()["state"] != "running":
        assert time.time() - t0 < 30
        time.sleep(0.05)
    client.post(f"/api/jobs/{slow['id']}/cancel")
    assert _wait(client, slow["id"])["state"] == "cancelled"

    failing = client.post("/api/jobs", json={"stage": "noop", "params": {"seconds": 0.1, "steps": 1, "fail": True}}).json()
    failed = _wait(client, failing["id"])
    assert failed["state"] == "failed" and "asked to fail" in failed["error"]
    assert client.post("/api/jobs", json={"stage": "analyze", "experiment": ""}).status_code in (400, 404)


def test_pipeline_job_on_demo_data(client: TestClient):
    settings = client.app.state.settings
    spec = DemoSpec(name="tiny", image_size=(320, 256), n_frames=5, n_calib_views=0, poi_step=24)
    make_demo_dataset(settings.data_root, spec, workspace=settings.workspace / "tiny", workers=1, exact_calibration=True)
    job = client.post("/api/experiments/tiny/jobs", json={"stage": "pipeline", "params": {"stages": ["stereo", "temporal", "reconstruct"]}}).json()
    done = _wait(client, job["id"], timeout=300)
    assert done["state"] == "succeeded", client.get(f"/api/jobs/{job['id']}/log").json()["text"][-2000:]
    st = client.get("/api/experiments/tiny/status").json()["stages"]
    assert [st[s]["state"] for s in ("stereo", "temporal", "reconstruct")] == ["done"] * 3


def test_import_endpoints(client: TestClient):
    import cv2
    import numpy as np

    settings = client.app.state.settings
    spec = DemoSpec(name="imp", image_size=(160, 128), n_frames=6, n_calib_views=0, poi_step=24)
    data_dir = make_demo_dataset(settings.data_root, spec, workspace=settings.workspace / "imp", workers=1, exact_calibration=True)

    probe = client.post("/api/sequences/probe", json={"path": str(data_dir / "left"), "pattern": "*.tif"}).json()
    assert probe["n_frames"] == 6 and probe["shape"] == [128, 160] and probe["reader"] == "tiff"
    assert probe["timestamp_candidates"] and probe["dtype"] == "uint16"
    assert client.post("/api/sequences/probe", json={"path": "/etc", "pattern": "*"}).status_code == 403

    seqs = client.get("/api/experiments/imp/sequences").json()
    assert seqs["n_frames"] == 6 and not seqs["length_mismatch"]
    assert seqs["cameras"]["left"]["full_scale"] >= seqs["cameras"]["left"]["value_max"]

    r = client.get("/api/experiments/imp/frames/left/3.png")
    assert r.status_code == 200 and r.headers["x-frame-name"].endswith(".tif")
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img.shape == (128, 160)
    r = client.get("/api/experiments/imp/frames/right/0.png", params={"overlay": "saturation", "lo": 0, "hi": 4095, "scale": 0.5})
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED).shape == (64, 80, 3)
    assert client.get("/api/experiments/imp/frames/left/99.png").status_code == 404

    rep = client.get("/api/experiments/imp/frames/left/0/report").json()
    assert 2.0 < rep["speckle_px"] < 7.0 and rep["saturation_fraction"] < 0.01 and rep["raw_index"] == 0

    sync = client.get("/api/experiments/imp/sync").json()
    assert sync["index_aligned"] and sync["n_pairs"] == 6 and sync["source"]["left"] == "file"

    sharp = client.get("/api/experiments/imp/sharpness", params={"points": 4}).json()
    assert len(sharp["frames"]) == 4 and len(sharp["left"]) == 4 and all(v > 0 for v in sharp["right"])


def test_calibration_endpoints(client: TestClient):
    import cv2
    import numpy as np

    settings = client.app.state.settings
    spec = DemoSpec(name="cal", n_frames=1, n_calib_views=8, oversample=1)
    make_demo_dataset(settings.data_root, spec, workspace=settings.workspace / "cal", workers=1)
    ov = client.get("/api/experiments/cal/calibration").json()
    assert ov["detections"] is None and ov["images"]["left"] == 8 and ov["status"]["state"] == "missing"
    board = client.get("/api/experiments/cal/calibration/board.png", params={"squares_x": 11})
    img = cv2.imdecode(np.frombuffer(board.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img.shape[1] > img.shape[0]
    assert client.get("/api/experiments/cal/calibration/board.png", params={"marker_mm": 9}).status_code == 422

    job = client.post("/api/experiments/cal/jobs", json={"stage": "calibrate"}).json()
    done = _wait(client, job["id"], timeout=300)
    assert done["state"] == "succeeded", client.get(f"/api/jobs/{job['id']}/log").json()["text"][-1500:]
    ov = client.get("/api/experiments/cal/calibration").json()
    assert ov["status"]["state"] == "done" and ov["detections"]["up_to_date"]
    assert len(ov["detections"]["views"]) == 8 and ov["report"]["stereo_rms_px"] < 0.1
    assert len(ov["pairs"]) >= 6 and len(ov["residuals"]["left"]) == sum(p["n"] for p in ov["pairs"])
    v = ov["pairs"][0]["view"]
    view = client.get(f"/api/experiments/cal/calibration/views/{v}").json()
    assert view["pair"] and len(view["pair"]["left"]["points"]) == len(view["pair"]["ids"])
    assert len(view["left"]["points"]) >= len(view["pair"]["ids"])
    png = client.get(f"/api/experiments/cal/calibration/views/{v}/right.png", params={"scale": 0.5})
    assert cv2.imdecode(np.frombuffer(png.content, np.uint8), cv2.IMREAD_UNCHANGED).shape == (208, 256)
    poses = client.get("/api/experiments/cal/calibration/poses").json()
    assert len(poses["boards"]) == len(ov["pairs"]) and abs(poses["baseline_mm"] - ov["report"]["baseline_mm"]) < 1e-9
    assert poses["cameras"][1]["center"][0] > 50  # right camera sits at +x
