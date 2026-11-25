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


def test_reference_endpoints(client: TestClient):
    settings = client.app.state.settings
    spec = DemoSpec(name="ref", image_size=(320, 256), n_frames=2, n_calib_views=0, poi_step=24)
    make_demo_dataset(settings.data_root, spec, workspace=settings.workspace / "ref", workers=1, exact_calibration=True)
    m = client.get("/api/experiments/ref/masks").json()
    assert m["image_shape"] == [256, 320] and m["poi"]["path"] is None and m["subset_size"] == 31

    full = client.post("/api/experiments/ref/poi-preview", json={"roi": None, "poi": None, "subset_size": 31, "step": 12}).json()
    region = [[60, 50], [260, 50], [260, 200], [60, 200]]
    part = client.post("/api/experiments/ref/poi-preview",
                       json={"roi": None, "poi": [{"op": "add", "points": region}], "subset_size": 31, "step": 12}).json()
    assert 0 < part["n"] < full["n"] and len(part["points"]) == 2 * part["n"]
    bad = client.post("/api/experiments/ref/poi-preview", json={"roi": None, "poi": None, "subset_size": 30, "step": 12})
    assert bad.status_code == 422
    assert client.get("/api/experiments/ref/reference").status_code == 404

    saved = client.put("/api/experiments/ref/masks", json={
        "roi": None, "poi": [{"op": "add", "points": region}],
        "rbm": [{"op": "add", "points": [[40, 40], [90, 40], [90, 90], [40, 90]]}], "subset_size": 31, "step": 12,
    }).json()
    assert saved["poi"]["drawn"] and saved["poi"]["exists"] and saved["rbm"]["drawn"] and saved["step"] == 12
    assert client.put("/api/experiments/ref/masks", json={"poi": [{"op": "add", "points": [[0, 0], [1, 0], [2, 0]]}]}).status_code == 422

    job = client.post("/api/experiments/ref/jobs", json={"stage": "stereo"}).json()
    assert _wait(client, job["id"], timeout=300)["state"] == "succeeded"
    ref = client.get("/api/experiments/ref/reference").json()
    assert ref["n"] == part["n"]  # the saved mask was used by the stereo stage
    assert len(ref["points_3d"]) == 3 * ref["n"] and len(ref["connectivity"]) % 3 == 0
    assert ref["status_counts"].get("OK", 0) >= ref["n"] - 2 and set(ref["mean_axes"]) == {"c", "s", "n"}
    assert 200 < ref["depth_range"][0] < ref["depth_range"][1] < 320

    cleared = client.put("/api/experiments/ref/masks", json={"roi": [10, 10, 300, 230], "poi": [], "rbm": []}).json()
    assert cleared["poi"]["path"] is None and cleared["rbm"]["path"] is None and cleared["roi"] == [10, 10, 300, 230]


def test_results_endpoints(client: TestClient):
    settings = client.app.state.settings
    spec = DemoSpec(name="res", image_size=(320, 256), n_frames=6, n_calib_views=0, poi_step=24)
    make_demo_dataset(settings.data_root, spec, workspace=settings.workspace / "res", workers=1, exact_calibration=True)
    assert client.get("/api/experiments/res/dic/summary").status_code == 404
    job = client.post("/api/experiments/res/jobs", json={"stage": "pipeline"}).json()
    assert _wait(client, job["id"], timeout=300)["state"] == "succeeded"

    s = client.get("/api/experiments/res/dic/summary").json()
    assert s["n_frames"] == 6 and s["frames_done"] == 6 and len(s["valid_left"]) == 6
    assert min(s["valid_left"]) > 0.9 and s["status_counts"]["left"]["OK"] > 0 and s["reference_frame"] == 0
    assert s["znssd_median"][0] == 0.0 and s["znssd_median"][3] > 0

    m = client.get("/api/experiments/res/dic/status-map", params={"t_bins": 3, "n_bins": 4}).json()
    assert len(m["valid"]) == 3 and len(m["valid"][0]) == 4 and m["valid"][0][0] > 0.9
    p = client.get("/api/experiments/res/dic/point/0").json()
    assert p["left"]["u"][0] == 0.0 and len(p["right"]["v"]) == 6
    assert client.get(f"/api/experiments/res/dic/point/{s['n_points']}").status_code == 404

    info = client.get("/api/experiments/res/displacement/info").json()
    assert {"local", "raw", "rbm_local"} <= set(info["kinds"]) and info["rbm"] and info["ranges"]["local"]["mag"] > 0
    fr = client.get("/api/experiments/res/displacement/frame/3", params={"kind": "local"}).json()
    assert len(fr["d"]) == s["n_points"] and len(fr["d"][0]) == 3
    assert client.get("/api/experiments/res/displacement/frame/3", params={"kind": "bogus"}).status_code == 400
    dp = client.get("/api/experiments/res/displacement/point/0", params={"max_samples": 4}).json()
    assert len(dp["local"]["n"]) <= 6 and "rbm_local" in dp
    r = client.get("/api/experiments/res/rbm").json()
    assert len(r["tx"]) == 6 and r["residual_rms"][0] is not None


def test_analysis_endpoints(client: TestClient):
    settings = client.app.state.settings
    spec = DemoSpec(name="vib", image_size=(320, 256), n_frames=40, n_calib_views=0, poi_step=32)
    make_demo_dataset(settings.data_root, spec, workspace=settings.workspace / "vib", workers=4, exact_calibration=True)
    assert _wait(client, client.post("/api/experiments/vib/jobs", json={"stage": "pipeline"}).json()["id"], 300)["state"] == "succeeded"
    assert client.get("/api/experiments/vib/vibration").status_code == 404
    job = client.post("/api/experiments/vib/jobs", json={"stage": "analyze", "params": {"options": {"fit_peaks": 2, "fdd": True}}}).json()
    done = _wait(client, job["id"], 300)
    assert done["state"] == "succeeded", client.get(f"/api/jobs/{job['id']}/log").json()["text"][-1500:]
    nf = client.post("/api/experiments/vib/jobs", json={"stage": "noise_floor"}).json()
    assert _wait(client, nf["id"], 120)["state"] == "succeeded"
    un = client.post("/api/experiments/vib/jobs", json={"stage": "uncertainty", "params": {"pixel_sigma": 0.02}}).json()
    assert _wait(client, un["id"], 120)["state"] == "succeeded"

    v = client.get("/api/experiments/vib/vibration", params={"noise_from": "vib"}).json()
    assert len(v["spectrum"]["amplitude"]) == 3 and v["peaks"] and len(v["harmonic"]) == 2 and v["fdd"] and v["noise"]
    key = v["harmonic"][0]["key"]
    h = client.get(f"/api/experiments/vib/vibration/harmonic/{key}", params={"noise_from": "vib"}).json()
    n = len(h["amplitude"])
    assert len(h["amplitude"][0]) == 3 and len(h["re"]) == n and h["significant"] is not None
    assert client.get("/api/experiments/vib/vibration/fdd/0").json()["re"]
    m = client.get("/api/experiments/vib/vibration/mac").json()
    assert len(m["labels"]) == len(m["mac"]) >= 3 and abs(m["mac"][0][0] - 1) < 1e-6
    q = client.get("/api/experiments/vib/quality").json()
    assert q["noise_floor"]["n_frames"] == 40 and q["uncertainty"][0]["pixel_sigma"] == 0.02

    assert client.get("/api/experiments/vib/export/peaks.csv").text.startswith("frequency_hz")
    assert client.get(f"/api/experiments/vib/export/harmonic/{key}.csv").text.count("\n") == n + 1
    ply = client.get("/api/experiments/vib/export/shape.ply").text
    assert ply.startswith("ply") and "element face" in ply
    assert client.get("/api/experiments/vib/export/point/0.csv").text.count("\n") == 41
    rep = client.get("/api/experiments/vib/report.html")
    assert rep.status_code == 200 and "振动分析" in rep.text and "data:image/png;base64" in rep.text
