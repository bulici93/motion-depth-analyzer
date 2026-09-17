from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from tests.conftest import write_color_video

from depth_capture.server import create_app


class FakeEngine:
    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        height, width = bgr_frame.shape[:2]
        xs = np.linspace(0, 1, width, dtype=np.float32)
        return np.tile(xs, (height, 1))


def _client(tmp_path: Path) -> tuple[TestClient, object]:
    app = create_app(root=tmp_path, engine_loader=lambda: FakeEngine())
    return TestClient(app), app


def test_upload_and_list_empty(tmp_path: Path) -> None:
    client, _app = _client(tmp_path)
    assert client.get("/api/jobs").json() == []
    video = write_color_video(tmp_path / "clip.mp4", frames=6, width=32, height=24)
    with video.open("rb") as handle:
        response = client.post("/api/upload", files={"file": ("clip.mp4", handle, "video/mp4")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["info"]["width"] == 32
    assert Path(payload["source_path"]).is_file()


def test_jobs_crud(tmp_path: Path) -> None:
    client, app = _client(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    source = write_color_video(tmp_path / "input" / "in.mp4", frames=4)
    output = write_color_video(tmp_path / "output" / "out.mp4", frames=4)
    store = app.state.store
    job = store.add(title="demo", source_path=source, output_path=output, params={"scale": 0.5})
    listed = client.get("/api/jobs").json()
    assert listed[0]["id"] == job.id
    detail = client.get(f"/api/jobs/{job.id}").json()
    assert detail["source"]["width"] == 64
    media = client.get(f"/api/media/source/{job.id}")
    assert media.status_code == 200
    deleted = client.delete(f"/api/jobs/{job.id}")
    assert deleted.status_code == 200
    assert client.get("/api/jobs").json() == []


def test_process_writes_history(tmp_path: Path) -> None:
    client, _app = _client(tmp_path)
    video = write_color_video(tmp_path / "clip.mp4", frames=6, width=32, height=24)
    with video.open("rb") as handle:
        uploaded = client.post("/api/upload", files={"file": ("clip.mp4", handle, "video/mp4")}).json()
    started = client.post(
        "/api/process",
        json={"source_path": uploaded["source_path"], "scale": 1.0, "stride": 1, "preserve_audio": False},
    )
    assert started.status_code == 200
    jobs = []
    for _ in range(80):
        jobs = client.get("/api/jobs").json()
        if jobs:
            break
        time.sleep(0.05)
    assert jobs, "process should create a history job"
    job = client.get(f"/api/jobs/{jobs[0]['id']}").json()
    assert job["output_exists"] is True
    assert job["params"]["scale"] == 1.0


def test_process_cancel(tmp_path: Path) -> None:
    class GateEngine:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
            self.entered.set()
            assert self.release.wait(timeout=8)
            height, width = bgr_frame.shape[:2]
            xs = np.linspace(0, 1, width, dtype=np.float32)
            return np.tile(xs, (height, 1))

    engine = GateEngine()
    client = TestClient(create_app(root=tmp_path, engine_loader=lambda: engine))
    video = write_color_video(tmp_path / "clip.mp4", frames=8, width=32, height=24)
    with video.open("rb") as handle:
        uploaded = client.post("/api/upload", files={"file": ("clip.mp4", handle, "video/mp4")}).json()
    started = client.post(
        "/api/process",
        json={"source_path": uploaded["source_path"], "scale": 1.0, "preserve_audio": False},
    )
    run_id = started.json()["run_id"]
    assert engine.entered.wait(timeout=8)
    cancelled = client.post(f"/api/process/{run_id}/cancel")
    assert cancelled.status_code == 200
    engine.release.set()

    body = ""
    with client.stream("GET", f"/api/process/{run_id}/events") as response:
        for chunk in response.iter_text():
            body += chunk
            if "event: cancelled" in body or "event: done" in body or "event: failed" in body:
                break
    assert "event: cancelled" in body
    assert client.get("/api/jobs").json() == []


def test_rejects_path_escape(tmp_path: Path) -> None:
    client, _app = _client(tmp_path)
    response = client.get("/api/media/file", params={"path": "/etc/passwd"})
    assert response.status_code in {403, 404}


def test_serves_spa_index(tmp_path: Path) -> None:
    dist = tmp_path / "web"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
    client, _app = _client(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert b"ok" in response.content


def test_clear_cache_deletes_input_and_output(tmp_path: Path) -> None:
    client, app = _client(tmp_path)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    nested = output_dir / "previews" / "clip"
    nested.mkdir(parents=True)
    source = write_color_video(input_dir / "in.mp4", frames=4)
    output = write_color_video(output_dir / "out.mp4", frames=4)
    (nested / "preview.jpg").write_bytes(b"x")
    store = app.state.store
    store.add(title="demo", source_path=source, output_path=output, params={"scale": 0.5})

    response = client.delete("/api/cache")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert list(input_dir.iterdir()) == []
    leftover = [path.name for path in output_dir.iterdir()]
    assert leftover == ["jobs.json"]
    assert store.list_jobs() == []
    assert not source.exists()
    assert not output.exists()
    assert not nested.exists()


def test_clear_cache_after_process_completes(tmp_path: Path) -> None:
    client, _app = _client(tmp_path)
    video = write_color_video(tmp_path / "clip.mp4", frames=4, width=32, height=24)
    with video.open("rb") as handle:
        uploaded = client.post("/api/upload", files={"file": ("clip.mp4", handle, "video/mp4")}).json()
    started = client.post(
        "/api/process",
        json={"source_path": uploaded["source_path"], "scale": 1.0, "preserve_audio": False},
    )
    assert started.status_code == 200
    for _ in range(80):
        if client.get("/api/jobs").json():
            break
        time.sleep(0.05)
    assert client.get("/api/process/active").json()["run_id"] is None
    response = client.delete("/api/cache")
    assert response.status_code == 200


def test_clear_cache_blocked_while_running(tmp_path: Path) -> None:
    class GateEngine:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
            self.entered.set()
            assert self.release.wait(timeout=8)
            height, width = bgr_frame.shape[:2]
            xs = np.linspace(0, 1, width, dtype=np.float32)
            return np.tile(xs, (height, 1))

    engine = GateEngine()
    client = TestClient(create_app(root=tmp_path, engine_loader=lambda: engine))
    video = write_color_video(tmp_path / "clip.mp4", frames=6, width=32, height=24)
    with video.open("rb") as handle:
        uploaded = client.post("/api/upload", files={"file": ("clip.mp4", handle, "video/mp4")}).json()
    started = client.post(
        "/api/process",
        json={"source_path": uploaded["source_path"], "scale": 1.0, "preserve_audio": False},
    )
    assert started.status_code == 200
    assert engine.entered.wait(timeout=8)
    blocked = client.delete("/api/cache")
    assert blocked.status_code == 409
    second = client.post(
        "/api/process",
        json={"source_path": uploaded["source_path"], "scale": 1.0, "preserve_audio": False},
    )
    assert second.status_code == 409
    engine.release.set()
    for _ in range(80):
        if client.get("/api/jobs").json():
            break
        time.sleep(0.05)
    assert client.delete("/api/cache").status_code == 200


def test_clear_cache_ignores_stale_running_status(tmp_path: Path) -> None:
    from depth_capture.server import RunState

    client, app = _client(tmp_path)
    stale = RunState(run_id="stale")
    stale.status = "running"
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    stale.thread = dead
    app.state.runs[stale.run_id] = stale
    response = client.delete("/api/cache")
    assert response.status_code == 200
