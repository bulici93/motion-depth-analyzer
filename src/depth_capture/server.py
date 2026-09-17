from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from depth_capture.config import DepthCaptureConfig, default_model_id
from depth_capture.engine import DepthAnythingEngine
from depth_capture.history import JobStore
from depth_capture.pipeline import DepthPipeline, PipelineCancelled, ProgressInfo, inspect_video, unique_path
from depth_capture.video_io import VIDEO_EXTENSIONS

MAX_UPLOAD_BYTES = 500 * 1024 * 1024

logger = logging.getLogger(__name__)

EngineLoader = Callable[[], Any]


class ProcessRequest(BaseModel):
    source_path: str
    scale: float = 0.5
    stride: int = 1
    gamma: float = 0.7
    crush: float = 35.0
    smooth: float = 0.4
    preserve_audio: bool = True


@dataclass
class RunState:
    run_id: str
    events: queue.Queue = field(default_factory=queue.Queue)
    status: str = "queued"
    job_id: str | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


def run_is_active(run: RunState) -> bool:
    if run.status not in {"queued", "running"}:
        return False
    if run.thread is None:
        return True
    return run.thread.is_alive()


def _project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]


def _web_root(root: Path) -> Path:
    for candidate in (root / "web", Path.cwd() / "web"):
        if (candidate / "index.html").is_file():
            return candidate
    return root / "web"


def _empty_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _info_payload(path: Path) -> dict[str, Any]:
    info = inspect_video(path)
    return {
        "path": str(path),
        "name": path.name,
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "frame_count": info.frame_count,
        "duration": info.duration,
        "duration_label": info.format_duration(),
    }


def create_app(
    root: Path | None = None,
    engine_loader: EngineLoader | None = None,
) -> FastAPI:
    root = (root or _project_root()).resolve()
    input_dir = root / "input"
    output_dir = root / "output"
    store = JobStore(output_dir / "jobs.json")
    runs: dict[str, RunState] = {}
    engine_lock = threading.Lock()
    engine_holder: dict[str, Any] = {"engine": None}

    def load_engine():
        if engine_loader is not None:
            return engine_loader()
        with engine_lock:
            if engine_holder["engine"] is None:
                engine_holder["engine"] = DepthAnythingEngine.load(default_model_id())
            return engine_holder["engine"]

    def safe_under_workspace(path: Path) -> Path:
        resolved = path.resolve()
        allowed = (input_dir.resolve(), output_dir.resolve())
        if not any(resolved == folder or folder in resolved.parents for folder in allowed):
            raise HTTPException(status_code=403, detail="Path is outside the workspace")
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return resolved

    app = FastAPI(title="Motion Depth Analyzer", version="0.1.0")
    web_dir = _web_root(root)

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [job.to_public_dict() for job in store.list_jobs()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        payload = job.to_public_dict()
        if job.source_exists():
            payload["source"] = _info_payload(Path(job.source_path))
        if job.output_exists():
            payload["output_name"] = Path(job.output_path).name
        return payload

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, bool]:
        if not store.delete(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return {"ok": True}

    def active_run() -> RunState | None:
        return next((run for run in runs.values() if run_is_active(run)), None)

    @app.delete("/api/cache")
    def clear_cache() -> dict[str, bool]:
        if active_run() is not None:
            raise HTTPException(status_code=409, detail="A job is running")
        _empty_dir(input_dir)
        _empty_dir(output_dir)
        store.clear()
        return {"ok": True}

    @app.get("/api/process/active")
    def get_active_process() -> dict[str, str | None]:
        run = active_run()
        if run is None:
            return {"run_id": None, "status": None}
        return {"run_id": run.run_id, "status": run.status}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        filename = Path(file.filename or "clip.mp4")
        suffix = filename.suffix.lower() or ".mp4"
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported video format")
        input_dir.mkdir(parents=True, exist_ok=True)
        target = unique_path(input_dir, filename.stem, suffix)
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="File is larger than 500MB")
        target.write_bytes(content)
        try:
            info = _info_payload(target)
        except Exception as exc:  # noqa: BLE001
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"source_path": str(target), "info": info}

    @app.get("/api/media/source/{job_id}")
    def media_source(job_id: str) -> FileResponse:
        job = store.get(job_id)
        if job is None or not job.source_exists():
            raise HTTPException(status_code=404, detail="Source not found")
        path = safe_under_workspace(Path(job.source_path))
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/api/media/output/{job_id}")
    def media_output(job_id: str) -> FileResponse:
        job = store.get(job_id)
        if job is None or not job.output_exists():
            raise HTTPException(status_code=404, detail="Output not found")
        path = safe_under_workspace(Path(job.output_path))
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/api/media/file")
    def media_file(path: str = Query(...)) -> FileResponse:
        file_path = safe_under_workspace(Path(path))
        suffix = file_path.suffix.lower()
        media = "image/jpeg" if suffix in {".jpg", ".jpeg", ".png"} else "video/mp4"
        return FileResponse(file_path, media_type=media, filename=file_path.name)

    @app.post("/api/process")
    def start_process(body: ProcessRequest) -> dict[str, str]:
        if active_run() is not None:
            raise HTTPException(status_code=409, detail="A job is running")
        source = safe_under_workspace(Path(body.source_path))
        run = RunState(run_id=uuid.uuid4().hex[:12])
        runs[run.run_id] = run
        thread = threading.Thread(
            target=_run_process,
            kwargs={
                "run": run,
                "source": source,
                "output_dir": output_dir,
                "store": store,
                "body": body,
                "load_engine": load_engine,
            },
            daemon=True,
        )
        run.thread = thread
        thread.start()
        return {"run_id": run.run_id}

    @app.post("/api/process/{run_id}/cancel")
    def cancel_process(run_id: str) -> dict[str, Any]:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        run.cancel.set()
        status = run.status if run.status in {"done", "error", "cancelled"} else "cancelling"
        return {"ok": True, "status": status}

    @app.get("/api/process/{run_id}/events")
    def process_events(run_id: str) -> StreamingResponse:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        def terminal_event() -> dict[str, Any] | None:
            if run.status == "done":
                return {"event": "done", "data": {"job_id": run.job_id}}
            if run.status == "cancelled":
                return {"event": "cancelled", "data": {}}
            if run.status == "error":
                return {"event": "failed", "data": {"message": run.error or "error"}}
            return None

        def stream():
            ended = terminal_event()
            if ended is not None:
                data = json.dumps(ended.get("data", {}), ensure_ascii=False)
                yield f"event: {ended['event']}\ndata: {data}\n\n"
                return
            while True:
                try:
                    event = run.events.get(timeout=20)
                except queue.Empty:
                    ended = terminal_event()
                    if ended is not None:
                        data = json.dumps(ended.get("data", {}), ensure_ascii=False)
                        yield f"event: {ended['event']}\ndata: {data}\n\n"
                        return
                    yield ": keepalive\n\n"
                    continue
                name = event.get("event", "message")
                data = json.dumps(event.get("data", {}), ensure_ascii=False)
                yield f"event: {name}\ndata: {data}\n\n"
                if name in {"done", "failed", "cancelled"}:
                    break

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/")
    def index() -> FileResponse:
        index_path = web_dir / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="Web UI is missing.")
        return FileResponse(index_path)

    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="ui")

    app.state.root = root
    app.state.store = store
    app.state.runs = runs
    return app


def _run_process(
    *,
    run: RunState,
    source: Path,
    output_dir: Path,
    store: JobStore,
    body: ProcessRequest,
    load_engine: EngineLoader,
) -> None:
    run.status = "running"
    try:
        config = DepthCaptureConfig(
            model_id=default_model_id(),
            scale=body.scale,
            stride=body.stride,
            gamma=body.gamma,
            crush_percentile=body.crush,
            temporal_smooth=body.smooth,
            preserve_audio=body.preserve_audio,
        )
        engine = load_engine()
        pipeline = DepthPipeline(config, engine=engine)
        output_path = unique_path(output_dir, f"{source.stem}_depth")
        preview_dir = output_dir / "previews" / output_path.stem

        def on_progress(update: ProgressInfo) -> None:
            payload = {
                "frame_index": update.frame_index,
                "total_frames": update.total_frames,
                "progress": update.progress,
                "fps": update.fps,
                "eta_seconds": update.eta_seconds,
                "preview_path": update.preview_path,
            }
            run.events.put({"event": "progress", "data": payload})

        result = pipeline.process(
            input_path=source,
            output_path=output_path,
            progress_callback=on_progress,
            preview_dir=preview_dir,
            cancel_check=run.cancel.is_set,
        )
        job = store.add(
            title=source.stem,
            source_path=source,
            output_path=result,
            params={
                "scale": body.scale,
                "stride": body.stride,
                "gamma": body.gamma,
                "crush": body.crush,
                "smooth": body.smooth,
                "preserve_audio": body.preserve_audio,
            },
            preview_dir=preview_dir,
        )
        run.status = "done"
        run.job_id = job.id
        run.events.put({"event": "done", "data": {"job_id": job.id}})
    except PipelineCancelled:
        run.status = "cancelled"
        run.events.put({"event": "cancelled", "data": {}})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Process failed")
        run.status = "error"
        run.error = str(exc)
        run.events.put({"event": "failed", "data": {"message": str(exc)}})
    finally:
        if run.status in {"queued", "running"}:
            run.status = "error"
            run.error = run.error or "Process ended unexpectedly"
            run.events.put({"event": "failed", "data": {"message": run.error}})


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        "depth_capture.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


app = create_app()


if __name__ == "__main__":
    main()
