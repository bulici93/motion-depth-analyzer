from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from depth_capture.config import DepthCaptureConfig
from depth_capture.engine import DepthAnythingEngine, DepthEngine
from depth_capture.ffmpeg import has_ffmpeg, merge_audio
from depth_capture.stylize import DepthStylizer
from depth_capture.video_io import EmptyVideoError, VideoInfo, VideoReader, VideoWriter

logger = logging.getLogger(__name__)

ProgressCallback = Callable[["ProgressInfo"], None]
CancelCheck = Callable[[], bool]


class PipelineCancelled(Exception):
    """Raised when the user stops a running process."""


@dataclass
class ProgressInfo:
    frame_index: int
    total_frames: int
    progress: float
    fps: float
    eta_seconds: float
    preview_path: str | None = None
    message: str = ""


def unique_path(directory: Path, stem: str, suffix: str = ".mp4") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def inspect_video(path: str | Path) -> VideoInfo:
    reader = VideoReader()
    try:
        return reader.open(path)
    finally:
        reader.release()


def _resize_for_inference(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return frame
    height, width = frame.shape[:2]
    infer_w = max(1, int(width * scale))
    infer_h = max(1, int(height * scale))
    if infer_w == width and infer_h == height:
        return frame
    return cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)


def _upsample_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth.shape[1] == width and depth.shape[0] == height:
        return depth
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)


class DepthPipeline:
    def __init__(
        self,
        config: DepthCaptureConfig | None = None,
        engine: DepthEngine | None = None,
    ) -> None:
        self.config = config or DepthCaptureConfig()
        self.engine = engine
        self.stylizer = DepthStylizer(
            gamma=self.config.gamma,
            crush_percentile=self.config.crush_percentile,
            temporal_smooth=self.config.temporal_smooth,
            polarity=self.config.polarity,
        )

    def _ensure_engine(self) -> DepthEngine:
        if self.engine is None:
            self.engine = DepthAnythingEngine.load(self.config.model_id)
        return self.engine

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        progress_callback: ProgressCallback | None = None,
        preview_dir: str | Path | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Path:
        self.stylizer = DepthStylizer(
            gamma=self.config.gamma,
            crush_percentile=self.config.crush_percentile,
            temporal_smooth=self.config.temporal_smooth,
            polarity=self.config.polarity,
        )
        engine = self._ensure_engine()

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preview_root = Path(preview_dir) if preview_dir else output_path.parent / "previews" / output_path.stem
        preview_root.mkdir(parents=True, exist_ok=True)

        reader = VideoReader()
        writer = VideoWriter()
        temp_video: Path | None = None
        cancelled = False

        try:
            info = reader.open(input_path)
            temp_video = output_path.with_name(f".{output_path.stem}_raw.mp4")
            writer.open(temp_video, info.fps, info.width, info.height)
            temp_video = writer.path
            processed = self._run_frames(
                reader=reader,
                writer=writer,
                info=info,
                engine=engine,
                progress_callback=progress_callback,
                preview_root=preview_root,
                cancel_check=cancel_check,
            )
            if processed == 0:
                raise EmptyVideoError("Video has no valid frames")
        except PipelineCancelled:
            cancelled = True
        finally:
            reader.release()
            writer.release()

        if cancelled:
            _cleanup_partial_output(temp_video, preview_root)
            raise PipelineCancelled()

        assert temp_video is not None
        return self._finalize_output(temp_video, input_path, output_path)

    def _run_frames(
        self,
        reader: VideoReader,
        writer: VideoWriter,
        info: VideoInfo,
        engine: DepthEngine,
        progress_callback: ProgressCallback | None,
        preview_root: Path,
        cancel_check: CancelCheck | None = None,
    ) -> int:
        total = info.frame_count
        preview_step = max(1, total // 10) if total > 0 else 30
        last_gray: np.ndarray | None = None
        processed = 0
        started = time.perf_counter()

        while True:
            if cancel_check is not None and cancel_check():
                raise PipelineCancelled()
            ok, frame = reader.read()
            if not ok or frame is None:
                break
            processed += 1
            try:
                gray = self._depth_frame(frame, engine, processed, last_gray)
            except Exception:
                logger.exception("Frame %s failed; writing a black frame", processed)
                gray = np.zeros((info.height, info.width), dtype=np.uint8)
            last_gray = gray
            writer.write(gray)

            preview_path = None
            if processed == 1 or processed % preview_step == 0:
                preview_path = str(preview_root / f"preview_{processed:04d}.jpg")
                cv2.imwrite(preview_path, gray)

            if progress_callback is not None:
                elapsed = max(time.perf_counter() - started, 1e-6)
                fps = processed / elapsed
                remaining = max(total - processed, 0) if total > 0 else 0
                eta = remaining / fps if fps > 0 else 0.0
                progress = min(processed / total, 1.0) if total > 0 else 0.0
                progress_callback(
                    ProgressInfo(
                        frame_index=processed,
                        total_frames=total,
                        progress=progress,
                        fps=fps,
                        eta_seconds=eta,
                        preview_path=preview_path,
                    )
                )

        return processed

    def _depth_frame(
        self,
        frame: np.ndarray,
        engine: DepthEngine,
        frame_index: int,
        last_gray: np.ndarray | None,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        reuse = self.config.stride > 1 and frame_index > 1 and (frame_index - 1) % self.config.stride != 0
        if reuse and last_gray is not None:
            if last_gray.shape[:2] == (height, width):
                return last_gray
            return cv2.resize(last_gray, (width, height), interpolation=cv2.INTER_LINEAR)

        infer_frame = _resize_for_inference(frame, self.config.scale)
        raw = engine.infer(infer_frame)
        raw = _upsample_depth(raw, width, height)
        return self.stylizer.stylize(raw)

    def _finalize_output(self, temp_video: Path, original: Path, output_path: Path) -> Path:
        if has_ffmpeg():
            try:
                result = merge_audio(
                    temp_video,
                    original,
                    output_path,
                    preserve_audio=self.config.preserve_audio,
                )
                if temp_video.exists() and temp_video.resolve() != Path(result).resolve():
                    temp_video.unlink(missing_ok=True)
                return Path(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FFmpeg finalize failed, keeping the OpenCV file: %s", exc)

        logger.warning(
            "FFmpeg not used; keeping the OpenCV file. "
            "Browsers may not play it. Install FFmpeg for H.264 + original audio."
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if temp_video.resolve() != output_path.resolve():
            if output_path.exists():
                output_path.unlink()
            shutil.move(str(temp_video), str(output_path))
        return output_path


def _cleanup_partial_output(temp_video: Path | None, preview_root: Path) -> None:
    if temp_video is not None:
        temp_video.unlink(missing_ok=True)
    if preview_root.exists():
        shutil.rmtree(preview_root, ignore_errors=True)
