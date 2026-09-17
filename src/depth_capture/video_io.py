from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class VideoOpenError(RuntimeError):
    pass


class EmptyVideoError(RuntimeError):
    pass


class EncoderUnavailableError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    fourcc: str

    def format_duration(self) -> str:
        total = int(round(self.duration))
        minutes, seconds = divmod(max(total, 0), 60)
        return f"{minutes:02d}:{seconds:02d}"


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return chars.strip() or "unknown"


def validate_video_path(path: str | Path) -> Path:
    video_path = Path(path)
    if not video_path.is_file():
        raise VideoOpenError(f"Video file not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise VideoOpenError(f"Unsupported video format: {video_path.suffix}")
    return video_path.resolve()


class VideoReader:
    def __init__(self) -> None:
        self.capture: cv2.VideoCapture | None = None
        self.path: Path | None = None
        self._info: VideoInfo | None = None

    def open(self, path: str | Path) -> VideoInfo:
        video_path = validate_video_path(path)
        self.release()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoOpenError(f"Unable to open video: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = _fourcc_to_str(capture.get(cv2.CAP_PROP_FOURCC))

        if width <= 0 or height <= 0:
            capture.release()
            raise VideoOpenError(f"Unable to read video: {video_path}")

        if fps <= 1e-3:
            fps = 25.0

        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        self.capture = capture
        self.path = video_path
        self._info = VideoInfo(
            path=str(video_path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            fourcc=fourcc,
        )
        return self._info

    def get_info(self) -> VideoInfo:
        if self._info is None:
            raise VideoOpenError("Video is not open")
        return self._info

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.capture is None:
            raise VideoOpenError("Video is not open")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.path = None
        self._info = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


_CODEC_CANDIDATES = (
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("XVID", ".avi"),
    ("MJPG", ".avi"),
)


class VideoWriter:
    def __init__(self) -> None:
        self.writer: cv2.VideoWriter | None = None
        self.path: Path | None = None
        self.size: tuple[int, int] | None = None
        self.used_codec: str | None = None

    def open(self, path: str | Path, fps: float, width: int, height: int) -> Path:
        self.release()
        requested = Path(path)
        requested.parent.mkdir(parents=True, exist_ok=True)
        fps = max(float(fps), 1.0)

        last_error = None
        for codec, suffix in _CODEC_CANDIDATES:
            candidate = requested.with_suffix(suffix)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(candidate), fourcc, fps, (width, height), True)
            if writer.isOpened():
                self.writer = writer
                self.path = candidate
                self.size = (width, height)
                self.used_codec = codec
                return candidate
            writer.release()
            last_error = codec

        raise EncoderUnavailableError(
            "No usable OpenCV video encoder in this environment "
            f"(last tried: {last_error}). Install FFmpeg and retry."
        )

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None or self.size is None:
            raise EncoderUnavailableError("VideoWriter is not open")
        output = frame
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        if output.shape[1] != self.size[0] or output.shape[0] != self.size[1]:
            output = cv2.resize(output, self.size, interpolation=cv2.INTER_LINEAR)
        self.writer.write(output)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
