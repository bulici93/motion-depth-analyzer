from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class FFmpegNotFoundError(RuntimeError):
    pass


class FFmpegError(RuntimeError):
    pass


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def require_ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise FFmpegNotFoundError(
            "FFmpeg was not found.\nInstall it first.\nmacOS:\n  brew install ffmpeg"
        )
    return binary


def run_ffmpeg(args: list[str]) -> None:
    binary = require_ffmpeg()
    command = [binary, "-y", *args]
    logger.info("Running FFmpeg: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or "FFmpeg failed")


def transcode_h264(input_path: str | Path, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ]
    )
    return output_path


def merge_audio(
    processed_video: str | Path,
    original_video: str | Path,
    output_path: str | Path,
    preserve_audio: bool = True,
) -> Path:
    """Mux processed frames with the original audio track into H.264 MP4."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not preserve_audio:
        return transcode_h264(processed_video, output_path)

    try:
        run_ffmpeg(
            [
                "-i",
                str(processed_video),
                "-i",
                str(original_video),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        return output_path
    except FFmpegError as exc:
        logger.warning("Audio mux failed, falling back to silent transcode: %s", exc)
        return transcode_h264(processed_video, output_path)
