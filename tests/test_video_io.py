from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import write_color_video

from depth_capture.video_io import VideoOpenError, VideoReader, VideoWriter, validate_video_path


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    source = write_color_video(tmp_path / "src.mp4", frames=8, width=64, height=48, fps=12.0)
    reader = VideoReader()
    info = reader.open(source)
    assert info.width == 64
    assert info.height == 48
    assert info.fps == pytest.approx(12.0, abs=0.5)

    frames = []
    while True:
        ok, frame = reader.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    reader.release()
    assert len(frames) == 8
    assert frames[0].shape == (48, 64, 3)

    out = tmp_path / "out.mp4"
    writer = VideoWriter()
    writer.open(out, info.fps, info.width, info.height)
    for frame in frames:
        writer.write(frame)
    written = writer.path
    writer.release()
    assert written is not None
    assert written.exists()

    check = VideoReader()
    written_info = check.open(written)
    count = 0
    while True:
        ok, frame = check.read()
        if not ok or frame is None:
            break
        count += 1
        assert frame.shape[1] == written_info.width
        assert frame.shape[0] == written_info.height
    check.release()
    assert count == 8


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(VideoOpenError):
        validate_video_path(tmp_path / "missing.mp4")


def test_unsupported_extension(tmp_path: Path) -> None:
    bogus = tmp_path / "clip.txt"
    bogus.write_text("nope")
    with pytest.raises(VideoOpenError):
        validate_video_path(bogus)
