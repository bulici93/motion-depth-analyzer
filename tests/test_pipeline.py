from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import write_color_video

from depth_capture.config import DepthCaptureConfig
from depth_capture.engine import DepthAnythingEngine, DepthEngineError
from depth_capture.pipeline import DepthPipeline, PipelineCancelled, ProgressInfo


class FakeEngine:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        self.calls += 1
        height, width = bgr_frame.shape[:2]
        xs = np.linspace(0, 1, width, dtype=np.float32)
        return np.tile(xs, (height, 1))


def test_pipeline_writes_video_and_previews(tmp_path: Path) -> None:
    source = write_color_video(tmp_path / "in.mp4", frames=8, width=64, height=48)
    engine = FakeEngine()
    config = DepthCaptureConfig(scale=1.0, stride=1, temporal_smooth=0.0, preserve_audio=False)
    pipeline = DepthPipeline(config, engine=engine)
    output = tmp_path / "out.mp4"
    preview_dir = tmp_path / "previews"
    updates: list[ProgressInfo] = []

    result = pipeline.process(
        source,
        output,
        progress_callback=updates.append,
        preview_dir=preview_dir,
    )

    assert result.exists()
    assert result.stat().st_size > 0
    assert engine.calls == 8
    assert updates
    assert updates[-1].frame_index == 8
    previews = list(preview_dir.glob("preview_*.jpg"))
    assert previews


def test_stride_reuses_depth(tmp_path: Path) -> None:
    source = write_color_video(tmp_path / "in.mp4", frames=6, width=32, height=24)
    engine = FakeEngine()
    config = DepthCaptureConfig(scale=0.5, stride=2, temporal_smooth=0.0, preserve_audio=False)
    pipeline = DepthPipeline(config, engine=engine)
    pipeline.process(source, tmp_path / "out.mp4", preview_dir=tmp_path / "previews")
    assert engine.calls == 3


def test_pipeline_cancel_stops_and_cleans(tmp_path: Path) -> None:
    source = write_color_video(tmp_path / "in.mp4", frames=8, width=32, height=24)
    engine = FakeEngine()
    config = DepthCaptureConfig(scale=1.0, stride=1, temporal_smooth=0.0, preserve_audio=False)
    pipeline = DepthPipeline(config, engine=engine)
    output = tmp_path / "out.mp4"
    preview_dir = tmp_path / "previews"

    def cancel_check() -> bool:
        return engine.calls >= 3

    with pytest.raises(PipelineCancelled):
        pipeline.process(source, output, preview_dir=preview_dir, cancel_check=cancel_check)

    assert not output.exists()
    assert not (tmp_path / ".out_raw.mp4").exists()
    assert not preview_dir.exists()
    assert engine.calls == 3


def test_infer_without_load_raises() -> None:
    engine = DepthAnythingEngine()
    with pytest.raises(DepthEngineError):
        engine.infer(np.zeros((8, 8, 3), dtype=np.uint8))
