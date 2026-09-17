from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def write_color_video(path: Path, frames: int = 8, width: int = 64, height: int = 48, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4v in this environment")
    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 40
        frame[:, :, 1] = 80
        frame[:, :, 2] = min(255, 20 * index)
        cv2.rectangle(frame, (8, 8), (24, 24), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path
