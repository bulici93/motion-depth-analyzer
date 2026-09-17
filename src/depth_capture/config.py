from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def default_model_id() -> str:
    return os.environ.get("DEPTH_CAPTURE_MODEL_ID", DEFAULT_MODEL_ID)


@dataclass
class DepthCaptureConfig:
    """Processing parameters shared by the CLI and Streamlit UI."""

    model_id: str = DEFAULT_MODEL_ID
    scale: float = 0.5
    stride: int = 1
    gamma: float = 0.7
    crush_percentile: float = 35.0
    temporal_smooth: float = 0.4
    preserve_audio: bool = True
    polarity: str = "near_white"

    def __post_init__(self) -> None:
        self.scale = float(min(max(self.scale, 0.25), 1.0))
        self.stride = int(max(self.stride, 1))
        self.gamma = float(min(max(self.gamma, 0.1), 2.0))
        self.crush_percentile = float(min(max(self.crush_percentile, 0.0), 90.0))
        self.temporal_smooth = float(min(max(self.temporal_smooth, 0.0), 0.95))
        self.preserve_audio = bool(self.preserve_audio)
        polarity = str(self.polarity or "near_white").lower()
        if polarity not in {"near_white", "near_black"}:
            polarity = "near_white"
        self.polarity = polarity
        if not str(self.model_id).strip():
            self.model_id = default_model_id()
