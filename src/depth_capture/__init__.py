"""Local video depth capture powered by Depth Anything V2 Small."""

from depth_capture.config import DEFAULT_MODEL_ID, DepthCaptureConfig
from depth_capture.pipeline import DepthPipeline, ProgressInfo

__all__ = [
    "DEFAULT_MODEL_ID",
    "DepthCaptureConfig",
    "DepthPipeline",
    "ProgressInfo",
    "__version__",
]

__version__ = "0.1.0"
