from __future__ import annotations

import logging
import os
from typing import Protocol

import numpy as np

from depth_capture.config import DEFAULT_MODEL_ID, default_model_id

logger = logging.getLogger(__name__)


class DepthEngine(Protocol):
    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return a float32 relative-depth map matching the input frame size."""


class DepthEngineError(RuntimeError):
    pass


def _pick_device(torch_mod) -> str:
    if torch_mod.cuda.is_available():
        return "cuda"
    mps = getattr(torch_mod.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _load_weights(model_cls, model_id: str):
    """Load real tensors; skip meta-device initializations when possible."""
    attempts = (
        {"low_cpu_mem_usage": False, "device_map": None},
        {"low_cpu_mem_usage": False},
        {},
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            model = model_cls.from_pretrained(model_id, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
        if any(getattr(param, "is_meta", False) for param in model.parameters()):
            continue
        return model
    raise DepthEngineError(
        "Could not load Depth Anything weights as real tensors. "
        "Reinstall the package dependencies and try again."
        + (f" Details: {last_error}" if last_error else "")
    )


class DepthAnythingEngine:
    """Depth Anything V2 Small inference wrapper."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device_name: str = "cpu") -> None:
        self.model_id = model_id
        self.device_name = device_name
        self._processor = None
        self._model = None
        self._torch = None

    @classmethod
    def load(cls, model_id: str | None = None) -> "DepthAnythingEngine":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        resolved_id = model_id or default_model_id()
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise DepthEngineError(
                "Depth capture needs PyTorch and Transformers.\n"
                "Install with:\n"
                "  pip install -e .\n"
                "The first run downloads "
                f"{DEFAULT_MODEL_ID} (~100MB) from Hugging Face."
            ) from exc

        engine = cls(model_id=resolved_id)
        engine._torch = torch
        engine.device_name = _pick_device(torch)

        logger.info("Loading Depth Anything: %s  device=%s", resolved_id, engine.device_name)
        engine._processor = AutoImageProcessor.from_pretrained(resolved_id)
        model = _load_weights(AutoModelForDepthEstimation, resolved_id)
        device = torch.device(engine.device_name)
        try:
            model = model.to(device)
        except (NotImplementedError, RuntimeError) as exc:
            logger.warning("Could not move model to %s, falling back to CPU: %s", engine.device_name, exc)
            engine.device_name = "cpu"
            if any(getattr(param, "is_meta", False) for param in model.parameters()):
                model = _load_weights(AutoModelForDepthEstimation, resolved_id)
            else:
                model = model.to(torch.device("cpu"))
        model.eval()
        engine._model = model
        return engine

    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        if self._model is None or self._processor is None or self._torch is None:
            raise DepthEngineError("Depth model is not loaded")

        from PIL import Image
        import cv2

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self._processor(images=image, return_tensors="pt")
        device = self._torch.device(self.device_name)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        torch = self._torch
        with torch.inference_mode():
            predicted = self._model(**inputs).predicted_depth

        if predicted.ndim == 2:
            predicted = predicted.unsqueeze(0)
        height, width = bgr_frame.shape[:2]
        resized = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze().detach().cpu().numpy().astype(np.float32)
