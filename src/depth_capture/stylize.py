from __future__ import annotations

import numpy as np


def normalize_relative_depth(raw_depth: np.ndarray) -> np.ndarray | None:
    depth = raw_depth.astype(np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return None
    valid = depth[finite]
    lo = float(np.percentile(valid, 1))
    hi = float(np.percentile(valid, 99))
    if hi - lo < 1e-6:
        return None
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[finite] = np.clip((valid - lo) / (hi - lo), 0.0, 1.0)
    return normalized


def invert_for_polarity(normalized: np.ndarray, polarity: str = "near_white") -> bool:
    """Background usually dominates. Near-white: make the median (background) dark."""
    majority_is_bright = float(np.median(normalized)) > 0.5
    if polarity == "near_black":
        return not majority_is_bright
    return majority_is_bright


class TemporalSmoother:
    def __init__(self, amount: float = 0.4) -> None:
        self.amount = float(min(max(amount, 0.0), 0.95))
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def apply(self, current: np.ndarray) -> np.ndarray:
        if self.amount <= 0 or self._prev is None or self._prev.shape != current.shape:
            blended = current.astype(np.float32, copy=True)
        else:
            blended = ((1.0 - self.amount) * current + self.amount * self._prev).astype(np.float32)
        self._prev = blended
        return blended


class DepthStylizer:
    """Map relative depth to 0–255 grayscale: near white, far black."""

    def __init__(
        self,
        gamma: float = 0.7,
        crush_percentile: float = 35.0,
        temporal_smooth: float = 0.4,
        polarity: str = "near_white",
    ) -> None:
        self.gamma = max(float(gamma), 0.1)
        self.crush_percentile = float(min(max(crush_percentile, 0.0), 90.0))
        self.polarity = polarity
        self.smoother = TemporalSmoother(temporal_smooth)
        self._invert: bool | None = None

    def reset(self) -> None:
        self.smoother.reset()
        self._invert = None

    def stylize(self, raw_depth: np.ndarray) -> np.ndarray:
        normalized = normalize_relative_depth(raw_depth)
        if normalized is None:
            return np.zeros(raw_depth.shape[:2], dtype=np.uint8)

        if self._invert is None:
            self._invert = invert_for_polarity(normalized, self.polarity)
        if self._invert:
            normalized = 1.0 - normalized

        normalized = self.smoother.apply(normalized)
        floor = float(np.percentile(normalized, self.crush_percentile))
        crushed = np.clip((normalized - floor) / (1.0 - floor + 1e-6), 0.0, 1.0)
        gray = np.power(crushed, self.gamma) * 255.0
        return np.clip(gray, 0, 255).astype(np.uint8)
