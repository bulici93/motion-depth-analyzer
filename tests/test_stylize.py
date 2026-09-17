from __future__ import annotations

import numpy as np

from depth_capture.stylize import DepthStylizer, TemporalSmoother, invert_for_polarity, normalize_relative_depth


def test_foreground_blob_is_brighter_than_background() -> None:
    depth = np.zeros((40, 40), dtype=np.float32)
    depth[12:28, 12:28] = 1.0
    gray = DepthStylizer(gamma=1.0, crush_percentile=0.0, temporal_smooth=0.0).stylize(depth)
    assert gray[20, 20] > gray[0, 0]
    assert gray[20, 20] > 200
    assert gray[0, 0] < 40


def test_constant_depth_is_black() -> None:
    depth = np.full((16, 16), 3.0, dtype=np.float32)
    gray = DepthStylizer(temporal_smooth=0.0).stylize(depth)
    assert gray.max() == 0


def test_inverted_raw_depth_still_maps_near_to_white() -> None:
    depth = np.ones((50, 50), dtype=np.float32)
    depth[20:30, 20:30] = 0.0
    gray = DepthStylizer(gamma=1.0, crush_percentile=0.0, temporal_smooth=0.0).stylize(depth)
    assert gray[25, 25] > gray[2, 2]


def test_polarity_lock_ignores_later_median_flip() -> None:
    stylizer = DepthStylizer(gamma=1.0, crush_percentile=0.0, temporal_smooth=0.0)
    first = np.zeros((30, 30), dtype=np.float32)
    first[10:20, 10:20] = 1.0
    stylizer.stylize(first)
    assert stylizer._invert is False

    second = np.full((30, 30), 1.0, dtype=np.float32)
    second[0:5, :] = 0.0
    gray = stylizer.stylize(second)
    assert gray[20, 15] > gray[1, 15]


def test_temporal_smoother_blends_previous_frame() -> None:
    smoother = TemporalSmoother(0.5)
    first = smoother.apply(np.ones((4, 4), dtype=np.float32))
    second = smoother.apply(np.zeros((4, 4), dtype=np.float32))
    assert np.allclose(first, 1.0)
    assert np.allclose(second, 0.5)


def test_normalize_rejects_empty_range() -> None:
    assert normalize_relative_depth(np.ones((4, 4), dtype=np.float32)) is None


def test_invert_for_polarity_near_white() -> None:
    bright_bg = np.ones((10, 10), dtype=np.float32)
    dark_bg = np.zeros((10, 10), dtype=np.float32)
    assert invert_for_polarity(bright_bg, "near_white") is True
    assert invert_for_polarity(dark_bg, "near_white") is False
