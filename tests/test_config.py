from depth_capture.config import DEFAULT_MODEL_ID, DepthCaptureConfig, default_model_id


def test_config_clamps_values() -> None:
    config = DepthCaptureConfig(
        scale=4.0,
        stride=0,
        gamma=0.01,
        crush_percentile=120,
        temporal_smooth=2.0,
        polarity="rainbow",
        model_id="  ",
    )
    assert config.scale == 1.0
    assert config.stride == 1
    assert config.gamma == 0.1
    assert config.crush_percentile == 90.0
    assert config.temporal_smooth == 0.95
    assert config.polarity == "near_white"
    assert config.model_id == default_model_id()


def test_default_model_id() -> None:
    assert DEFAULT_MODEL_ID.endswith("Depth-Anything-V2-Small-hf")
