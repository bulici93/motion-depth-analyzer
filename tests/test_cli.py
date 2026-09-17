from depth_capture.cli import _build_parser


def test_cli_parser_defaults() -> None:
    args = _build_parser().parse_args(["-i", "clip.mp4"])
    assert args.input == "clip.mp4"
    assert args.scale == 0.5
    assert args.stride == 1
    assert args.no_audio is False
