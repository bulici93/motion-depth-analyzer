from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from depth_capture.config import DepthCaptureConfig, default_model_id
from depth_capture.pipeline import DepthPipeline, ProgressInfo, unique_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depth-capture",
        description="Frame-by-frame depth capture with Depth Anything V2 Small.",
    )
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", help="Output MP4 path")
    parser.add_argument("--scale", type=float, default=0.5, help="Inference scale (0.25–1.0)")
    parser.add_argument("--stride", type=int, default=1, help="Run the model every N frames")
    parser.add_argument("--gamma", type=float, default=0.7, help="Grayscale gamma")
    parser.add_argument("--crush", type=float, default=35.0, help="Far-background crush percentile")
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.4,
        help="Temporal EMA amount (0=off, higher=more stable)",
    )
    parser.add_argument("--no-audio", action="store_true", help="Do not copy the original audio")
    parser.add_argument(
        "--model-id",
        default=default_model_id(),
        help="Hugging Face model id (default: Depth Anything V2 Small)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Less logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = unique_path(Path("output"), f"{input_path.stem}_depth")

    config = DepthCaptureConfig(
        model_id=args.model_id,
        scale=args.scale,
        stride=args.stride,
        gamma=args.gamma,
        crush_percentile=args.crush,
        temporal_smooth=args.smooth,
        preserve_audio=not args.no_audio,
    )
    pipeline = DepthPipeline(config)
    bar: tqdm | None = None

    def on_progress(update: ProgressInfo) -> None:
        nonlocal bar
        total = update.total_frames or None
        if bar is None:
            bar = tqdm(total=total, unit="frame", disable=args.quiet)
        if total and bar.total != total:
            bar.total = total
        bar.n = update.frame_index
        bar.set_postfix(fps=f"{update.fps:.1f}")
        bar.refresh()

    try:
        result = pipeline.process(input_path, output_path, progress_callback=on_progress)
    except Exception as exc:  # noqa: BLE001
        if bar is not None:
            bar.close()
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if bar is not None:
        bar.close()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
