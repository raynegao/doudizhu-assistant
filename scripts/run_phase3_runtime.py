from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.crop_hand_roi_cards import parse_crop_size
from scripts.replay_phase2 import parse_roi_box
from src.pipeline import (
    DEFAULT_CARD_COUNT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CROP_SIZE,
    DEFAULT_ROI_BOX,
    DEFAULT_START_X,
    DEFAULT_START_Y,
    DEFAULT_STEP_X,
    Phase3Runtime,
    RuntimeCaptureError,
    RuntimeSettings,
    format_runtime_event,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3 fixed-ROI Mac screen runtime.")
    parser.add_argument("--model", default="models/card_cnn.pt", help="PyTorch checkpoint path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda.")
    parser.add_argument("--roi-box", default=",".join(str(value) for value in DEFAULT_ROI_BOX), help="Screen ROI box: left,top,right,bottom.")
    parser.add_argument("--count", type=int, default=DEFAULT_CARD_COUNT, help="Number of hand cards to crop from ROI.")
    parser.add_argument("--start-x", type=int, default=DEFAULT_START_X)
    parser.add_argument("--start-y", type=int, default=DEFAULT_START_Y)
    parser.add_argument("--step-x", type=int, default=DEFAULT_STEP_X)
    parser.add_argument("--crop-size", type=parse_crop_size, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--last-play", default="", help="Previous play, empty means lead play.")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames; omit for continuous runtime.")
    parser.add_argument("--log-file", default="logs/phase3_runtime.jsonl", help="JSONL runtime log path.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between frames.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = RuntimeSettings(
        model_path=Path(args.model),
        device_name=args.device,
        roi_box=parse_roi_box(args.roi_box),
        count=args.count,
        start_x=args.start_x,
        start_y=args.start_y,
        step_x=args.step_x,
        crop_size=args.crop_size,
        confidence_threshold=args.confidence_threshold,
        last_play=args.last_play,
        log_file=Path(args.log_file) if args.log_file else None,
    )

    try:
        runtime = Phase3Runtime(settings)
        for event in runtime.run_loop(max_frames=args.max_frames, interval=args.interval):
            if not args.no_clear:
                print("\033[2J\033[H", end="")
            print(format_runtime_event(event), flush=True)
    except KeyboardInterrupt:
        print("\nPhase 3 runtime stopped.")
        return 0
    except RuntimeCaptureError as exc:
        print(f"截图错误: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"文件错误: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"运行参数错误: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
