from __future__ import annotations

import argparse
from pathlib import Path

import sys

from src.pipeline import DEFAULT_APP_NAME, WindowLookupError, calibrate_roi, find_window, save_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate Phase 3.5 runtime ROI from a visible Dou Dizhu window.")
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME, help="macOS process/app name to inspect.")
    parser.add_argument("--save-config", default="configs/phase3_runtime.local.json", help="Output runtime config path.")
    parser.add_argument("--offset-x", type=int, default=33, help="Hand ROI x offset from the window left.")
    parser.add_argument("--offset-y", type=int, default=473, help="Hand ROI y offset from the window top.")
    parser.add_argument("--roi-height", type=int, default=203, help="Hand ROI height in screen points.")
    parser.add_argument("--count", type=int, default=17)
    parser.add_argument("--start-x", type=int, default=0)
    parser.add_argument("--start-y", type=int, default=10)
    parser.add_argument("--step-x", type=int, default=60)
    parser.add_argument("--crop-size", default="63x105", help="Visible crop size WIDTHxHEIGHT.")
    return parser


def _parse_crop_size(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        window = find_window(args.app_name)
    except WindowLookupError as exc:
        print(f"窗口错误: {exc}", file=sys.stderr)
        return 2
    calibration = calibrate_roi(
        window,
        offset=(args.offset_x, args.offset_y),
        roi_height=args.roi_height,
        count=args.count,
        start_x=args.start_x,
        start_y=args.start_y,
        step_x=args.step_x,
        crop_size=_parse_crop_size(args.crop_size),
    )
    path = Path(args.save_config)
    save_runtime_config(path, calibration)
    print(f"window_name: {calibration.window_name}")
    print(f"window_box: {calibration.window_box}")
    print(f"roi_box: {calibration.roi_box}")
    print(f"count: {calibration.count}")
    print(f"step_x: {calibration.step_x}")
    print(f"crop_size: {calibration.crop_size}")
    print(f"saved_config: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
