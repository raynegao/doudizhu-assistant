from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.capture.screen_geometry import MacWindowCapture
from src.pipeline.live_layout import load_live_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record local-only Phase 6 full-window and ROI frames."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/live_game.local.json"),
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--interval", type=float)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/live_game/recordings"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    config = load_live_layout(args.config)
    interval = config.interval_seconds if args.interval is None else args.interval
    if interval <= 0:
        raise SystemExit("--interval must be positive")
    session_dir = args.output_root / args.session
    frames_dir = session_dir / "frames"
    roi_root = session_dir / "rois"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "manifest.jsonl"
    capture = MacWindowCapture(config.app_name)
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for frame_id in range(1, args.frames + 1):
            frame = capture.capture(frame_id)
            frame_name = f"{frame_id:06d}.png"
            full_path = frames_dir / frame_name
            frame.image.save(full_path)
            roi_paths: dict[str, str] = {}
            for name in sorted(config.rois):
                directory = roi_root / name
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / frame_name
                config.crop(frame.image, name).save(path)
                roi_paths[name] = path.relative_to(session_dir).as_posix()
            manifest.write(json.dumps({
                "event": "recorded_frame",
                "session": args.session,
                "frame_id": frame_id,
                "timestamp": frame.timestamp,
                "full_image": full_path.relative_to(session_dir).as_posix(),
                "rois": roi_paths,
                "labels": {},
            }, ensure_ascii=False) + "\n")
            manifest.flush()
            print(f"\rrecorded {frame_id}/{args.frames}", end="", flush=True)
            if frame_id < args.frames:
                time.sleep(interval)
    print(f"\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
