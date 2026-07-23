from __future__ import annotations

import argparse
from pathlib import Path

from src.capture.recorded_window import RecordedWindowFrameSource
from src.pipeline.live_layout import live_layout_from_dict, load_live_layout
from src.pipeline.live_runtime import LiveGameRuntime, format_live_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one recorded Phase 6 full-window session."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/live_game.local.json"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/live-replay"),
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = load_live_layout(args.config)
    payload = base.to_payload()
    payload["log_file"] = (args.output_dir / "events.jsonl").as_posix()
    payload["error_frames_dir"] = (args.output_dir / "errors").as_posix()
    config = live_layout_from_dict(payload)
    source = RecordedWindowFrameSource(args.manifest, app_name=config.app_name)
    max_frames = source.frame_count
    if args.max_frames is not None:
        max_frames = min(max_frames, args.max_frames)
    runtime = LiveGameRuntime(
        config,
        frame_source=source,
        sleeper=lambda _: None,
    )
    last = None
    try:
        for snapshot in runtime.run_loop(max_frames=max_frames):
            last = snapshot
            if not args.quiet:
                print(format_live_snapshot(snapshot))
                print()
    finally:
        runtime.close()
    if last is None:
        raise SystemExit("recorded replay produced no frames")
    print(f"replayed_frames: {max_frames}")
    print(f"events_log: {config.log_file}")
    print(f"final_mode: {last.tracker_update.mode.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
