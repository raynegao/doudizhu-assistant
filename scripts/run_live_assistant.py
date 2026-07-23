from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path

from src.pipeline.live_layout import load_live_layout
from src.pipeline.live_runtime import LiveGameRuntime, LiveRuntimeSnapshot, format_live_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 6 read-only macOS live Dou Dizhu assistant."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/live_game.local.json"),
    )
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-clear", action="store_true")
    parser.add_argument("--overlay-geometry", default="250x430+0+70")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_live_layout(args.config)
    runtime = LiveGameRuntime(config)
    if args.no_ui:
        try:
            for snapshot in runtime.run_loop(max_frames=args.max_frames):
                if not args.no_clear:
                    print("\033[2J\033[H", end="")
                print(format_live_snapshot(snapshot), flush=True)
        finally:
            runtime.close()
        return 0

    from src.ui.live_overlay import LiveAssistantOverlay

    snapshots: "queue.Queue[LiveRuntimeSnapshot]" = queue.Queue(maxsize=2)
    stopped = threading.Event()
    failure: list[BaseException] = []

    def produce() -> None:
        try:
            for snapshot in runtime.run_loop(max_frames=args.max_frames):
                if stopped.is_set():
                    break
                while True:
                    try:
                        snapshots.put_nowait(snapshot)
                        break
                    except queue.Full:
                        try:
                            snapshots.get_nowait()
                        except queue.Empty:
                            pass
        except BaseException as exc:  # noqa: BLE001
            failure.append(exc)
            stopped.set()
        finally:
            runtime.close()

    worker = threading.Thread(
        target=produce,
        daemon=True,
        name="doudizhu-live-runtime",
    )
    worker.start()

    def stop() -> None:
        stopped.set()
        runtime.close()

    overlay = LiveAssistantOverlay(
        snapshots,
        on_close=stop,
        geometry=args.overlay_geometry,
    )
    overlay.run()
    worker.join(timeout=2.0)
    if failure:
        raise failure[0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
