from __future__ import annotations

import argparse
import multiprocessing
import queue
import signal
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from src.pipeline.live_layout import load_live_layout
from src.pipeline.live_runtime import LiveGameRuntime, format_live_snapshot

_RESTART_WINDOW_SECONDS = 30.0
_MAX_FAILURES_IN_WINDOW = 3


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
    parser.add_argument("--overlay-geometry", default="260x480+0+70")
    return parser


def _produce_live_snapshots(
    config_path: Path,
    max_frames: int | None,
    snapshots: Any,
    commands: Any,
    stopped: Any,
    runtime_session_id: str,
) -> None:
    """Run macOS capture outside the Tk process.

    Tk initializes CoreFoundation in the UI process.  Keeping all
    ``screencapture``/Vision helper launches in this separate process avoids
    unsafe fork-after-GUI behavior on macOS.
    """

    config = load_live_layout(config_path)
    runtime = LiveGameRuntime(
        config,
        resume_session_id=runtime_session_id,
    )
    try:
        produced = 0
        while (
            not stopped.is_set()
            and (max_frames is None or produced < max_frames)
        ):
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break
                if command == "scan_current":
                    runtime.request_current_scan()
            snapshot = runtime.run_once()
            while True:
                try:
                    snapshots.put_nowait(snapshot)
                    break
                except queue.Full:
                    try:
                        snapshots.get_nowait()
                    except queue.Empty:
                        pass
            produced += 1
            if (
                not stopped.is_set()
                and (max_frames is None or produced < max_frames)
            ):
                time.sleep(config.interval_seconds)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        raise
    finally:
        runtime.close()


class _RuntimeProcessController:
    def __init__(
        self,
        config_path: Path,
        *,
        max_frames: int | None,
    ) -> None:
        self.context = multiprocessing.get_context("spawn")
        self.config_path = config_path
        self.max_frames = max_frames
        self.snapshots = self.context.Queue(maxsize=2)
        self.commands = self.context.Queue(maxsize=4)
        self.stopped = self.context.Event()
        self.runtime_session_id = uuid.uuid4().hex
        self.process: Any | None = None
        self.started_at = 0.0
        self.failure_times: list[float] = []

    def start(self) -> None:
        process = self.context.Process(
            target=_produce_live_snapshots,
            args=(
                self.config_path,
                self.max_frames,
                self.snapshots,
                self.commands,
                self.stopped,
                self.runtime_session_id,
            ),
            name="doudizhu-live-runtime",
        )
        process.start()
        self.process = process
        self.started_at = time.monotonic()
        print(
            f"[live-assistant] recognition process started pid={process.pid}",
            flush=True,
        )

    def ensure_running(self) -> str | None:
        process = self.process
        if self.stopped.is_set() or process is None:
            return None
        if process.is_alive():
            if time.monotonic() - self.started_at >= _RESTART_WINDOW_SECONDS:
                self.failure_times.clear()
            return None
        process.join(timeout=0)
        if self.max_frames is not None and process.exitcode == 0:
            return None

        now = time.monotonic()
        self.failure_times = [
            failed_at
            for failed_at in self.failure_times
            if now - failed_at < _RESTART_WINDOW_SECONDS
        ]
        self.failure_times.append(now)
        exit_code = process.exitcode
        if len(self.failure_times) >= _MAX_FAILURES_IN_WINDOW:
            return (
                "识别进程连续异常退出，已停止自动恢复；"
                f"最后退出码={exit_code}，请查看 logs/live_assistant.stdout.log"
            )
        print(
            "[live-assistant] recognition process exited "
            f"code={exit_code}; restarting "
            f"({len(self.failure_times)}/{_MAX_FAILURES_IN_WINDOW})",
            flush=True,
        )
        self.start()
        return None

    def request_current_scan(self) -> None:
        if self.stopped.is_set():
            return
        process = self.process
        if process is None or not process.is_alive():
            if process is not None:
                process.join(timeout=0)
            self.failure_times.clear()
            self.start()
        try:
            self.commands.put_nowait("scan_current")
        except queue.Full:
            try:
                self.commands.get_nowait()
            except queue.Empty:
                pass
            self.commands.put_nowait("scan_current")
        print("[live-assistant] current-game scan requested", flush=True)

    def stop(self) -> None:
        if self.stopped.is_set():
            return
        self.stopped.set()
        process = self.process
        if process is not None:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for channel in (self.snapshots, self.commands):
            try:
                channel.close()
                channel.join_thread()
            except (OSError, ValueError):
                pass
        print("[live-assistant] stopped", flush=True)


def _make_gui_shutdown_handler(
    overlay: Any,
    controller: _RuntimeProcessController,
) -> Any:
    """Route process signals through the normal Tk/controller cleanup path."""

    def handle_signal(_signum: int, _frame: Any) -> None:
        try:
            overlay.root.after_idle(overlay.close)
        except Exception:  # noqa: BLE001
            controller.stop()

    return handle_signal


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_ui:
        config = load_live_layout(args.config)
        runtime = LiveGameRuntime(
            config,
            resume_session_id=uuid.uuid4().hex,
        )
        try:
            for snapshot in runtime.run_loop(max_frames=args.max_frames):
                if not args.no_clear:
                    print("\033[2J\033[H", end="")
                print(format_live_snapshot(snapshot), flush=True)
        finally:
            runtime.close()
        return 0

    from src.ui.live_overlay import LiveAssistantOverlay

    controller = _RuntimeProcessController(
        args.config,
        max_frames=args.max_frames,
    )
    controller.start()
    overlay = LiveAssistantOverlay(
        controller.snapshots,
        on_close=controller.stop,
        on_scan=controller.request_current_scan,
        health_check=controller.ensure_running,
        geometry=args.overlay_geometry,
    )
    shutdown_handler = _make_gui_shutdown_handler(overlay, controller)
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    try:
        overlay.run()
    finally:
        controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
