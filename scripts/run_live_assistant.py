from __future__ import annotations

import argparse
import multiprocessing
import os
import queue
import signal
import time
import traceback
import uuid
from pathlib import Path
from threading import Event
from typing import Any

from src.config.live_layout import load_live_layout
from src.pipeline.frame_schedule import FrameSchedule
from src.pipeline.live_runtime import LiveGameRuntime, format_live_snapshot

_RESTART_WINDOW_SECONDS = 30.0
_MAX_FAILURES_IN_WINDOW = 3


class _ScanStartRollbackError(RuntimeError):
    """Starting failed and the spawned capture worker could not be reaped."""

    scan_state_unconfirmed = True


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
    parser.add_argument(
        "--start-scanning",
        action="store_true",
        help=(
            "Explicitly allow screen capture in --no-ui mode. "
            "The desktop UI always starts paused until its button is clicked."
        ),
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-clear", action="store_true")
    parser.add_argument("--overlay-geometry", default="260x480+0+70")
    parser.add_argument("--pid-file", type=Path)
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
        schedule = FrameSchedule(config.interval_seconds)
        while not stopped.is_set() and (max_frames is None or produced < max_frames):
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
            if not stopped.is_set() and (max_frames is None or produced < max_frames):
                schedule.wait_next(stopped.wait)
    except BaseException as exc:  # noqa: BLE001
        runtime.record_runtime_failure(exc)
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
        # This lifetime flag is never sent to a child. A threading Event avoids
        # retaining five named semaphores in GUI/controller callback cycles.
        self.stopped = Event()
        self.worker_stopped: Any | None = None
        self.runtime_session_id: str | None = None
        self.process: Any | None = None
        self.scanning = False
        self.stop_requested = False
        self.restart_suspended = False
        self.started_at = 0.0
        self.failure_times: list[float] = []

    def start(self) -> None:
        """Start a user-authorized scanning session.

        Constructing the controller never creates a capture process.  This
        method is only wired to the overlay's explicit ``开始扫描`` button.
        """

        if self.stopped.is_set():
            raise RuntimeError("assistant controller is already closed")
        if self.scanning and self.process is not None and self.process.is_alive():
            return
        self.scanning = True
        self.stop_requested = False
        self.restart_suspended = False
        self.failure_times.clear()
        self.runtime_session_id = uuid.uuid4().hex
        self.worker_stopped = self.context.Event()
        try:
            _drain_queue(self.snapshots)
            _drain_queue(self.commands)
            self._start_worker()
            self._enqueue_command("scan_current")
            print("[live-assistant] scanning started by user", flush=True)
        except Exception as start_error:
            # Starting is transactional from the privacy boundary's point of
            # view. If anything after process.start() fails, first revoke the
            # worker before telling the UI that no scan was started.
            try:
                self.stop_scanning()
            except Exception as stop_error:
                raise _ScanStartRollbackError(
                    "启动扫描失败，且无法确认已启动的扫描进程已经停止："
                    f"{stop_error}"
                ) from start_error
            raise

    def _start_worker(self) -> None:
        if not self.scanning or self.worker_stopped is None:
            return
        assert self.runtime_session_id is not None
        process = self.context.Process(
            target=_produce_live_snapshots,
            args=(
                self.config_path,
                self.max_frames,
                self.snapshots,
                self.commands,
                self.worker_stopped,
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
        if self.stopped.is_set() or not self.scanning:
            return None
        if self.stop_requested:
            return "停止请求已发送，但尚未确认扫描进程退出；请再次点击“结束扫描”"
        if self.restart_suspended:
            return "识别进程连续异常退出，已停止自动恢复；请点击“重新扫描当前牌局”手动恢复"
        if process is None:
            self._start_worker()
            return None
        if process.is_alive():
            if time.monotonic() - self.started_at >= _RESTART_WINDOW_SECONDS:
                self.failure_times.clear()
            return None
        process.join(timeout=0)
        if self.max_frames is not None and process.exitcode == 0:
            self.process = None
            self.scanning = False
            _close_process_handle(process)
            return None

        now = time.monotonic()
        self.failure_times = [
            failed_at
            for failed_at in self.failure_times
            if now - failed_at < _RESTART_WINDOW_SECONDS
        ]
        self.failure_times.append(now)
        exit_code = process.exitcode
        self.process = None
        _close_process_handle(process)
        if len(self.failure_times) >= _MAX_FAILURES_IN_WINDOW:
            self.restart_suspended = True
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
        self._start_worker()
        return None

    def request_current_scan(self) -> None:
        if self.stopped.is_set() or not self.scanning:
            raise RuntimeError("请先点击“开始扫描”")
        if self.stop_requested:
            raise RuntimeError("正在停止屏幕扫描，不能重新扫描")
        process = self.process
        if process is None or not process.is_alive():
            if process is not None:
                process.join(timeout=0)
                self.process = None
                _close_process_handle(process)
            self.failure_times.clear()
            self.restart_suspended = False
            self._start_worker()
        self._enqueue_command("scan_current")
        print("[live-assistant] current-game scan requested", flush=True)

    def _enqueue_command(self, command: str) -> None:
        try:
            self.commands.put_nowait(command)
        except queue.Full:
            try:
                self.commands.get_nowait()
            except queue.Empty:
                pass
            self.commands.put_nowait(command)

    def stop_scanning(self) -> None:
        """Stop capture while keeping the privacy-safe overlay open."""

        if not self.scanning and self.process is None:
            self.worker_stopped = None
            self.runtime_session_id = None
            return
        self.stop_requested = True
        if self.worker_stopped is not None:
            self.worker_stopped.set()
        process = self.process
        if process is not None:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive():
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
                    process.join(timeout=2.0)
            if process.is_alive():
                # Never discard the only handle or claim that capture is off
                # until the OS confirms the worker has exited. The UI remains
                # in scanning state so the user can retry stopping it.
                self.scanning = True
                raise RuntimeError(
                    "无法确认屏幕扫描进程已经停止"
                    + (f"（pid={process.pid}）" if process.pid is not None else "")
                )
            self.process = None
            _close_process_handle(process)
        self.scanning = False
        self.stop_requested = False
        self.worker_stopped = None
        self.runtime_session_id = None
        self.failure_times.clear()
        self.restart_suspended = False
        _drain_queue(self.snapshots)
        _drain_queue(self.commands)
        print("[live-assistant] scanning stopped; screen capture is off", flush=True)

    def stop(self) -> None:
        if self.stopped.is_set() and self.process is None:
            return
        self.stopped.set()
        try:
            self.stop_scanning()
        except BaseException:
            # Keep the controller operable and retain the process handle when
            # shutdown could not be confirmed.
            self.stopped.clear()
            raise
        channels = (self.snapshots, self.commands)
        self.snapshots = None
        self.commands = None
        for channel in channels:
            try:
                channel.close()
                channel.join_thread()
            except (OSError, ValueError):
                pass
        print("[live-assistant] stopped", flush=True)


def _drain_queue(channel: Any) -> None:
    while True:
        try:
            channel.get_nowait()
        except queue.Empty:
            return
        except (OSError, ValueError):
            return


def _close_process_handle(process: Any) -> None:
    """Release a joined multiprocessing handle before replacing its worker."""

    close = getattr(process, "close", None)
    if not callable(close) or process.is_alive():
        return
    try:
        close()
    except (OSError, ValueError):
        pass


def _write_pid_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"{os.getpid()}\n", encoding="utf-8")
    temporary.replace(path)


def _remove_owned_pid_file(path: Path) -> None:
    try:
        recorded_pid = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return
    if recorded_pid != os.getpid():
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


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


def _run(args: argparse.Namespace) -> int:
    if args.no_ui:
        if not args.start_scanning:
            print(
                "屏幕扫描未启动；无界面模式需要显式添加 --start-scanning。",
                flush=True,
            )
            return 0
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
    overlay = LiveAssistantOverlay(
        controller.snapshots,
        on_close=controller.stop,
        on_start=controller.start,
        on_stop=controller.stop_scanning,
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pid_file is not None:
        _write_pid_file(args.pid_file)
    try:
        return _run(args)
    finally:
        if args.pid_file is not None:
            _remove_owned_pid_file(args.pid_file)


if __name__ == "__main__":
    raise SystemExit(main())
