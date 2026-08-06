from __future__ import annotations

import os
from pathlib import Path

from scripts.run_live_assistant import (
    _make_gui_shutdown_handler,
    _remove_owned_pid_file,
    _RuntimeProcessController,
    _write_pid_file,
)


class _Root:
    def __init__(self) -> None:
        self.callback = None

    def after_idle(self, callback) -> None:
        self.callback = callback


class _Overlay:
    def __init__(self, *, failing_root: bool = False) -> None:
        self.closed = False
        if failing_root:
            self.root = _FailingRoot()
        else:
            self.root = _Root()

    def close(self) -> None:
        self.closed = True


class _FailingRoot:
    def after_idle(self, _callback) -> None:
        raise RuntimeError("Tk already unavailable")


class _Controller:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _DeadProcess:
    exitcode = 1

    def __init__(self) -> None:
        self.joined = False
        self.closed = False

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float = 0) -> None:
        self.joined = True

    def close(self) -> None:
        self.closed = True


def test_gui_shutdown_signal_uses_normal_overlay_close_path() -> None:
    overlay = _Overlay()
    controller = _Controller()

    handler = _make_gui_shutdown_handler(overlay, controller)
    handler(15, None)

    assert overlay.root.callback is not None
    assert controller.stopped is False
    overlay.root.callback()
    assert overlay.closed is True


def test_gui_shutdown_signal_falls_back_to_controller_cleanup() -> None:
    overlay = _Overlay(failing_root=True)
    controller = _Controller()

    handler = _make_gui_shutdown_handler(overlay, controller)
    handler(15, None)

    assert controller.stopped is True


def test_scan_button_restarts_worker_after_restart_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    controller = _RuntimeProcessController(
        tmp_path / "live.json",
        max_frames=None,
    )
    dead = _DeadProcess()
    controller.process = dead
    controller.failure_times = [1.0, 2.0, 3.0]
    starts: list[bool] = []

    def fake_start() -> None:
        starts.append(True)

    controller.start = fake_start  # type: ignore[method-assign]
    try:
        controller.request_current_scan()
        command = controller.commands.get(timeout=1)
    finally:
        controller.stop()

    assert dead.joined is True
    assert dead.closed is True
    assert starts == [True]
    assert controller.failure_times == []
    assert command == "scan_current"


def test_pid_file_is_atomic_and_removed_only_by_owner(tmp_path: Path) -> None:
    path = tmp_path / "live.pid"

    _write_pid_file(path)

    assert path.read_text(encoding="utf-8") == f"{os.getpid()}\n"
    _remove_owned_pid_file(path)
    assert not path.exists()

    path.write_text(f"{os.getpid() + 1}\n", encoding="utf-8")
    _remove_owned_pid_file(path)
    assert path.exists()
