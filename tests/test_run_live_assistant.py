from __future__ import annotations

import gc
import os
import time
import warnings
import weakref
from multiprocessing import resource_tracker
from pathlib import Path
from threading import Event

import pytest

from scripts.run_live_assistant import (
    _make_gui_shutdown_handler,
    _remove_owned_pid_file,
    _run,
    _RuntimeProcessController,
    _ScanStartRollbackError,
    _write_pid_file,
    build_parser,
)


class _Root:
    def __init__(self) -> None:
        self.callback = None

    def after_idle(self, callback) -> None:
        self.callback = callback


@pytest.mark.parametrize("natural_end", [False, True])
def test_stopped_controller_releases_ipc_even_when_callbacks_form_a_cycle(
    tmp_path: Path, natural_end: bool,
) -> None:
    gc.collect()
    controller = _RuntimeProcessController(tmp_path / "unused.json", max_frames=1)
    references = [weakref.ref(controller.snapshots), weakref.ref(controller.commands)]
    if natural_end:
        process = _DeadProcess()
        process.exitcode = 0
        controller.process = process
        controller.scanning = True
        controller.worker_stopped = controller.context.Event()
        references.append(weakref.ref(controller.worker_stopped))
        assert controller.ensure_running() is None
        assert not controller.scanning
    controller._start_worker = lambda owner=controller: None  # type: ignore[assignment]
    controller.stop()
    assert all(reference() is None for reference in references)
    assert isinstance(controller.stopped, Event)
    del controller
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        # Force the old failure window rather than depending on GC timing in
        # an unrelated later test. No child is started and no screen is read.
        with resource_tracker._resource_tracker._lock:
            gc.collect()
    assert not any("ResourceTracker called reentrantly" in str(item.message) for item in observed)


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


class _StubbornProcess:
    exitcode = None
    pid = 4242

    def __init__(self) -> None:
        self.alive = True
        self.closed = False
        self.terminate_calls = 0
        self.kill_calls = 0

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float = 0) -> None:
        del timeout

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def close(self) -> None:
        self.closed = True


class _TerminableProcess(_StubbornProcess):
    def terminate(self) -> None:
        super().terminate()
        self.alive = False


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
    controller.scanning = True
    controller.worker_stopped = controller.context.Event()
    controller.runtime_session_id = "test-session"
    controller.failure_times = [1.0, 2.0, 3.0]
    starts: list[bool] = []

    def fake_start() -> None:
        starts.append(True)

    controller._start_worker = fake_start  # type: ignore[method-assign]
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


def test_controller_starts_privacy_safe_until_explicit_user_action(
    tmp_path: Path,
) -> None:
    controller = _RuntimeProcessController(
        tmp_path / "live.json",
        max_frames=None,
    )
    starts: list[bool] = []
    controller._start_worker = lambda: starts.append(True)  # type: ignore[method-assign]
    try:
        assert controller.process is None
        assert controller.scanning is False
        assert controller.runtime_session_id is None
        with pytest.raises(RuntimeError, match="开始扫描"):
            controller.request_current_scan()

        controller.start()
        command = controller.commands.get(timeout=1)

        assert starts == [True]
        assert controller.scanning is True
        assert controller.runtime_session_id is not None
        assert command == "scan_current"

        controller.stop_scanning()

        assert controller.scanning is False
        assert controller.process is None
        assert controller.runtime_session_id is None
    finally:
        controller.stop()


def test_stop_scanning_keeps_handle_and_never_claims_success_for_live_worker(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    controller = _RuntimeProcessController(
        tmp_path / "live.json",
        max_frames=None,
    )
    stubborn = _StubbornProcess()
    controller.process = stubborn
    controller.scanning = True
    controller.worker_stopped = controller.context.Event()
    try:
        with pytest.raises(RuntimeError, match="无法确认屏幕扫描进程已经停止"):
            controller.stop_scanning()

        assert controller.process is stubborn
        assert controller.scanning is True
        assert controller.stop_requested is True
        assert controller.restart_suspended is False
        assert stubborn.terminate_calls == 1
        assert stubborn.kill_calls == 1
        assert stubborn.closed is False
        assert "screen capture is off" not in capsys.readouterr().out
        assert "尚未确认" in (controller.ensure_running() or "")
        with pytest.raises(RuntimeError, match="正在停止屏幕扫描"):
            controller.request_current_scan()
    finally:
        stubborn.alive = False
        controller.stop()


def test_start_rolls_back_live_worker_when_command_enqueue_fails(
    tmp_path: Path,
) -> None:
    controller = _RuntimeProcessController(
        tmp_path / "live.json",
        max_frames=None,
    )
    process = _TerminableProcess()

    def fake_start() -> None:
        controller.process = process

    def fail_enqueue(_command: str) -> None:
        raise OSError("queue unavailable")

    controller._start_worker = fake_start  # type: ignore[method-assign]
    controller._enqueue_command = fail_enqueue  # type: ignore[method-assign]

    with pytest.raises(OSError, match="queue unavailable"):
        controller.start()

    assert process.alive is False
    assert process.closed is True
    assert controller.process is None
    assert controller.scanning is False
    assert controller.runtime_session_id is None
    controller.stop()


def test_start_reports_unconfirmed_state_when_rollback_cannot_reap_worker(
    tmp_path: Path,
) -> None:
    controller = _RuntimeProcessController(
        tmp_path / "live.json",
        max_frames=None,
    )
    process = _StubbornProcess()

    def fake_start() -> None:
        controller.process = process

    def fail_enqueue(_command: str) -> None:
        raise OSError("queue unavailable")

    controller._start_worker = fake_start  # type: ignore[method-assign]
    controller._enqueue_command = fail_enqueue  # type: ignore[method-assign]
    try:
        with pytest.raises(_ScanStartRollbackError) as captured:
            controller.start()

        assert captured.value.scan_state_unconfirmed is True
        assert controller.process is process
        assert controller.scanning is True
        assert controller.stop_requested is True
        assert process.alive is True
    finally:
        process.alive = False
        controller.stop()


def test_no_ui_mode_does_not_read_config_without_explicit_consent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_config = tmp_path / "must-not-be-read.json"
    args = build_parser().parse_args(
        [
            "--no-ui",
            "--config",
            str(missing_config),
        ]
    )

    assert _run(args) == 0
    assert not missing_config.exists()
    assert "屏幕扫描未启动" in capsys.readouterr().out


def test_gui_run_does_not_start_capture_before_button_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.ui.live_overlay as overlay_module

    missing_config = tmp_path / "must-not-be-read.json"
    starts: list[bool] = []
    callbacks: dict[str, object] = {}

    class Overlay:
        root = _Root()

        def __init__(self, _snapshots, **kwargs: object) -> None:
            callbacks.update(kwargs)

        def run(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(overlay_module, "LiveAssistantOverlay", Overlay)
    monkeypatch.setattr(
        _RuntimeProcessController,
        "_start_worker",
        lambda _self: starts.append(True),
    )
    monkeypatch.setattr("scripts.run_live_assistant.signal.signal", lambda *_args: None)
    args = build_parser().parse_args(["--config", str(missing_config)])

    assert _run(args) == 0
    assert starts == []
    assert not missing_config.exists()
    assert callable(callbacks["on_start"])
    assert callable(callbacks["on_stop"])


def test_restart_budget_stays_suspended_until_manual_scan(
    tmp_path: Path,
) -> None:
    controller = _RuntimeProcessController(
        tmp_path / "live.json",
        max_frames=None,
    )
    dead = _DeadProcess()
    controller.process = dead
    controller.scanning = True
    controller.worker_stopped = controller.context.Event()
    controller.runtime_session_id = "test-session"
    now = time.monotonic()
    controller.failure_times = [now - 1.0, now - 0.5]
    starts: list[bool] = []
    controller._start_worker = lambda: starts.append(True)  # type: ignore[method-assign]
    try:
        message = controller.ensure_running()

        assert message is not None
        assert controller.restart_suspended is True
        assert starts == []

        repeated = controller.ensure_running()

        assert repeated is not None
        assert starts == []

        controller.request_current_scan()
        command = controller.commands.get(timeout=1)

        assert controller.restart_suspended is False
        assert starts == [True]
        assert command == "scan_current"
    finally:
        controller.stop()


def test_pid_file_is_atomic_and_removed_only_by_owner(tmp_path: Path) -> None:
    path = tmp_path / "live.pid"

    _write_pid_file(path)

    assert path.read_text(encoding="utf-8") == f"{os.getpid()}\n"
    _remove_owned_pid_file(path)
    assert not path.exists()

    path.write_text(f"{os.getpid() + 1}\n", encoding="utf-8")
    _remove_owned_pid_file(path)
    assert path.exists()
