"""Real macOS integration checks which never read any screen pixels."""

from __future__ import annotations

import multiprocessing
import os
import platform
import queue
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="macOS integration")


def test_real_tk_starts_paused_and_buttons_only_call_explicit_hooks() -> None:
    from src.ui.live_overlay import LiveAssistantOverlay

    calls: list[str] = []
    with patch(
        "src.pipeline.live_runtime.MacWindowCapture",
        side_effect=AssertionError("smoke test must not capture the screen"),
    ):
        overlay = LiveAssistantOverlay(
            queue.Queue(),
            on_start=lambda: calls.append("start"),
            on_stop=lambda: calls.append("stop"),
            on_close=lambda: calls.append("close"),
        )
        try:
            overlay.root.withdraw()
            overlay.root.update_idletasks()
            assert not overlay._scanning
            assert calls == []
            assert overlay.scan_toggle_button.cget("text") == "开始扫描"
            overlay.scan_toggle_button.invoke()
            assert calls == ["start"]
            assert overlay._scanning
            overlay.scan_toggle_button.invoke()
            assert calls == ["start", "stop"]
            assert not overlay._scanning
        finally:
            overlay.close()
    assert calls == ["start", "stop", "close"]


def test_real_spawn_worker_exits_after_tk_initialization() -> None:
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        # The production app creates spawned workers after Tk initializes.
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            child_pid = executor.submit(os.getpid).result(timeout=20)
        assert child_pid != os.getpid()
        assert child_pid not in {child.pid for child in multiprocessing.active_children()}
    finally:
        root.destroy()


def test_native_swift_capture_helper_compiles_without_being_started(tmp_path: Path) -> None:
    from src.capture.macos_stream import _ensure_native_binary

    source = Path(__file__).resolve().parents[1] / "native" / "macos_window_stream.swift"
    # Compile and check cache reuse only; executing this binary would request capture.
    binary = _ensure_native_binary(source, tmp_path, compiler=Path("/usr/bin/swiftc"))
    assert binary.is_file() and os.access(binary, os.X_OK)
    assert _ensure_native_binary(source, tmp_path, compiler=Path("/usr/bin/swiftc")) == binary
