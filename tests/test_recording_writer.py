from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from threading import Event, Thread

import pytest

from src.pipeline.recording_writer import (
    RecordingBackpressure,
    RecordingWriter,
    RecordingWriterError,
)


def test_queue_does_not_block_producer_and_close_drains_fifo() -> None:
    entered, release, closed = Event(), Event(), Event()
    written: list[int] = []

    def persist(value: int) -> None:
        if value == 1:
            entered.set()
            assert release.wait(3)
        written.append(value)

    writer = RecordingWriter(persist, max_pending=2)
    closer = Thread(target=lambda: (writer.close(), closed.set()))
    try:
        writer.submit(1)
        assert entered.wait(1)
        # These submissions must return while the sink remains blocked.
        writer.submit(2)
        writer.submit(3)
        with pytest.raises(RecordingBackpressure, match="queue is full"):
            writer.submit(4)
        assert written == []
        closer.start()
        assert not closed.wait(0.02)
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(3)
        writer.close()
    assert closed.is_set()
    assert written == [1, 2, 3]
    diagnostics = writer.diagnostics()
    assert diagnostics["queue_high_water_frames"] == 2
    assert diagnostics["accepted_frames"] == diagnostics["written_frames"] == 3
    assert diagnostics["unwritten_frames"] == 0
    with pytest.raises(RecordingWriterError, match="closed"):
        writer.submit(5)


def test_worker_failure_is_propagated_without_deadlock_or_retry() -> None:
    entered, release = Event(), Event()
    calls: list[int] = []

    def persist(value: int) -> None:
        calls.append(value)
        entered.set()
        assert release.wait(3)
        raise OSError("injected disk failure")

    writer = RecordingWriter(persist, max_pending=1)
    try:
        writer.submit(1)
        assert entered.wait(1)
        writer.submit(2)
    finally:
        release.set()
        with pytest.raises(RecordingWriterError, match="injected disk failure") as exc:
            writer.close()
    assert isinstance(exc.value.__cause__, OSError)
    assert calls == [1]
    assert writer.diagnostics()["unwritten_frames"] == 2
    with pytest.raises(RecordingWriterError, match="injected disk failure"):
        writer.submit(3)
    with pytest.raises(RecordingWriterError, match="injected disk failure"):
        writer.close()


def test_empty_writer_closes_and_invalid_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        RecordingWriter(lambda _: None, max_pending=0)
    writer = RecordingWriter(lambda _: None, max_pending=1)
    writer.close()
    writer.close()
    assert writer.diagnostics()["accepted_frames"] == 0


def test_full_unconsumed_stdout_pipe_does_not_block_progress_or_writer_exit() -> None:
    probe = (
        "from scripts.record_live_game import _progress_print\n"
        "from src.pipeline.recording_writer import RecordingWriter\n"
        "def persist(_):\n"
        "    for _ in range(2000):\n"
        "        _progress_print('x' * 10000, flush=True)\n"
        "writer = RecordingWriter(persist, max_pending=1)\n"
        "writer.submit(1)\n"
        "writer.close()\n"
        "assert writer.diagnostics()['written_frames'] == 1\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Do not consume stdout until the child exits: the pipe must fill.
        process.wait(timeout=5)
        _stdout, stderr = process.communicate(timeout=1)
        assert process.returncode == 0, stderr.decode(errors="replace")
    except subprocess.TimeoutExpired:
        pytest.fail("full progress pipe blocked the evidence writer")
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=2)
