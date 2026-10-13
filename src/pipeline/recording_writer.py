"""Bounded FIFO persistence, isolated from the recording acquisition clock."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from threading import Condition, Thread, current_thread
from typing import Generic, TypeVar

T = TypeVar("T")


class RecordingWriterError(RuntimeError):
    """Persistence failed; retained partial evidence must not be finalized."""


class RecordingBackpressure(RecordingWriterError):
    """The queue filled; stop instead of dropping or retiming frames."""


class RecordingWriter(Generic[T]):
    """Transfer ownership from one producer to one ordered consumer.

    submit never waits for IO. close drains accepted items and propagates worker
    failure. On failure, unwritten items are counted, not retried into a broken
    sink. The caller closes the sink after close, even if close raises. Callbacks
    must not submit to or close this writer. Capacity excludes the active item.
    """

    def __init__(self, write: Callable[[T], None], *, max_pending: int) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self._write = write
        self._max_pending = max_pending
        self._condition = Condition()
        self._pending: deque[T] = deque()
        self._closing = False
        self._error: BaseException | None = None
        self._accepted = 0
        self._written = 0
        self._high_water = 0
        self._max_write_seconds = 0.0
        self._thread = Thread(target=self._run, name="recording-writer")
        self._thread.start()

    def submit(self, item: T) -> None:
        with self._condition:
            self._raise_failure()
            if self._closing:
                raise RecordingWriterError("recording writer is closed")
            if len(self._pending) >= self._max_pending:
                raise RecordingBackpressure(
                    f"recording persistence queue is full ({self._max_pending} frames); "
                    "stopping without dropping accepted frames"
                )
            self._pending.append(item)
            self._accepted += 1
            self._high_water = max(self._high_water, len(self._pending))
            self._condition.notify()

    def check(self) -> None:
        with self._condition:
            self._raise_failure()

    def close(self) -> None:
        if current_thread() is self._thread:
            raise RecordingWriterError("recording writer cannot close itself")
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join()
        self.check()

    def diagnostics(self) -> dict[str, int | float]:
        with self._condition:
            return {
                "queue_capacity_frames": self._max_pending,
                "queue_high_water_frames": self._high_water,
                "accepted_frames": self._accepted,
                "written_frames": self._written,
                "unwritten_frames": self._accepted - self._written,
                "max_write_seconds": self._max_write_seconds,
            }

    def _raise_failure(self) -> None:
        if self._error is not None:
            raise RecordingWriterError(
                f"recording persistence failed: {type(self._error).__name__}: {self._error}"
            ) from self._error

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: bool(self._pending) or self._closing)
                if not self._pending:
                    return
                item = self._pending.popleft()
            started = time.monotonic()
            try:
                self._write(item)
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._closing = True
                    self._pending.clear()
                    self._condition.notify_all()
                return
            finally:
                elapsed = time.monotonic() - started
                with self._condition:
                    self._max_write_seconds = max(self._max_write_seconds, elapsed)
            with self._condition:
                self._written += 1
            del item
