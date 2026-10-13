"""Shared target-period scheduling for desktop and headless capture loops."""

from __future__ import annotations

import math
import time
from collections.abc import Callable


class FrameSchedule:
    """Deduct processing time and avoid bursts of catch-up frames after overruns.

    Create immediately before the first frame. ``wait`` is injected so desktop
    callers can use a stop event's interruptible wait without depending on Tk.
    """

    def __init__(
        self,
        interval_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("frame interval must be finite and positive")
        self.interval_seconds = interval_seconds
        self._clock = clock or time.monotonic
        self._next_frame_at = self._clock()

    def wait_next(self, wait: Callable[[float], object]) -> None:
        self._next_frame_at += self.interval_seconds
        now = self._clock()
        delay = self._next_frame_at - now
        if delay > 0:
            wait(delay)
        else:
            self._next_frame_at = now
