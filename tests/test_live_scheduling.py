from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_live_assistant
from src.pipeline.frame_schedule import FrameSchedule
from src.pipeline.live_runtime import LiveGameRuntime


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.delays: list[float] = []

    def __call__(self) -> float:
        return self.now

    def wait(self, delay: float) -> None:
        self.delays.append(delay)
        self.now += delay


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_schedule_rejects_invalid_intervals(interval: float) -> None:
    with pytest.raises(ValueError):
        FrameSchedule(interval)


def test_schedule_deducts_processing_time_and_skips_catchup_bursts() -> None:
    clock = Clock()
    schedule = FrameSchedule(0.1, clock=clock)
    clock.now += 0.04
    schedule.wait_next(clock.wait)
    assert clock.delays == pytest.approx([0.06])
    clock.now += 0.35
    schedule.wait_next(clock.wait)
    assert len(clock.delays) == 1
    clock.now += 0.04
    schedule.wait_next(clock.wait)
    assert clock.delays == pytest.approx([0.06, 0.06])


@pytest.mark.parametrize("stop_during_wait", [False, True])
def test_desktop_uses_target_period_and_interruptible_stop(monkeypatch, stop_during_wait) -> None:
    clock = Clock()
    starts: list[float] = []
    closed: list[bool] = []

    class Runtime:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run_once(self):
            starts.append(clock.now)
            clock.now += 0.04
            return len(starts)

        def close(self) -> None:
            closed.append(True)

    class Stop:
        stopped = False

        def is_set(self) -> bool:
            return self.stopped

        def wait(self, delay: float) -> None:
            if stop_during_wait:
                self.stopped = True
            else:
                clock.wait(delay)

    monkeypatch.setattr(run_live_assistant, "LiveGameRuntime", Runtime)
    monkeypatch.setattr(
        run_live_assistant, "load_live_layout", lambda _: SimpleNamespace(interval_seconds=0.1)
    )
    monkeypatch.setattr(
        run_live_assistant, "FrameSchedule", lambda interval: FrameSchedule(interval, clock=clock)
    )
    run_live_assistant._produce_live_snapshots(
        Path("unused.json"), 3, queue.Queue(maxsize=2), queue.Queue(), Stop(), "synthetic"
    )
    assert starts == pytest.approx([0.0] if stop_during_wait else [0.0, 0.1, 0.2])
    assert closed == [True]


def test_headless_uses_the_same_schedule(monkeypatch) -> None:
    from src.pipeline import live_runtime

    clock = Clock()
    starts: list[float] = []

    def run_once():
        starts.append(clock.now)
        clock.now += 0.04
        return len(starts)

    runtime = object.__new__(LiveGameRuntime)
    runtime.config = SimpleNamespace(interval_seconds=0.1)
    runtime._sleeper = clock.wait
    runtime.run_once = run_once
    monkeypatch.setattr(
        live_runtime, "FrameSchedule", lambda interval: FrameSchedule(interval, clock=clock)
    )
    assert list(runtime.run_loop(max_frames=3)) == [1, 2, 3]
    assert starts == pytest.approx([0.0, 0.1, 0.2])
