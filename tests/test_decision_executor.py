from __future__ import annotations

import multiprocessing
import os
import pickle
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.connection import Connection
from threading import Event, Thread

import pytest

from src.logic.monte_carlo import MonteCarloSettings, recommend_phase4
from src.pipeline.decision_executor import DecisionExecutor
from src.state.observable_state import ObservableGameState


def _worker_identity(value: str) -> tuple[int, str, str]:
    return os.getpid(), multiprocessing.get_start_method(), value


def _raise_worker_error() -> None:
    raise ValueError("decision failed")


def _hold_process(connection: Connection) -> str:
    connection.send(("running", os.getpid()))
    if not connection.poll(15):
        raise TimeoutError("parent did not release worker")
    assert connection.recv() == "release"
    return "running"


def _report_process(connection: Connection, value: str) -> str:
    connection.send((value, os.getpid()))
    return value


def _exit_process(connection: Connection) -> None:
    connection.send(("running", os.getpid()))
    if not connection.poll(15):
        raise TimeoutError("parent did not release worker")
    assert connection.recv() == "release"
    os._exit(17)


def _nested_executor_probe(connection: Connection) -> None:
    with DecisionExecutor() as executor:
        identity = executor.submit(
            "nested",
            _worker_identity,
            "nested-ready",
        ).result(timeout=15)
        connection.send(identity)
    connection.close()


def test_default_executor_uses_spawn_process_and_reaps_it(
    phase4_ready_state: ObservableGameState,
) -> None:
    executor = DecisionExecutor()

    worker_pid, start_method, value = executor.submit(
        ("round-1", 0),
        _worker_identity,
        "ready",
    ).result(timeout=15)

    assert value == "ready"
    assert worker_pid != os.getpid()
    assert start_method == "spawn"

    decision = executor.submit(
        ("round-1", 1),
        recommend_phase4,
        phase4_ready_state,
        MonteCarloSettings(
            simulations=2,
            max_depth=4,
            time_budget_ms=0,
            top_k=1,
            max_candidates=2,
            min_rollouts_per_action=1,
        ),
    ).result(timeout=15)

    assert decision.requested_simulations == 2
    assert decision.completed_simulations == 2

    executor.close()

    assert executor.closed is True
    assert worker_pid not in {child.pid for child in multiprocessing.active_children()}


def test_worker_exception_is_propagated_by_future() -> None:
    with DecisionExecutor() as executor:
        future = executor.submit(("round-1", 1), _raise_worker_error)

        with pytest.raises(ValueError, match="decision failed"):
            future.result(timeout=15)


def test_executor_can_spawn_and_reap_worker_inside_spawned_runtime() -> None:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    runtime = context.Process(target=_nested_executor_probe, args=(sender,))
    runtime.start()
    sender.close()
    try:
        assert receiver.poll(20)
        worker_pid, start_method, value = receiver.recv()
        runtime.join(timeout=20)

        assert runtime.exitcode == 0
        assert value == "nested-ready"
        assert start_method == "spawn"
        assert worker_pid != runtime.pid
    finally:
        receiver.close()
        if runtime.is_alive():
            runtime.terminate()
            runtime.join(timeout=5)


def test_process_backend_reports_pickle_error_without_breaking_pool() -> None:
    local_value = 7

    with DecisionExecutor() as executor:
        unpickleable = executor.submit("closure", lambda: local_value)

        with pytest.raises((AttributeError, pickle.PicklingError), match="pickle|local"):
            unpickleable.result(timeout=15)

        worker_pid, start_method, value = executor.submit(
            "recovery",
            _worker_identity,
            "ready",
        ).result(timeout=15)

        assert worker_pid != os.getpid()
        assert start_method == "spawn"
        assert value == "ready"


def test_submit_cancels_queued_stale_task_and_keeps_latest() -> None:
    release_running = Event()
    running_started = Event()

    def hold_worker() -> str:
        running_started.set()
        assert release_running.wait(timeout=5)
        return "running"

    backend = ThreadPoolExecutor(max_workers=1)
    executor = DecisionExecutor(executor=backend)
    try:
        running = executor.submit("running", hold_worker)
        assert running_started.wait(timeout=2)

        stale = executor.submit("stale", lambda: "stale")
        latest = executor.submit("latest", lambda: "latest")

        assert stale.cancelled() is True
        with pytest.raises(CancelledError):
            stale.result()
        assert executor.latest_key == "latest"
        assert executor.tracked_keys == ("latest",)

        release_running.set()
        assert running.result(timeout=2) == "running"
        assert latest.result(timeout=2) == "latest"
    finally:
        release_running.set()
        executor.close()


def test_same_key_is_idempotent_while_task_is_pending() -> None:
    release_running = Event()
    running_started = Event()

    def hold_worker() -> str:
        running_started.set()
        assert release_running.wait(timeout=5)
        return "done"

    executor = DecisionExecutor(executor=ThreadPoolExecutor(max_workers=1))
    try:
        first = executor.submit(("round-1", 2), hold_worker)
        assert running_started.wait(timeout=2)
        duplicate = executor.submit(("round-1", 2), lambda: "duplicate")

        assert duplicate is first
        release_running.set()
        assert first.result(timeout=2) == "done"
    finally:
        release_running.set()
        executor.close()


def test_cancel_stale_can_invalidate_latest_running_task() -> None:
    release_running = Event()
    running_started = Event()

    def hold_worker() -> str:
        running_started.set()
        assert release_running.wait(timeout=5)
        return "done"

    executor = DecisionExecutor(executor=ThreadPoolExecutor(max_workers=1))
    try:
        future = executor.submit("old", hold_worker)
        assert running_started.wait(timeout=2)

        assert executor.cancel_stale() == 0
        assert executor.latest_key is None
        assert executor.tracked_keys == ()

        release_running.set()
        assert future.result(timeout=2) == "done"
    finally:
        release_running.set()
        executor.close()


def test_thread_executor_accepts_non_pickleable_callable() -> None:
    offset = 4
    with DecisionExecutor(executor=ThreadPoolExecutor(max_workers=1)) as executor:
        result = executor.submit("closure", lambda value: value + offset, 3)

        assert result.result(timeout=2) == 7


def test_shutdown_is_idempotent_and_rejects_new_tasks() -> None:
    executor = DecisionExecutor(executor=ThreadPoolExecutor(max_workers=1))

    executor.shutdown()
    executor.shutdown()

    with pytest.raises(RuntimeError, match="shut down"):
        executor.submit("late", _worker_identity, "late")


def test_default_process_backend_never_receives_replaced_pending_tasks() -> None:
    parent, child = multiprocessing.get_context("spawn").Pipe()
    executor = DecisionExecutor()
    try:
        running = executor.submit("running", _hold_process, child)
        assert parent.poll(15)
        _, worker_pid = parent.recv()

        stale = [
            executor.submit(f"stale-{index}", _report_process, child, f"stale-{index}")
            for index in range(3)
        ]
        latest = executor.submit("latest", _report_process, child, "latest")
        duplicate = executor.submit("latest", _report_process, child, "duplicate")

        assert duplicate is latest
        assert not latest.running()
        assert all(future.cancelled() for future in stale)
        # Cancelled tasks discarded by the wrapper must still wake wait() and
        # as_completed(), without ever entering the backend's work queue.
        assert wait(stale, timeout=1).not_done == set()
        assert set(as_completed(stale, timeout=1)) == set(stale)
        parent.send("release")

        assert running.result(timeout=15) == "running"
        assert latest.result(timeout=15) == "latest"
        assert parent.poll(2)
        assert parent.recv() == ("latest", worker_pid)
        assert not parent.poll(0.05)
    finally:
        parent.send("release")
        executor.close()
        parent.close()
        child.close()

    assert worker_pid not in {process.pid for process in multiprocessing.active_children()}


def test_external_pending_cancel_notifies_waiters_and_allows_same_key_retry() -> None:
    release = Event()
    started = Event()

    def hold_worker() -> str:
        started.set()
        assert release.wait(5)
        return "running"

    executor = DecisionExecutor(executor=ThreadPoolExecutor(max_workers=1))
    try:
        executor.submit("running", hold_worker)
        assert started.wait(2)
        pending = executor.submit("pending", lambda: "never")

        assert executor.cancel_stale(keep_key="pending") == 0
        assert tuple(executor.tracked_keys) == ("pending",)
        assert pending.cancel()
        assert wait((pending,), timeout=1).not_done == set()
        assert list(as_completed((pending,), timeout=1)) == [pending]
        assert executor.tracked_keys == ()
        assert executor.latest_key is None

        replacement = executor.submit("pending", lambda: "replacement")
        assert replacement is not pending
        release.set()
        assert replacement.result(timeout=2) == "replacement"
    finally:
        release.set()
        executor.close()


def test_cancellation_callbacks_do_not_hold_dispatch_lock() -> None:
    release = Event()
    started = Event()
    callback_blocked: list[bool] = []
    readers: list[Thread] = []

    def hold_worker() -> None:
        started.set()
        assert release.wait(5)

    def read_from_another_thread(_future: object) -> None:
        reader = Thread(target=lambda: executor.latest_key)
        readers.append(reader)
        reader.start()
        reader.join(timeout=1)
        callback_blocked.append(reader.is_alive())

    executor = DecisionExecutor(executor=ThreadPoolExecutor(max_workers=1))
    try:
        executor.submit("running", hold_worker)
        assert started.wait(2)
        stale = executor.submit("stale", lambda: "never")
        stale.add_done_callback(read_from_another_thread)
        latest = executor.submit("latest", lambda: "latest")

        assert callback_blocked == [False]
        release.set()
        assert latest.result(timeout=2) == "latest"
    finally:
        release.set()
        executor.close()
        for reader in readers:
            reader.join(timeout=2)


def test_shutdown_from_completion_callback_does_not_join_dispatcher_itself() -> None:
    release = Event()
    callback_finished = Event()

    def hold_worker() -> str:
        assert release.wait(5)
        return "done"

    def close_from_callback(_future: object) -> None:
        executor.close()
        callback_finished.set()

    executor = DecisionExecutor(executor=ThreadPoolExecutor(max_workers=1))
    try:
        future = executor.submit("running", hold_worker)
        future.add_done_callback(close_from_callback)
        release.set()
        assert future.result(timeout=2) == "done"
        assert callback_finished.wait(2)
        assert executor.closed
    finally:
        release.set()
        executor.close()


@pytest.mark.parametrize("cancel_pending", [False, True])
def test_nonblocking_shutdown_can_be_joined_and_reaps_spawn_worker(cancel_pending: bool) -> None:
    parent, child = multiprocessing.get_context("spawn").Pipe()
    executor = DecisionExecutor()
    try:
        running = executor.submit("running", _hold_process, child)
        assert parent.poll(15)
        _, worker_pid = parent.recv()
        pending = executor.submit("pending", _report_process, child, "pending")

        executor.shutdown(wait=False, cancel_futures=cancel_pending)

        assert executor.closed
        assert not running.done()
        assert pending.cancelled() is cancel_pending
        if cancel_pending:
            assert wait((pending,), timeout=1).not_done == set()
        with pytest.raises(RuntimeError, match="shut down"):
            executor.submit("late", _worker_identity, "late")
        parent.send("release")
        executor.shutdown(wait=True, cancel_futures=cancel_pending)

        assert running.result(timeout=1) == "running"
        if cancel_pending:
            with pytest.raises(CancelledError):
                pending.result(timeout=1)
            assert not parent.poll(0.05)
        else:
            assert pending.result(timeout=1) == "pending"
            assert parent.poll(2)
            assert parent.recv() == ("pending", worker_pid)
    finally:
        parent.send("release")
        executor.close()
        parent.close()
        child.close()

    assert worker_pid not in {process.pid for process in multiprocessing.active_children()}


def test_broken_process_pool_completes_active_and_pending_futures_and_reaps_worker() -> None:
    parent, child = multiprocessing.get_context("spawn").Pipe()
    executor = DecisionExecutor()
    try:
        crashing = executor.submit("crashing", _exit_process, child)
        assert parent.poll(15)
        _, worker_pid = parent.recv()
        pending = executor.submit("pending", _worker_identity, "pending")
        parent.send("release")

        with pytest.raises(BrokenProcessPool):
            crashing.result(timeout=15)
        with pytest.raises(BrokenProcessPool):
            pending.result(timeout=15)
        after_break = executor.submit("after-break", _worker_identity, "never")
        with pytest.raises(BrokenProcessPool):
            after_break.result(timeout=15)
    finally:
        parent.send("release")
        executor.close()
        parent.close()
        child.close()

    assert worker_pid not in {process.pid for process in multiprocessing.active_children()}


def test_cancel_dispatch_race_never_executes_a_successfully_cancelled_future() -> None:
    called: list[int] = []
    submitted = {}
    with DecisionExecutor(executor=ThreadPoolExecutor(max_workers=1)) as executor:
        for index in range(100):
            future = executor.submit(index, called.append, index)
            submitted[index] = future
            future.cancel()

    assert wait(submitted.values(), timeout=2).not_done == set()
    assert not {key for key, future in submitted.items() if future.cancelled()}.intersection(called)
