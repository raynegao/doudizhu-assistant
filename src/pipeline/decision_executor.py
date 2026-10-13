"""Latest-only background execution for CPU-bound live decisions.

The live capture loop must not run pure-Python Monte Carlo work in a thread:
that work can retain the GIL long enough to stall frame acquisition.  This
module therefore uses one ``spawn`` process by default.  Callables and their
arguments must consequently be pickleable and should normally be defined at
module scope.

An existing :class:`concurrent.futures.Executor` can be injected for tests or
for deliberately non-pickleable callables.  A thread executor is convenient in
those cases, but it does not provide GIL isolation.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Hashable
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from threading import Condition, Thread, current_thread
from typing import Any, TypeVar, cast

ResultT = TypeVar("ResultT")
TaskKey = Hashable


@dataclass
class _DecisionTask:
    future: Future[Any]
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    claimed: bool = False


class DecisionExecutor:
    """Run only the newest logical decision on a replaceable executor.

    One lightweight parent-side thread dispatches one backend task at a time.
    Until that task finishes, only the newest pending task is retained here;
    superseded tasks never enter the process pool's non-cancellable IPC queue.
    A dispatched task is considered running and cannot be cancelled, so callers
    must still validate its task key before publishing a result.

    The returned ``Future`` keeps the normal ``concurrent.futures`` contract:
    worker and serialization exceptions are re-raised by ``Future.result()``.
    """

    def __init__(self, executor: Executor | None = None) -> None:
        self._executor = executor if executor is not None else _spawn_process_executor()
        self._condition = Condition()
        self._futures: dict[TaskKey, Future[Any]] = {}
        self._latest_key: TaskKey | None = None
        self._pending: _DecisionTask | None = None
        self._closed = False
        self._shutdown_error: BaseException | None = None
        self._dispatcher = Thread(
            target=self._dispatch,
            name="live-decision-dispatch",
            daemon=True,
        )
        self._dispatcher.start()

    def submit(
        self,
        task_key: TaskKey,
        fn: Callable[..., ResultT],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[ResultT]:
        """Submit ``fn`` as the newest task and return its future.

        Reusing a key while its task is still tracked is idempotent and returns
        the original future.  A different key automatically makes all older
        tasks stale.
        """

        with self._condition:
            self._ensure_open()
            existing = self._futures.get(task_key)
            if existing is not None:
                return cast(Future[ResultT], existing)

            stale = self._retire_stale_locked(keep_key=None)
            future: Future[ResultT] = Future()
            self._futures[task_key] = future
            self._latest_key = task_key
            task = _DecisionTask(future, fn, args, kwargs)
            self._pending = task
            future.add_done_callback(partial(self._forget_finished, task_key, task))
            self._condition.notify()

        # Future callbacks may call back into this executor from another thread.
        # Never invoke them while holding the scheduling condition.
        for old_future in stale:
            old_future.cancel()
        return future

    def cancel_stale(self, keep_key: TaskKey | None = None) -> int:
        """Forget stale tasks and request cancellation where still possible.

        ``keep_key`` remains tracked.  Passing ``None`` invalidates every
        tracked task.  The return value is the number of futures whose queued
        execution was successfully cancelled; already-running work is merely
        forgotten and its eventual result must not be published.
        """

        with self._condition:
            stale = self._retire_stale_locked(keep_key=keep_key)
        return sum(future.cancel() for future in stale)

    def cancel(self, task_key: TaskKey) -> bool:
        """Forget one task and request cancellation of its future."""

        with self._condition:
            future = self._futures.pop(task_key, None)
            if future is None:
                return False
            if self._pending is not None and self._pending.future is future:
                self._pending = None
            if self._latest_key == task_key:
                self._latest_key = None
        return future.cancel()

    @property
    def latest_key(self) -> TaskKey | None:
        """Return the newest still-tracked task key."""

        with self._condition:
            return self._latest_key

    @property
    def tracked_keys(self) -> tuple[TaskKey, ...]:
        """Return a snapshot of logical tasks still owned by this wrapper."""

        with self._condition:
            return tuple(self._futures)

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def close(self) -> None:
        """Cancel queued tasks and synchronously reap the worker process."""

        self.shutdown(wait=True, cancel_futures=True)

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = True,
    ) -> None:
        """Stop accepting work, optionally discard pending work, and reap it.

        ``wait=True`` is the safe normal-exit path: it lets the current bounded
        decision finish and reaps the spawned worker and its IPC resources. A
        later ``wait=True`` also joins an earlier non-blocking shutdown. With
        ``cancel_futures=False``, the last accepted pending task is drained.

        A completion callback runs on the dispatcher thread. Shutdown from that
        callback requests exit but cannot join itself; reaping completes after
        the callback returns, and an external caller may join with ``wait=True``.
        """

        with self._condition:
            self._closed = True
            stale = self._retire_stale_locked(keep_key=None) if cancel_futures else ()
            self._condition.notify()

        for future in stale:
            future.cancel()
        if wait and current_thread() is not self._dispatcher:
            self._dispatcher.join()
            if self._shutdown_error is not None:
                raise self._shutdown_error

    def __enter__(self) -> DecisionExecutor:
        with self._condition:
            self._ensure_open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("decision executor is shut down")

    def _retire_stale_locked(self, *, keep_key: TaskKey | None) -> tuple[Future[Any], ...]:
        stale: list[Future[Any]] = []
        for key, future in tuple(self._futures.items()):
            if keep_key is not None and key == keep_key:
                continue
            self._futures.pop(key, None)
            if self._pending is not None and self._pending.future is future:
                self._pending = None
            stale.append(future)
        self._latest_key = keep_key if keep_key is not None and keep_key in self._futures else None
        return tuple(stale)

    def _dispatch(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending is not None or self._closed)
                    task = self._pending
                    if task is None:
                        return
                    self._pending = None
                    # This method invokes no user callbacks. Serialize it with
                    # cancellation acknowledgement, so Future waiters are notified
                    # exactly once even when a pending task is never dispatched.
                    if task.claimed:
                        continue
                    task.claimed = True
                    if not task.future.set_running_or_notify_cancel():
                        continue
                try:
                    result = self._executor.submit(task.fn, *task.args, **task.kwargs).result()
                except BaseException as exc:
                    task.future.set_exception(exc)
                else:
                    task.future.set_result(result)
        finally:
            try:
                # Only the dispatcher owns backend shutdown. In particular, an
                # asynchronous close cannot race a pending backend submission.
                self._executor.shutdown(wait=True, cancel_futures=True)
            except BaseException as exc:
                self._shutdown_error = exc

    def _forget_finished(
        self,
        task_key: TaskKey,
        task: _DecisionTask,
        future: Future[Any],
    ) -> None:
        with self._condition:
            if future.cancelled() and not task.claimed:
                task.claimed = True
                future.set_running_or_notify_cancel()
            if self._futures.get(task_key) is not future:
                return
            self._futures.pop(task_key, None)
            if self._pending is not None and self._pending.future is future:
                self._pending = None
            if self._latest_key == task_key:
                self._latest_key = None


def _spawn_process_executor() -> ProcessPoolExecutor:
    """Create a fork-safe, single-worker backend for macOS GUI runtimes."""

    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    )
