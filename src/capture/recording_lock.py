"""Exclusive ownership of a recording session across recorder/finalize calls."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def recording_session_lock(session_dir: Path) -> Iterator[None]:
    if not session_dir.is_dir():
        raise ValueError(f"recording session directory is missing: {session_dir}")
    # Keep this inode in place. Unlinking a flock file could allow another
    # process to lock a replacement while a waiter still holds the old inode.
    with (session_dir / ".recording.lock").open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"recording session is busy: {session_dir.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
