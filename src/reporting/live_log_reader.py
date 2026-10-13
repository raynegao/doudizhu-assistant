"""Read the retained slices of one live JSONL stream, without loading them at once."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LogIdentity = tuple[int, int, int, int]


@dataclass(frozen=True)
class LiveLogSource:
    path: Path
    identity: LogIdentity


@dataclass(frozen=True)
class LiveLogRecord:
    source: Path
    line_number: int
    payload: dict[str, object]


def log_file_identity(path: Path) -> LogIdentity:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"live log must be a plain file: {path}")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _archive_key(base: Path, candidate: Path) -> tuple[str, int] | None:
    match = re.fullmatch(
        rf"{re.escape(base.name)}\.(\d{{8}}-\d{{6}})(?:\.([1-9]\d*))?",
        candidate.name,
    )
    if match is None:
        return None
    try:
        datetime.strptime(match[1], "%Y%m%d-%H%M%S")
    except ValueError:
        return None
    return match[1], int(match[2] or 0)


def list_live_log_sources(base: Path, *, include_current: bool = True) -> tuple[LiveLogSource, ...]:
    """Return chronological archives (numeric collision suffixes), then current.

    Only direct siblings with the writer's exact naming grammar are accepted.
    A matching symlink fails closed instead of silently hiding a log slice.
    Absence of old slices is never interpreted as evidence of an empty history.
    """

    if ".." in base.parts or base.parent.is_symlink():
        raise ValueError(f"unsafe live log path: {base}")
    if not base.parent.exists():
        return ()
    archives: list[tuple[tuple[str, int], LiveLogSource]] = []
    for candidate in base.parent.iterdir():
        key = _archive_key(base, candidate)
        if key is not None:
            archives.append((key, LiveLogSource(candidate, log_file_identity(candidate))))
    sources = [source for _, source in sorted(archives, key=lambda item: item[0])]
    if include_current and (base.exists() or base.is_symlink()):
        sources.append(LiveLogSource(base, log_file_identity(base)))
    return tuple(sources)


def iter_live_log_records(sources: tuple[LiveLogSource, ...]) -> Iterator[LiveLogRecord]:
    """Yield source/line-addressable records; reject unsafe or concurrently changed files."""

    for source in sources:
        descriptor = os.open(source.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            value = os.fstat(handle.fileno())
            identity = (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
            if not stat.S_ISREG(value.st_mode) or identity != source.identity:
                raise ValueError(f"live log changed before reading: {source.path}")
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{source.path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"{source.path}:{line_number} must be a JSON object")
                yield LiveLogRecord(source.path, line_number, payload)
            if log_file_identity(source.path) != source.identity:
                raise ValueError(f"live log changed while reading: {source.path}")


__all__ = [
    "LiveLogRecord",
    "LiveLogSource",
    "iter_live_log_records",
    "list_live_log_sources",
    "log_file_identity",
]
