"""Bounded current-run frame cache; diagnostic errors/results are never evicted."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from src.reporting.live_log_reader import LogIdentity, log_file_identity


@dataclass(frozen=True)
class BufferedFrame:
    frame_id: int
    path: Path
    kind: str
    critical: bool
    identity: LogIdentity


@dataclass(frozen=True)
class FrameSaveResult:
    path: Path | None = None
    omitted: str | None = None
    evicted: BufferedFrame | None = None
    reused: bool = False
    promoted: bool = False
    error: str | None = None


class LiveFrameBuffer:
    """Own only PNGs written by this instance, never adopt existing files.

    Ordinary observations form a rolling cache. Critical frames can reclaim
    ordinary slots but never each other's slots. Once all slots are critical
    (or the session is pinned), callers receive an explicit omission reason.
    """

    def __init__(self, directory: Path, *, limit: int) -> None:
        if limit <= 0:
            raise ValueError("frame buffer limit must be positive")
        self.directory = directory
        self.limit = limit
        self.frames: dict[int, BufferedFrame] = {}
        value = directory.lstat()
        self._directory_identity = (value.st_dev, value.st_ino)

    @property
    def paths(self) -> dict[int, Path]:
        return {key: frame.path for key, frame in self.frames.items()}

    def save(
        self,
        image: Image.Image,
        *,
        frame_id: int,
        kind: str,
        can_evict: Callable[[], bool],
    ) -> FrameSaveResult:
        critical = kind == "uncertain" or kind.startswith("result-")
        old = self.frames.get(frame_id)
        if old is not None:
            if not self._unchanged(old):
                return FrameSaveResult(omitted="existing_frame_changed")
            promoted = critical and not old.critical
            if promoted:
                self.frames[frame_id] = replace(old, kind=kind, critical=True)
            return FrameSaveResult(path=old.path, reused=True, promoted=promoted)
        if not self._directory_unchanged():
            return FrameSaveResult(omitted="unsafe_frame_directory")
        evicted = None
        if len(self.frames) >= self.limit:
            if not can_evict():
                return FrameSaveResult(omitted="session_protected")
            evicted = next(
                (
                    frame
                    for frame in self.frames.values()
                    if not frame.critical and self._unchanged(frame)
                ),
                None,
            )
            if evicted is None:
                return FrameSaveResult(omitted="critical_or_modified_frames_fill_limit")
            evicted.path.unlink()
            del self.frames[evicted.frame_id]

        safe_kind = re.sub(r"[^a-z0-9_-]+", "-", kind.lower()).strip("-") or "evidence"
        stem = f"frame-{frame_id:06d}-{safe_kind}"
        suffix = 0
        while True:
            path = self.directory / f"{stem}{'-' + str(suffix) if suffix else ''}.png"
            try:
                descriptor = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
                )
                break
            except FileExistsError:
                suffix += 1
            except OSError as exc:
                return FrameSaveResult(
                    omitted="frame_write_failed", evicted=evicted, error=str(exc)
                )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                image.save(handle, format="PNG")
            self.frames[frame_id] = BufferedFrame(
                frame_id, path, kind, critical, log_file_identity(path)
            )
        except (OSError, ValueError) as exc:
            # The exclusively created partial file remains owned and counted,
            # so even a persistent encoder failure cannot leak unlimited PNGs.
            self.frames[frame_id] = BufferedFrame(
                frame_id, path, kind, True, log_file_identity(path)
            )
            return FrameSaveResult(
                omitted="frame_encoding_failed", evicted=evicted, error=str(exc)
            )
        return FrameSaveResult(path=path, evicted=evicted)

    def _unchanged(self, frame: BufferedFrame) -> bool:
        try:
            return (
                self._directory_unchanged()
                and frame.path.parent.resolve() == self.directory.resolve()
                and log_file_identity(frame.path) == frame.identity
            )
        except (OSError, ValueError):
            return False

    def _directory_unchanged(self) -> bool:
        try:
            value = self.directory.lstat()
            return (
                not self.directory.is_symlink()
                and self.directory.is_dir()
                and (value.st_dev, value.st_ino) == self._directory_identity
            )
        except OSError:
            return False


__all__ = ["BufferedFrame", "FrameSaveResult", "LiveFrameBuffer"]
