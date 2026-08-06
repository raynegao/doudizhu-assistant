from __future__ import annotations

import hashlib
import json
import os
import platform
import select
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image

from src.pipeline.calibration import WindowInfo

_PROTOCOL_VERSION = "screen-capture-kit-stream-v1"
_FRAME_HEADER = struct.Struct("<4sIIIQQ")
_FRAME_MAGIC = b"SCF1"
_MAX_FRAME_BYTES = 128 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 10.0


class MacOSStreamCaptureError(RuntimeError):
    """Raised when the native ScreenCaptureKit stream cannot provide a frame."""


@dataclass(frozen=True)
class StreamedWindowFrame:
    timestamp: float
    image: Image.Image
    window_id: int
    window: WindowInfo


class ScreenCaptureKitWindowStream:
    """Persistent, lossless macOS window stream backed by ScreenCaptureKit."""

    def __init__(
        self,
        app_name: str,
        *,
        fps: int = 12,
        project_root: Path | None = None,
        cache_dir: Path | None = None,
        compiler: Path = Path("/usr/bin/swiftc"),
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not app_name.strip():
            raise ValueError("app_name cannot be empty")
        if not 1 <= fps <= 60:
            raise ValueError("fps must be between 1 and 60")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        source_path = root / "native" / "macos_window_stream.swift"
        native_cache = (
            cache_dir or root / "data" / "live_game" / "native"
        ).resolve()
        binary_path = _ensure_native_binary(
            source_path,
            native_cache,
            compiler=compiler,
        )
        self.app_name = app_name
        self.fps = fps
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[bytes] | None = subprocess.Popen(
            [binary_path.as_posix(), app_name, str(fps)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        try:
            metadata = _read_metadata(self._process, timeout_seconds)
            self._window_id = _strict_positive_int(
                metadata.get("window_id"),
                "window_id",
            )
            window_box = cast(
                tuple[int, int, int, int],
                _integer_tuple(
                    metadata.get("window_box"),
                    length=4,
                    label="window_box",
                ),
            )
            if window_box[2] <= window_box[0] or window_box[3] <= window_box[1]:
                raise MacOSStreamCaptureError("native stream returned invalid window bounds")
            pixel_size = _integer_tuple(
                metadata.get("pixel_size"),
                length=2,
                label="pixel_size",
            )
            if min(pixel_size) <= 0:
                raise MacOSStreamCaptureError("native stream returned invalid pixel size")
            if metadata.get("protocol") != _PROTOCOL_VERSION:
                raise MacOSStreamCaptureError("native stream protocol version mismatch")
            if metadata.get("app_name") != app_name:
                raise MacOSStreamCaptureError("native stream selected the wrong application")
            self._pixel_size = pixel_size
            self._window = WindowInfo(
                app_name=app_name,
                window_name=str(metadata.get("window_name") or app_name),
                window_box=window_box,
            )
        except Exception:
            self.close()
            raise

    @property
    def window_id(self) -> int:
        return self._window_id

    @property
    def window(self) -> WindowInfo:
        return self._window

    def capture(self) -> StreamedWindowFrame:
        process = self._require_process()
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            process.stdin.write(b"C\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._process_error("cannot request a native capture frame") from exc
        header = _read_exact(
            process.stdout.fileno(),
            _FRAME_HEADER.size,
            timeout_seconds=self.timeout_seconds,
        )
        magic, width, height, bytes_per_row, timestamp_ns, byte_count = (
            _FRAME_HEADER.unpack(header)
        )
        if magic != _FRAME_MAGIC:
            raise MacOSStreamCaptureError("native frame header magic is invalid")
        if (width, height) != self._pixel_size:
            raise MacOSStreamCaptureError(
                "native stream frame size changed; restart capture after resizing the window"
            )
        if bytes_per_row < width * 4 or byte_count != bytes_per_row * height:
            raise MacOSStreamCaptureError("native frame byte layout is invalid")
        if byte_count <= 0 or byte_count > _MAX_FRAME_BYTES:
            raise MacOSStreamCaptureError("native frame byte count is unsafe")
        pixels = _read_exact(
            process.stdout.fileno(),
            byte_count,
            timeout_seconds=self.timeout_seconds,
        )
        image = Image.frombuffer(
            "RGBA",
            (width, height),
            pixels,
            "raw",
            "BGRA",
            bytes_per_row,
            1,
        ).convert("RGB")
        return StreamedWindowFrame(
            timestamp=timestamp_ns / 1_000_000_000,
            image=image,
            window_id=self._window_id,
            window=self._window,
        )

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        self._process = None
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(b"Q\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()

    def _require_process(self) -> subprocess.Popen[bytes]:
        process = self._process
        if process is None:
            raise MacOSStreamCaptureError("native capture stream is closed")
        if process.poll() is not None:
            raise self._process_error("native capture stream exited")
        return process

    def _process_error(self, prefix: str) -> MacOSStreamCaptureError:
        process = self._process
        detail = ""
        if process is not None and process.stderr is not None and process.poll() is not None:
            try:
                detail = process.stderr.read().decode("utf-8", errors="replace").strip()
            except OSError:
                detail = ""
        return MacOSStreamCaptureError(f"{prefix}: {detail}" if detail else prefix)

    def __enter__(self) -> ScreenCaptureKitWindowStream:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def _ensure_native_binary(
    source_path: Path,
    cache_dir: Path,
    *,
    compiler: Path,
) -> Path:
    if platform.system() != "Darwin":
        raise MacOSStreamCaptureError("ScreenCaptureKit is available only on macOS")
    if not source_path.is_file():
        raise MacOSStreamCaptureError(f"native capture source is missing: {source_path}")
    if not compiler.is_file():
        raise MacOSStreamCaptureError(f"Swift compiler is missing: {compiler}")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    architecture = platform.machine() or "unknown"
    binary_path = cache_dir / (
        f"macos-window-stream-{architecture}-{source_sha256[:16]}"
    )
    if binary_path.is_file() and os.access(binary_path, os.X_OK):
        return binary_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = binary_path.with_name(
        f"{binary_path.name}.{os.getpid()}.tmp"
    )
    result = subprocess.run(
        [
            compiler.as_posix(),
            "-parse-as-library",
            "-O",
            source_path.as_posix(),
            "-o",
            temporary.as_posix(),
            "-framework",
            "AppKit",
            "-framework",
            "ScreenCaptureKit",
            "-framework",
            "CoreMedia",
            "-framework",
            "CoreVideo",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not temporary.is_file():
        raise MacOSStreamCaptureError(
            "cannot compile native ScreenCaptureKit helper: "
            + (result.stderr.strip() or result.stdout.strip() or "unknown compiler error")
        )
    temporary.chmod(0o700)
    temporary.replace(binary_path)
    return binary_path


def _read_metadata(
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> dict[str, object]:
    if process.stdout is None:
        raise MacOSStreamCaptureError("native stream stdout is unavailable")
    raw = _read_line(
        process.stdout.fileno(),
        timeout_seconds=timeout_seconds,
        maximum_bytes=16 * 1024,
    )
    if not raw:
        detail = ""
        if process.stderr is not None and process.poll() is not None:
            detail = process.stderr.read().decode("utf-8", errors="replace").strip()
        raise MacOSStreamCaptureError(
            "native stream did not become ready" + (f": {detail}" if detail else "")
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MacOSStreamCaptureError("native stream metadata is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise MacOSStreamCaptureError("native stream metadata must be an object")
    return payload


def _read_line(
    file_descriptor: int,
    *,
    timeout_seconds: float,
    maximum_bytes: int,
) -> bytes:
    deadline = time.monotonic() + timeout_seconds
    result = bytearray()
    while len(result) < maximum_bytes:
        chunk = _read_with_deadline(file_descriptor, 1, deadline)
        if not chunk or chunk == b"\n":
            return bytes(result)
        result.extend(chunk)
    raise MacOSStreamCaptureError("native stream metadata line is too large")


def _read_exact(
    file_descriptor: int,
    size: int,
    *,
    timeout_seconds: float,
) -> bytes:
    deadline = time.monotonic() + timeout_seconds
    result = bytearray()
    while len(result) < size:
        chunk = _read_with_deadline(file_descriptor, size - len(result), deadline)
        if not chunk:
            raise MacOSStreamCaptureError("native capture stream closed unexpectedly")
        result.extend(chunk)
    return bytes(result)


def _read_with_deadline(
    file_descriptor: int,
    size: int,
    deadline: float,
) -> bytes:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise MacOSStreamCaptureError("native capture stream timed out")
    readable, _, _ = select.select([file_descriptor], [], [], remaining)
    if not readable:
        raise MacOSStreamCaptureError("native capture stream timed out")
    try:
        return os.read(file_descriptor, size)
    except OSError as exc:
        raise MacOSStreamCaptureError("cannot read native capture stream") from exc


def _strict_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MacOSStreamCaptureError(f"native stream {label} is invalid")
    return value


def _integer_tuple(
    value: object,
    *,
    length: int,
    label: str,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise MacOSStreamCaptureError(f"native stream {label} is invalid")
    converted: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise MacOSStreamCaptureError(f"native stream {label} is invalid")
        converted.append(item)
    return tuple(converted)


__all__ = [
    "MacOSStreamCaptureError",
    "ScreenCaptureKitWindowStream",
    "StreamedWindowFrame",
]
