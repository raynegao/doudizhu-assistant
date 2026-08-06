from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

LOSSLESS_VIDEO_SCHEMA_VERSION = "phase6-lossless-rgb-video-v1"
LOSSLESS_VIDEO_CODEC = "libx264rgb"
LOSSLESS_VIDEO_CONTAINER = "matroska"
LOSSLESS_VIDEO_PIXEL_FORMAT = "rgb24"


class LosslessVideoError(RuntimeError):
    """Raised when the local lossless video transport cannot be used safely."""


def ffmpeg_path() -> Path:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise LosslessVideoError(
            "ffmpeg is required for lossless high-frame-rate recording"
        )
    return Path(executable).resolve()


def ffmpeg_version() -> str:
    executable = ffmpeg_path()
    completed = subprocess.run(
        (str(executable), "-version"),
        check=False,
        capture_output=True,
        text=True,
    )
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    if completed.returncode != 0 or not first_line.startswith("ffmpeg version "):
        detail = completed.stderr.strip() or "version probe failed"
        raise LosslessVideoError(f"cannot inspect ffmpeg: {detail}")
    return first_line


class LosslessRGBVideoWriter:
    """Stream exact RGB frames into a lossless inter-frame H.264 container."""

    def __init__(
        self,
        path: Path,
        *,
        image_size: tuple[int, int],
        frames_per_second: float,
    ) -> None:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("video image dimensions must be positive")
        if frames_per_second <= 0:
            raise ValueError("video frame rate must be positive")
        if path.exists():
            raise LosslessVideoError(f"video segment already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        executable = ffmpeg_path()
        command = (
            str(executable),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            LOSSLESS_VIDEO_PIXEL_FORMAT,
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{frames_per_second:.9f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            LOSSLESS_VIDEO_CODEC,
            "-preset",
            "ultrafast",
            "-crf",
            "0",
            "-pix_fmt",
            LOSSLESS_VIDEO_PIXEL_FORMAT,
            "-f",
            LOSSLESS_VIDEO_CONTAINER,
            "-y",
            str(path),
        )
        self.path = path
        self.image_size = image_size
        self.frames_per_second = frames_per_second
        self.frame_count = 0
        self._closed = False
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, image: Image.Image) -> None:
        if self._closed:
            raise LosslessVideoError("cannot write to a closed video segment")
        rgb = image.convert("RGB")
        if rgb.size != self.image_size:
            raise LosslessVideoError(
                "video frame dimensions changed from "
                f"{self.image_size} to {rgb.size}"
            )
        stdin = self._process.stdin
        if stdin is None:
            raise LosslessVideoError("ffmpeg input pipe is unavailable")
        try:
            stdin.write(rgb.tobytes())
        except BrokenPipeError as exc:
            raise LosslessVideoError(self._failure_detail()) from exc
        self.frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stdin = self._process.stdin
        if stdin is not None and not stdin.closed:
            stdin.close()
        stderr = self._process.stderr.read() if self._process.stderr is not None else b""
        return_code = self._process.wait()
        if return_code != 0:
            detail = stderr.decode(errors="replace").strip() or "unknown ffmpeg error"
            raise LosslessVideoError(
                f"lossless video encoder exited with {return_code}: {detail}"
            )
        if self.frame_count <= 0 or not self.path.is_file():
            raise LosslessVideoError("lossless video segment contains no frames")

    def _failure_detail(self) -> str:
        stderr = self._process.stderr.read() if self._process.stderr is not None else b""
        detail = stderr.decode(errors="replace").strip() or "ffmpeg input closed"
        return f"lossless video encoder failed: {detail}"

    def __enter__(self) -> LosslessRGBVideoWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def iter_lossless_rgb_frames(
    path: Path,
    *,
    image_size: tuple[int, int],
) -> Iterator[Image.Image]:
    """Decode a lossless segment sequentially without materializing it in RAM."""

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("video image dimensions must be positive")
    if not path.is_file():
        raise LosslessVideoError(f"lossless video segment is missing: {path}")
    executable = ffmpeg_path()
    command = (
        str(executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        LOSSLESS_VIDEO_PIXEL_FORMAT,
        "pipe:1",
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    bytes_per_frame = width * height * 3
    completed = False
    try:
        assert process.stdout is not None
        while True:
            payload = _read_exact_or_eof(process.stdout, bytes_per_frame)
            if payload is None:
                completed = True
                break
            yield Image.frombytes("RGB", image_size, payload)
    finally:
        if not completed and process.poll() is None:
            process.terminate()
        if process.stdout is not None:
            process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
        if completed and return_code != 0:
            detail = stderr.decode(errors="replace").strip() or "unknown ffmpeg error"
            raise LosslessVideoError(
                f"lossless video decoder exited with {return_code}: {detail}"
            )


def _read_exact_or_eof(stream: object, size: int) -> bytes | None:
    read = getattr(stream, "read", None)
    if not callable(read):
        raise LosslessVideoError("ffmpeg output pipe is unavailable")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = read(remaining)
        if not chunk:
            if not chunks:
                return None
            raise LosslessVideoError("lossless video ended inside an RGB frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = [
    "LOSSLESS_VIDEO_CODEC",
    "LOSSLESS_VIDEO_CONTAINER",
    "LOSSLESS_VIDEO_PIXEL_FORMAT",
    "LOSSLESS_VIDEO_SCHEMA_VERSION",
    "LosslessRGBVideoWriter",
    "LosslessVideoError",
    "ffmpeg_path",
    "ffmpeg_version",
    "iter_lossless_rgb_frames",
]
