from __future__ import annotations

import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from src.capture import lossless_video
from src.capture.lossless_video import (
    LosslessRGBVideoWriter,
    LosslessVideoError,
    iter_lossless_rgb_frames,
)


@pytest.mark.parametrize("timeout", (0.0, -1.0, float("nan"), float("inf")))
def test_writer_rejects_non_positive_or_non_finite_timeout(
    tmp_path: Path, timeout: float,
) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        LosslessRGBVideoWriter(
            tmp_path / "invalid.mkv", image_size=(10, 10), frames_per_second=10,
            io_timeout_seconds=timeout,
        )
    assert not (tmp_path / "invalid.mkv").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires real ffmpeg")
def test_real_encoder_preserves_each_synthetic_rgb_pixel(tmp_path: Path) -> None:
    rng = random.Random(17)
    frames = [Image.frombytes("RGB", (64, 48), rng.randbytes(64 * 48 * 3)) for _ in range(5)]
    path = tmp_path / "pixels.mkv"
    writer = LosslessRGBVideoWriter(path, image_size=(64, 48), frames_per_second=10)
    with writer:
        for frame in frames:
            writer.write(frame)
    decoded = list(iter_lossless_rgb_frames(path, image_size=(64, 48)))

    assert writer.frame_count == len(frames)
    assert len(decoded) == len(frames)
    assert all(original.tobytes() == restored.tobytes()
               for original, restored in zip(frames, decoded, strict=True))
    assert writer._closed
    assert writer._process.poll() == 0
    assert writer._stderr.closed
    assert writer._process.stdin is not None and writer._process.stdin.closed
    assert writer._process.stderr is None
    writer.close()
    with pytest.raises(LosslessVideoError, match="closed"):
        writer.write(frames[0])


def _reap_only_this_writer(writer: LosslessRGBVideoWriter) -> None:
    if writer._process.poll() is None:
        writer._process.kill()
        writer._process.wait(timeout=2)
    try:
        writer.close()
    except LosslessVideoError:
        pass


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires real ffmpeg")
@pytest.mark.parametrize("early_stop", [False, True])
def test_decoder_closes_owned_pipes_on_eof_and_early_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, early_stop: bool,
) -> None:
    path = tmp_path / "decode-cleanup.mkv"
    with LosslessRGBVideoWriter(path, image_size=(64, 48), frames_per_second=10) as writer:
        writer.write(Image.new("RGB", (64, 48), "red"))
        writer.write(Image.new("RGB", (64, 48), "blue"))
    original_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def track_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(lossless_video.subprocess, "Popen", track_process)
    decoded = iter_lossless_rgb_frames(path, image_size=(64, 48))
    try:
        assert next(decoded).getpixel((0, 0)) == (255, 0, 0)
        if not early_stop:
            assert len(list(decoded)) == 1
    finally:
        decoded.close()
    assert len(processes) == 1
    process = processes[0]
    assert process.poll() is not None
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("ffmpeg") is None,
    reason="stopped encoder regression requires POSIX and ffmpeg",
)
def test_stopped_owned_encoder_write_times_out_and_is_reaped(tmp_path: Path) -> None:
    writer = LosslessRGBVideoWriter(
        tmp_path / "stopped-write.mkv", image_size=(1024, 1024), frames_per_second=10,
        io_timeout_seconds=0.15,
    )
    try:
        # This PID comes from the newly constructed test-owned encoder only.
        assert os.getpgid(writer._process.pid) == writer._process.pid
        assert os.getpgid(writer._process.pid) != os.getpgrp()
        writer._process.send_signal(signal.SIGSTOP)
        started = time.monotonic()
        with pytest.raises(LosslessVideoError, match="timed out writing"):
            writer.write(Image.new("RGB", (1024, 1024), (30, 90, 20)))
        assert time.monotonic() - started < 3
        assert writer.frame_count == 0
        assert writer._closed
        assert writer._process.poll() is not None
        assert writer._stderr.closed
        assert writer._process.stdin is not None and writer._process.stdin.closed
    finally:
        _reap_only_this_writer(writer)


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("ffmpeg") is None,
    reason="stopped encoder regression requires POSIX and ffmpeg",
)
def test_stopped_owned_encoder_close_times_out_and_is_reaped(tmp_path: Path) -> None:
    writer = LosslessRGBVideoWriter(
        tmp_path / "stopped-close.mkv", image_size=(64, 48), frames_per_second=10,
        io_timeout_seconds=0.15,
    )
    try:
        writer.write(Image.new("RGB", (64, 48), (30, 90, 20)))
        writer._process.send_signal(signal.SIGSTOP)
        started = time.monotonic()
        with pytest.raises(LosslessVideoError, match="timed out closing"):
            writer.close()
        assert time.monotonic() - started < 3
        assert writer._closed
        assert writer._process.poll() is not None
        assert writer._stderr.closed
        assert writer._process.stdin is not None and writer._process.stdin.closed
    finally:
        _reap_only_this_writer(writer)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires real ffmpeg")
def test_stdin_close_error_still_reaps_owned_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = LosslessRGBVideoWriter(
        tmp_path / "close-error.mkv", image_size=(64, 48), frames_per_second=10,
        io_timeout_seconds=0.15,
    )
    stdin = writer._process.stdin
    assert stdin is not None
    real_close = stdin.close
    failed = False

    def close_fails_once() -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected stdin close failure")
        real_close()

    monkeypatch.setattr(stdin, "close", close_fails_once)
    try:
        with pytest.raises(LosslessVideoError, match="injected stdin close failure"):
            writer.close()
        assert writer._process.poll() is not None
        assert writer._stderr.closed
        assert stdin.closed
    finally:
        _reap_only_this_writer(writer)


def test_stderr_flood_does_not_deadlock_and_failure_detail_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_popen = subprocess.Popen

    def noisy_encoder(_command: object, **kwargs: object) -> subprocess.Popen[bytes]:
        return original_popen(
            [sys.executable, "-c", (
                "import sys; sys.stderr.write('x' * 2_000_000 + 'encoder-tail'); "
                "sys.stderr.flush(); sys.stdin.buffer.read(); sys.exit(17)"
            )],
            **kwargs,
        )

    monkeypatch.setattr(lossless_video, "ffmpeg_path", lambda: Path(sys.executable))
    monkeypatch.setattr(lossless_video.subprocess, "Popen", noisy_encoder)
    writer = LosslessRGBVideoWriter(
        tmp_path / "noisy.mkv", image_size=(64, 48), frames_per_second=10,
        io_timeout_seconds=2,
    )
    try:
        writer.write(Image.new("RGB", (64, 48)))
        with pytest.raises(LosslessVideoError, match="encoder-tail") as failure:
            writer.close()
        assert len(str(failure.value)) < 8400
        assert writer._process.returncode == 17
        assert writer._stderr.closed
    finally:
        _reap_only_this_writer(writer)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires real ffmpeg")
def test_broken_input_pipe_reaps_process_and_closes_all_owned_handles(tmp_path: Path) -> None:
    writer = LosslessRGBVideoWriter(
        tmp_path / "broken-pipe.mkv", image_size=(64, 48), frames_per_second=10,
        io_timeout_seconds=0.15,
    )
    try:
        writer._process.kill()
        writer._process.wait(timeout=2)
        with pytest.raises(LosslessVideoError, match="encoder failed"):
            writer.write(Image.new("RGB", (64, 48)))
        assert writer.frame_count == 0
        assert writer._closed
        assert writer._stderr.closed
        assert writer._process.stdin is not None and writer._process.stdin.closed
    finally:
        _reap_only_this_writer(writer)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires real ffmpeg")
def test_context_exception_reaps_encoder_without_masking_original_error(tmp_path: Path) -> None:
    writer = LosslessRGBVideoWriter(
        tmp_path / "context-failure.mkv", image_size=(64, 48), frames_per_second=10,
        io_timeout_seconds=0.15,
    )
    with pytest.raises(RuntimeError, match="original frame failure"):
        with writer:
            raise RuntimeError("original frame failure")
    assert writer._process.poll() is not None
    assert writer._stderr.closed
    assert writer._process.stdin is not None and writer._process.stdin.closed


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires real ffmpeg")
def test_failure_after_encoder_spawn_reaps_new_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def track_process(*args, **kwargs) -> subprocess.Popen[bytes]:
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    def fail_nonblocking(_descriptor: int, _blocking: bool) -> None:
        raise OSError("injected nonblocking setup failure")

    monkeypatch.setattr(lossless_video.subprocess, "Popen", track_process)
    monkeypatch.setattr(lossless_video.os, "set_blocking", fail_nonblocking)
    with pytest.raises(OSError, match="injected nonblocking setup failure"):
        LosslessRGBVideoWriter(
            tmp_path / "startup-failure.mkv", image_size=(64, 48), frames_per_second=10,
            io_timeout_seconds=0.15,
        )
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdin is not None and processes[0].stdin.closed
