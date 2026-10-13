from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

import pytest

from src.capture.recording_integrity import inspect_recording_session
from src.config.live_layout import LiveLayoutConfig, save_live_layout

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RECORDER_PROBE = dedent(
    """
    import json
    import os
    import sys
    import time
    from pathlib import Path

    from PIL import Image
    from src.capture import lossless_video
    from src.capture.screen_geometry import CapturedWindow, ScreenGeometry
    from src.pipeline import recording_writer
    from src.pipeline.calibration import WindowInfo

    fixture_root = Path(sys.argv[1])
    pause_phase = sys.argv[2]
    original_writer = lossless_video.LosslessRGBVideoWriter
    original_queue = recording_writer.RecordingWriter

    def wait_for_release(name):
        deadline = time.monotonic() + 15
        while not (fixture_root / name).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("test did not release " + name)
            time.sleep(0.01)

    class ProbeQueue(original_queue):
        def submit(self, item):
            super().submit(item)
            accepted = self.diagnostics()["accepted_frames"]
            (fixture_root / ("accepted-" + str(accepted))).touch()

    class ProbeWriter(original_writer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            (fixture_root / "encoder.json").write_text(json.dumps({
                "pid": self._process.pid,
                "group": os.getpgid(self._process.pid),
                "session": os.getsid(self._process.pid),
            }))

        def write(self, image):
            if pause_phase == "drain" and self.frame_count == 2:
                (fixture_root / "writer-blocked").touch()
                wait_for_release("release-write")
            super().write(image)

        def close(self):
            if not self._closed:
                (fixture_root / "closing").touch()
                wait_for_release("release-close")
            super().close()

    # Patch before importing the CLI, so an extracted pipeline writer also
    # uses this probe without requiring a production-only injection hook.
    lossless_video.LosslessRGBVideoWriter = ProbeWriter
    recording_writer.RecordingWriter = ProbeQueue
    from scripts import record_live_game

    class SyntheticCapture:
        def __init__(self, _app_name):
            self.started_at = time.time()

        def capture(self, frame_id):
            # Cadence is synthetic and deterministic; this test measures
            # lifecycle correctness, not real ScreenCaptureKit performance.
            return CapturedWindow(
                frame_id=frame_id,
                timestamp=self.started_at + frame_id * 0.1,
                image=Image.new("RGB", (100, 100), (frame_id % 256, 20, 80)),
                window=WindowInfo(
                    app_name="synthetic", window_name="signal-test",
                    window_box=(0, 0, 100, 100),
                ),
                pixel_box=(0, 0, 100, 100),
                geometry=ScreenGeometry(
                    logical_size=(100, 100), pixel_size=(100, 100),
                ),
                capture_backend="screen_capture_kit_stream",
            )

        def close(self):
            (fixture_root / "capture-closed").touch()

    record_live_game.MacWindowCapture = SyntheticCapture
    raise SystemExit(record_live_game.main([
        "--config", str(fixture_root / "config.json"),
        "--session", "synthetic-signals", "--until-interrupt",
        "--interval", "0.1", "--evidence-split", "acceptance",
        "--output-root", str(fixture_root / "recordings"),
    ]))
    """
)


def _wait_for_probe(
    predicate: Callable[[], bool], process: subprocess.Popen[str], *, timeout: float = 20
) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=2)
            pytest.fail(f"recorder exited early ({process.returncode}):\n{stdout}\n{stderr}")
        if time.monotonic() >= deadline:
            pytest.fail("timed out waiting for the synthetic recorder probe")
        time.sleep(0.01)


def _manifest_has_two_frames(path: Path) -> bool:
    if not path.exists():
        return False
    # Count complete lines only; the writer may currently be appending a row.
    return path.read_bytes().count(b"\n") >= 2


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("ffmpeg") is None,
    reason="real process-group signal regression requires POSIX and ffmpeg",
)
@pytest.mark.parametrize(
    ("stop_signal", "exit_code"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
@pytest.mark.parametrize("pause_phase", ["drain", "close"])
def test_group_signal_and_repeated_signal_drain_and_seal_without_encoder_orphans(
    tmp_path: Path, stop_signal: signal.Signals, exit_code: int, pause_phase: str,
) -> None:
    save_live_layout(tmp_path / "config.json", LiveLayoutConfig(interval_seconds=0.1))
    session = tmp_path / "recordings" / "synthetic-signals"
    manifest = session / "manifest.jsonl"
    process = subprocess.Popen(
        [sys.executable, "-c", _RECORDER_PROBE, str(tmp_path), pause_phase],
        cwd=_PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    encoder_pid: int | None = None
    try:
        _wait_for_probe(lambda: _manifest_has_two_frames(manifest), process)
        if pause_phase == "drain":
            # Frame 3 is in the sink and frame 4 is definitely accepted into
            # the FIFO. Neither can finish until we release the write gate.
            _wait_for_probe(
                lambda: (tmp_path / "writer-blocked").exists()
                and (tmp_path / "accepted-4").exists(),
                process,
            )
        encoder = json.loads((tmp_path / "encoder.json").read_text())
        encoder_pid = int(encoder["pid"])
        # Never signal the test runner's or the user's terminal process group.
        assert os.getpgid(process.pid) == process.pid
        assert process.pid != os.getpgrp()
        os.killpg(process.pid, stop_signal)

        marker = "capture-closed" if pause_phase == "drain" else "closing"
        _wait_for_probe(lambda: (tmp_path / marker).exists(), process)
        assert (tmp_path / "capture-closed").exists()
        if pause_phase == "drain":
            assert not (tmp_path / "closing").exists()
        os.killpg(process.pid, stop_signal)
        (tmp_path / "release-write").touch()
        _wait_for_probe(lambda: (tmp_path / "closing").exists(), process)
        (tmp_path / "release-close").touch()
        stdout, stderr = process.communicate(timeout=20)

        assert process.returncode == exit_code, f"{stdout}\n{stderr}"
        assert encoder["group"] != process.pid
        assert encoder["session"] != process.pid
        assert not _process_exists(encoder_pid), f"owned ffmpeg {encoder_pid} was not reaped"
        inspection = inspect_recording_session(session)
        assert inspection.valid, inspection.issues
        assert inspection.metadata["recording_state"] == "interrupted"
        assert inspection.metadata["complete_game"] is False
        rows = inspection.manifest_rows
        assert len(rows) >= (4 if pause_phase == "drain" else 2)
        assert inspection.metadata["recorded_frames"] == len(rows)
        diagnostics = inspection.metadata["recording_diagnostics"]
        assert diagnostics["accepted_frames"] == diagnostics["written_frames"] == len(rows)
        assert diagnostics["unwritten_frames"] == 0
        segments = inspection.metadata["frame_storage"]["segments"]
        assert sum(segment["frame_count"] for segment in segments) == len(rows)
        assert [row["frame_id"] for row in rows] == list(range(1, len(rows) + 1))
        assert [row["video_frame_index"] for row in rows] == list(range(len(rows)))
    finally:
        (tmp_path / "release-write").touch()
        (tmp_path / "release-close").touch()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=5)
        # The PID is emitted by this probe's own freshly created encoder.
        # Clean up test-created resources even when an assertion exposes a bug.
        if encoder_pid is None and (tmp_path / "encoder.json").exists():
            try:
                encoder_pid = int(json.loads((tmp_path / "encoder.json").read_text())["pid"])
            except (OSError, ValueError, KeyError):
                pass
        if encoder_pid is not None and _process_exists(encoder_pid):
            os.kill(encoder_pid, signal.SIGTERM)
            deadline = time.monotonic() + 3
            while _process_exists(encoder_pid) and time.monotonic() < deadline:
                time.sleep(0.01)
            if _process_exists(encoder_pid):
                os.kill(encoder_pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason="signal regression requires POSIX")
def test_signal_handler_does_not_reenter_the_stop_events_nonreentrant_lock() -> None:
    probe = dedent(
        """
        import os
        import signal
        from threading import Event
        from scripts.record_live_game import _stop_on_signal

        stop = Event()
        with _stop_on_signal(stop) as received:
            # Reproduce a signal arriving in the brief lock-held part of
            # Event.wait(). Event.set() in the handler would deadlock here.
            with stop._cond:
                os.kill(os.getpid(), signal.SIGINT)
                os.kill(os.getpid(), signal.SIGTERM)
            assert received == [signal.SIGINT], received
        print("signal-handled-without-reentry")
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=_PROJECT_ROOT,
            capture_output=True, text=True, timeout=5, start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for its own timed-out child.
        pytest.fail("signal handler deadlocked on the stop Event's nonreentrant lock")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "signal-handled-without-reentry"
