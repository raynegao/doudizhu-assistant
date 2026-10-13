from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.reporting.live_frame_buffer import LiveFrameBuffer


def test_cache_continues_across_rounds_and_reserves_critical_evidence(tmp_path: Path) -> None:
    buffer = LiveFrameBuffer(tmp_path, limit=4)
    image = Image.new("RGB", (8, 8))
    for frame_id in range(20):
        result = buffer.save(image, frame_id=frame_id, kind="heartbeat", can_evict=lambda: True)
        assert result.path is not None
        assert len(buffer.frames) <= 4
    protected = []
    for frame_id, kind in [(20, "uncertain"), (21, "result-defeat")]:
        protected.append(buffer.save(image, frame_id=frame_id, kind=kind, can_evict=lambda: True).path)
    for frame_id in range(22, 42):
        buffer.save(image, frame_id=frame_id, kind="scene-change", can_evict=lambda: True)
    for frame_id, kind in [(42, "uncertain"), (43, "result-finished")]:
        protected.append(buffer.save(image, frame_id=frame_id, kind=kind, can_evict=lambda: True).path)
    assert all(path is not None and path.is_file() for path in protected)
    omitted = buffer.save(image, frame_id=44, kind="result-defeat", can_evict=lambda: True)
    assert omitted.omitted == "critical_or_modified_frames_fill_limit"
    assert len(tuple(tmp_path.glob("*.png"))) == 4


def test_cached_frame_promotes_to_critical_without_duplicating(tmp_path: Path) -> None:
    buffer = LiveFrameBuffer(tmp_path, limit=1)
    image = Image.new("RGB", (8, 8))
    saved = buffer.save(image, frame_id=1, kind="scene-change", can_evict=lambda: True)
    promoted = buffer.save(image, frame_id=1, kind="uncertain", can_evict=lambda: True)
    assert promoted.promoted and promoted.path == saved.path
    omitted = buffer.save(image, frame_id=2, kind="heartbeat", can_evict=lambda: True)
    assert omitted.path is None and saved.path.is_file()


def test_cache_does_not_adopt_existing_files_or_evict_pinned_modified_or_symlink_frames(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "frame-000001-heartbeat.png"
    existing.write_bytes(b"old user evidence")
    image = Image.new("RGB", (8, 8))
    buffer = LiveFrameBuffer(tmp_path, limit=1)
    saved = buffer.save(image, frame_id=1, kind="heartbeat", can_evict=lambda: True)
    assert saved.path != existing
    assert buffer.save(image, frame_id=2, kind="uncertain", can_evict=lambda: False).omitted == "session_protected"
    saved.path.write_bytes(b"manual change")
    assert buffer.save(image, frame_id=2, kind="uncertain", can_evict=lambda: True).path is None
    saved.path.unlink()
    saved.path.symlink_to(existing)
    assert buffer.save(image, frame_id=3, kind="uncertain", can_evict=lambda: True).path is None
    assert saved.path.is_symlink() and existing.read_bytes() == b"old user evidence"


def test_cache_refuses_a_replaced_output_directory(tmp_path: Path) -> None:
    directory = tmp_path / "frames"
    directory.mkdir()
    buffer = LiveFrameBuffer(directory, limit=1)
    moved = tmp_path / "user-frames"
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    assert buffer.save(
        Image.new("RGB", (8, 8)), frame_id=1, kind="uncertain", can_evict=lambda: True
    ).omitted == "unsafe_frame_directory"
    assert not tuple(moved.iterdir())
