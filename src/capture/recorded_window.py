from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from src.capture.lossless_video import iter_lossless_rgb_frames
from src.capture.recording_integrity import (
    VIDEO_FRAME_SCHEMA_VERSION,
    sha256_image_pixels,
)
from src.capture.screen_geometry import CapturedWindow, ScreenGeometry
from src.pipeline.calibration import WindowInfo


class RecordedWindowFrameSource:
    """Read full-window frames emitted by scripts.record_live_game."""

    def __init__(self, manifest: Path, *, app_name: str = "斗地主") -> None:
        self.manifest = manifest
        self.session_dir = manifest.parent
        self.app_name = app_name
        self.records = self._load_records(manifest)
        self._video_segment: Path | None = None
        self._video_iterator: Iterator[Image.Image] | None = None
        self._video_frame_index = -1

    @property
    def frame_count(self) -> int:
        return len(self.records)

    def capture(self, frame_id: int) -> CapturedWindow:
        try:
            record = self.records[frame_id]
        except KeyError as exc:
            raise IndexError(f"recorded frame {frame_id} is not available") from exc
        if record.get("schema_version") == VIDEO_FRAME_SCHEMA_VERSION:
            image = self._capture_video_frame(record, frame_id=frame_id)
        else:
            relative_path = _safe_relative_path(record.get("full_image"))
            path = self.session_dir / relative_path
            expected_sha256 = record.get("full_image_sha256")
            if expected_sha256 is not None:
                actual_sha256 = _sha256_file(path)
                if actual_sha256 != str(expected_sha256):
                    raise ValueError(
                        "recorded frame checksum mismatch for frame "
                        f"{frame_id}: {path}"
                    )
            image = Image.open(path).convert("RGB")
        width, height = image.size
        raw_timestamp = record.get("timestamp")
        timestamp = (
            float(raw_timestamp)
            if isinstance(raw_timestamp, (int, float))
            and not isinstance(raw_timestamp, bool)
            else float(frame_id)
        )
        return CapturedWindow(
            frame_id=frame_id,
            timestamp=timestamp,
            image=image,
            window=WindowInfo(
                app_name=self.app_name,
                window_name=f"recorded:{self.session_dir.name}",
                window_box=(0, 0, width, height),
            ),
            pixel_box=(0, 0, width, height),
            geometry=ScreenGeometry(
                logical_size=(width, height),
                pixel_size=(width, height),
            ),
            capture_backend=str(record.get("capture_backend") or "recorded"),
        )

    def close(self) -> None:
        iterator = self._video_iterator
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
        self._video_iterator = None
        self._video_segment = None
        self._video_frame_index = -1

    def _capture_video_frame(
        self,
        record: dict[str, object],
        *,
        frame_id: int,
    ) -> Image.Image:
        relative_path = _safe_relative_path(record.get("video_segment"))
        path = self.session_dir / relative_path
        raw_size = record.get("image_size")
        if (
            not isinstance(raw_size, list)
            or len(raw_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_size)
        ):
            raise ValueError(f"recorded video frame {frame_id} has invalid image_size")
        image_size = (raw_size[0], raw_size[1])
        target_index = record.get("video_frame_index")
        if isinstance(target_index, bool) or not isinstance(target_index, int) or target_index < 0:
            raise ValueError(f"recorded video frame {frame_id} has invalid index")
        if path != self._video_segment or target_index <= self._video_frame_index:
            self.close()
            self._video_segment = path
            self._video_iterator = iter_lossless_rgb_frames(
                path,
                image_size=image_size,
            )
        iterator = self._video_iterator
        if iterator is None:
            raise ValueError(f"recorded video frame {frame_id} cannot be decoded")
        try:
            while self._video_frame_index < target_index:
                image = next(iterator)
                self._video_frame_index += 1
        except StopIteration as exc:
            raise ValueError(
                f"recorded video ended before frame {frame_id}: {path}"
            ) from exc
        expected = record.get("full_image_pixel_sha256")
        if sha256_image_pixels(image) != expected:
            raise ValueError(
                f"recorded frame pixel checksum mismatch for frame {frame_id}: {path}"
            )
        return image

    @staticmethod
    def _load_records(manifest: Path) -> dict[int, dict[str, object]]:
        records: dict[int, dict[str, object]] = {}
        for line_number, line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"manifest line {line_number} must be an object")
            frame_id = int(payload["frame_id"])
            if frame_id in records:
                raise ValueError(f"duplicate recorded frame_id: {frame_id}")
            is_video = payload.get("schema_version") == VIDEO_FRAME_SCHEMA_VERSION
            if is_video and "video_segment" not in payload:
                raise ValueError(
                    f"manifest line {line_number} is missing video_segment"
                )
            if not is_video and "full_image" not in payload:
                raise ValueError(f"manifest line {line_number} is missing full_image")
            records[frame_id] = payload
        if not records:
            raise ValueError(f"recorded manifest is empty: {manifest}")
        expected = list(range(1, len(records) + 1))
        if sorted(records) != expected:
            raise ValueError("recorded frame ids must be contiguous and start at 1")
        return records


def _safe_relative_path(value: object) -> Path:
    relative_path = Path(str(value or ""))
    if (
        not relative_path.parts
        or relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise ValueError(
            f"recorded frame path must stay inside session: {relative_path}"
        )
    return relative_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["RecordedWindowFrameSource"]
