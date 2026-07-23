from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.capture.screen_geometry import CapturedWindow, ScreenGeometry
from src.pipeline.calibration import WindowInfo


class RecordedWindowFrameSource:
    """Read full-window frames emitted by scripts.record_live_game."""

    def __init__(self, manifest: Path, *, app_name: str = "斗地主") -> None:
        self.manifest = manifest
        self.session_dir = manifest.parent
        self.app_name = app_name
        self.records = self._load_records(manifest)

    @property
    def frame_count(self) -> int:
        return len(self.records)

    def capture(self, frame_id: int) -> CapturedWindow:
        try:
            record = self.records[frame_id]
        except KeyError as exc:
            raise IndexError(f"recorded frame {frame_id} is not available") from exc
        path = self.session_dir / str(record["full_image"])
        image = Image.open(path).convert("RGB")
        width, height = image.size
        return CapturedWindow(
            frame_id=frame_id,
            timestamp=float(record.get("timestamp", frame_id)),
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
        )

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
            if "full_image" not in payload:
                raise ValueError(f"manifest line {line_number} is missing full_image")
            records[frame_id] = payload
        if not records:
            raise ValueError(f"recorded manifest is empty: {manifest}")
        expected = list(range(1, len(records) + 1))
        if sorted(records) != expected:
            raise ValueError("recorded frame ids must be contiguous and start at 1")
        return records


__all__ = ["RecordedWindowFrameSource"]
