from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from PIL import Image

from src.capture.lossless_video import (
    LOSSLESS_VIDEO_CODEC,
    LOSSLESS_VIDEO_CONTAINER,
    LOSSLESS_VIDEO_PIXEL_FORMAT,
    LOSSLESS_VIDEO_SCHEMA_VERSION,
    LosslessVideoError,
    iter_lossless_rgb_frames,
)
from src.config.live_layout import NormalizedBox

SESSION_SCHEMA_VERSION = "phase6-recording-session-v3"
FRAME_SCHEMA_VERSION = "phase6-recorded-frame-v3"
VIDEO_FRAME_SCHEMA_VERSION = "phase6-recorded-frame-video-v1"
LEGACY_FRAME_SCHEMA_VERSION = "phase6-recorded-frame-v2"
DERIVED_ROI_STORAGE = "derived_from_full_rgb"
ROI_HASH_FORMAT = "rgb-size-pixels-v1"
REPLAY_SCHEMA_VERSION = "phase6-live-replay-run-v1"


@dataclass(frozen=True)
class RecordingInspection:
    session_dir: Path
    metadata: dict[str, object]
    manifest_rows: tuple[dict[str, object], ...]
    full_frame_hashes: tuple[str, ...]
    first_frame_timestamp: float | None
    last_frame_timestamp: float | None
    issues: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_image_pixels(image: Image.Image) -> str:
    """Hash decoded RGB pixels with dimensions, independent of PNG encoding."""

    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(ROI_HASH_FORMAT.encode("ascii"))
    digest.update(rgb.width.to_bytes(8, "big"))
    digest.update(rgb.height.to_bytes(8, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def sha256_json_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_directory(path: Path) -> str | None:
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError(f"expected a directory: {path}")
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        child_digest = sha256_file(child).encode()
        digest.update(child_digest)
    return digest.hexdigest()


def sha256_python_implementation(project_root: Path) -> str:
    project_root = project_root.resolve()
    candidates = [
        *sorted((project_root / "src").rglob("*.py")),
        *sorted((project_root / "scripts").rglob("*.py")),
        *sorted((project_root / "native").rglob("*.swift")),
        *(
            path
            for name in (
                "Makefile",
                "pyproject.toml",
                "requirements.txt",
                "requirements-dev.txt",
            )
            if (path := project_root / name).is_file()
        ),
    ]
    if not candidates:
        raise ValueError(f"implementation source is missing under: {project_root}")
    digest = hashlib.sha256()
    for child in candidates:
        if not child.is_file():
            raise ValueError(f"implementation source is missing: {child}")
        relative = child.relative_to(project_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(sha256_file(child).encode())
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in ("numpy", "pillow", "pydantic", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def inspect_recording_session(
    session_dir: Path,
    *,
    ignore_recorded_frame_count: bool = False,
) -> RecordingInspection:
    issues: list[str] = []
    metadata = _read_json_object(session_dir / "session.json", issues)
    rows = _read_jsonl(session_dir / "manifest.jsonl", issues)
    session_name = session_dir.name

    if metadata.get("schema_version") != SESSION_SCHEMA_VERSION:
        issues.append("unsupported or missing session metadata schema")
    if metadata.get("session") != session_name:
        issues.append("session metadata name does not match directory")
    if not ignore_recorded_frame_count and _integer(metadata.get("recorded_frames")) != len(rows):
        issues.append("session recorded_frames does not match manifest length")
    recording_state = metadata.get("recording_state")
    if recording_state not in {"recording", "interrupted", "failed", "captured", "complete"}:
        issues.append("session recording_state is missing or invalid")
    if bool(metadata.get("complete_game")) != (recording_state == "complete"):
        issues.append("session complete_game and recording_state disagree")
    created_at = _number(metadata.get("created_at"))
    if created_at is None:
        issues.append("session created_at is missing or invalid")

    config_sha256 = str(metadata.get("config_sha256") or "")
    snapshot = _safe_child(
        session_dir,
        metadata.get("config_snapshot"),
        "config snapshot",
        issues,
    )
    expected_roi_names: set[str] = set()
    expected_rois: dict[str, NormalizedBox] = {}
    if snapshot is not None and snapshot.is_file():
        if sha256_file(snapshot) != config_sha256:
            issues.append("config snapshot checksum does not match session metadata")
        try:
            snapshot_payload = json.loads(snapshot.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"config snapshot is not valid JSON: {exc}")
        else:
            if not isinstance(snapshot_payload, dict):
                issues.append("config snapshot root must be an object")
            else:
                rois = snapshot_payload.get("rois")
                if not isinstance(rois, dict) or not rois:
                    issues.append("config snapshot must contain non-empty rois")
                else:
                    expected_roi_names = {str(name) for name in rois}
                    try:
                        expected_rois = {
                            str(name): NormalizedBox.from_value(value)
                            for name, value in rois.items()
                        }
                    except ValueError as exc:
                        issues.append(f"config snapshot contains invalid rois: {exc}")
    elif snapshot is not None:
        issues.append("config snapshot file is missing")

    video_segments = _validate_video_storage(
        session_dir,
        metadata.get("frame_storage"),
        interval_seconds=_number(metadata.get("interval_seconds")),
        issues=issues,
    )
    video_rows: dict[str, list[dict[str, object]]] = {}
    full_hashes: list[str] = []
    frame_timestamps: list[float] = []
    seen_full_paths: set[Path] = set()
    observed_capture_backends: set[str] = set()
    for expected_frame_id, row in enumerate(rows, start=1):
        label = f"frame {expected_frame_id}"
        frame_schema = row.get("schema_version")
        if frame_schema not in {
            FRAME_SCHEMA_VERSION,
            VIDEO_FRAME_SCHEMA_VERSION,
            LEGACY_FRAME_SCHEMA_VERSION,
        }:
            issues.append(f"{label} has unsupported schema")
        if row.get("event") != "recorded_frame":
            issues.append(f"{label} has invalid event")
        if row.get("session") != session_name:
            issues.append(f"{label} session does not match directory")
        if _integer(row.get("frame_id")) != expected_frame_id:
            issues.append("recorded frame ids must be contiguous and start at 1")
        timestamp = _number(row.get("timestamp"))
        if timestamp is None:
            issues.append(f"{label} timestamp is missing or invalid")
        else:
            frame_timestamps.append(timestamp)
        if str(row.get("config_sha256") or "") != config_sha256:
            issues.append(f"{label} config checksum does not match session")

        full_path: Path | None = None
        if frame_schema == VIDEO_FRAME_SCHEMA_VERSION:
            segment_value = row.get("video_segment")
            segment_path = _safe_child(
                session_dir,
                segment_value,
                f"{label} video segment",
                issues,
            )
            segment_name = str(segment_value or "")
            if segment_path is not None and segment_name not in video_segments:
                issues.append(f"{label} references an undeclared video segment")
            frame_index = _integer(row.get("video_frame_index"))
            if frame_index is None or frame_index < 0:
                issues.append(f"{label} video frame index is missing or invalid")
            pixel_digest = row.get("full_image_pixel_sha256")
            if not _sha256_string(pixel_digest):
                issues.append(f"{label} full image pixel checksum is invalid")
            _validate_image_size_value(row.get("image_size"), label, issues)
            video_rows.setdefault(segment_name, []).append(row)
        else:
            full_path = _safe_child(
                session_dir,
                row.get("full_image"),
                f"{label} full image",
                issues,
            )
            if full_path is not None:
                if full_path in seen_full_paths:
                    issues.append(f"{label} reuses a full-image path")
                seen_full_paths.add(full_path)
                digest = _validate_file_checksum(
                    full_path,
                    row.get("full_image_sha256"),
                    f"{label} full image",
                    issues,
                )
                if digest is not None:
                    full_hashes.append(digest)
                _validate_image_size(full_path, row.get("image_size"), label, issues)

        rois = row.get("rois")
        roi_hashes = row.get("roi_sha256")
        if not isinstance(rois, Mapping) or not isinstance(roi_hashes, Mapping):
            issues.append(f"{label} must contain ROI paths and checksums")
            continue
        roi_names = {str(name) for name in rois}
        hash_names = {str(name) for name in roi_hashes}
        if roi_names != hash_names:
            issues.append(f"{label} ROI paths and checksum keys differ")
        if expected_roi_names and roi_names != expected_roi_names:
            issues.append(f"{label} ROI names do not match config snapshot")
        if frame_schema in {FRAME_SCHEMA_VERSION, VIDEO_FRAME_SCHEMA_VERSION}:
            capture_backend = row.get("capture_backend")
            if isinstance(capture_backend, str) and capture_backend:
                observed_capture_backends.add(capture_backend)
            else:
                observed_capture_backends.add("legacy_unspecified")
            _validate_derived_rois(
                full_path=full_path,
                rois=rois,
                roi_hashes=roi_hashes,
                expected_rois=expected_rois,
                storage=row.get("roi_storage"),
                hash_format=row.get("roi_hash_format"),
                label=label,
                issues=issues,
            )
        else:
            for name in sorted(roi_names & hash_names):
                roi_path = _safe_child(
                    session_dir,
                    rois[name],
                    f"{label} ROI {name}",
                    issues,
                )
                if roi_path is not None:
                    _validate_file_checksum(
                        roi_path,
                        roi_hashes[name],
                        f"{label} ROI {name}",
                        issues,
                    )

    _validate_video_frames(
        video_segments=video_segments,
        rows_by_segment=video_rows,
        expected_rois=expected_rois,
        full_hashes=full_hashes,
        issues=issues,
    )

    if not rows:
        issues.append("recording manifest is empty")
    if any(
        current <= previous
        for previous, current in zip(frame_timestamps, frame_timestamps[1:], strict=False)
    ):
        issues.append("recorded frame timestamps must be strictly increasing")
    if frame_timestamps and created_at is not None and created_at > frame_timestamps[0]:
        issues.append("session created_at is later than its first frame")
    if recording_state == "complete":
        completed_at = _number(metadata.get("completed_at"))
        if completed_at is None:
            issues.append("complete session completed_at is missing or invalid")
        elif frame_timestamps and completed_at < frame_timestamps[-1]:
            issues.append("session completed_at is earlier than its last frame")
    uses_derived_rois = any(
        row.get("schema_version") in {
            FRAME_SCHEMA_VERSION,
            VIDEO_FRAME_SCHEMA_VERSION,
        }
        for row in rows
    )
    if uses_derived_rois:
        evidence_split = metadata.get("evidence_split")
        if evidence_split not in {"development", "acceptance"}:
            issues.append("recording evidence_split is missing or invalid")
        allowed_gap = _number(metadata.get("max_capture_gap_seconds"))
        if allowed_gap is None or allowed_gap <= 0:
            issues.append("recording max_capture_gap_seconds is missing or invalid")
        gaps = [
            current - previous
            for previous, current in zip(
                frame_timestamps,
                frame_timestamps[1:],
                strict=False,
            )
        ]
        if gaps and allowed_gap is not None and max(gaps) > allowed_gap:
            issues.append(
                "recording capture gap exceeds acceptance limit: "
                f"max={max(gaps):.3f}s, allowed={allowed_gap:.3f}s"
            )
        cadence = metadata.get("capture_cadence")
        if not isinstance(cadence, Mapping):
            issues.append("recording capture_cadence is missing or invalid")
        elif cadence.get("sample_count") != len(gaps):
            issues.append("recording capture_cadence sample count is invalid")
        elif gaps:
            _validate_capture_cadence(
                metadata=metadata,
                cadence=cadence,
                gaps=gaps,
                evidence_split=evidence_split,
                issues=issues,
            )
        if evidence_split == "acceptance":
            if any(
                row.get("schema_version") != VIDEO_FRAME_SCHEMA_VERSION
                for row in rows
            ):
                issues.append(
                    "acceptance recording must use lossless inter-frame video storage"
                )
            declared_backends = metadata.get("capture_backends")
            if declared_backends != sorted(observed_capture_backends):
                issues.append("acceptance recording capture backend metadata is invalid")
            if observed_capture_backends != {"screen_capture_kit_stream"}:
                issues.append(
                    "acceptance recording must use the persistent ScreenCaptureKit stream"
                )
    return RecordingInspection(
        session_dir=session_dir,
        metadata=metadata,
        manifest_rows=tuple(rows),
        full_frame_hashes=tuple(full_hashes),
        first_frame_timestamp=frame_timestamps[0] if frame_timestamps else None,
        last_frame_timestamp=frame_timestamps[-1] if frame_timestamps else None,
        issues=tuple(dict.fromkeys(issues)),
    )


def finalize_recording_session(session_dir: Path) -> RecordingInspection:
    inspection = inspect_recording_session(
        session_dir,
        ignore_recorded_frame_count=True,
    )
    if not inspection.valid:
        raise ValueError("; ".join(inspection.issues))
    metadata = dict(inspection.metadata)
    metadata.update({
        "recorded_frames": len(inspection.manifest_rows),
        "recording_state": "complete",
        "complete_game": True,
        "completed_at": time.time(),
        "updated_at": time.time(),
    })
    write_json_atomic(session_dir / "session.json", metadata)
    finalized = inspect_recording_session(session_dir)
    if not finalized.valid:
        raise ValueError("; ".join(finalized.issues))
    return finalized


def _validate_capture_cadence(
    *,
    metadata: Mapping[str, object],
    cadence: Mapping[str, object],
    gaps: list[float],
    evidence_split: object,
    issues: list[str],
) -> None:
    median_gap = statistics.median(gaps)
    ordered = sorted(gaps)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    p95_gap = ordered[p95_index]
    reported_median = _number(cadence.get("median_gap_seconds"))
    reported_p95 = _number(cadence.get("p95_gap_seconds"))
    reported_max = _number(cadence.get("max_gap_seconds"))
    if (
        reported_median is None
        or abs(reported_median - median_gap) > 1e-5
        or reported_p95 is None
        or abs(reported_p95 - p95_gap) > 1e-5
        or reported_max is None
        or abs(reported_max - max(gaps)) > 1e-5
    ):
        issues.append("recording capture_cadence statistics are invalid")
    if evidence_split != "acceptance":
        return
    target_interval = _number(metadata.get("interval_seconds"))
    if target_interval is None or target_interval > 0.10:
        issues.append("acceptance recording target interval must be at most 0.10s")
    if median_gap > 0.15:
        issues.append(
            "acceptance recording median capture gap exceeds 0.15s: "
            f"{median_gap:.3f}s"
        )
    if p95_gap > 0.20:
        issues.append(
            "acceptance recording p95 capture gap exceeds 0.20s: "
            f"{p95_gap:.3f}s"
        )


def _read_json_object(path: Path, issues: list[str]) -> dict[str, object]:
    if not path.exists():
        issues.append(f"missing {path.name}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"invalid {path.name}: {exc}")
        return {}
    if not isinstance(payload, dict):
        issues.append(f"{path.name} root must be an object")
        return {}
    return payload


def _read_jsonl(path: Path, issues: list[str]) -> list[dict[str, object]]:
    if not path.exists():
        issues.append(f"missing {path.name}")
        return []
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        issues.append(f"cannot read {path.name}: {exc}")
        return []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"invalid manifest line {line_number}: {exc}")
            continue
        if not isinstance(payload, dict):
            issues.append(f"manifest line {line_number} must be an object")
            continue
        rows.append(payload)
    return rows


def _safe_child(
    session_dir: Path,
    value: object,
    label: str,
    issues: list[str],
) -> Path | None:
    if not isinstance(value, str) or not value:
        issues.append(f"{label} path is missing")
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        issues.append(f"{label} path escapes the session")
        return None
    return session_dir / relative


def _validate_file_checksum(
    path: Path,
    expected: object,
    label: str,
    issues: list[str],
) -> str | None:
    if not path.is_file():
        issues.append(f"{label} file is missing")
        return None
    digest = sha256_file(path)
    if not isinstance(expected, str) or digest != expected:
        issues.append(f"{label} checksum mismatch")
    return digest


def _validate_video_storage(
    session_dir: Path,
    raw_storage: object,
    *,
    interval_seconds: float | None,
    issues: list[str],
) -> dict[str, dict[str, object]]:
    if raw_storage is None:
        return {}
    if not isinstance(raw_storage, Mapping):
        issues.append("recording frame_storage is invalid")
        return {}
    expected_values = {
        "schema_version": LOSSLESS_VIDEO_SCHEMA_VERSION,
        "mode": "lossless_video_segments",
        "codec": LOSSLESS_VIDEO_CODEC,
        "container": LOSSLESS_VIDEO_CONTAINER,
        "decoded_pixel_format": LOSSLESS_VIDEO_PIXEL_FORMAT,
    }
    for name, expected in expected_values.items():
        if raw_storage.get(name) != expected:
            issues.append(f"recording frame_storage {name} is invalid")
    encoder = raw_storage.get("encoder")
    if not isinstance(encoder, Mapping):
        issues.append("recording frame_storage encoder evidence is invalid")
    else:
        if not isinstance(encoder.get("path"), str) or not encoder.get("path"):
            issues.append("recording frame_storage encoder path is invalid")
        if not _sha256_string(encoder.get("sha256")):
            issues.append("recording frame_storage encoder checksum is invalid")
        if not isinstance(encoder.get("version"), str) or not encoder.get("version"):
            issues.append("recording frame_storage encoder version is invalid")
    raw_segments = raw_storage.get("segments")
    if not isinstance(raw_segments, list):
        issues.append("recording frame_storage segments are invalid")
        return {}
    segments: dict[str, dict[str, object]] = {}
    next_frame_id = 1
    for number, raw_segment in enumerate(raw_segments, start=1):
        label = f"video segment {number}"
        if not isinstance(raw_segment, Mapping):
            issues.append(f"{label} metadata is invalid")
            continue
        relative_value = raw_segment.get("path")
        path = _safe_child(session_dir, relative_value, label, issues)
        relative_name = str(relative_value or "")
        if relative_name in segments:
            issues.append(f"{label} path is duplicated")
            continue
        if path is not None:
            _validate_file_checksum(
                path,
                raw_segment.get("sha256"),
                label,
                issues,
            )
        first_frame_id = _integer(raw_segment.get("first_frame_id"))
        frame_count = _integer(raw_segment.get("frame_count"))
        image_size = _image_size_value(raw_segment.get("image_size"))
        frames_per_second = _number(raw_segment.get("frames_per_second"))
        if first_frame_id is None or first_frame_id != next_frame_id:
            issues.append(f"{label} first_frame_id is not contiguous")
        if frame_count is None or frame_count <= 0:
            issues.append(f"{label} frame_count is missing or invalid")
        if image_size is None:
            issues.append(f"{label} image_size is missing or invalid")
        if frames_per_second is None or frames_per_second <= 0:
            issues.append(f"{label} frame rate is missing or invalid")
        elif (
            interval_seconds is not None
            and interval_seconds > 0
            and abs(frames_per_second - 1.0 / interval_seconds) > 1e-4
        ):
            issues.append(f"{label} frame rate does not match recording interval")
        if frame_count is not None and frame_count > 0:
            next_frame_id = (first_frame_id or next_frame_id) + frame_count
        segments[relative_name] = {
            "path": path,
            "first_frame_id": first_frame_id,
            "frame_count": frame_count,
            "image_size": image_size,
        }
    return segments


def _validate_video_frames(
    *,
    video_segments: Mapping[str, Mapping[str, object]],
    rows_by_segment: Mapping[str, list[dict[str, object]]],
    expected_rois: Mapping[str, NormalizedBox],
    full_hashes: list[str],
    issues: list[str],
) -> None:
    for undeclared in sorted(set(rows_by_segment) - set(video_segments)):
        if undeclared:
            issues.append(f"manifest references undeclared video segment: {undeclared}")
    for segment_name, segment in video_segments.items():
        rows = rows_by_segment.get(segment_name, [])
        expected_count = segment.get("frame_count")
        expected_first = segment.get("first_frame_id")
        image_size = segment.get("image_size")
        path = segment.get("path")
        label = f"video segment {segment_name}"
        if not rows:
            issues.append(f"{label} is not referenced by the manifest")
            continue
        if expected_count != len(rows):
            issues.append(f"{label} frame_count does not match the manifest")
        indexes = [_integer(row.get("video_frame_index")) for row in rows]
        if indexes != list(range(len(rows))):
            issues.append(f"{label} frame indexes are not contiguous")
        first_manifest_id = _integer(rows[0].get("frame_id"))
        if expected_first != first_manifest_id:
            issues.append(f"{label} first frame does not match the manifest")
        if not isinstance(path, Path) or not path.is_file() or not isinstance(
            image_size, tuple
        ):
            continue
        sentinel = object()
        try:
            decoded = iter_lossless_rgb_frames(path, image_size=image_size)
            for offset, pair in enumerate(
                zip_longest(rows, decoded, fillvalue=sentinel)
            ):
                row, image = pair
                if row is sentinel:
                    issues.append(f"{label} contains extra decoded frames")
                    break
                if image is sentinel:
                    issues.append(f"{label} ended before all manifest frames")
                    break
                assert isinstance(row, dict)
                assert isinstance(image, Image.Image)
                frame_id = _integer(row.get("frame_id")) or offset + 1
                frame_label = f"frame {frame_id}"
                declared_size = _image_size_value(row.get("image_size"))
                if declared_size is not None and image.size != declared_size:
                    issues.append(
                        f"{frame_label} decoded image size does not match manifest"
                    )
                pixel_digest = sha256_image_pixels(image)
                if pixel_digest != row.get("full_image_pixel_sha256"):
                    issues.append(f"{frame_label} full image pixel checksum mismatch")
                full_hashes.append(pixel_digest)
                rois = row.get("rois")
                roi_hashes = row.get("roi_sha256")
                if isinstance(rois, Mapping) and isinstance(roi_hashes, Mapping):
                    _validate_derived_roi_pixels(
                        image=image,
                        rois=rois,
                        roi_hashes=roi_hashes,
                        expected_rois=expected_rois,
                        label=frame_label,
                        issues=issues,
                    )
        except (LosslessVideoError, OSError, ValueError) as exc:
            issues.append(f"{label} cannot be decoded losslessly: {exc}")


def _validate_image_size_value(
    expected: object,
    label: str,
    issues: list[str],
) -> None:
    if _image_size_value(expected) is None:
        issues.append(f"{label} image_size is missing or invalid")


def _image_size_value(value: object) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    width, height = value
    if width <= 0 or height <= 0:
        return None
    return width, height


def _sha256_string(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _validate_image_size(
    path: Path,
    expected: object,
    label: str,
    issues: list[str],
) -> None:
    if not path.is_file():
        return
    if not isinstance(expected, list) or len(expected) != 2:
        issues.append(f"{label} image_size is missing or invalid")
        return
    try:
        expected_size = tuple(int(value) for value in expected)
        with Image.open(path) as image:
            actual_size = image.size
    except (OSError, TypeError, ValueError) as exc:
        issues.append(f"{label} image cannot be inspected: {exc}")
        return
    if actual_size != expected_size:
        issues.append(f"{label} image_size does not match the file")


def _validate_derived_rois(
    *,
    full_path: Path | None,
    rois: Mapping[object, object],
    roi_hashes: Mapping[object, object],
    expected_rois: Mapping[str, NormalizedBox],
    storage: object,
    hash_format: object,
    label: str,
    issues: list[str],
) -> None:
    if storage != DERIVED_ROI_STORAGE:
        issues.append(f"{label} derived ROI storage is missing or invalid")
    if hash_format != ROI_HASH_FORMAT:
        issues.append(f"{label} derived ROI hash format is missing or invalid")
    if full_path is None or not full_path.is_file() or not expected_rois:
        return
    try:
        with Image.open(full_path) as source:
            image = source.convert("RGB")
    except OSError as exc:
        issues.append(f"{label} full image cannot derive ROIs: {exc}")
        return
    _validate_derived_roi_pixels(
        image=image,
        rois=rois,
        roi_hashes=roi_hashes,
        expected_rois=expected_rois,
        label=label,
        issues=issues,
    )


def _validate_derived_roi_pixels(
    *,
    image: Image.Image,
    rois: Mapping[object, object],
    roi_hashes: Mapping[object, object],
    expected_rois: Mapping[str, NormalizedBox],
    label: str,
    issues: list[str],
) -> None:
    for name, box in sorted(expected_rois.items()):
        descriptor = rois.get(name)
        if not isinstance(descriptor, Mapping):
            issues.append(f"{label} ROI {name} derivation descriptor is invalid")
            continue
        if descriptor.get("source") != "full_image":
            issues.append(f"{label} ROI {name} derivation source is invalid")
        normalized_box = descriptor.get("normalized_box")
        if normalized_box != list(box.to_tuple()):
            issues.append(f"{label} ROI {name} box differs from config snapshot")
        crop = image.crop(box.to_pixel_box(image.size))
        expected_digest = roi_hashes.get(name)
        if (
            not isinstance(expected_digest, str)
            or sha256_image_pixels(crop) != expected_digest
        ):
            issues.append(f"{label} ROI {name} checksum mismatch")


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


__all__ = [
    "FRAME_SCHEMA_VERSION",
    "VIDEO_FRAME_SCHEMA_VERSION",
    "DERIVED_ROI_STORAGE",
    "LEGACY_FRAME_SCHEMA_VERSION",
    "REPLAY_SCHEMA_VERSION",
    "ROI_HASH_FORMAT",
    "SESSION_SCHEMA_VERSION",
    "RecordingInspection",
    "finalize_recording_session",
    "inspect_recording_session",
    "sha256_directory",
    "sha256_file",
    "sha256_json_payload",
    "sha256_image_pixels",
    "sha256_python_implementation",
    "runtime_versions",
    "write_bytes_atomic",
    "write_json_atomic",
]
