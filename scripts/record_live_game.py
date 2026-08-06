from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from itertools import count
from pathlib import Path

from src.capture.lossless_video import (
    LOSSLESS_VIDEO_CODEC,
    LOSSLESS_VIDEO_CONTAINER,
    LOSSLESS_VIDEO_PIXEL_FORMAT,
    LOSSLESS_VIDEO_SCHEMA_VERSION,
    LosslessRGBVideoWriter,
    LosslessVideoError,
    ffmpeg_path,
    ffmpeg_version,
)
from src.capture.recording_integrity import (
    DERIVED_ROI_STORAGE,
    FRAME_SCHEMA_VERSION,
    ROI_HASH_FORMAT,
    SESSION_SCHEMA_VERSION,
    VIDEO_FRAME_SCHEMA_VERSION,
    inspect_recording_session,
    sha256_directory,
    sha256_file,
    sha256_image_pixels,
    sha256_python_implementation,
    write_bytes_atomic,
    write_json_atomic,
)
from src.capture.screen_geometry import MacWindowCapture
from src.config.live_layout import load_live_layout

_DEFAULT_RECORDING_INTERVAL_SECONDS = 0.1
_PNG_STORAGE_MODE = "png_frames"
_VIDEO_STORAGE_MODE = "lossless_video_segments"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record local-only Phase 6 full-window and ROI frames."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/live_game.local.json"),
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument(
        "--until-interrupt",
        action="store_true",
        help="Record until Ctrl-C, then persist the interrupted session safely.",
    )
    parser.add_argument("--interval", type=float)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted session after validating its manifest.",
    )
    parser.add_argument(
        "--mark-complete",
        action="store_true",
        help="Mark this recording as one complete game for acceptance auditing.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/live_game/recordings"),
    )
    parser.add_argument(
        "--evidence-split",
        choices=("development", "acceptance"),
        default="development",
        help=(
            "Development recordings may be used for debugging. Only fresh "
            "acceptance recordings are eligible for the formal gate."
        ),
    )
    return parser


_SESSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def _validate_session_name(value: str) -> str:
    if not _SESSION_RE.fullmatch(value):
        raise ValueError(
            "session must use 1-64 letters, digits, dots, underscores, or dashes"
        )
    return value


def _read_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(payload)
    expected_ids = list(range(1, len(rows) + 1))
    actual_ids = [int(str(row["frame_id"])) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("recorded frame ids must be contiguous and start at 1")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames is not None and args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.frames is not None and args.until_interrupt:
        raise SystemExit("choose either --frames or --until-interrupt")
    if args.mark_complete and args.until_interrupt:
        raise SystemExit("--until-interrupt must be finalized explicitly after Ctrl-C")
    frame_limit = args.frames if args.frames is not None else (
        None if args.until_interrupt else 200
    )
    try:
        session_name = _validate_session_name(args.session)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = load_live_layout(args.config)
    config_sha256 = sha256_file(args.config)
    interval = (
        min(config.interval_seconds, _DEFAULT_RECORDING_INTERVAL_SECONDS)
        if args.interval is None
        else args.interval
    )
    if interval <= 0:
        raise SystemExit("--interval must be positive")
    session_dir = args.output_root / session_name
    frames_dir = session_dir / "frames"
    video_dir = session_dir / "video"
    manifest_path = session_dir / "manifest.jsonl"
    metadata_path = session_dir / "session.json"
    config_snapshot_path = session_dir / "config.snapshot.json"
    existing_rows = _read_manifest(manifest_path)
    if (existing_rows or metadata_path.exists()) and not args.resume:
        raise SystemExit(
            f"session already exists with {len(existing_rows)} frames; "
            "use --resume or choose a new --session"
        )
    if args.resume and not metadata_path.exists():
        raise SystemExit("--resume requires an existing session.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise SystemExit("session metadata must be a JSON object")
        if metadata.get("schema_version") != SESSION_SCHEMA_VERSION:
            raise SystemExit("recording session uses an unsupported legacy schema")
        if metadata.get("config_sha256") != config_sha256:
            raise SystemExit("cannot resume with a different live layout config")
        existing_split = str(metadata.get("evidence_split") or "development")
        if existing_split != args.evidence_split:
            raise SystemExit("cannot resume with a different evidence split")
        frame_storage = metadata.get("frame_storage")
        if isinstance(frame_storage, dict):
            storage_mode = str(frame_storage.get("mode") or "")
        else:
            storage_mode = _PNG_STORAGE_MODE
        if args.evidence_split == "acceptance" and storage_mode != _VIDEO_STORAGE_MODE:
            raise SystemExit(
                "legacy PNG acceptance recordings cannot be resumed; start a fresh "
                "session with lossless video storage"
            )
        if bool(metadata.get("complete_game")):
            raise SystemExit("cannot resume a session already marked complete")
        if existing_rows:
            inspection = inspect_recording_session(
                session_dir,
                ignore_recorded_frame_count=True,
            )
            if not inspection.valid:
                raise SystemExit(
                    "cannot resume an invalid recording session: "
                    + "; ".join(inspection.issues)
                )
        elif (
            not config_snapshot_path.is_file()
            or sha256_file(config_snapshot_path) != config_sha256
        ):
            raise SystemExit("cannot resume: config snapshot is missing or changed")
        metadata["recorded_frames"] = len(existing_rows)
    else:
        storage_mode = (
            _VIDEO_STORAGE_MODE
            if args.evidence_split == "acceptance"
            else _PNG_STORAGE_MODE
        )
        write_bytes_atomic(config_snapshot_path, args.config.read_bytes())
        project_root = Path(__file__).resolve().parents[1]
        model_path = config.model_path.resolve()
        templates_path = config.templates_dir.resolve()
        metadata = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session": session_name,
            "created_at": time.time(),
            "config": args.config.as_posix(),
            "config_snapshot": config_snapshot_path.relative_to(session_dir).as_posix(),
            "config_sha256": config_sha256,
            "app_name": config.app_name,
            "interval_seconds": interval,
            "max_capture_gap_seconds": (
                max(0.30, interval * 3.0)
                if args.evidence_split == "acceptance"
                else max(0.75, interval * 3.0)
            ),
            "evidence_split": args.evidence_split,
            "evidence_seal": {
                "implementation_sha256": sha256_python_implementation(
                    project_root
                ),
                "model_sha256": (
                    sha256_file(model_path) if model_path.is_file() else None
                ),
                "templates_sha256": sha256_directory(templates_path),
            },
            "complete_game": False,
            "recorded_frames": 0,
            "recording_state": "recording",
        }
        if storage_mode == _VIDEO_STORAGE_MODE:
            encoder_path = ffmpeg_path()
            metadata["frame_storage"] = {
                "schema_version": LOSSLESS_VIDEO_SCHEMA_VERSION,
                "mode": _VIDEO_STORAGE_MODE,
                "codec": LOSSLESS_VIDEO_CODEC,
                "container": LOSSLESS_VIDEO_CONTAINER,
                "decoded_pixel_format": LOSSLESS_VIDEO_PIXEL_FORMAT,
                "encoder": {
                    "path": encoder_path.as_posix(),
                    "sha256": sha256_file(encoder_path),
                    "version": ffmpeg_version(),
                },
                "segments": [],
            }
    if storage_mode == _VIDEO_STORAGE_MODE:
        video_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir.mkdir(parents=True, exist_ok=True)
    metadata.update({
        "recording_state": "recording",
        "complete_game": False,
        "updated_at": time.time(),
    })
    write_json_atomic(metadata_path, metadata)
    capture = MacWindowCapture(config.app_name)
    first_frame_id = len(existing_rows) + 1
    recorded = len(existing_rows)
    recording_state = "failed"
    interrupted = False
    captured_timestamps = [
        float(str(row["timestamp"]))
        for row in existing_rows
        if isinstance(row.get("timestamp"), (int, float))
    ]
    capture_backends = {
        str(row.get("capture_backend") or "legacy_unspecified")
        for row in existing_rows
    }
    video_writer: LosslessRGBVideoWriter | None = None
    video_segment_path: Path | None = None
    video_segment_first_frame_id = first_frame_id
    video_segment_number = 0
    if storage_mode == _VIDEO_STORAGE_MODE:
        raw_storage = metadata.get("frame_storage")
        if not isinstance(raw_storage, dict):
            raise SystemExit("lossless video storage metadata is missing")
        raw_segments = raw_storage.get("segments")
        if not isinstance(raw_segments, list):
            raise SystemExit("lossless video segment metadata is invalid")
        video_segment_number = len(raw_segments) + 1
    next_capture_at = time.monotonic()
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest:
            offsets = count() if frame_limit is None else range(frame_limit)
            for offset in offsets:
                delay = next_capture_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                frame_id = first_frame_id + offset
                frame = capture.capture(frame_id)
                if (
                    args.evidence_split == "acceptance"
                    and frame.capture_backend != "screen_capture_kit_stream"
                ):
                    raise SystemExit(
                        "acceptance recording requires the persistent "
                        "ScreenCaptureKit stream; fallback capture is not eligible"
                    )
                next_capture_at += interval
                frame_record: dict[str, object]
                if storage_mode == _VIDEO_STORAGE_MODE:
                    if video_writer is None:
                        video_segment_path = (
                            video_dir / f"segment-{video_segment_number:03d}.mkv"
                        )
                        video_writer = LosslessRGBVideoWriter(
                            video_segment_path,
                            image_size=frame.image.size,
                            frames_per_second=1.0 / interval,
                        )
                    video_frame_index = video_writer.frame_count
                    video_writer.write(frame.image)
                    assert video_segment_path is not None
                    frame_record = {
                        "schema_version": VIDEO_FRAME_SCHEMA_VERSION,
                        "video_segment": video_segment_path.relative_to(
                            session_dir
                        ).as_posix(),
                        "video_frame_index": video_frame_index,
                        "full_image_pixel_sha256": sha256_image_pixels(frame.image),
                    }
                else:
                    frame_name = f"{frame_id:06d}.png"
                    full_path = frames_dir / frame_name
                    frame.image.save(full_path, compress_level=1)
                    frame_record = {
                        "schema_version": FRAME_SCHEMA_VERSION,
                        "full_image": full_path.relative_to(session_dir).as_posix(),
                        "full_image_sha256": sha256_file(full_path),
                    }
                roi_descriptors: dict[str, dict[str, object]] = {}
                roi_sha256: dict[str, str] = {}
                for name in sorted(config.rois):
                    box = config.roi(name)
                    roi_descriptors[name] = {
                        "source": "full_image",
                        "normalized_box": list(box.to_tuple()),
                    }
                    roi_sha256[name] = sha256_image_pixels(
                        config.crop(frame.image, name)
                    )
                manifest.write(json.dumps({
                    **frame_record,
                    "event": "recorded_frame",
                    "session": session_name,
                    "frame_id": frame_id,
                    "timestamp": frame.timestamp,
                    "config_sha256": config_sha256,
                    "image_size": list(frame.image.size),
                    "window_pixel_box": list(frame.pixel_box),
                    "window_logical_box": list(frame.window.window_box),
                    "capture_backend": frame.capture_backend,
                    "roi_storage": DERIVED_ROI_STORAGE,
                    "roi_hash_format": ROI_HASH_FORMAT,
                    "rois": roi_descriptors,
                    "roi_sha256": roi_sha256,
                    "labels": {},
                }, ensure_ascii=False) + "\n")
                manifest.flush()
                recorded = frame_id
                captured_timestamps.append(frame.timestamp)
                capture_backends.add(frame.capture_backend)
                print(
                    (
                        f"\rrecorded {offset + 1} (frame {frame_id})"
                        if frame_limit is None
                        else f"\rrecorded {offset + 1}/{frame_limit} (frame {frame_id})"
                    ),
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        recording_state = "interrupted"
    else:
        recording_state = "complete" if args.mark_complete else "captured"
    finally:
        encoder_error: LosslessVideoError | None = None
        if video_writer is not None:
            try:
                video_writer.close()
            except LosslessVideoError as exc:
                encoder_error = exc
                recording_state = "failed"
            if video_segment_path is not None and video_segment_path.is_file():
                raw_storage = metadata.get("frame_storage")
                assert isinstance(raw_storage, dict)
                raw_segments = raw_storage.get("segments")
                assert isinstance(raw_segments, list)
                raw_segments.append({
                    "path": video_segment_path.relative_to(session_dir).as_posix(),
                    "sha256": sha256_file(video_segment_path),
                    "first_frame_id": video_segment_first_frame_id,
                    "frame_count": video_writer.frame_count,
                    "image_size": list(video_writer.image_size),
                    "frames_per_second": video_writer.frames_per_second,
                })
        close_capture = getattr(capture, "close", None)
        if callable(close_capture):
            close_capture()
        cadence = _capture_cadence(captured_timestamps, interval=interval)
        metadata.update({
            "updated_at": time.time(),
            "recorded_frames": recorded,
            "recording_state": recording_state,
            "complete_game": recording_state == "complete",
            "capture_cadence": cadence,
                    "capture_backends": sorted(capture_backends),
                })
        if metadata["complete_game"]:
            metadata["completed_at"] = time.time()
        write_json_atomic(metadata_path, metadata)
        if encoder_error is not None:
            raise encoder_error
    print(f"\nmanifest: {manifest_path}")
    print(f"session_metadata: {metadata_path}")
    if interrupted:
        print("recording interrupted safely; finalize it after confirming the game is complete")
        return 130
    return 0


def _capture_cadence(
    timestamps: list[float],
    *,
    interval: float,
) -> dict[str, object]:
    gaps = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
        if current > previous
    ]
    if not gaps:
        return {
            "target_interval_seconds": interval,
            "sample_count": 0,
            "median_gap_seconds": None,
            "p95_gap_seconds": None,
            "max_gap_seconds": None,
            "effective_fps": None,
        }
    ordered = sorted(gaps)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    median_gap = statistics.median(gaps)
    return {
        "target_interval_seconds": interval,
        "sample_count": len(gaps),
        "mean_gap_seconds": round(statistics.fmean(gaps), 6),
        "median_gap_seconds": round(median_gap, 6),
        "p95_gap_seconds": round(ordered[p95_index], 6),
        "max_gap_seconds": round(max(gaps), 6),
        "effective_fps": round(1.0 / median_gap, 6),
    }


if __name__ == "__main__":
    raise SystemExit(main())
