from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from src.capture.screen_geometry import (
    CapturedWindow,
    MacWindowCapture,
    ScreenGeometryError,
    WindowAvailabilityError,
    WindowCaptureStatus,
)
from src.config.live_layout import LiveLayoutConfig
from src.logic.monte_carlo import (
    MonteCarloSettings,
    Phase4DecisionResult,
    recommend_phase4,
)
from src.logic.rules import Play, legal_actions
from src.reporting.live_evidence import (
    EvidenceRetentionPolicy,
    EvidenceSessionStatus,
    FixedIssueRegistry,
    LiveEvidenceError,
    LiveEvidenceStore,
    load_fixed_issue_registry,
    stable_issue_signature,
)
from src.reporting.live_frame_buffer import LiveFrameBuffer
from src.reporting.live_log_reader import list_live_log_sources, log_file_identity
from src.state.cards import CardSet, parse_cards
from src.state.events import PlayerSeat, RoundPhase
from src.state.observable_state import ObservableGameState
from src.tracking.visual_events import (
    VisualEventTracker,
    VisualTrackerMode,
    VisualTrackerUpdate,
)
from src.vision.scene_recognizer import (
    RoundResultOutcome,
    SceneObservation,
    SceneRecognizer,
)

from .decision_executor import DecisionExecutor
from .frame_schedule import FrameSchedule


class WindowCapture(Protocol):
    def capture(self, frame_id: int) -> CapturedWindow: ...


DecisionFn = Callable[[ObservableGameState, MonteCarloSettings], Phase4DecisionResult]
_ROUND_RESUME_MAX_AGE_SECONDS = 30 * 60
_CURRENT_SCAN_MAX_ATTEMPTS = 8
_UNAVAILABLE_LOG_HEARTBEAT_FRAMES = 20
_AVAILABLE_LOG_HEARTBEAT_FRAMES = 20
_TRANSIENT_CAPTURE_ERROR_LIMIT = 1
_MAX_LIVE_LOG_BYTES = 64 * 1024 * 1024
_MAX_LIVE_LOG_ARCHIVES = 3
_EVIDENCE_HEARTBEAT_FRAMES = 20
_LATEST_EVIDENCE_POINTER = "latest-session.json"
_SUPPORTED_FIXED_RUNTIME_VERSIONS = frozenset({"phase6-live-v1"})


@dataclass(frozen=True)
class LiveDecisionRecord:
    round_id: str
    state_revision: int
    landlord: PlayerSeat
    result: Phase4DecisionResult

    @property
    def estimated_win_rate(self) -> float | None:
        return self.result.rankings[0].estimated_win_rate if self.result.rankings else None

    def to_log_payload(self) -> dict[str, object]:
        return {
            "event": "live_decision",
            "round_id": self.round_id,
            "state_revision": self.state_revision,
            "win_rate_scope": ("personal" if self.landlord is PlayerSeat.SELF else "farmer_team"),
            **self.result.to_log_payload(),
        }


def _evaluate_live_decision(
    decision_fn: DecisionFn,
    state: ObservableGameState,
    settings: MonteCarloSettings,
) -> LiveDecisionRecord:
    """Module-level spawn target for an isolated Monte Carlo worker."""

    result = decision_fn(state, settings)
    return LiveDecisionRecord(
        round_id=state.round_id,
        state_revision=state.revision,
        landlord=state.landlord,
        result=result,
    )


@dataclass(frozen=True)
class LiveRuntimeSnapshot:
    frame_id: int
    scene: SceneObservation
    tracker_update: VisualTrackerUpdate
    decision: LiveDecisionRecord | None
    decision_pending: bool
    capture_latency_ms: float
    total_latency_ms: float
    current_scan_pending: bool = False
    window_status: WindowCaptureStatus = WindowCaptureStatus.AVAILABLE
    window_message: str = ""
    decision_block_reason: str = ""

    @property
    def state(self) -> ObservableGameState | None:
        return self.tracker_update.state

    @property
    def window_available(self) -> bool:
        return self.window_status is WindowCaptureStatus.AVAILABLE

    def to_log_payload(self) -> dict[str, object]:
        return {
            "event": "live_runtime_snapshot",
            "frame_id": self.frame_id,
            "window_status": self.window_status.value,
            "window_message": self.window_message,
            "tracker_mode": self.tracker_update.mode.value,
            "decision_pending": self.decision_pending,
            "current_scan_pending": self.current_scan_pending,
            "decision_block_reason": self.decision_block_reason,
            "capture_latency_ms": round(self.capture_latency_ms, 3),
            "total_latency_ms": round(self.total_latency_ms, 3),
            "round_id": self.state.round_id if self.state else None,
            "state_revision": self.state.revision if self.state else None,
            "warnings": (
                list(self.state.warnings) if self.state is not None else list(self.scene.warnings)
            ),
        }


class LiveGameRuntime:
    """Orchestrate capture, scene recognition, event tracking, and live decisions."""

    def __init__(
        self,
        config: LiveLayoutConfig,
        *,
        frame_source: WindowCapture | None = None,
        recognizer: SceneRecognizer | None = None,
        tracker: VisualEventTracker | None = None,
        decision_fn: DecisionFn = recommend_phase4,
        sleeper: Callable[[float], None] = time.sleep,
        resume_session_id: str | None = None,
        error_frame_clock: Callable[[], float] = time.monotonic,
        log_every_scene_frame: bool = False,
        decision_executor: DecisionExecutor | None = None,
        diagnostic_evidence: bool | None = None,
    ) -> None:
        production_components = frame_source is None and recognizer is None and tracker is None
        use_process_decisions = decision_fn is recommend_phase4 and production_components
        self.config = config
        self.resume_session_id = resume_session_id
        _rotate_jsonl_log(config.log_file)
        self.frame_source = frame_source or MacWindowCapture(config.app_name)
        self.recognizer = recognizer or SceneRecognizer(config)
        if tracker is not None:
            self.tracker = tracker
        else:
            resumed_state = _load_round_context(
                config,
                self.recognizer,
                runtime_session_id=resume_session_id,
            )
            self.tracker = VisualEventTracker(
                stability_frames=config.stability_frames,
                initial_stability_frames=config.initial_stability_frames,
                confidence_threshold=config.confidence_threshold,
                initial_state=resumed_state,
            )
        self.decision_fn = decision_fn
        self._sleeper = sleeper
        self._error_frame_clock = error_frame_clock
        self._log_every_scene_frame = log_every_scene_frame
        self._next_frame_id = 1
        if decision_executor is not None:
            self._executor = decision_executor
        elif use_process_decisions:
            # Pure-Python Monte Carlo retains the GIL. A spawned worker keeps
            # the capture loop responsive while the recommendation is built.
            self._executor = DecisionExecutor()
        else:
            # Tests and embedders often inject closures, which cannot be sent
            # through a spawn process. The production function above always
            # uses process isolation.
            self._executor = DecisionExecutor(
                ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="doudizhu-live-decision-test",
                )
            )
        self._decision_future: Future[LiveDecisionRecord] | None = None
        self._decision_revision: tuple[str, int] | None = None
        self._decision: LiveDecisionRecord | None = None
        self._logged_decisions: set[tuple[str, int]] = set()
        self._error_frame_saved_at: dict[tuple[str | None, str], float] = {}
        self._error_frame_round_counts: Counter[str | None] = Counter()
        self._saved_error_frame_count = 0
        self._recorded_evidence_issues: set[tuple[str | None, str]] = set()
        self._last_window_status: WindowCaptureStatus | None = None
        self._last_scene_log_signature: tuple[object, ...] | None = None
        self._last_update_log_signature: tuple[object, ...] | None = None
        self._last_snapshot_log_signature: tuple[object, ...] | None = None
        self._current_scan_attempts_remaining = 0
        self._consecutive_capture_errors = 0
        self._closed = False
        self._evidence_persistence_enabled = (
            config.evidence_enabled if diagnostic_evidence is None else diagnostic_evidence
        )
        self._diagnostic_evidence = self._evidence_persistence_enabled and (
            production_components or diagnostic_evidence is True
        )
        self._evidence_store: LiveEvidenceStore | None = None
        self._evidence_registry = FixedIssueRegistry(version=0)
        self._evidence_registry_trusted = True
        self._evidence_policy = EvidenceRetentionPolicy(
            max_total_bytes=config.evidence_max_total_bytes,
            max_sessions=config.evidence_max_sessions,
            retention_days=config.evidence_retention_days,
            protect_recent_unresolved_days=config.evidence_protect_recent_days,
        )
        self._evidence_session_id: str | None = None
        self._evidence_log_file: Path | None = None
        self._evidence_frames_dir: Path | None = None
        self._evidence_frame_paths: dict[int, Path] = {}
        self._evidence_frame_buffer: LiveFrameBuffer | None = None
        self._last_evidence_result_round_id: str | None = None
        self._evidence_event_count = 0
        self._start_evidence_session()

    def request_current_scan(self) -> None:
        """Retry available frames until the current scene can be rebuilt."""

        self._current_scan_attempts_remaining = _CURRENT_SCAN_MAX_ATTEMPTS
        self._decision = None

    def _start_evidence_session(self) -> None:
        """Create one managed diagnostic bundle per authorized scan worker."""

        if not self._diagnostic_evidence or self.resume_session_id is None:
            return
        registry_error: str | None = None
        try:
            self._evidence_registry = load_fixed_issue_registry(self.config.fixed_issue_registry)
        except (KeyError, TypeError, ValueError) as exc:
            registry_error = str(exc)
            self._evidence_registry = FixedIssueRegistry(version=0)
            self._evidence_registry_trusted = False
        store = LiveEvidenceStore(self.config.evidence_dir)
        try:
            recovered_sessions = store.recover_abandoned_sessions()
            cleanup_payload: dict[str, object] = store.cleanup(
                self._evidence_policy,
                fixed_registry=self._evidence_registry,
                supported_runtime_versions=_SUPPORTED_FIXED_RUNTIME_VERSIONS,
                dry_run=False,
            ).to_payload()
            cleanup_payload["recovered_abandoned_sessions"] = list(recovered_sessions)
        except (OSError, ValueError) as exc:
            # Diagnostics are deliberately best-effort: a damaged evidence
            # directory must never prevent the user from starting recognition.
            cleanup_payload = {"error": str(exc), "performed": False}
        session_id = f"scan-{int(time.time())}-{self.resume_session_id[:12]}-{uuid.uuid4().hex[:8]}"
        runtime_version = _live_runtime_version()
        session_started = False
        try:
            store.start_session(
                session_id,
                runtime_version=runtime_version,
                registry_version=self._evidence_registry.version,
                owner_pid=os.getpid(),
                registry_trusted=self._evidence_registry_trusted,
            )
            session_started = True
            session_path = store.session_path(session_id)
            frames_dir = session_path / "frames"
            frames_dir.mkdir(mode=0o700)
            _write_atomic_json(
                session_path / "runtime-config.json",
                {
                    "schema_version": "phase6-live-evidence-config-v1",
                    "runtime_version": runtime_version,
                    "fixed_issue_registry_version": self._evidence_registry.version,
                    "fixed_issue_registry_trusted": self._evidence_registry_trusted,
                    "capture_started_only_after_user_action": True,
                    "config": self.config.to_payload(),
                },
            )
        except (OSError, ValueError) as exc:
            if session_started:
                try:
                    store.close_session(session_id)
                except (OSError, ValueError):
                    pass
            _write_jsonl_best_effort(
                self.config.log_file,
                {
                    "event": "live_evidence_error",
                    "operation": "start_session",
                    "error": str(exc),
                    "timestamp": time.time(),
                },
            )
            return
        self._evidence_store = store
        self._evidence_session_id = session_id
        self._evidence_log_file = session_path / "events.jsonl"
        self._evidence_frames_dir = frames_dir
        self._evidence_frame_buffer = LiveFrameBuffer(
            frames_dir, limit=self.config.evidence_max_frames_per_session
        )
        self._log_event(
            {
                "event": "scan_session_started",
                "timestamp": time.time(),
                "runtime_version": runtime_version,
                "fixed_issue_registry_version": self._evidence_registry.version,
                "fixed_issue_registry_trusted": self._evidence_registry_trusted,
                "registry_warning": registry_error,
                "cleanup": cleanup_payload,
            }
        )
        self._write_latest_evidence_pointer(status="active")

    def _log_event(self, payload: dict[str, object]) -> None:
        event = dict(payload)
        if self.resume_session_id is not None:
            event.setdefault("runtime_session_id", self.resume_session_id)
        if self._evidence_session_id is not None:
            event.setdefault("scan_session_id", self._evidence_session_id)
            self._evidence_event_count += 1
            event["scan_event_index"] = self._evidence_event_count
        global_log_error: OSError | None = None
        try:
            _write_jsonl(self.config.log_file, event)
        except OSError as exc:
            global_log_error = exc
        if self._evidence_log_file is not None and self._evidence_log_file != self.config.log_file:
            try:
                _write_jsonl(self._evidence_log_file, event)
            except OSError as exc:
                if global_log_error is None:
                    try:
                        _write_jsonl(
                            self.config.log_file,
                            {
                                "event": "live_evidence_error",
                                "operation": "write_event",
                                "error": str(exc),
                                "timestamp": time.time(),
                            },
                        )
                    except OSError:
                        pass

    def _save_evidence_frame(
        self,
        frame: CapturedWindow,
        *,
        kind: str,
        round_id: str | None,
    ) -> Path | None:
        buffer = self._evidence_frame_buffer
        if buffer is None:
            return None
        try:
            result = buffer.save(
                frame.image,
                frame_id=frame.frame_id,
                kind=kind,
                can_evict=self._can_evict_current_frame_cache,
            )
        except (OSError, ValueError) as exc:
            self._log_event(
                {
                    "event": "live_evidence_error",
                    "operation": "save_frame",
                    "frame_id": frame.frame_id,
                    "error": str(exc),
                    "timestamp": time.time(),
                }
            )
            return None
        self._evidence_frame_paths = buffer.paths
        if result.evicted is not None:
            self._log_event(
                {
                    "event": "diagnostic_frame_evicted",
                    "frame_id": result.evicted.frame_id,
                    "kind": result.evicted.kind,
                    "path": result.evicted.path.as_posix(),
                    "reason": "ordinary_cache_capacity",
                    "replacement_frame_id": frame.frame_id,
                    "timestamp": time.time(),
                }
            )
        path = result.path
        if path is None:
            self._log_event(
                {
                    "event": "diagnostic_frame_omitted",
                    "frame_id": frame.frame_id,
                    "round_id": round_id,
                    "kind": kind,
                    "frame_omitted": result.omitted,
                    "error": result.error,
                    "timestamp": time.time(),
                }
            )
            return None
        if result.reused and not result.promoted:
            return path
        store = self._evidence_store
        session_id = self._evidence_session_id
        if store is not None and session_id is not None and round_id is not None:
            try:
                store.touch_round(session_id, round_id)
            except (OSError, LiveEvidenceError) as exc:
                self._log_event(
                    {
                        "event": "live_evidence_error",
                        "operation": "touch_round",
                        "round_id": round_id,
                        "error": str(exc),
                        "timestamp": time.time(),
                    }
                )
        self._log_event(
            {
                "event": "diagnostic_frame_promoted" if result.promoted else "diagnostic_frame_saved",
                "frame_id": frame.frame_id,
                "round_id": round_id,
                "kind": kind,
                "retention_class": "critical" if buffer.frames[frame.frame_id].critical else "cache",
                "path": path.as_posix(),
                "timestamp": time.time(),
            }
        )
        return path

    def _can_evict_current_frame_cache(self) -> bool:
        """Only the current unpinned owner may recycle its own ordinary cache."""

        store, session_id = self._evidence_store, self._evidence_session_id
        if store is None or session_id is None:
            return False
        try:
            session_path = store.session_path(session_id)
            if session_path.is_symlink() or session_path.resolve().parent != store.root.resolve():
                return False
            metadata = store.load_session(session_id)
            return (
                metadata.status is EvidenceSessionStatus.ACTIVE
                and not metadata.pinned
                and metadata.owner_pid == os.getpid()
            )
        except (OSError, ValueError):
            return False

    def _record_evidence_issue(
        self,
        *,
        round_id: str | None,
        reason_key: str,
        summary: str,
    ) -> str | None:
        store = self._evidence_store
        session_id = self._evidence_session_id
        if store is None or session_id is None:
            return None
        signature = stable_issue_signature(reason_key)
        try:
            store.record_issue(
                session_id,
                round_id or "uninitialized",
                signature=signature,
                summary=summary,
            )
        except (OSError, LiveEvidenceError) as exc:
            self._log_event(
                {
                    "event": "live_evidence_error",
                    "operation": "record_issue",
                    "round_id": round_id,
                    "error": str(exc),
                    "timestamp": time.time(),
                }
            )
            return None
        return signature

    def _close_evidence_session(self) -> None:
        store = self._evidence_store
        session_id = self._evidence_session_id
        if store is None or session_id is None:
            return
        self._log_event(
            {
                "event": "scan_session_stopped",
                "timestamp": time.time(),
                "saved_frame_count": len(self._evidence_frame_paths),
            }
        )
        session_closed = False
        try:
            store.close_session(session_id, sealed_event_count=self._evidence_event_count)
            session_closed = True
            cleanup = store.cleanup(
                self._evidence_policy,
                fixed_registry=self._evidence_registry,
                supported_runtime_versions=_SUPPORTED_FIXED_RUNTIME_VERSIONS,
                dry_run=False,
            )
        except (OSError, LiveEvidenceError) as exc:
            _write_jsonl_best_effort(
                self.config.log_file,
                {
                    "event": "live_evidence_error",
                    "operation": "close_session",
                    "scan_session_id": session_id,
                    "error": str(exc),
                    "timestamp": time.time(),
                },
            )
        else:
            _write_jsonl_best_effort(
                self.config.log_file,
                {
                    "event": "live_evidence_cleanup",
                    "scan_session_id": session_id,
                    "report": cleanup.to_payload(),
                    "timestamp": time.time(),
                },
            )
        self._write_latest_evidence_pointer(status="closed" if session_closed else "close_error")
        self._evidence_store = None
        self._evidence_session_id = None
        self._evidence_log_file = None
        self._evidence_frames_dir = None
        self._evidence_frame_buffer = None

    def _write_latest_evidence_pointer(self, *, status: str) -> None:
        store = self._evidence_store
        session_id = self._evidence_session_id
        if store is None or session_id is None:
            return
        session_path = store.session_path(session_id)
        try:
            _write_atomic_json(
                store.root / _LATEST_EVIDENCE_POINTER,
                {
                    "schema_version": "phase6-live-evidence-latest-v1",
                    "session_id": session_id,
                    "status": status,
                    "session_dir": session_path.as_posix(),
                    "events_file": (session_path / "events.jsonl").as_posix(),
                    "metadata_file": (session_path / "evidence-session.json").as_posix(),
                    "updated_at": time.time(),
                    "available": session_path.exists(),
                },
            )
        except OSError as exc:
            _write_jsonl_best_effort(
                self.config.log_file,
                {
                    "event": "live_evidence_error",
                    "operation": "write_latest_pointer",
                    "scan_session_id": session_id,
                    "error": str(exc),
                    "timestamp": time.time(),
                },
            )

    def run_once(self, frame_id: int | None = None) -> LiveRuntimeSnapshot:
        current_frame_id = self._allocate_frame_id(frame_id)
        started = time.perf_counter()
        try:
            frame = self.frame_source.capture(current_frame_id)
        except WindowAvailabilityError as exc:
            return self._window_unavailable_snapshot(
                current_frame_id,
                started,
                status=exc.status,
                message=str(exc),
            )
        except ScreenGeometryError as exc:
            return self._window_unavailable_snapshot(
                current_frame_id,
                started,
                status=WindowCaptureStatus.CAPTURE_ERROR,
                message=f"斗地主窗口当前无法识别：{exc}",
            )
        capture_latency_ms = (time.perf_counter() - started) * 1000
        self._consecutive_capture_errors = 0
        status_changed = self._log_window_transition(
            WindowCaptureStatus.AVAILABLE,
            "已连接斗地主窗口，正在识别",
            frame_id=current_frame_id,
        )
        scene = self.recognizer.observe(frame)
        scene_signature = _scene_log_signature(scene)
        scene_changed = self._last_scene_log_signature != scene_signature
        if self._should_log_available(
            "scene",
            scene_signature,
            frame_id=current_frame_id,
            force=status_changed or self._log_every_scene_frame,
        ):
            self._log_event(scene.to_log_payload())

        state_before_update = self.tracker.state
        manual_scan = self._current_scan_attempts_remaining > 0
        if scene.round_result is not None:
            update = self.tracker.update(scene)
            if update.round_result is not None:
                self._current_scan_attempts_remaining = 0
        elif manual_scan:
            update = self.tracker.scan_current_scene(scene)
            if update.initialized:
                self._current_scan_attempts_remaining = 0
                self._decision = None
            else:
                self._current_scan_attempts_remaining -= 1
            self._log_event(
                {
                    "event": "manual_current_scan",
                    "frame_id": current_frame_id,
                    "success": update.initialized,
                    "retrying": self._current_scan_attempts_remaining > 0,
                    "attempts_remaining": self._current_scan_attempts_remaining,
                    "message": update.message,
                    "warnings": list(update.warnings),
                }
            )
        else:
            update = self.tracker.update(scene)
        self._persist_round_context(frame, update)
        if update.event is not None and update.mode is not VisualTrackerMode.UNCERTAIN:
            event_payload = update.event.to_log_payload()
            event_payload["round_id"] = update.state.round_id if update.state is not None else None
            event_payload["frame_id"] = current_frame_id
            self._log_event(event_payload)
        elif update.event is not None:
            self._log_event(
                {
                    "event": "visual_event_rejected",
                    "round_id": (update.state.round_id if update.state is not None else None),
                    "frame_id": current_frame_id,
                    "candidate": update.event.to_log_payload(),
                    "reason": update.message,
                }
            )
        if update.round_result is not None and update.completed_round_id is not None:
            _clear_round_context(self.config)
            winner, winning_side = _visual_result_winner(
                update.round_result.outcome,
                state_before_update,
            )
            self._log_event(
                {
                    "event": "round_result_detected",
                    "round_id": update.completed_round_id,
                    "state_revision": (
                        state_before_update.revision if state_before_update is not None else None
                    ),
                    "winner": winner,
                    "landlord": (
                        state_before_update.landlord.value
                        if state_before_update is not None else None
                    ),
                    "winning_side": winning_side,
                    "outcome": update.round_result.outcome.value,
                    "confidence": update.round_result.confidence,
                    "source": "visual_result_overlay",
                    "frame_id": current_frame_id,
                }
            )
            if update.round_result.outcome is RoundResultOutcome.UNKNOWN:
                self._record_evidence_issue(
                    round_id=update.completed_round_id,
                    reason_key="round result overlay unknown",
                    summary="结算画面存在，但当前匿名签名无法判断胜负",
                )
        elif (
            update.state is not None
            and update.state.phase is RoundPhase.FINISHED
            and update.state.winner is not None
        ):
            self._log_event(
                {
                    "event": "round_result_detected",
                    "round_id": update.state.round_id,
                    "state_revision": update.state.revision,
                    "winner": update.state.winner.value,
                    "landlord": update.state.landlord.value,
                    "winning_side": (
                        "landlord" if update.state.winner is update.state.landlord else "farmers"
                    ),
                    "outcome": _self_outcome(update.state),
                    "source": (
                        update.event.source if update.event is not None else "state_invariant"
                    ),
                    "frame_id": current_frame_id,
                }
            )
        force_update_log = bool(
            status_changed or manual_scan or update.initialized or update.event is not None
        )
        if self._should_log_available(
            "update",
            _update_log_signature(update),
            frame_id=current_frame_id,
            force=force_update_log,
        ):
            update_payload = update.to_log_payload()
            update_payload["frame_id"] = current_frame_id
            self._log_event(update_payload)
        evidence_round_id = update.completed_round_id or (
            update.state.round_id if update.state is not None else None
        )
        if update.mode is not VisualTrackerMode.UNCERTAIN:
            evidence_kind: str | None = None
            if (
                update.round_result is not None
                and update.completed_round_id is not None
            ):
                evidence_kind = f"result-{update.round_result.outcome.value}"
            elif update.state is not None and update.state.phase is RoundPhase.FINISHED:
                evidence_kind = "result-finished"
            elif update.event is not None:
                evidence_kind = "observed-action"
            elif update.initialized:
                evidence_kind = "round-initialized"
            elif scene_changed:
                evidence_kind = "scene-change"
            elif current_frame_id % _EVIDENCE_HEARTBEAT_FRAMES == 0:
                evidence_kind = "heartbeat"
            if (
                evidence_kind is not None
                and not (
                    evidence_kind.startswith("result-")
                    and evidence_round_id == self._last_evidence_result_round_id
                )
            ):
                self._save_evidence_frame(
                    frame,
                    kind=evidence_kind,
                    round_id=evidence_round_id,
                )
                if evidence_kind.startswith("result-"):
                    self._last_evidence_result_round_id = evidence_round_id
        self._handle_uncertain_frame(frame, update)

        state = update.state
        scan_pending = self._current_scan_attempts_remaining > 0
        decision_block_reason = _decision_scene_block_reason(scene, state)
        if scan_pending:
            decision = None
            decision_pending = False
        elif decision_block_reason:
            # A completed result is only safe to display while the live hand
            # still agrees with the revision that produced it.  This final
            # guard prevents a stale or mis-modelled state from recommending
            # cards that are not visible in the player's current hand.
            self._poll_decision(None)
            decision = None
            decision_pending = False
        else:
            self._poll_decision(state)
            self._schedule_decision(state)
            decision = self._decision_for_state(state)
            decision_pending = self._decision_is_pending_for(state)
        snapshot = LiveRuntimeSnapshot(
            frame_id=current_frame_id,
            scene=scene,
            tracker_update=update,
            decision=decision,
            decision_pending=decision_pending,
            capture_latency_ms=capture_latency_ms,
            total_latency_ms=(time.perf_counter() - started) * 1000,
            current_scan_pending=scan_pending,
            decision_block_reason=decision_block_reason,
        )
        if self._should_log_available(
            "snapshot",
            _snapshot_log_signature(snapshot),
            frame_id=current_frame_id,
            force=force_update_log,
        ):
            self._log_event(snapshot.to_log_payload())
        return snapshot

    def _should_log_available(
        self,
        stream: str,
        signature: tuple[object, ...],
        *,
        frame_id: int,
        force: bool = False,
    ) -> bool:
        attribute = f"_last_{stream}_log_signature"
        previous = getattr(self, attribute)
        changed = previous != signature
        setattr(self, attribute, signature)
        return bool(force or changed or frame_id % _AVAILABLE_LOG_HEARTBEAT_FRAMES == 0)

    def _window_unavailable_snapshot(
        self,
        frame_id: int,
        started: float,
        *,
        status: WindowCaptureStatus,
        message: str,
    ) -> LiveRuntimeSnapshot:
        if status is WindowCaptureStatus.CAPTURE_ERROR:
            self._consecutive_capture_errors += 1
        else:
            self._consecutive_capture_errors = 0
        status_changed = self._log_window_transition(
            status,
            message,
            frame_id=frame_id,
        )
        handler = getattr(self.tracker, "handle_window_unavailable", None)
        tracker_state = getattr(self.tracker, "state", None)
        tracker_mode = getattr(
            self.tracker,
            "mode",
            VisualTrackerMode.WAITING_FOR_ROUND,
        )
        transient_capture_failure = bool(
            status is WindowCaptureStatus.CAPTURE_ERROR
            and self._consecutive_capture_errors <= _TRANSIENT_CAPTURE_ERROR_LIMIT
            and tracker_state is not None
            and tracker_mode is VisualTrackerMode.TRACKING
        )
        if transient_capture_failure:
            update = VisualTrackerUpdate(
                mode=VisualTrackerMode.TRACKING,
                message=(
                    "WindowServer 单帧捕获失败，已暂停本帧推荐并保留牌局状态；下一帧将自动重试"
                ),
                state=tracker_state,
                warnings=(
                    message,
                    "transient_capture_failure_preserved_state",
                ),
            )
        elif callable(handler):
            update = handler(message)
        else:
            update = VisualTrackerUpdate(
                mode=getattr(
                    self.tracker,
                    "mode",
                    VisualTrackerMode.WAITING_FOR_ROUND,
                ),
                message=message,
                state=getattr(self.tracker, "state", None),
                warnings=(message,),
            )
        scene = SceneObservation(
            frame_id=frame_id,
            timestamp=time.time(),
            window_pixel_box=(0, 0, 0, 0),
            self_hand=(),
            seats=(),
            self_turn=None,
            self_turn_confidence=0.0,
            confidence=0.0,
            warnings=(message,),
        )
        snapshot = LiveRuntimeSnapshot(
            frame_id=frame_id,
            scene=scene,
            tracker_update=update,
            decision=None,
            decision_pending=False,
            capture_latency_ms=(time.perf_counter() - started) * 1000,
            total_latency_ms=(time.perf_counter() - started) * 1000,
            current_scan_pending=self._current_scan_attempts_remaining > 0,
            window_status=status,
            window_message=message,
        )
        if (
            status_changed
            or (
                status is WindowCaptureStatus.CAPTURE_ERROR
                and self._consecutive_capture_errors == _TRANSIENT_CAPTURE_ERROR_LIMIT + 1
            )
            or frame_id % _UNAVAILABLE_LOG_HEARTBEAT_FRAMES == 0
        ):
            self._log_event(snapshot.to_log_payload())
        return snapshot

    def _log_window_transition(
        self,
        status: WindowCaptureStatus,
        message: str,
        *,
        frame_id: int,
    ) -> bool:
        if status is self._last_window_status:
            return False
        self._last_window_status = status
        self._log_event(
            {
                "event": "live_window_status",
                "frame_id": frame_id,
                "status": status.value,
                "message": message,
                "timestamp": time.time(),
            }
        )
        return True

    def run_loop(
        self,
        *,
        max_frames: int | None = None,
    ) -> Iterator[LiveRuntimeSnapshot]:
        produced = 0
        schedule = FrameSchedule(self.config.interval_seconds)
        while max_frames is None or produced < max_frames:
            yield self.run_once()
            produced += 1
            if max_frames is None or produced < max_frames:
                schedule.wait_next(self._sleeper)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            # Stop the persistent ScreenCaptureKit stream before waiting for a
            # bounded Monte Carlo task, so clicking "结束扫描" revokes capture
            # access immediately instead of after decision shutdown.
            close_frame_source = getattr(self.frame_source, "close", None)
            if callable(close_frame_source):
                close_frame_source()
        finally:
            try:
                self._executor.close()
            finally:
                self._close_evidence_session()

    def record_runtime_failure(self, error: BaseException) -> None:
        """Persist a crash marker before the worker's normal cleanup."""

        state = getattr(self.tracker, "state", None)
        round_id = state.round_id if state is not None else None
        reason_key = f"runtime worker crash {type(error).__name__}"
        summary = f"{type(error).__name__}: {error}"
        issue_signature = self._record_evidence_issue(
            round_id=round_id,
            reason_key=reason_key,
            summary=summary,
        )
        self._log_event(
            {
                "event": "live_runtime_crash",
                "round_id": round_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "issue_signature": issue_signature,
                "timestamp": time.time(),
            }
        )

    def _allocate_frame_id(self, frame_id: int | None) -> int:
        if frame_id is not None:
            self._next_frame_id = max(self._next_frame_id, frame_id + 1)
            return frame_id
        value = self._next_frame_id
        self._next_frame_id += 1
        return value

    def _schedule_decision(self, state: ObservableGameState | None) -> None:
        if not _decision_allowed(state):
            return
        assert state is not None
        key = (state.round_id, state.revision)
        if key in self._logged_decisions or self._decision_revision == key:
            return
        settings = MonteCarloSettings(
            simulations=self.config.simulations,
            max_depth=self.config.max_depth,
            time_budget_ms=self.config.time_budget_ms,
            seed=_revision_seed(state),
            top_k=self.config.top_k,
            max_candidates=self.config.max_candidates,
            min_rollouts_per_action=self.config.min_rollouts_per_action,
            enforce_min_rollouts=True,
        )
        self._decision_revision = key
        self._decision_future = self._executor.submit(
            key,
            _evaluate_live_decision,
            self.decision_fn,
            state,
            settings,
        )

    def _poll_decision(self, state: ObservableGameState | None) -> None:
        future = self._decision_future
        if future is None or not future.done():
            return
        self._decision_future = None
        try:
            record = future.result()
        except Exception as exc:  # noqa: BLE001
            round_id = self._decision_revision[0] if self._decision_revision else None
            issue_signature = self._record_evidence_issue(
                round_id=round_id,
                reason_key=f"decision worker error {type(exc).__name__}",
                summary=f"{type(exc).__name__}: {exc}",
            )
            self._log_event(
                {
                    "event": "live_decision_error",
                    "round_id": round_id,
                    "state_revision": self._decision_revision[1]
                    if self._decision_revision
                    else None,
                    "error": str(exc),
                    "issue_signature": issue_signature,
                }
            )
            self._decision_revision = None
            return
        current_key = (
            (state.round_id, state.revision)
            if state is not None and _decision_allowed(state)
            else None
        )
        record_key = (record.round_id, record.state_revision)
        if record_key != current_key:
            self._log_event(
                {
                    "event": "live_decision_stale",
                    "round_id": record.round_id,
                    "state_revision": record.state_revision,
                    "current": list(current_key) if current_key is not None else None,
                }
            )
            self._decision_revision = None
            return
        assert state is not None
        validation_error = _decision_validation_error(record, state)
        if validation_error is not None:
            issue_signature = self._record_evidence_issue(
                round_id=record.round_id,
                reason_key=(f"decision validation rejected {_error_reason_key(validation_error)}"),
                summary=validation_error,
            )
            self._log_event(
                {
                    "event": "live_decision_rejected",
                    "round_id": record.round_id,
                    "state_revision": record.state_revision,
                    "error": validation_error,
                    "issue_signature": issue_signature,
                }
            )
            self._decision = None
            self._logged_decisions.add(record_key)
            self._decision_revision = None
            return
        self._decision = record
        self._logged_decisions.add(record_key)
        self._decision_revision = record_key
        self._log_event(record.to_log_payload())

    def _decision_for_state(
        self,
        state: ObservableGameState | None,
    ) -> LiveDecisionRecord | None:
        if not _decision_allowed(state) or self._decision is None:
            return None
        assert state is not None
        key = (state.round_id, state.revision)
        decision_key = (
            self._decision.round_id,
            self._decision.state_revision,
        )
        return self._decision if key == decision_key else None

    def _decision_is_pending_for(
        self,
        state: ObservableGameState | None,
    ) -> bool:
        if not _decision_allowed(state) or self._decision_future is None:
            return False
        assert state is not None
        return not self._decision_future.done() and self._decision_revision == (
            state.round_id,
            state.revision,
        )

    def _handle_uncertain_frame(
        self,
        frame: CapturedWindow,
        update: VisualTrackerUpdate,
    ) -> None:
        if update.mode is not VisualTrackerMode.UNCERTAIN:
            return
        round_id = update.state.round_id if update.state else None
        reason = update.warnings[0] if update.warnings else update.message
        reason_key = _error_reason_key(reason)
        signature = (round_id, reason_key)
        issue_signature: str | None = None
        if signature not in self._recorded_evidence_issues:
            issue_signature = self._record_evidence_issue(
                round_id=round_id,
                reason_key=reason_key,
                summary=reason,
            )
            if issue_signature is not None:
                self._recorded_evidence_issues.add(signature)
        elif self._evidence_store is not None:
            issue_signature = stable_issue_signature(reason_key)
        if not self._evidence_persistence_enabled:
            self._log_event(
                {
                    "event": "diagnostic_issue_recorded",
                    "round_id": round_id,
                    "frame_id": frame.frame_id,
                    "reason": reason,
                    "reason_key": reason_key,
                    "issue_signature": issue_signature,
                    "path": None,
                    "frame_omitted": "evidence_persistence_disabled",
                    "timestamp": time.time(),
                }
            )
            return
        now = self._error_frame_clock()
        last_saved_at = self._error_frame_saved_at.get(signature)
        if (
            last_saved_at is not None
            and now - last_saved_at < self.config.error_frame_cooldown_seconds
        ):
            return
        quota_reason = (
            "error_frame_round_limit_reached"
            if self._error_frame_round_counts[round_id] >= self.config.max_error_frames_per_round
            else "error_frame_session_limit_reached"
            if self._saved_error_frame_count >= self.config.max_error_frames_per_session
            else None
        )
        if quota_reason is not None:
            self._error_frame_saved_at[signature] = now
            self._log_event(
                {
                    "event": "diagnostic_issue_recorded",
                    "round_id": round_id,
                    "frame_id": frame.frame_id,
                    "reason": reason,
                    "reason_key": reason_key,
                    "issue_signature": issue_signature,
                    "path": None,
                    "frame_omitted": quota_reason,
                    "timestamp": time.time(),
                }
            )
            return
        frame_omitted: str | None = None
        if self._evidence_frames_dir is not None:
            path = self._save_evidence_frame(
                frame,
                kind="uncertain",
                round_id=round_id,
            )
        elif self._diagnostic_evidence:
            # A production scan requested managed evidence, but its isolated
            # session could not be created. Do not silently fall back to the
            # legacy unbounded screenshot directory: retain the JSONL marker
            # and try again in the next user-authorized scan session.
            path = None
            frame_omitted = "managed_evidence_unavailable"
        else:
            self.config.error_frames_dir.mkdir(parents=True, exist_ok=True)
            safe_round_id = (round_id or "uninitialized").replace("/", "_")
            path = _unique_error_frame_path(
                self.config.error_frames_dir,
                f"{safe_round_id}-frame-{frame.frame_id:06d}",
            )
            frame.image.save(path)
            path.chmod(0o600)
        self._error_frame_saved_at[signature] = now
        if path is None:
            self._log_event(
                {
                    "event": "diagnostic_issue_recorded",
                    "round_id": round_id,
                    "frame_id": frame.frame_id,
                    "reason": reason,
                    "reason_key": reason_key,
                    "issue_signature": issue_signature,
                    "path": None,
                    "frame_omitted": frame_omitted or "evidence_frame_limit_reached",
                    "timestamp": time.time(),
                }
            )
            return
        self._error_frame_round_counts[round_id] += 1
        self._saved_error_frame_count += 1
        self._log_event(
            {
                "event": "error_frame_saved",
                "round_id": round_id,
                "frame_id": frame.frame_id,
                "reason": reason,
                "reason_key": reason_key,
                "issue_signature": issue_signature,
                "path": path.as_posix(),
                "session_saved_count": self._saved_error_frame_count,
                "round_saved_count": self._error_frame_round_counts[round_id],
                "timestamp": time.time(),
            }
        )

    def _persist_round_context(
        self,
        frame: CapturedWindow,
        update: VisualTrackerUpdate,
    ) -> None:
        state = update.state
        if state is not None and state.phase is RoundPhase.FINISHED:
            _clear_round_context(self.config)
            return
        if (
            state is None
            or state.phase is not RoundPhase.PLAYING
            or (not update.initialized and update.event is None)
        ):
            return
        if update.initialized:
            seed_path = _round_seed_path(self.config)
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            frame.image.save(seed_path)
            seed_path.chmod(0o600)
        _write_round_checkpoint(
            _round_state_path(self.config),
            state,
            runtime_session_id=self.resume_session_id,
        )


def _round_state_path(config: LiveLayoutConfig) -> Path:
    return config.log_file.with_suffix(".state.json")


def _round_seed_path(config: LiveLayoutConfig) -> Path:
    return config.error_frames_dir.parent / "runtime" / "current_round_initial.png"


def _load_round_context(
    config: LiveLayoutConfig,
    recognizer: SceneRecognizer,
    *,
    runtime_session_id: str | None = None,
) -> ObservableGameState | None:
    state_path = _round_state_path(config)
    seed_path = _round_seed_path(config)
    if not state_path.exists() or not seed_path.exists():
        return None
    try:
        latest_activity = state_path.stat().st_mtime
        if config.log_file.exists():
            latest_activity = max(
                latest_activity,
                config.log_file.stat().st_mtime,
            )
        age_seconds = max(0.0, time.time() - latest_activity)
        if age_seconds > _ROUND_RESUME_MAX_AGE_SECONDS:
            return None
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            runtime_session_id is not None
            and payload.get("runtime_session_id") != runtime_session_id
        ):
            return None
        state = _state_from_checkpoint(payload)
        with Image.open(seed_path) as source:
            seed_cards = recognizer.seed_hand_references(source.convert("RGB"))
    except (
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None
    if not seed_cards:
        return None
    seed_counts = Counter(card.rank for card in seed_cards)
    if Counter(state.self_hand.cards) - seed_counts:
        return None
    return state


def _state_from_checkpoint(payload: object) -> ObservableGameState:
    if not isinstance(payload, dict):
        raise ValueError("live round checkpoint must be an object")
    if payload.get("phase") != RoundPhase.PLAYING.value:
        raise ValueError("live round checkpoint is not active")
    remaining_payload = payload.get("remaining_cards")
    if not isinstance(remaining_payload, dict):
        raise ValueError("live round checkpoint remaining_cards is invalid")
    revision = int(payload.get("revision", 0))
    return ObservableGameState(
        round_id=str(payload["round_id"]),
        revision=revision,
        phase=RoundPhase.PLAYING,
        landlord=PlayerSeat(str(payload["landlord"])),
        turn_order=tuple(PlayerSeat(str(seat)) for seat in payload.get("turn_order", ())),
        self_hand=CardSet.parse(payload.get("self_hand", ())),
        remaining_cards=tuple((seat, int(remaining_payload[seat.value])) for seat in PlayerSeat),
        current_actor=PlayerSeat(str(payload["current_actor"])),
        trick_target=CardSet.parse(payload.get("trick_target", ())),
        trick_leader=(
            PlayerSeat(str(payload["trick_leader"]))
            if payload.get("trick_leader") is not None
            else None
        ),
        consecutive_passes=int(payload.get("consecutive_passes", 0)),
        played_cards=parse_cards(payload.get("played_cards", ())),
        hidden_played_count=int(payload.get("hidden_played_count", 0)),
        last_sequence_no=int(payload.get("last_sequence_no", revision)),
        state_confidence=float(payload.get("state_confidence", 1.0)),
        warnings=tuple(str(item) for item in payload.get("warnings", ())),
    )


def _write_round_checkpoint(
    path: Path,
    state: ObservableGameState,
    *,
    runtime_session_id: str | None = None,
) -> None:
    payload = {
        **state.to_log_payload(),
        "last_sequence_no": state.last_sequence_no,
        "saved_at": time.time(),
    }
    if runtime_session_id is not None:
        payload["runtime_session_id"] = runtime_session_id
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _clear_round_context(config: LiveLayoutConfig) -> None:
    for path in (_round_state_path(config), _round_seed_path(config)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _decision_allowed(state: ObservableGameState | None) -> bool:
    return bool(
        state is not None
        and state.phase is RoundPhase.PLAYING
        and state.current_actor is PlayerSeat.SELF
        and state.decision_ready
    )


def _self_outcome(state: ObservableGameState) -> str:
    winner = state.winner
    if winner is None:
        return "unknown"
    if state.landlord is PlayerSeat.SELF:
        return "victory" if winner is PlayerSeat.SELF else "defeat"
    return "victory" if winner is not state.landlord else "defeat"


def _visual_result_winner(
    outcome: RoundResultOutcome,
    state: ObservableGameState | None,
) -> tuple[str | None, str]:
    """Return only winner information proven by a visual outcome overlay."""

    if state is None or outcome is RoundResultOutcome.UNKNOWN:
        return None, "unknown"
    self_is_landlord = state.landlord is PlayerSeat.SELF
    landlord_won = (
        outcome is RoundResultOutcome.VICTORY
        if self_is_landlord
        else outcome is RoundResultOutcome.DEFEAT
    )
    if landlord_won:
        return state.landlord.value, "landlord"
    # When farmers win, the result panel does not prove which farmer emptied
    # their hand. Avoid inventing a seat while preserving the team outcome.
    return None, "farmers"


def _decision_validation_error(
    record: LiveDecisionRecord,
    state: ObservableGameState,
) -> str | None:
    """Reject recommendations that are not legal for the exact live revision."""

    previous = Play.parse(state.trick_target.cards)
    legal = set(legal_actions(state.self_hand, previous))
    rankings = record.result.rankings
    if not rankings:
        return "decision returned no ranked actions"
    if record.result.action not in legal:
        return f"recommended action is not legal for the tracked hand: {record.result.action}"
    if rankings[0].action != record.result.action:
        return "recommended action does not match Top-1"
    illegal_ranked = [
        evaluation.action for evaluation in rankings if evaluation.action not in legal
    ]
    if illegal_ranked:
        return "Top-K contains actions that are not legal for the tracked hand: " + ", ".join(
            str(action) for action in illegal_ranked
        )
    return None


def _scene_log_signature(scene: SceneObservation) -> tuple[object, ...]:
    return (
        tuple(card.rank for card in scene.self_hand),
        tuple(
            (
                seat.seat.value,
                seat.signal.value,
                tuple(card.rank for card in seat.cards),
                seat.remaining_count,
                seat.role.value if seat.role is not None else None,
            )
            for seat in scene.seats
        ),
        scene.self_turn,
        scene.warnings,
        scene.self_hand_empty_verified,
        (scene.self_table.complete, scene.self_table.reason,
         tuple((card.rank, card.confidence > 0) for card in scene.self_table.cards))
        if scene.self_table is not None else None,
    )


def _state_log_signature(
    state: ObservableGameState | None,
) -> tuple[object, ...] | None:
    if state is None:
        return None
    return (
        state.round_id,
        state.revision,
        state.phase.value,
        state.landlord.value,
        state.current_actor.value,
        state.self_hand.cards,
        tuple((seat.value, count) for seat, count in state.remaining_cards),
        state.trick_target.cards,
        state.trick_leader.value if state.trick_leader is not None else None,
        state.consecutive_passes,
        state.hidden_played_count,
        state.warnings,
    )


def _update_log_signature(
    update: VisualTrackerUpdate,
) -> tuple[object, ...]:
    return (
        update.mode.value,
        update.message,
        update.initialized,
        (
            update.event.event_id,
            update.event.sequence_no,
            update.event.actor.value,
            update.event.cards.cards,
            update.event.source,
        )
        if update.event is not None
        else None,
        _state_log_signature(update.state),
        update.warnings,
    )


def _snapshot_log_signature(
    snapshot: LiveRuntimeSnapshot,
) -> tuple[object, ...]:
    decision = snapshot.decision
    return (
        snapshot.window_status.value,
        snapshot.window_message,
        snapshot.tracker_update.mode.value,
        snapshot.decision_pending,
        snapshot.current_scan_pending,
        snapshot.decision_block_reason,
        (
            decision.round_id,
            decision.state_revision,
            decision.result.action,
        )
        if decision is not None
        else None,
        _state_log_signature(snapshot.state),
    )


def _decision_scene_block_reason(
    scene: SceneObservation,
    state: ObservableGameState | None,
) -> str:
    """Require the latest live hand to support every displayed recommendation."""

    if (
        state is None
        or state.phase is not RoundPhase.PLAYING
        or state.current_actor is not PlayerSeat.SELF
    ):
        return ""
    if not scene.self_hand:
        return "当前手牌画面尚未稳定，已暂停推荐"
    try:
        observed_hand = scene.self_hand_set
    except ValueError:
        return "当前手牌识别不符合物理牌组，已暂停推荐"
    if observed_hand != state.self_hand:
        return "当前手牌画面与牌局模型不一致，已暂停推荐并等待校准"
    return ""


def _revision_seed(state: ObservableGameState) -> int:
    payload = f"{state.round_id}:{state.revision}".encode()
    value = 2166136261
    for byte in payload:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _error_reason_key(reason: str) -> str:
    """Collapse volatile card/confidence details into a stable incident key."""

    category = reason.split(":", 1)[0].strip().lower()
    category = re.sub(r"\d+(?:\.\d+)?", "#", category)
    return re.sub(r"\s+", " ", category) or "unknown_error"


def _unique_error_frame_path(directory: Path, stem: str) -> Path:
    """Return a non-overwriting path even when a worker restarts frame ids."""

    candidate = directory / f"{stem}.png"
    suffix = 1
    while candidate.exists():
        candidate = directory / f"{stem}-{suffix}.png"
        suffix += 1
    return candidate


def _write_jsonl(path: Path, payload: dict[str, object]) -> None:
    _rotate_jsonl_log(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    path.chmod(0o600)


def _write_jsonl_best_effort(path: Path, payload: dict[str, object]) -> None:
    try:
        _write_jsonl(path, payload)
    except OSError:
        pass


def _write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o600)


def _live_runtime_version() -> str:
    """Fingerprint source that changes live CV, tracking, or decisions."""

    root = Path(__file__).parents[2]
    digest = hashlib.sha256()
    for relative in (
        "src/pipeline/live_runtime.py",
        "src/pipeline/decision_executor.py",
        "src/pipeline/frame_schedule.py",
        "src/reporting/live_frame_buffer.py",
        "src/reporting/live_log_reader.py",
        "src/reporting/live_evidence.py",
        "src/vision/__init__.py",
        "src/vision/scene_recognizer.py",
        "src/vision/scene_types.py",
        "src/tracking/visual_events.py",
        "src/tracking/observation_evidence.py",
        "src/logic/monte_carlo.py",
        "configs/live_evidence_fixed_issues.json",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return f"phase6-live-{digest.hexdigest()[:16]}"


def _rotate_jsonl_log(
    path: Path,
    *,
    max_bytes: int = _MAX_LIVE_LOG_BYTES,
    timestamp: float | None = None,
    archive_limit: int = _MAX_LIVE_LOG_ARCHIVES,
) -> Path | None:
    """Archive an oversized live log and retain a bounded archive window."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if archive_limit < 0:
        raise ValueError("archive_limit cannot be negative")
    if path.is_symlink():
        raise OSError(f"live log must be a plain file: {path}")
    try:
        if path.stat().st_size < max_bytes:
            return None
    except FileNotFoundError:
        return None
    try:
        previous_sources = list_live_log_sources(path)
    except ValueError as exc:
        raise OSError(str(exc)) from exc
    value = time.time() if timestamp is None else timestamp
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(value))
    prefix = f"{path.name}.{stamp}"
    previous_suffixes = [
        int(source.path.name[len(prefix) + 1:]) if source.path.name != prefix else 0
        for source in previous_sources
        if source.path.name == prefix or source.path.name.startswith(prefix + ".")
    ]
    suffix = max(previous_suffixes, default=-1) + 1
    archive = path.with_name(prefix + (f".{suffix}" if suffix else ""))
    path.replace(archive)
    archive.chmod(0o600)
    archives = list_live_log_sources(path, include_current=False)
    expired_archives = archives[:-archive_limit] if archive_limit else archives
    if expired_archives:
        # Preserve the knowledge of missing historical references before
        # deleting a slice. Fixed-issue cleanup must then fail closed.
        with path.open("x", encoding="utf-8") as current:
            current.write(json.dumps({
                "event": "live_log_history_pruned",
                "timestamp": value,
                "archive_count": len(expired_archives),
                "history_scope": "retained_log_slices_only",
            }) + "\n")
        path.chmod(0o600)
    for expired in expired_archives:
        if log_file_identity(expired.path) != expired.identity:
            raise OSError("refusing to remove a changed live log archive")
        expired.path.unlink()
    return archive


def format_live_snapshot(snapshot: LiveRuntimeSnapshot) -> str:
    if not snapshot.window_available:
        return "\n".join(
            [
                "Dou Dizhu Phase 6 Live Assistant",
                f"frame={snapshot.frame_id} window={snapshot.window_status.value}",
                snapshot.window_message,
                "状态：无法识别，已暂停推荐",
            ]
        )
    state = snapshot.state
    lines = [
        "Dou Dizhu Phase 6 Live Assistant",
        (
            f"frame={snapshot.frame_id} mode={snapshot.tracker_update.mode.value} "
            f"latency={snapshot.total_latency_ms:.1f}ms"
        ),
        snapshot.tracker_update.message,
    ]
    if state is None:
        lines.append("状态：等待地主/加倍完成后的完整开局")
        return "\n".join(lines)
    remaining = state.remaining_by_player
    lines.extend(
        [
            (
                f"地主={state.landlord.value} 当前={state.current_actor.value} "
                f"置信度={state.state_confidence:.3f}"
            ),
            (
                f"余牌 self={remaining[PlayerSeat.SELF]} "
                f"right={remaining[PlayerSeat.RIGHT]} "
                f"left={remaining[PlayerSeat.LEFT]}"
            ),
            (
                "当前待压："
                + (" ".join(state.trick_target.cards) if state.trick_target else "自由出牌")
            ),
        ]
    )
    if snapshot.decision_block_reason:
        lines.append(snapshot.decision_block_reason)
    elif snapshot.decision_pending:
        lines.append("胜率计算中…")
    elif snapshot.decision is None:
        lines.append("当前不输出推荐")
    else:
        result = snapshot.decision.result
        best = result.rankings[0]
        scope = "个人" if state.landlord is PlayerSeat.SELF else "农民团队"
        lines.append(
            f"最佳：{result.action} | 估计{scope}胜率={best.estimated_win_rate:.1%}"
            + (
                f" (95% CI {best.win_rate_ci95_low:.1%}–{best.win_rate_ci95_high:.1%})"
                if best.win_rate_ci95_low is not None and best.win_rate_ci95_high is not None
                else ""
            )
        )
        for index, evaluation in enumerate(result.rankings, start=1):
            lines.append(
                f"{index}. {evaluation.action} | "
                f"胜率={evaluation.estimated_win_rate:.1%} "
                f"样本={evaluation.simulations}"
            )
    if state.warnings:
        lines.append("WARNING:")
        lines.extend(f"  {warning}" for warning in state.warnings)
    return "\n".join(lines)


__all__ = [
    "LiveDecisionRecord",
    "LiveGameRuntime",
    "LiveRuntimeSnapshot",
    "format_live_snapshot",
]
