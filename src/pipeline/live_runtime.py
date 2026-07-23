from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol

from src.capture.screen_geometry import CapturedWindow, MacWindowCapture
from src.logic.monte_carlo import (
    MonteCarloSettings,
    Phase4DecisionResult,
    recommend_phase4,
)
from src.pipeline.live_layout import LiveLayoutConfig
from src.state.events import PlayerSeat, RoundPhase
from src.state.observable_state import ObservableGameState
from src.tracking.visual_events import (
    VisualEventTracker,
    VisualTrackerMode,
    VisualTrackerUpdate,
)
from src.vision.scene_recognizer import SceneObservation, SceneRecognizer


class WindowCapture(Protocol):
    def capture(self, frame_id: int) -> CapturedWindow:
        ...


DecisionFn = Callable[[ObservableGameState, MonteCarloSettings], Phase4DecisionResult]


@dataclass(frozen=True)
class LiveDecisionRecord:
    round_id: str
    state_revision: int
    landlord: PlayerSeat
    result: Phase4DecisionResult

    @property
    def estimated_win_rate(self) -> float | None:
        return (
            self.result.rankings[0].estimated_win_rate
            if self.result.rankings
            else None
        )

    def to_log_payload(self) -> dict[str, object]:
        return {
            "event": "live_decision",
            "round_id": self.round_id,
            "state_revision": self.state_revision,
            "win_rate_scope": (
                "personal" if self.landlord is PlayerSeat.SELF else "farmer_team"
            ),
            **self.result.to_log_payload(),
        }


@dataclass(frozen=True)
class LiveRuntimeSnapshot:
    frame_id: int
    scene: SceneObservation
    tracker_update: VisualTrackerUpdate
    decision: LiveDecisionRecord | None
    decision_pending: bool
    capture_latency_ms: float
    total_latency_ms: float

    @property
    def state(self) -> ObservableGameState | None:
        return self.tracker_update.state

    def to_log_payload(self) -> dict[str, object]:
        return {
            "event": "live_runtime_snapshot",
            "frame_id": self.frame_id,
            "tracker_mode": self.tracker_update.mode.value,
            "decision_pending": self.decision_pending,
            "capture_latency_ms": round(self.capture_latency_ms, 3),
            "total_latency_ms": round(self.total_latency_ms, 3),
            "round_id": self.state.round_id if self.state else None,
            "state_revision": self.state.revision if self.state else None,
            "warnings": (
                list(self.state.warnings)
                if self.state is not None
                else list(self.scene.warnings)
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
    ) -> None:
        self.config = config
        self.frame_source = frame_source or MacWindowCapture(config.app_name)
        self.recognizer = recognizer or SceneRecognizer(config)
        self.tracker = tracker or VisualEventTracker(
            stability_frames=config.stability_frames,
            confidence_threshold=config.confidence_threshold,
        )
        self.decision_fn = decision_fn
        self._sleeper = sleeper
        self._next_frame_id = 1
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="doudizhu-live-decision",
        )
        self._decision_future: Future[LiveDecisionRecord] | None = None
        self._decision_revision: tuple[str, int] | None = None
        self._decision: LiveDecisionRecord | None = None
        self._logged_decisions: set[tuple[str, int]] = set()
        self._last_error_frame: tuple[str | None, int] | None = None

    def run_once(self, frame_id: int | None = None) -> LiveRuntimeSnapshot:
        current_frame_id = self._allocate_frame_id(frame_id)
        started = time.perf_counter()
        frame = self.frame_source.capture(current_frame_id)
        capture_latency_ms = (time.perf_counter() - started) * 1000
        scene = self.recognizer.observe(frame)
        _write_jsonl(self.config.log_file, scene.to_log_payload())

        update = self.tracker.update(scene)
        if update.event is not None:
            _write_jsonl(self.config.log_file, update.event.to_log_payload())
        _write_jsonl(self.config.log_file, update.to_log_payload())
        self._handle_uncertain_frame(frame, update)

        state = update.state
        self._poll_decision(state)
        self._schedule_decision(state)
        decision = self._decision_for_state(state)
        snapshot = LiveRuntimeSnapshot(
            frame_id=current_frame_id,
            scene=scene,
            tracker_update=update,
            decision=decision,
            decision_pending=self._decision_is_pending_for(state),
            capture_latency_ms=capture_latency_ms,
            total_latency_ms=(time.perf_counter() - started) * 1000,
        )
        _write_jsonl(self.config.log_file, snapshot.to_log_payload())
        return snapshot

    def run_loop(
        self,
        *,
        max_frames: int | None = None,
    ) -> Iterator[LiveRuntimeSnapshot]:
        produced = 0
        while max_frames is None or produced < max_frames:
            yield self.run_once()
            produced += 1
            if max_frames is None or produced < max_frames:
                self._sleeper(self.config.interval_seconds)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

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
        if self._decision_future is not None and not self._decision_future.done():
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
            self._evaluate_state,
            state,
            settings,
        )

    def _evaluate_state(
        self,
        state: ObservableGameState,
        settings: MonteCarloSettings,
    ) -> LiveDecisionRecord:
        result = self.decision_fn(state, settings)
        return LiveDecisionRecord(
            round_id=state.round_id,
            state_revision=state.revision,
            landlord=state.landlord,
            result=result,
        )

    def _poll_decision(self, state: ObservableGameState | None) -> None:
        future = self._decision_future
        if future is None or not future.done():
            return
        self._decision_future = None
        try:
            record = future.result()
        except Exception as exc:  # noqa: BLE001
            _write_jsonl(
                self.config.log_file,
                {
                    "event": "live_decision_error",
                    "round_id": self._decision_revision[0] if self._decision_revision else None,
                    "state_revision": self._decision_revision[1] if self._decision_revision else None,
                    "error": str(exc),
                },
            )
            self._decision_revision = None
            return
        current_key = (
            (state.round_id, state.revision)
            if _decision_allowed(state)
            else None
        )
        record_key = (record.round_id, record.state_revision)
        if record_key != current_key:
            _write_jsonl(
                self.config.log_file,
                {
                    "event": "live_decision_stale",
                    "round_id": record.round_id,
                    "state_revision": record.state_revision,
                    "current": list(current_key) if current_key is not None else None,
                },
            )
            self._decision_revision = None
            return
        self._decision = record
        self._logged_decisions.add(record_key)
        self._decision_revision = record_key
        _write_jsonl(self.config.log_file, record.to_log_payload())

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
        return (
            not self._decision_future.done()
            and self._decision_revision == (state.round_id, state.revision)
        )

    def _handle_uncertain_frame(
        self,
        frame: CapturedWindow,
        update: VisualTrackerUpdate,
    ) -> None:
        if update.mode is not VisualTrackerMode.UNCERTAIN:
            return
        round_id = update.state.round_id if update.state else None
        key = (round_id, frame.frame_id)
        if self._last_error_frame == key:
            return
        self.config.error_frames_dir.mkdir(parents=True, exist_ok=True)
        safe_round_id = (round_id or "uninitialized").replace("/", "_")
        path = self.config.error_frames_dir / (
            f"{safe_round_id}-frame-{frame.frame_id:06d}.png"
        )
        frame.image.save(path)
        self._last_error_frame = key


def _decision_allowed(state: ObservableGameState | None) -> bool:
    return bool(
        state is not None
        and state.phase is RoundPhase.PLAYING
        and state.current_actor is PlayerSeat.SELF
        and state.decision_ready
    )


def _revision_seed(state: ObservableGameState) -> int:
    payload = f"{state.round_id}:{state.revision}".encode("utf-8")
    value = 2166136261
    for byte in payload:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _write_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def format_live_snapshot(snapshot: LiveRuntimeSnapshot) -> str:
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
        lines.append("状态：等待完整新局")
        return "\n".join(lines)
    remaining = state.remaining_by_player
    lines.extend([
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
    ])
    if snapshot.decision_pending:
        lines.append("胜率计算中…")
    elif snapshot.decision is None:
        lines.append("当前不输出推荐")
    else:
        result = snapshot.decision.result
        best = result.rankings[0]
        scope = "个人" if state.landlord is PlayerSeat.SELF else "农民团队"
        lines.append(
            f"最佳：{result.action} | 估计{scope}胜率={best.estimated_win_rate:.1%}"
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
