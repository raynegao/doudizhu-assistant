from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image

from src.capture.screen_geometry import (
    CapturedWindow,
    ScreenGeometry,
    WindowAvailabilityError,
    WindowCaptureStatus,
)
from src.logic.monte_carlo import MonteCarloSettings, recommend_phase4
from src.logic.rules import Play
from src.pipeline.calibration import WindowInfo
from src.pipeline.live_layout import LiveLayoutConfig
from src.pipeline.live_runtime import (
    LiveGameRuntime,
    _rotate_jsonl_log,
    _round_seed_path,
    _round_state_path,
    _write_round_checkpoint,
)
from src.state.cards import CardSet
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.observable_state import ObservableGameState
from src.tracking.visual_events import VisualTrackerMode, VisualTrackerUpdate
from src.ui.live_overlay import LiveOverlayViewModel, _advance_frame_cursor
from src.vision.scene_recognizer import (
    SceneObservation,
    SeatObservation,
    SeatRole,
    VisualCard,
    VisualSignal,
)


POST_BIDDING_HAND = (
    "3", "3", "3", "3",
    "4", "4", "4", "4",
    "5", "5", "5", "5",
    "6", "6", "6", "6",
    "7", "7", "7", "7",
)


class _FrameSource:
    def capture(self, frame_id: int) -> CapturedWindow:
        return CapturedWindow(
            frame_id=frame_id,
            timestamp=float(frame_id),
            image=Image.new("RGB", (200, 100), "navy"),
            window=WindowInfo(
                app_name="斗地主",
                window_name="斗地主",
                window_box=(0, 0, 200, 100),
            ),
            pixel_box=(0, 0, 200, 100),
            geometry=ScreenGeometry((200, 100), (200, 100)),
        )


class _UnavailableThenFrameSource:
    def __init__(self, status: WindowCaptureStatus) -> None:
        self.status = status
        self.calls = 0

    def capture(self, frame_id: int) -> CapturedWindow:
        self.calls += 1
        if self.calls == 1:
            raise WindowAvailabilityError(
                self.status,
                "斗地主窗口当前无法识别",
            )
        return _FrameSource().capture(frame_id)


class _AlwaysUnavailableFrameSource:
    def capture(self, frame_id: int) -> CapturedWindow:
        raise WindowAvailabilityError(
            WindowCaptureStatus.NOT_OPEN,
            "未检测到“斗地主”窗口，请打开斗地主",
        )


class _FrameCaptureErrorFrameSource:
    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.calls = 0

    def capture(self, frame_id: int) -> CapturedWindow:
        self.calls += 1
        if 2 <= self.calls <= self.failures + 1:
            raise WindowAvailabilityError(
                WindowCaptureStatus.CAPTURE_ERROR,
                "window-level capture failed",
            )
        return _FrameSource().capture(frame_id)


class _Recognizer:
    def observe(self, frame: CapturedWindow) -> SceneObservation:
        return SceneObservation(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            window_pixel_box=frame.pixel_box,
            self_hand=(),
            seats=(),
            self_turn=True,
            self_turn_confidence=1.0,
            confidence=1.0,
        )


class _HandRecognizer(_Recognizer):
    def __init__(self, hand: tuple[str, ...]) -> None:
        self.hand = hand

    def observe(self, frame: CapturedWindow) -> SceneObservation:
        scene = super().observe(frame)
        return replace(
            scene,
            self_hand=tuple(
                VisualCard(
                    rank=rank,
                    confidence=0.99,
                    box=(0, 0, 10, 20),
                )
                for rank in self.hand
            ),
        )


class _PostBiddingRecognizer:
    def observe(self, frame: CapturedWindow) -> SceneObservation:
        cards = tuple(
            VisualCard(
                rank=rank,
                confidence=0.99,
                box=(0, 0, 10, 20),
            )
            for rank in POST_BIDDING_HAND
        )
        seats = tuple(
            SeatObservation(
                seat=seat,
                signal=VisualSignal.NEUTRAL,
                remaining_count=20 if seat is PlayerSeat.SELF else 17,
                role=(
                    SeatRole.LANDLORD
                    if seat is PlayerSeat.SELF
                    else SeatRole.FARMER
                ),
                confidence=0.99,
                remaining_confidence=0.99,
                role_confidence=0.99,
            )
            for seat in PlayerSeat
        )
        return SceneObservation(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            window_pixel_box=frame.pixel_box,
            self_hand=cards,
            seats=seats,
            self_turn=True,
            self_turn_confidence=0.99,
            confidence=0.99,
        )


class _FlakyPostBiddingRecognizer(_PostBiddingRecognizer):
    def __init__(self) -> None:
        self.calls = 0

    def observe(self, frame: CapturedWindow) -> SceneObservation:
        self.calls += 1
        scene = super().observe(frame)
        if self.calls >= 3:
            return scene
        seats = tuple(
            (
                replace(
                    seat,
                    remaining_count=None,
                    remaining_confidence=0.0,
                    remaining_verified=False,
                )
                if seat.seat is PlayerSeat.RIGHT
                else seat
            )
            for seat in scene.seats
        )
        return replace(scene, seats=seats)


class _ResumeRecognizer(_Recognizer):
    def __init__(self, hand: tuple[str, ...]) -> None:
        self.hand = hand
        self.seed_calls = 0

    def seed_hand_references(self, image: Image.Image):
        self.seed_calls += 1
        return tuple(
            VisualCard(
                rank=rank,
                confidence=0.99,
                box=(0, 0, 10, 20),
            )
            for rank in self.hand
        )


class _Tracker:
    def __init__(self, state) -> None:
        self.state = state

    def update(self, scene: SceneObservation) -> VisualTrackerUpdate:
        return VisualTrackerUpdate(
            mode=VisualTrackerMode.TRACKING,
            message="ready",
            state=self.state,
        )


class _AvailabilityAwareTracker(_Tracker):
    def __init__(self, state) -> None:
        super().__init__(state)
        self.mode = VisualTrackerMode.TRACKING
        self.unavailable_calls = 0

    def handle_window_unavailable(self, reason: str) -> VisualTrackerUpdate:
        self.unavailable_calls += 1
        self.mode = VisualTrackerMode.UNCERTAIN
        return VisualTrackerUpdate(
            mode=self.mode,
            message=reason,
            state=self.state,
            warnings=(reason,),
        )


class _EventTracker(_Tracker):
    def update(self, scene: SceneObservation) -> VisualTrackerUpdate:
        return VisualTrackerUpdate(
            mode=VisualTrackerMode.TRACKING,
            message="self 出牌：3",
            state=self.state,
            event=ObservedAction(
                event_id=f"{self.state.round_id}:1:self",
                sequence_no=1,
                actor=PlayerSeat.SELF,
                cards=CardSet(("3",)),
                source="test",
            ),
        )


class _FinishedEventTracker(_Tracker):
    def update(self, scene: SceneObservation) -> VisualTrackerUpdate:
        return VisualTrackerUpdate(
            mode=VisualTrackerMode.FINISHED,
            message="self 出完最后一手：J；检测到胜利",
            state=self.state,
            event=ObservedAction(
                event_id=f"{self.state.round_id}:1:self",
                sequence_no=1,
                actor=PlayerSeat.SELF,
                cards=CardSet(("J",)),
                source="live_hand_diff",
            ),
        )


def test_live_runtime_schedules_and_logs_revision_scoped_decision(
    tmp_path: Path,
    phase4_ready_state,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
        simulations=2,
        max_depth=4,
        time_budget_ms=0,
        min_rollouts_per_action=1,
        top_k=1,
        max_candidates=2,
    )

    def decision(state, settings: MonteCarloSettings):
        return recommend_phase4(state, settings)

    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=_HandRecognizer(phase4_ready_state.self_hand.cards),
        tracker=_Tracker(phase4_ready_state),
        decision_fn=decision,
        sleeper=lambda _: None,
    )
    try:
        first = runtime.run_once()
        assert first.decision_pending or first.decision is None
        snapshot = first
        for _ in range(20):
            time.sleep(0.01)
            snapshot = runtime.run_once()
            if snapshot.decision is not None:
                break
    finally:
        runtime.close()

    assert snapshot.decision is not None
    assert snapshot.decision.state_revision == phase4_ready_state.revision
    assert snapshot.decision.result.rankings[0].estimated_win_rate is not None
    view = LiveOverlayViewModel.from_snapshot(snapshot)
    assert "最佳" in view.best
    assert view.top_k
    assert f"R{phase4_ready_state.revision}" in view.status
    assert f"F{snapshot.frame_id}" in view.status
    events = [
        json.loads(line)["event"]
        for line in config.log_file.read_text(encoding="utf-8").splitlines()
    ]
    assert "scene_observation" in events
    assert "state_update" in events
    assert "live_decision" in events


def test_live_runtime_blocks_bad_hand_recommendation_then_recovers(
    tmp_path: Path,
    phase4_ready_state,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
        simulations=2,
        max_depth=4,
        time_budget_ms=0,
        min_rollouts_per_action=1,
        top_k=1,
        max_candidates=2,
    )
    mismatched_hand = list(phase4_ready_state.self_hand.cards)
    mismatched_hand[0] = "BJ"
    decision_calls = 0

    def decision(state, settings: MonteCarloSettings):
        nonlocal decision_calls
        decision_calls += 1
        return recommend_phase4(state, settings)

    recognizer = _HandRecognizer(tuple(mismatched_hand))
    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=recognizer,
        tracker=_Tracker(phase4_ready_state),
        decision_fn=decision,
        sleeper=lambda _: None,
    )
    try:
        blocked_snapshots = [runtime.run_once() for _ in range(4)]
        assert decision_calls == 0
        assert all(
            snapshot.decision is None
            for snapshot in blocked_snapshots
        )
        recognizer.hand = phase4_ready_state.self_hand.cards
        recovered = runtime.run_once()
        for _ in range(20):
            if recovered.decision is not None:
                break
            time.sleep(0.01)
            recovered = runtime.run_once()
        recognizer.hand = tuple(mismatched_hand)
        stale_hidden = runtime.run_once()
    finally:
        runtime.close()

    assert all(
        "手牌画面与牌局模型不一致" in snapshot.decision_block_reason
        for snapshot in blocked_snapshots
    )
    view = LiveOverlayViewModel.from_snapshot(blocked_snapshots[-1])
    assert "已暂停" in view.best
    assert "手牌画面与牌局模型不一致" in view.best
    assert decision_calls == 1
    assert recovered.decision_block_reason == ""
    assert recovered.decision is not None
    assert stale_hidden.decision is None
    assert "手牌画面与牌局模型不一致" in stale_hidden.decision_block_reason
    rows = [
        json.loads(line)
        for line in config.log_file.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        "手牌画面与牌局模型不一致"
        in row.get("decision_block_reason", "")
        for row in rows
        if row["event"] == "live_runtime_snapshot"
    )


def test_live_runtime_rejects_impossible_decision_action(
    tmp_path: Path,
    phase4_ready_state,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
        simulations=2,
        max_depth=4,
        time_budget_ms=0,
        min_rollouts_per_action=1,
        top_k=1,
        max_candidates=2,
    )

    def invalid_decision(state, settings: MonteCarloSettings):
        valid = recommend_phase4(state, settings)
        return replace(valid, action=Play.parse(("BJ",)))

    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=_HandRecognizer(phase4_ready_state.self_hand.cards),
        tracker=_Tracker(phase4_ready_state),
        decision_fn=invalid_decision,
        sleeper=lambda _: None,
    )
    events: list[dict[str, object]] = []
    snapshots = []
    try:
        for _ in range(20):
            snapshots.append(runtime.run_once())
            events = [
                json.loads(line)
                for line in config.log_file.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            if any(
                event["event"] == "live_decision_rejected"
                for event in events
            ):
                break
            time.sleep(0.01)
    finally:
        runtime.close()

    assert all(snapshot.decision is None for snapshot in snapshots)
    rejected = [
        event
        for event in events
        if event["event"] == "live_decision_rejected"
    ]
    assert len(rejected) == 1
    assert "not legal" in str(rejected[0]["error"])


def test_live_runtime_logs_observed_action_without_crashing(
    tmp_path: Path,
    phase4_ready_state,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
    )
    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=_Recognizer(),
        tracker=_EventTracker(phase4_ready_state),
        sleeper=lambda _: None,
    )
    try:
        snapshot = runtime.run_once()
    finally:
        runtime.close()

    assert snapshot.tracker_update.event is not None
    rows = [
        json.loads(line)
        for line in config.log_file.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows].count("play_observed") == 1
    observed_action = next(
        row for row in rows
        if row["event"] == "play_observed"
    )
    assert observed_action["round_id"] == phase4_ready_state.round_id
    assert observed_action["frame_id"] == 1
    state_updates = [
        row for row in rows
        if row["event"] == "state_update"
    ]
    assert state_updates[0]["frame_id"] == 1
    assert state_updates[0]["observed_action"]["cards"] == ["3"]


def test_live_runtime_waits_for_window_and_recovers(tmp_path: Path) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
    )
    runtime = LiveGameRuntime(
        config,
        frame_source=_UnavailableThenFrameSource(
            WindowCaptureStatus.NOT_OPEN,
        ),
        recognizer=_Recognizer(),
        tracker=_Tracker(None),
        sleeper=lambda _: None,
    )
    try:
        waiting = runtime.run_once()
        recovered = runtime.run_once()
    finally:
        runtime.close()

    assert waiting.window_status is WindowCaptureStatus.NOT_OPEN
    assert waiting.window_available is False
    assert waiting.decision is None
    waiting_view = LiveOverlayViewModel.from_snapshot(waiting)
    assert "不可用" in waiting_view.roles
    assert "已暂停" in waiting_view.best

    assert recovered.window_status is WindowCaptureStatus.AVAILABLE
    assert recovered.window_available is True
    events = [
        json.loads(line)
        for line in config.log_file.read_text(encoding="utf-8").splitlines()
    ]
    transitions = [
        event["status"]
        for event in events
        if event["event"] == "live_window_status"
    ]
    assert transitions == ["not_open", "available"]


def test_live_runtime_preserves_round_across_one_capture_error(
    tmp_path: Path,
    phase4_ready_state,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
    )
    tracker = _AvailabilityAwareTracker(phase4_ready_state)
    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameCaptureErrorFrameSource(failures=1),
        recognizer=_HandRecognizer(phase4_ready_state.self_hand.cards),
        tracker=tracker,
        sleeper=lambda _: None,
    )
    try:
        before = runtime.run_once()
        failed = runtime.run_once()
        recovered = runtime.run_once()
    finally:
        runtime.close()

    assert before.state is not None
    assert failed.window_status is WindowCaptureStatus.CAPTURE_ERROR
    assert failed.decision is None
    assert failed.state is phase4_ready_state
    assert failed.tracker_update.mode is VisualTrackerMode.TRACKING
    assert "保留牌局状态" in failed.tracker_update.message
    assert tracker.unavailable_calls == 0
    assert recovered.window_status is WindowCaptureStatus.AVAILABLE
    assert recovered.state is phase4_ready_state
    assert recovered.state.round_id == before.state.round_id
    assert recovered.state.revision == before.state.revision


def test_live_runtime_marks_round_uncertain_after_repeated_capture_errors(
    tmp_path: Path,
    phase4_ready_state,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
    )
    tracker = _AvailabilityAwareTracker(phase4_ready_state)
    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameCaptureErrorFrameSource(failures=2),
        recognizer=_HandRecognizer(phase4_ready_state.self_hand.cards),
        tracker=tracker,
        sleeper=lambda _: None,
    )
    try:
        runtime.run_once()
        transient = runtime.run_once()
        uncertain = runtime.run_once()
    finally:
        runtime.close()

    assert transient.tracker_update.mode is VisualTrackerMode.TRACKING
    assert uncertain.tracker_update.mode is VisualTrackerMode.UNCERTAIN
    assert tracker.unavailable_calls == 1


def test_live_runtime_throttles_repeated_unavailable_snapshots(
    tmp_path: Path,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
    )
    runtime = LiveGameRuntime(
        config,
        frame_source=_AlwaysUnavailableFrameSource(),
        recognizer=_Recognizer(),
        tracker=_Tracker(None),
        sleeper=lambda _: None,
    )
    try:
        for frame_id in range(1, 22):
            runtime.run_once(frame_id)
    finally:
        runtime.close()

    rows = [
        json.loads(line)
        for line in config.log_file.read_text(encoding="utf-8").splitlines()
    ]
    snapshots = [
        row for row in rows
        if row["event"] == "live_runtime_snapshot"
    ]
    assert [row["frame_id"] for row in snapshots] == [1, 20]


def test_live_runtime_throttles_repeated_available_telemetry(
    tmp_path: Path,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
    )
    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=_Recognizer(),
        tracker=_Tracker(None),
        sleeper=lambda _: None,
    )
    try:
        for frame_id in range(1, 22):
            runtime.run_once(frame_id)
    finally:
        runtime.close()

    rows = [
        json.loads(line)
        for line in config.log_file.read_text(encoding="utf-8").splitlines()
    ]
    for event in (
        "scene_observation",
        "state_update",
        "live_runtime_snapshot",
    ):
        assert [
            row["frame_id"]
            for row in rows
            if row["event"] == event
        ] == [1, 20]


def test_live_runtime_archives_oversized_log_without_deleting_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "live.jsonl"
    path.write_text("historical evidence\n", encoding="utf-8")

    archive = _rotate_jsonl_log(
        path,
        max_bytes=4,
        timestamp=1_785_000_000.0,
    )

    assert archive is not None
    assert archive.exists()
    assert archive.read_text(encoding="utf-8") == "historical evidence\n"
    assert not path.exists()


def test_live_overlay_exposes_background_runtime_error() -> None:
    view = LiveOverlayViewModel.from_runtime_error(
        "ScreenGeometryError: capture failed"
    )

    assert view.status == "识别线程异常停止"
    assert "已暂停" in view.best
    assert "capture failed" in view.warnings


def test_live_overlay_frame_cursor_resets_after_worker_restart() -> None:
    assert _advance_frame_cursor(420, 421) == (421, False)
    assert _advance_frame_cursor(421, 1) == (1, True)
    assert _advance_frame_cursor(1, 2) == (2, False)


def test_live_runtime_bootstraps_after_switching_to_post_bidding_table(
    tmp_path: Path,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
        initial_stability_frames=2,
        stability_frames=3,
        simulations=2,
        max_depth=4,
        time_budget_ms=0,
        min_rollouts_per_action=1,
        top_k=1,
        max_candidates=2,
    )
    runtime = LiveGameRuntime(
        config,
        frame_source=_UnavailableThenFrameSource(
            WindowCaptureStatus.CAPTURE_ERROR,
        ),
        recognizer=_PostBiddingRecognizer(),
        sleeper=lambda _: None,
    )
    try:
        outside_game = runtime.run_once()
        stabilizing = runtime.run_once()
        initialized = runtime.run_once()
        decided = initialized
        for _ in range(20):
            time.sleep(0.01)
            decided = runtime.run_once()
            if decided.decision is not None:
                break
    finally:
        runtime.close()

    assert outside_game.window_available is False
    assert stabilizing.state is None
    assert "正在建立牌局 1/2" in stabilizing.tracker_update.message
    assert initialized.state is not None
    assert initialized.tracker_update.initialized is True
    assert initialized.state.landlord is PlayerSeat.SELF
    assert initialized.state.current_actor is PlayerSeat.SELF
    assert initialized.state.decision_ready is True
    assert decided.decision is not None
    assert decided.decision.state_revision == 0


def test_live_runtime_restores_recent_round_checkpoint(
    tmp_path: Path,
    phase4_ready_state,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
    )
    _write_round_checkpoint(
        _round_state_path(config),
        phase4_ready_state,
    )
    seed_path = _round_seed_path(config)
    seed_path.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100), "navy").save(seed_path)
    recognizer = _ResumeRecognizer(phase4_ready_state.self_hand.cards)

    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=recognizer,
        sleeper=lambda _: None,
    )
    try:
        restored = runtime.tracker.state
    finally:
        runtime.close()

    assert recognizer.seed_calls == 1
    assert restored is not None
    assert restored.round_id == phase4_ready_state.round_id
    assert restored.revision == phase4_ready_state.revision
    assert restored.self_hand == phase4_ready_state.self_hand
    assert restored.current_actor is phase4_ready_state.current_actor


def test_live_runtime_rejects_checkpoint_from_previous_ui_session(
    tmp_path: Path,
    phase4_ready_state,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
    )
    _write_round_checkpoint(
        _round_state_path(config),
        phase4_ready_state,
        runtime_session_id="previous-session",
    )
    seed_path = _round_seed_path(config)
    seed_path.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100), "navy").save(seed_path)

    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=_ResumeRecognizer(phase4_ready_state.self_hand.cards),
        sleeper=lambda _: None,
        resume_session_id="new-session",
    )
    try:
        restored = runtime.tracker.state
    finally:
        runtime.close()

    assert restored is None


def test_live_runtime_clears_checkpoint_when_round_finishes(
    tmp_path: Path,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
    )
    finished_state = ObservableGameState.from_inputs(
        (),
        round_id="finished-round",
        landlord=PlayerSeat.SELF,
        remaining_cards={
            PlayerSeat.SELF: 0,
            PlayerSeat.RIGHT: 17,
            PlayerSeat.LEFT: 17,
        },
        hidden_played_count=20,
    )
    state_path = _round_state_path(config)
    seed_path = _round_seed_path(config)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{}\n", encoding="utf-8")
    Image.new("RGB", (200, 100), "navy").save(seed_path)
    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=_Recognizer(),
        tracker=_Tracker(finished_state),
        sleeper=lambda _: None,
    )
    try:
        runtime._persist_round_context(  # noqa: SLF001
            _FrameSource().capture(1),
            VisualTrackerUpdate(
                mode=VisualTrackerMode.FINISHED,
                message="本局已结束",
                state=finished_state,
            ),
        )
    finally:
        runtime.close()

    assert not state_path.exists()
    assert not seed_path.exists()


def test_live_runtime_logs_victory_and_overlay_displays_result(
    tmp_path: Path,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
    )
    finished_state = ObservableGameState.from_inputs(
        (),
        round_id="victory-round",
        landlord=PlayerSeat.SELF,
        remaining_cards={
            PlayerSeat.SELF: 0,
            PlayerSeat.RIGHT: 11,
            PlayerSeat.LEFT: 10,
        },
        hidden_played_count=33,
    )
    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=_Recognizer(),
        tracker=_FinishedEventTracker(finished_state),
        sleeper=lambda _: None,
    )
    try:
        snapshot = runtime.run_once()
    finally:
        runtime.close()

    rows = [
        json.loads(line)
        for line in config.log_file.read_text(encoding="utf-8").splitlines()
    ]
    result = next(
        row for row in rows
        if row["event"] == "round_result_detected"
    )
    assert result["round_id"] == "victory-round"
    assert result["winner"] == "self"
    assert result["outcome"] == "victory"
    assert result["source"] == "live_hand_diff"
    view = LiveOverlayViewModel.from_snapshot(snapshot)
    assert view.best == "结果：胜利"
    assert "断点已自动清除" in view.warnings


def test_live_runtime_manual_scan_initializes_without_stability_wait(
    tmp_path: Path,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
        initial_stability_frames=3,
        stability_frames=3,
        simulations=2,
        max_depth=4,
        time_budget_ms=0,
        min_rollouts_per_action=1,
        top_k=1,
        max_candidates=2,
    )
    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=_PostBiddingRecognizer(),
        sleeper=lambda _: None,
    )
    try:
        runtime.request_current_scan()
        scanned = runtime.run_once()
    finally:
        runtime.close()

    assert scanned.tracker_update.initialized is True
    assert scanned.state is not None
    assert scanned.state.current_actor is PlayerSeat.SELF
    assert scanned.state.decision_ready is True
    assert "已扫描当前牌局" in scanned.tracker_update.message
    events = [
        json.loads(line)
        for line in config.log_file.read_text(encoding="utf-8").splitlines()
    ]
    scan_events = [
        event for event in events
        if event["event"] == "manual_current_scan"
    ]
    assert len(scan_events) == 1
    assert scan_events[0]["success"] is True


def test_live_runtime_manual_scan_retries_transient_ocr_failures(
    tmp_path: Path,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
        simulations=2,
        max_depth=4,
        time_budget_ms=0,
        min_rollouts_per_action=1,
        top_k=1,
        max_candidates=2,
    )
    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=_FlakyPostBiddingRecognizer(),
        sleeper=lambda _: None,
    )
    try:
        runtime.request_current_scan()
        first = runtime.run_once()
        second = runtime.run_once()
        third = runtime.run_once()
    finally:
        runtime.close()

    assert first.current_scan_pending is True
    assert second.current_scan_pending is True
    assert third.current_scan_pending is False
    assert third.tracker_update.initialized is True
    events = [
        json.loads(line)
        for line in config.log_file.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "manual_current_scan"
    ]
    assert [event["retrying"] for event in events] == [True, True, False]


def test_live_runtime_saves_one_error_frame_per_uncertain_episode(
    tmp_path: Path,
    phase4_ready_state,
) -> None:
    config = LiveLayoutConfig(
        log_file=tmp_path / "live.jsonl",
        error_frames_dir=tmp_path / "errors",
    )
    runtime = LiveGameRuntime(
        config,
        frame_source=_FrameSource(),
        recognizer=_Recognizer(),
        tracker=_Tracker(phase4_ready_state),
        sleeper=lambda _: None,
    )
    uncertain_state = replace(
        phase4_ready_state,
        phase=RoundPhase.UNCERTAIN,
    )
    uncertain = VisualTrackerUpdate(
        mode=VisualTrackerMode.UNCERTAIN,
        message="same recognition failure",
        state=uncertain_state,
        warnings=("same recognition failure",),
    )
    recovered = VisualTrackerUpdate(
        mode=VisualTrackerMode.TRACKING,
        message="recovered",
        state=phase4_ready_state,
    )
    try:
        runtime._handle_uncertain_frame(  # noqa: SLF001
            _FrameSource().capture(1),
            uncertain,
        )
        runtime._handle_uncertain_frame(  # noqa: SLF001
            _FrameSource().capture(2),
            uncertain,
        )
        assert len(tuple(config.error_frames_dir.glob("*.png"))) == 1

        runtime._handle_uncertain_frame(  # noqa: SLF001
            _FrameSource().capture(3),
            recovered,
        )
        runtime._handle_uncertain_frame(  # noqa: SLF001
            _FrameSource().capture(4),
            uncertain,
        )
    finally:
        runtime.close()

    assert len(tuple(config.error_frames_dir.glob("*.png"))) == 2
