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
from src.pipeline.calibration import WindowInfo
from src.pipeline.live_layout import LiveLayoutConfig
from src.pipeline.live_runtime import (
    LiveGameRuntime,
    _round_seed_path,
    _round_state_path,
    _write_round_checkpoint,
)
from src.state.events import PlayerSeat
from src.tracking.visual_events import VisualTrackerMode, VisualTrackerUpdate
from src.ui.live_overlay import LiveOverlayViewModel
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
        recognizer=_Recognizer(),
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


def test_live_overlay_exposes_background_runtime_error() -> None:
    view = LiveOverlayViewModel.from_runtime_error(
        "ScreenGeometryError: capture failed"
    )

    assert view.status == "识别线程异常停止"
    assert "已暂停" in view.best
    assert "capture failed" in view.warnings


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
