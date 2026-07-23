from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from src.capture.screen_geometry import CapturedWindow, ScreenGeometry
from src.logic.monte_carlo import MonteCarloSettings, recommend_phase4
from src.pipeline.calibration import WindowInfo
from src.pipeline.live_layout import LiveLayoutConfig
from src.pipeline.live_runtime import LiveGameRuntime
from src.tracking.visual_events import VisualTrackerMode, VisualTrackerUpdate
from src.ui.live_overlay import LiveOverlayViewModel
from src.vision.scene_recognizer import SceneObservation


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
    events = [
        json.loads(line)["event"]
        for line in config.log_file.read_text(encoding="utf-8").splitlines()
    ]
    assert "scene_observation" in events
    assert "state_update" in events
    assert "live_decision" in events
