from __future__ import annotations

import queue
import sys
from dataclasses import replace

import pytest

from src.logic.monte_carlo import ActionEvaluation, Phase4DecisionResult
from src.logic.rules import Play
from src.pipeline.live_runtime import LiveDecisionRecord, LiveRuntimeSnapshot
from src.state.observable_state import ObservableGameState
from src.tracking.visual_events import VisualTrackerMode, VisualTrackerUpdate
from src.ui.live_overlay import LiveAssistantOverlay, LiveOverlayViewModel, _format_state_warnings
from src.vision.scene_recognizer import SceneObservation


def _snapshot(state: ObservableGameState) -> LiveRuntimeSnapshot:
    action = Play.parse("5")
    evaluation = ActionEvaluation(
        action=action,
        strategy_score=0.7,
        estimated_win_rate=0.75,
        terminal_win_rate=0.75,
        terminal_rate=1.0,
        simulations=4,
        average_plies=5.0,
        average_margin=0.2,
        reason_factors=(),
        risk_flags=("uniform_opponent_model",),
        win_rate_ci95_low=0.3,
        win_rate_ci95_high=0.95,
        win_rate_interval_method="wilson",
        win_rate_estimate_kind="terminal_rollout_win_rate",
    )
    result = Phase4DecisionResult(
        action=action,
        rankings=(evaluation,),
        all_candidate_count=2,
        evaluated_candidate_count=2,
        requested_simulations=200,
        completed_simulations=4,
        seed=1,
        elapsed_ms=500.0,
        reason="test",
        warnings=(
            "uniform_opponent_model",
            "low_sample_count: only 4 rollouts completed per action",
            "top_action_confidence_intervals_overlap",
            "time_budget_exhausted: completed 4 of 200 sampled worlds",
        ),
        opponent_estimate=None,
    )
    return LiveRuntimeSnapshot(
        frame_id=1,
        scene=SceneObservation(
            frame_id=1,
            timestamp=1.0,
            window_pixel_box=(0, 0, 100, 100),
            self_hand=(),
            seats=(),
            self_turn=True,
            self_turn_confidence=1.0,
            confidence=1.0,
        ),
        tracker_update=VisualTrackerUpdate(
            mode=VisualTrackerMode.TRACKING, message="ready", state=state,
        ),
        decision=LiveDecisionRecord(
            round_id=state.round_id,
            state_revision=state.revision,
            landlord=state.landlord,
            result=result,
        ),
        decision_pending=False,
        capture_latency_ms=1.0,
        total_latency_ms=10.0,
    )


def test_overlay_displays_decision_sampling_and_model_caveats(
    phase4_ready_state: ObservableGameState,
) -> None:
    view = LiveOverlayViewModel.from_snapshot(_snapshot(phase4_ready_state))

    assert "样本不足" in view.warnings
    assert "排序不稳定" in view.warnings
    assert "计算预算耗尽" in view.warnings
    assert "均匀对手模型近似" in view.warnings
    assert "区间不含模型偏差" in view.warnings
    assert view.warning_color == "#fdb022"
    assert "模型内" in view.best
    assert "95%区间" in view.best


def test_overlay_state_errors_have_priority_and_warning_duplicates_are_removed() -> None:
    text, color = _format_state_warnings(
        (
            "automatic_current_game_scan",
            "historical_played_cards_unknown=31",
            "invalid card count",
            "invalid card count",
        ),
        decision_warnings=(
            "invalid card count",
            "uniform_opponent_model",
            "uniform_opponent_model",
            "low_sample_count: only 1 rollout",
        ),
        block_reason="decision rejected",
    )

    assert text.splitlines()[:2] == ["decision rejected", "invalid card count"]
    assert text.count("invalid card count") == 1
    assert text.count("均匀对手模型近似") == 1
    assert "历史牌未知" in text
    assert "样本不足" in text
    assert color == "#f97066"


def test_overlay_blocked_state_hides_old_decision_and_its_caveats(
    phase4_ready_state: ObservableGameState,
) -> None:
    snapshot = replace(_snapshot(phase4_ready_state), decision_block_reason="hand mismatch")
    view = LiveOverlayViewModel.from_snapshot(snapshot)

    assert "已暂停" in view.best
    assert view.top_k == ()
    assert view.warnings.splitlines()[0] == "hand mismatch"
    assert "样本不足" not in view.warnings
    assert view.warning_color == "#f97066"


def test_overlay_uncertain_state_keeps_errors_above_decision_advice(
    phase4_ready_state: ObservableGameState,
) -> None:
    state = replace(
        phase4_ready_state,
        state_confidence=0.2,
        warnings=("state rejected visual event",),
    )
    view = LiveOverlayViewModel.from_snapshot(_snapshot(state))

    assert "已暂停" in view.best
    assert view.warnings == "state rejected visual event"
    assert view.warning_color == "#f97066"


def test_overlay_labels_truncated_estimates_as_scores(
    phase4_ready_state: ObservableGameState,
) -> None:
    snapshot = _snapshot(phase4_ready_state)
    assert snapshot.decision is not None
    result = snapshot.decision.result
    result = replace(
        result,
        rankings=(replace(
            result.rankings[0],
            terminal_rate=0.5,
            win_rate_estimate_kind="bounded_rollout_score",
            win_rate_interval_method="hoeffding_anytime",
            risk_flags=("depth_limit_reached",),
        ),),
        warnings=(*result.warnings, "heuristic_score_not_win_probability"),
    )
    snapshot = replace(snapshot, decision=replace(snapshot.decision, result=result))
    view = LiveOverlayViewModel.from_snapshot(snapshot)

    assert "模拟评分" in view.best
    assert "95%评分区间" in view.best
    assert "胜率" not in view.best
    assert "模拟评分" in view.top_k[0]
    assert "不等同真实胜率" in view.warnings


def test_overlay_handles_missing_win_rate_without_formatting_error(
    phase4_ready_state: ObservableGameState,
) -> None:
    snapshot = _snapshot(phase4_ready_state)
    assert snapshot.decision is not None
    result = snapshot.decision.result
    result = replace(
        result,
        rankings=(replace(result.rankings[0], estimated_win_rate=None),),
    )
    view = LiveOverlayViewModel.from_snapshot(
        replace(snapshot, decision=replace(snapshot.decision, result=result)),
    )

    assert "胜率不可用" in view.best
    assert "胜率不可用" in view.top_k[0]


def test_overlay_distinguishes_rule_proven_finish_from_sampling_confidence(
    phase4_ready_state: ObservableGameState,
) -> None:
    snapshot = _snapshot(phase4_ready_state)
    assert snapshot.decision is not None
    result = snapshot.decision.result
    result = replace(
        result,
        rankings=(replace(
            result.rankings[0],
            reason_factors=("exact_finish",),
            estimated_win_rate=1.0,
            win_rate_ci95_low=1.0,
            win_rate_ci95_high=1.0,
            win_rate_estimate_kind="rule_proven_finish",
            win_rate_interval_method="rule_proven_finish",
        ),),
    )
    view = LiveOverlayViewModel.from_snapshot(
        replace(snapshot, decision=replace(snapshot.decision, result=result)),
    )

    assert "可直接出完（规则已确认）" in view.best
    assert "95%" not in view.best


@pytest.mark.skipif(sys.platform != "darwin", reason="native Tk layout smoke runs on macOS")
def test_long_overlay_caveats_remain_scrollable_without_hiding_capture_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tkinter as tk

    original_tk = tk.Tk

    def hidden_tk() -> tk.Tk:
        root = original_tk()
        root.withdraw()
        return root

    monkeypatch.setattr(tk, "Tk", hidden_tk)
    overlay = LiveAssistantOverlay(queue.Queue())
    try:
        overlay.warnings.set("\n".join(f"合成警告 {index}" for index in range(30)))
        overlay.root.update_idletasks()
        assert overlay.scan_toggle_button.master is overlay.root
        assert overlay.scan_button.master is overlay.root
        assert overlay.warning_label.master is overlay._content
        before = overlay.content_canvas.yview()
        overlay.content_canvas.yview_moveto(1.0)
        after = overlay.content_canvas.yview()
        assert after[0] > before[0]
        assert after[1] == pytest.approx(1.0)
        assert overlay.root.state() == "withdrawn"
    finally:
        overlay.close()
