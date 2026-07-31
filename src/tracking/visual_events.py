from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Sequence

from src.logic.action_validation import validate_observed_action
from src.logic.rules import Play, PlayType
from src.state.cards import FULL_DECK, CardSet
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.game_tracker import GameStateTracker, StateUpdateStatus
from src.state.observable_state import ObservableGameState
from src.vision.scene_recognizer import (
    SceneObservation,
    SeatObservation,
    SeatRole,
    VisualCard,
    VisualSignal,
)


class VisualTrackerMode(str, Enum):
    WAITING_FOR_ROUND = "waiting_for_round"
    TRACKING = "tracking"
    UNCERTAIN = "uncertain"
    FINISHED = "finished"


@dataclass(frozen=True)
class VisualTrackerUpdate:
    mode: VisualTrackerMode
    message: str
    state: ObservableGameState | None
    event: ObservedAction | None = None
    initialized: bool = False
    warnings: tuple[str, ...] = ()

    def to_log_payload(self) -> dict[str, object]:
        return {
            "event": "state_update",
            "mode": self.mode.value,
            "message": self.message,
            "initialized": self.initialized,
            "observed_action": (
                self.event.to_log_payload() if self.event is not None else None
            ),
            "state": self.state.to_log_payload() if self.state is not None else None,
            "warnings": list(self.warnings),
        }


@dataclass
class _StableValue:
    fingerprint: tuple[object, ...] | None = None
    count: int = 0

    def update(self, fingerprint: tuple[object, ...]) -> int:
        if fingerprint == self.fingerprint:
            self.count += 1
        else:
            self.fingerprint = fingerprint
            self.count = 1
        return self.count


RoundIdFactory = Callable[[SceneObservation], str]


class VisualEventTracker:
    """Convert stable scene observations into validated Phase 4 game events."""

    def __init__(
        self,
        *,
        stability_frames: int = 3,
        initial_stability_frames: int | None = None,
        confidence_threshold: float = 0.70,
        round_id_factory: RoundIdFactory | None = None,
        initial_state: ObservableGameState | None = None,
    ) -> None:
        if stability_frames <= 0:
            raise ValueError("stability_frames must be positive")
        if initial_stability_frames is not None and initial_stability_frames <= 0:
            raise ValueError("initial_stability_frames must be positive")
        self.stability_frames = stability_frames
        self.initial_stability_frames = (
            stability_frames
            if initial_stability_frames is None
            else initial_stability_frames
        )
        self.confidence_threshold = confidence_threshold
        self.round_id_factory = round_id_factory or (
            lambda scene: f"live-{int(scene.timestamp * 1000)}"
        )
        if (
            initial_state is not None
            and initial_state.phase is not RoundPhase.PLAYING
        ):
            raise ValueError("resumed visual state must be in playing phase")
        self.mode = (
            VisualTrackerMode.TRACKING
            if initial_state is not None
            else VisualTrackerMode.WAITING_FOR_ROUND
        )
        self._tracker: GameStateTracker | None = (
            GameStateTracker(
                initial_state,
                validator=validate_observed_action,
                confidence_threshold=confidence_threshold,
            )
            if initial_state is not None
            else None
        )
        self._initial_stable = _StableValue()
        self._auto_scan_stable = _StableValue()
        self._active_scan_stable = _StableValue()
        self._role_stable = _StableValue()
        self._seat_stable = {seat: _StableValue() for seat in PlayerSeat}
        self._armed = {seat: False for seat in PlayerSeat}
        self._uncertain_reason: str | None = None

    @property
    def state(self) -> ObservableGameState | None:
        if self._tracker is None:
            return None
        state = self._tracker.state
        if self.mode is VisualTrackerMode.UNCERTAIN:
            warning = self._uncertain_reason or "visual event tracker is uncertain"
            return replace(
                state,
                phase=RoundPhase.UNCERTAIN,
                warnings=tuple(dict.fromkeys((*state.warnings, warning))),
            )
        return state

    def update(self, scene: SceneObservation) -> VisualTrackerUpdate:
        initial = _initial_state_payload(
            scene,
            confidence_threshold=self.confidence_threshold,
        )
        if initial is not None:
            stable_count = self._initial_stable.update(initial.fingerprint)
            should_initialize = self._tracker is None or self.mode in {
                VisualTrackerMode.UNCERTAIN,
                VisualTrackerMode.FINISHED,
            }
            if (
                self._tracker is not None
                and self.mode is VisualTrackerMode.TRACKING
                and _is_new_initial_scene(self._tracker.state, initial)
            ):
                should_initialize = True
            if should_initialize and stable_count >= self.initial_stability_frames:
                return self._initialize(scene, initial)
            if should_initialize:
                return VisualTrackerUpdate(
                    mode=VisualTrackerMode.WAITING_FOR_ROUND,
                    message=(
                        "已识别地主、角色和完整初始手牌，正在建立牌局 "
                        f"{stable_count}/{self.initial_stability_frames}"
                    ),
                    state=None,
                    warnings=tuple(
                        dict.fromkeys((*scene.warnings, *initial.warnings))
                    ),
                )
        else:
            self._initial_stable.update(("not_initial",))

        if self._tracker is None or self.mode is VisualTrackerMode.UNCERTAIN:
            auto_payload, auto_error = _current_scan_payload(
                scene,
                confidence_threshold=self.confidence_threshold,
                scan_source="automatic",
            )
            if auto_payload is not None:
                stable_count = self._auto_scan_stable.update(
                    auto_payload.fingerprint
                )
                if stable_count >= self.initial_stability_frames:
                    return self._initialize_current_scan(
                        scene,
                        auto_payload,
                        automatic=True,
                    )
                return VisualTrackerUpdate(
                    mode=self.mode,
                    message=(
                        "已识别到自己回合，正在自动扫描当前牌局 "
                        f"{stable_count}/{self.initial_stability_frames}"
                    ),
                    state=self.state,
                    warnings=tuple(
                        dict.fromkeys(
                            (*scene.warnings, *auto_payload.warnings)
                        )
                    ),
                )
            self._auto_scan_stable.update(("not_scannable", auto_error))

        if self._tracker is None:
            return VisualTrackerUpdate(
                mode=VisualTrackerMode.WAITING_FOR_ROUND,
                message=(
                    "等待地主和加倍完成后的完整初始场面；"
                    "也支持由地主首手安全重建"
                ),
                state=None,
                warnings=scene.warnings,
            )
        if self.mode is VisualTrackerMode.UNCERTAIN:
            return VisualTrackerUpdate(
                mode=self.mode,
                message=(
                    "当前牌局已不确定；为避免伪胜率已暂停，"
                    "将在下次自己回合自动重新扫描"
                ),
                state=self.state,
                warnings=(self._uncertain_reason or "uncertain",),
            )
        if self.mode is VisualTrackerMode.FINISHED:
            finished_state = self._tracker.state
            outcome = (
                "胜利"
                if _self_team_won(finished_state)
                else "本局结束"
            )
            self._clear_round()
            return VisualTrackerUpdate(
                mode=VisualTrackerMode.WAITING_FOR_ROUND,
                message=f"已检测到{outcome}，旧牌局状态已清除，等待下一局",
                state=None,
            )

        state = self._tracker.state
        # A seat can become blank before it is that seat's turn. Remember the
        # edge immediately instead of only watching the currently expected
        # actor. Otherwise two opponents acting between adjacent captures can
        # hide the blank frame for the second actor and make its new cards look
        # like a stale table prompt.
        for seat in PlayerSeat:
            if scene.seat(seat).signal is VisualSignal.NEUTRAL:
                self._armed[seat] = True

        observed_landlord = _observed_landlord(
            scene,
            confidence_threshold=self.confidence_threshold,
        )
        role_fingerprint = (
            ("landlord", observed_landlord.value)
            if observed_landlord is not None
            else ("landlord", "unavailable")
        )
        role_stable_count = self._role_stable.update(role_fingerprint)
        if (
            observed_landlord is not None
            and observed_landlord is not state.landlord
        ):
            reason = (
                "检测到地主位置已变化："
                f"状态={state.landlord.value}，画面={observed_landlord.value}"
            )
            if role_stable_count >= self.stability_frames:
                return self._mark_uncertain(
                    f"{reason}；判定已进入新牌局，等待完整开局重新建模"
                )
            paused_state = replace(
                state,
                phase=RoundPhase.UNCERTAIN,
                warnings=tuple(
                    dict.fromkeys((*state.warnings, reason))
                ),
            )
            return VisualTrackerUpdate(
                mode=self.mode,
                message=(
                    f"{reason}，正在确认 "
                    f"{role_stable_count}/{self.stability_frames}"
                ),
                state=paused_state,
                warnings=(reason,),
            )

        expected = state.current_actor
        observation = scene.seat(expected)
        fingerprint = _seat_fingerprint(
            observation,
            self_hand=scene.self_hand if expected is PlayerSeat.SELF else (),
            self_turn=scene.self_turn if expected is PlayerSeat.SELF else None,
        )
        stable_count = self._seat_stable[expected].update(fingerprint)
        hand_change = (
            _self_hand_change(
                state,
                scene,
                confidence_threshold=self.confidence_threshold,
            )
            if expected is PlayerSeat.SELF
            else None
        )

        # A genuine self-hand difference is a normal event and must get the
        # first opportunity to advance the reducer. The generic drift recovery
        # previously ran first and could replace a known self play plus two
        # fast opponent passes with a fresh approximate round.
        if hand_change is None:
            recovery = self._recover_drifted_state(scene, state)
            if recovery is not None:
                return recovery

        if expected is PlayerSeat.SELF:
            if hand_change is not None:
                required = min(2, self.stability_frames)
                if stable_count < required:
                    if (
                        scene.self_turn is True
                        and not all(
                            scene.seat(seat).signal is VisualSignal.PASS
                            for seat in (PlayerSeat.RIGHT, PlayerSeat.LEFT)
                        )
                    ):
                        recovery = self._recover_drifted_state(scene, state)
                        if recovery is not None:
                            return recovery
                    return VisualTrackerUpdate(
                        mode=self.mode,
                        message=(
                            f"稳定自己的手牌变化 {stable_count}/{required}"
                        ),
                        state=state,
                        warnings=scene.warnings,
                    )
                if hand_change.error is not None:
                    failure_frames = max(
                        self.stability_frames * 2,
                        self.initial_stability_frames + 2,
                    )
                    if stable_count < failure_frames:
                        return _paused_visual_update(
                            state,
                            message=(
                                "自己的手牌识别与跟踪状态不一致，"
                                "正在等待视觉恢复 "
                                f"{stable_count}/{failure_frames}"
                            ),
                            warning=hand_change.error,
                            scene_warnings=scene.warnings,
                        )
                    return self._mark_uncertain(hand_change.error)
                assert hand_change.cards is not None
                opponent_cycle_complete = all(
                    scene.seat(seat).signal is VisualSignal.PASS
                    for seat in (PlayerSeat.RIGHT, PlayerSeat.LEFT)
                )
                if (
                    scene.self_turn is True
                    and observation.signal is not VisualSignal.PLAY
                    and not any(
                        _remaining_confirms_play(
                            state,
                            scene.seat(seat),
                        )
                        for seat in (PlayerSeat.RIGHT, PlayerSeat.LEFT)
                    )
                    and not opponent_cycle_complete
                ):
                    recovery = self._recover_drifted_state(scene, state)
                    if recovery is not None:
                        return recovery
                    return _paused_visual_update(
                        state,
                        message=(
                            "检测到自己的手牌减少，但出牌控件仍显示；"
                            "正在等待场面状态交叉确认"
                        ),
                        warning=(
                            "self hand changed while the screen still reports "
                            "the self turn"
                        ),
                        scene_warnings=scene.warnings,
                    )
                event = ObservedAction(
                    event_id=(
                        f"{state.round_id}:{state.last_sequence_no + 1}:"
                        f"{expected.value}"
                    ),
                    sequence_no=state.last_sequence_no + 1,
                    actor=expected,
                    cards=hand_change.cards,
                    confidence=hand_change.confidence,
                    source="live_hand_diff",
                )
                return self._apply_event(
                    scene,
                    observation,
                    event,
                )

        if _can_infer_pass(
            state,
            scene,
            observation,
            confidence_threshold=self.confidence_threshold,
            stable_count=stable_count,
            stability_frames=self.stability_frames,
        ):
            event = ObservedAction(
                event_id=(
                    f"{state.round_id}:{state.last_sequence_no + 1}:"
                    f"{expected.value}"
                ),
                sequence_no=state.last_sequence_no + 1,
                actor=expected,
                cards=CardSet(()),
                confidence=_inferred_pass_confidence(scene, observation),
                source="live_turn_inferred_pass",
            )
            return self._apply_event(scene, observation, event)

        if (
            expected is PlayerSeat.SELF
            and observation.signal is VisualSignal.PASS
        ):
            return VisualTrackerUpdate(
                mode=self.mode,
                message=(
                    "检测到自己的“不出”文字或按钮，"
                    "等待回合控件稳定消失后再确认过牌"
                ),
                state=state,
                warnings=scene.warnings,
            )

        corroborated_confidence = _verified_low_confidence_play(
            state,
            observation,
            confidence_threshold=self.confidence_threshold,
        )
        if corroborated_confidence is not None:
            if stable_count < self.stability_frames:
                return VisualTrackerUpdate(
                    mode=self.mode,
                    message=(
                        f"用余牌变化交叉确认 {expected.value} 动作 "
                        f"{stable_count}/{self.stability_frames}"
                    ),
                    state=state,
                    warnings=scene.warnings,
                )
            try:
                cards = observation.card_set
            except ValueError as exc:
                return self._mark_uncertain(
                    f"{expected.value} low-confidence cards are invalid: {exc}"
                )
            event = ObservedAction(
                event_id=(
                    f"{state.round_id}:{state.last_sequence_no + 1}:"
                    f"{expected.value}"
                ),
                sequence_no=state.last_sequence_no + 1,
                actor=expected,
                cards=cards,
                confidence=corroborated_confidence,
                source="live_visual_remaining_correlated",
            )
            return self._apply_event(scene, observation, event)

        if observation.signal is VisualSignal.NEUTRAL:
            self._armed[expected] = True
            return VisualTrackerUpdate(
                mode=self.mode,
                message=f"等待 {expected.value} 出牌或过牌",
                state=state,
                warnings=scene.warnings,
            )
        if observation.signal is VisualSignal.UNKNOWN:
            if stable_count >= self.stability_frames:
                if (
                    observation.remaining_verified
                    and observation.remaining_count
                    == state.remaining_for(expected)
                ):
                    return VisualTrackerUpdate(
                        mode=self.mode,
                        message=(
                            f"{expected.value} 视觉区域暂不清晰，"
                            "但余牌未减少，继续等待有效动作"
                        ),
                        state=state,
                        warnings=scene.warnings,
                    )
                return self._mark_uncertain(
                    f"{expected.value} action remained low-confidence for "
                    f"{stable_count} frames"
                )
            return VisualTrackerUpdate(
                mode=self.mode,
                message=f"等待 {expected.value} 视觉结果稳定",
                state=state,
                warnings=scene.warnings,
            )
        if _remaining_disproves_play(state, observation):
            return VisualTrackerUpdate(
                mode=self.mode,
                message=(
                    f"忽略 {expected.value} 的疑似牌影："
                    "已确认余牌数没有减少"
                ),
                state=state,
                warnings=scene.warnings,
            )
        if (
            not self._armed[expected]
            and not _remaining_confirms_play(state, observation)
        ):
            return VisualTrackerUpdate(
                mode=self.mode,
                message=f"忽略 {expected.value} 的旧场面提示，等待空白到动作的新变化",
                state=state,
            )
        if stable_count < self.stability_frames:
            return VisualTrackerUpdate(
                mode=self.mode,
                message=(
                    f"稳定 {expected.value} 动作 "
                    f"{stable_count}/{self.stability_frames}"
                ),
                state=state,
            )

        cards = observation.card_set
        confidence = _action_confidence(observation)
        if confidence < self.confidence_threshold:
            return self._mark_uncertain(
                f"{expected.value} action confidence {confidence:.3f} is below "
                f"{self.confidence_threshold:.3f}"
            )
        event = ObservedAction(
            event_id=f"{state.round_id}:{state.last_sequence_no + 1}:{expected.value}",
            sequence_no=state.last_sequence_no + 1,
            actor=expected,
            cards=cards,
            confidence=confidence,
            source="live_visual",
        )
        return self._apply_event(scene, observation, event)

    def scan_current_scene(
        self,
        scene: SceneObservation,
    ) -> VisualTrackerUpdate:
        """Build a safe approximate state from a user-requested current scan."""

        payload, error = _current_scan_payload(
            scene,
            confidence_threshold=self.confidence_threshold,
            scan_source="manual",
        )
        if payload is None:
            reason = f"扫描当前牌局失败：{error}"
            return VisualTrackerUpdate(
                mode=self.mode,
                message=reason,
                state=self.state,
                warnings=(reason,),
            )
        return self._initialize_current_scan(
            scene,
            payload,
            automatic=False,
        )

    def _initialize_current_scan(
        self,
        scene: SceneObservation,
        payload: "_CurrentScanPayload",
        *,
        automatic: bool,
    ) -> VisualTrackerUpdate:
        prefix = "auto-scan" if automatic else "scan"
        round_id = f"{prefix}-{int(scene.timestamp * 1000)}"
        try:
            state = ObservableGameState.from_inputs(
                payload.hand,
                round_id=round_id,
                landlord=payload.landlord,
                current_actor=PlayerSeat.SELF,
                remaining_cards=payload.remaining,
                played_cards=payload.trick_target,
                hidden_played_count=payload.hidden_played_count,
                last_play=payload.trick_target,
                last_player=payload.trick_leader,
                consecutive_passes=payload.consecutive_passes,
                state_confidence=payload.confidence,
            )
        except ValueError as exc:
            reason = f"扫描当前牌局失败：{exc}"
            return VisualTrackerUpdate(
                mode=self.mode,
                message=reason,
                state=self.state,
                warnings=(reason,),
            )
        state = replace(
            state,
            warnings=tuple(
                dict.fromkeys((*state.warnings, *payload.warnings))
            ),
        )
        self._tracker = GameStateTracker(
            state,
            validator=validate_observed_action,
            confidence_threshold=self.confidence_threshold,
        )
        self.mode = VisualTrackerMode.TRACKING
        self._uncertain_reason = None
        self._initial_stable = _StableValue()
        self._auto_scan_stable = _StableValue()
        self._active_scan_stable = _StableValue()
        self._role_stable = _StableValue()
        self._seat_stable = {seat: _StableValue() for seat in PlayerSeat}
        self._armed = {
            seat: scene.seat(seat).signal is VisualSignal.NEUTRAL
            for seat in PlayerSeat
        }
        trick = (
            " ".join(payload.trick_target)
            if payload.trick_target
            else "自由出牌"
        )
        scope = (
            "中途近似模型"
            if payload.hidden_played_count
            else "完整当前场面"
        )
        action = "已自动扫描" if automatic else "已扫描"
        return VisualTrackerUpdate(
            mode=self.mode,
            message=f"{action}当前牌局（{scope}），待压：{trick}",
            state=state,
            initialized=True,
            warnings=payload.warnings,
        )

    def _recover_drifted_state(
        self,
        scene: SceneObservation,
        state: ObservableGameState,
    ) -> VisualTrackerUpdate | None:
        """Rebuild a stale active model after the table proves a stable drift.

        Normal event inference gets the first opportunity to catch up.  A
        rebuild only happens on a stable self-turn scene whose hand is a
        physical subset of the tracked hand, so a transient rank
        misclassification cannot silently replace the model.
        """

        payload, _ = _current_scan_payload(
            scene,
            confidence_threshold=self.confidence_threshold,
            scan_source="automatic_resync",
        )
        if payload is None or _current_scan_matches_state(payload, state):
            self._active_scan_stable = _StableValue()
            return None
        if not _current_scan_is_monotonic(payload, state):
            self._active_scan_stable = _StableValue()
            return None
        if not _current_scan_has_material_progress(payload, state):
            # Old table cards and pass labels can remain visible after the
            # reducer has already completed a trick.  A differing target alone
            # is not evidence that state is stale; require an actual hand/count
            # decrease before replacing a healthy active model.
            self._active_scan_stable = _StableValue()
            return None

        stable_count = self._active_scan_stable.update(payload.fingerprint)
        recovery_frames = max(
            self.stability_frames * 2,
            self.initial_stability_frames + 1,
        )
        if stable_count >= recovery_frames:
            update = self._initialize_current_scan(
                scene,
                payload,
                automatic=True,
            )
            if not update.initialized:
                return update
            return replace(
                update,
                message=(
                    "检测到桌面与跟踪状态持续不一致，"
                    f"已自动重建当前牌局；{update.message}"
                ),
                warnings=tuple(
                    dict.fromkeys(
                        (*update.warnings, "active_state_drift_recovered")
                    )
                ),
            )

        # While the expected actor is not self, normal play/pass inference may
        # still close the gap.  Once both the model and the screen say it is our
        # turn, pause recommendations until the drift is resolved.
        if (
            state.current_actor is PlayerSeat.SELF
            and stable_count
            >= max(self.initial_stability_frames, self.stability_frames)
        ):
            warning = "active table state differs from the tracked model"
            return _paused_visual_update(
                state,
                message=(
                    "桌面与跟踪状态不一致，正在自动校准 "
                    f"{stable_count}/{recovery_frames}"
                ),
                warning=warning,
                scene_warnings=scene.warnings,
            )
        return None

    def _apply_event(
        self,
        scene: SceneObservation,
        observation: SeatObservation,
        event: ObservedAction,
    ) -> VisualTrackerUpdate:
        assert self._tracker is not None
        state = self._tracker.state
        expected = event.actor
        count_error = _remaining_count_error(state, observation, event)
        if count_error is not None:
            return self._mark_uncertain(count_error, event=event)

        result = self._tracker.apply(event)
        if result.status is not StateUpdateStatus.APPLIED:
            return self._mark_uncertain(
                f"state rejected visual event: {result.message}",
                event=event,
            )

        self._armed[expected] = False
        self._active_scan_stable = _StableValue()
        next_actor = result.state.current_actor
        if scene.seat(next_actor).signal is VisualSignal.NEUTRAL:
            self._armed[next_actor] = True
        if result.state.phase is RoundPhase.FINISHED:
            self.mode = VisualTrackerMode.FINISHED
            outcome = (
                "胜利"
                if _self_team_won(result.state)
                else "本局结束"
            )
            message = (
                f"{expected.value} 出完最后一手："
                f"{' '.join(event.cards.cards)}；检测到{outcome}"
            )
        else:
            message = (
                f"{expected.value} 过牌"
                if event.is_pass
                else f"{expected.value} 出牌：{' '.join(event.cards.cards)}"
            )
        return VisualTrackerUpdate(
            mode=self.mode,
            message=message,
            state=self.state,
            event=event,
            warnings=result.warnings,
        )

    def handle_window_unavailable(self, reason: str) -> VisualTrackerUpdate:
        if self._tracker is not None and self.mode is VisualTrackerMode.TRACKING:
            return self._mark_uncertain(
                f"{reason}；可能漏过场上事件，等待下一局完整初始化"
            )
        return VisualTrackerUpdate(
            mode=self.mode,
            message=reason,
            state=self.state,
            warnings=(reason,),
        )

    def _clear_round(self) -> None:
        self._tracker = None
        self.mode = VisualTrackerMode.WAITING_FOR_ROUND
        self._uncertain_reason = None
        self._initial_stable = _StableValue()
        self._auto_scan_stable = _StableValue()
        self._active_scan_stable = _StableValue()
        self._role_stable = _StableValue()
        self._seat_stable = {seat: _StableValue() for seat in PlayerSeat}
        self._armed = {seat: False for seat in PlayerSeat}

    def _initialize(
        self,
        scene: SceneObservation,
        initial: "_InitialStatePayload",
    ) -> VisualTrackerUpdate:
        round_id = self.round_id_factory(scene)
        try:
            state = ObservableGameState.from_inputs(
                initial.hand,
                round_id=round_id,
                landlord=initial.landlord,
                current_actor=initial.landlord,
                remaining_cards=initial.remaining,
                state_confidence=initial.confidence,
            )
        except ValueError as exc:
            return self._mark_uncertain(f"cannot initialize visual round: {exc}")
        if initial.warnings:
            state = replace(
                state,
                warnings=tuple(
                    dict.fromkeys((*state.warnings, *initial.warnings))
                ),
            )
        tracker = GameStateTracker(
            state,
            validator=validate_observed_action,
            confidence_threshold=self.confidence_threshold,
        )
        opening_event: ObservedAction | None = None
        if initial.opening_cards:
            opening_event = ObservedAction(
                event_id=f"{round_id}:1:{initial.landlord.value}",
                sequence_no=1,
                actor=initial.landlord,
                cards=initial.opening_card_set,
                confidence=initial.opening_confidence,
                source="live_visual_bootstrap",
            )
            result = tracker.apply(opening_event)
            if result.status is not StateUpdateStatus.APPLIED:
                return self._mark_uncertain(
                    f"cannot reconstruct landlord opening play: {result.message}",
                    event=opening_event,
                )
            state = result.state
        self._tracker = tracker
        self.mode = VisualTrackerMode.TRACKING
        self._uncertain_reason = None
        self._auto_scan_stable = _StableValue()
        self._active_scan_stable = _StableValue()
        self._role_stable = _StableValue()
        self._seat_stable = {seat: _StableValue() for seat in PlayerSeat}
        self._armed = {
            seat: scene.seat(seat).signal is VisualSignal.NEUTRAL
            for seat in PlayerSeat
        }
        return VisualTrackerUpdate(
            mode=self.mode,
            message=_initialization_message(initial, state),
            state=state,
            event=opening_event,
            initialized=True,
            warnings=tuple(
                dict.fromkeys((*scene.warnings, *initial.warnings))
            ),
        )

    def _mark_uncertain(
        self,
        reason: str,
        *,
        event: ObservedAction | None = None,
    ) -> VisualTrackerUpdate:
        self.mode = VisualTrackerMode.UNCERTAIN
        self._uncertain_reason = reason
        return VisualTrackerUpdate(
            mode=self.mode,
            message=reason,
            state=self.state,
            event=event,
            warnings=(reason,),
        )


@dataclass(frozen=True)
class _InitialStatePayload:
    landlord: PlayerSeat
    hand: tuple[str, ...]
    remaining: dict[PlayerSeat, int]
    confidence: float
    opening_cards: tuple[str, ...] = ()
    opening_confidence: float = 1.0
    warnings: tuple[str, ...] = ()

    @property
    def opening_card_set(self) -> CardSet:
        return CardSet.parse(self.opening_cards)

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.landlord.value,
            self.hand,
            tuple((seat.value, self.remaining[seat]) for seat in PlayerSeat),
            self.opening_cards,
        )


@dataclass(frozen=True)
class _CurrentScanPayload:
    landlord: PlayerSeat
    hand: tuple[str, ...]
    remaining: dict[PlayerSeat, int]
    trick_target: tuple[str, ...]
    trick_leader: PlayerSeat | None
    consecutive_passes: int
    hidden_played_count: int
    confidence: float
    warnings: tuple[str, ...]

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.landlord.value,
            self.hand,
            tuple((seat.value, self.remaining[seat]) for seat in PlayerSeat),
            self.trick_target,
            self.trick_leader.value if self.trick_leader is not None else None,
            self.consecutive_passes,
        )


@dataclass(frozen=True)
class _SelfHandChange:
    cards: CardSet | None
    confidence: float
    error: str | None = None


def _self_team_won(state: ObservableGameState) -> bool:
    winner = state.winner
    if winner is None:
        return False
    if state.landlord is PlayerSeat.SELF:
        return winner is PlayerSeat.SELF
    return winner is not state.landlord


def _paused_visual_update(
    state: ObservableGameState,
    *,
    message: str,
    warning: str,
    scene_warnings: Sequence[str] = (),
) -> VisualTrackerUpdate:
    warnings = tuple(
        dict.fromkeys((*state.warnings, *scene_warnings, warning))
    )
    return VisualTrackerUpdate(
        mode=VisualTrackerMode.TRACKING,
        message=message,
        state=replace(
            state,
            phase=RoundPhase.UNCERTAIN,
            warnings=warnings,
        ),
        warnings=tuple(dict.fromkeys((*scene_warnings, warning))),
    )


def _current_scan_matches_state(
    payload: _CurrentScanPayload,
    state: ObservableGameState,
) -> bool:
    try:
        hand = CardSet.parse(payload.hand)
        trick_target = CardSet.parse(payload.trick_target)
    except ValueError:
        return False
    return bool(
        payload.landlord is state.landlord
        and state.current_actor is PlayerSeat.SELF
        and hand == state.self_hand
        and trick_target == state.trick_target
        and payload.trick_leader is state.trick_leader
        and payload.consecutive_passes == state.consecutive_passes
        and all(
            payload.remaining[seat] == state.remaining_for(seat)
            for seat in PlayerSeat
        )
    )


def _current_scan_is_monotonic(
    payload: _CurrentScanPayload,
    state: ObservableGameState,
) -> bool:
    observed = Counter(payload.hand)
    tracked = Counter(state.self_hand.cards)
    return bool(
        not (observed - tracked)
        and all(
            payload.remaining[seat] <= state.remaining_for(seat)
            for seat in PlayerSeat
        )
    )


def _current_scan_has_material_progress(
    payload: _CurrentScanPayload,
    state: ObservableGameState,
) -> bool:
    return bool(
        len(payload.hand) < len(state.self_hand)
        or any(
            payload.remaining[seat] < state.remaining_for(seat)
            for seat in PlayerSeat
        )
    )


def _is_new_initial_scene(
    state: ObservableGameState,
    initial: _InitialStatePayload,
) -> bool:
    """Tell a resumed/active round apart from a newly dealt full hand."""

    # Before the first action, clicking cards raises them inside the hand ROI.
    # The changed crop geometry can remain stable for several frames and still
    # contain 20 physically valid ranks.  It is not a new deal.  A pristine
    # initial model therefore stays authoritative until an action advances it,
    # the landlord changes (handled by the role-boundary guard), or the tracker
    # becomes uncertain/finished.  This prevents selected-card animation from
    # replacing the correct opening hand and corrupting the first hand diff.
    if _is_pristine_initial_state(state):
        return False
    if state.landlord is not initial.landlord:
        return True
    if state.self_hand != CardSet.parse(initial.hand):
        return True
    expected_remaining = dict(initial.remaining)
    expected_revision = 0
    expected_actor = initial.landlord
    expected_trick = CardSet(())
    expected_played: tuple[str, ...] = ()
    if initial.opening_cards:
        expected_revision = 1
        expected_remaining[initial.landlord] -= len(initial.opening_cards)
        expected_actor = state.next_player(initial.landlord)
        expected_trick = initial.opening_card_set
        expected_played = initial.opening_card_set.cards
    return not (
        state.revision == expected_revision
        and state.current_actor is expected_actor
        and state.trick_target == expected_trick
        and state.played_cards == expected_played
        and all(
            state.remaining_for(seat) == expected_remaining[seat]
            for seat in PlayerSeat
        )
    )


def _is_pristine_initial_state(state: ObservableGameState) -> bool:
    expected_remaining = {
        seat: 20 if seat is state.landlord else 17
        for seat in PlayerSeat
    }
    return bool(
        state.revision == 0
        and state.last_sequence_no == 0
        and state.current_actor is state.landlord
        and not state.trick_target
        and state.trick_leader is None
        and not state.played_cards
        and state.hidden_played_count == 0
        and state.consecutive_passes == 0
        and all(
            state.remaining_for(seat) == expected_remaining[seat]
            for seat in PlayerSeat
        )
    )


def _initial_state_payload(
    scene: SceneObservation,
    *,
    confidence_threshold: float,
) -> _InitialStatePayload | None:
    landlords = [
        observation.seat
        for observation in scene.seats
        if observation.role is SeatRole.LANDLORD
    ]
    farmers = [
        observation.seat
        for observation in scene.seats
        if observation.role is SeatRole.FARMER
    ]
    if len(landlords) != 1 or len(farmers) != 2:
        return None
    landlord = landlords[0]
    expected = {
        seat: 20 if seat is landlord else 17
        for seat in PlayerSeat
    }
    hand = tuple(card.rank for card in scene.self_hand)
    if len(hand) != expected[PlayerSeat.SELF]:
        return None
    try:
        hand_set = scene.self_hand_set
    except ValueError:
        return None
    if len(hand_set) != len(hand):
        return None
    hand_confidence, hand_warnings = _stable_hand_confidence(
        scene.self_hand,
        confidence_threshold=confidence_threshold,
    )
    if hand_confidence < confidence_threshold:
        return None

    if any(
        observation.signal is not VisualSignal.NEUTRAL
        for observation in scene.seats
    ):
        return _opening_state_payload(
            scene,
            landlord=landlord,
            hand=hand,
            expected=expected,
            hand_confidence=hand_confidence,
            hand_warnings=hand_warnings,
            confidence_threshold=confidence_threshold,
        )

    remaining: dict[PlayerSeat, int] = {}
    inferred_warnings: list[str] = []
    for observation in scene.seats:
        expected_count = expected[observation.seat]
        if observation.remaining_count is None:
            if observation.seat is not landlord:
                return None
            remaining[observation.seat] = expected_count
            inferred_warnings.append(
                f"inferred_initial_{observation.seat.value}_remaining={expected_count}"
            )
            continue
        if observation.remaining_count != expected_count:
            if observation.seat is landlord and not observation.remaining_verified:
                remaining[observation.seat] = expected_count
                inferred_warnings.append(
                    f"ignored_unverified_initial_{observation.seat.value}_remaining="
                    f"{observation.remaining_count}"
                )
                continue
            return None
        remaining[observation.seat] = observation.remaining_count

    confidence = _initial_observation_confidence(
        scene,
        hand_confidence=hand_confidence,
        include_action=None,
    )
    if confidence < confidence_threshold:
        return None
    return _InitialStatePayload(
        landlord=landlord,
        hand=hand,
        remaining=remaining,
        confidence=confidence,
        warnings=tuple((*hand_warnings, *inferred_warnings)),
    )


def _observed_landlord(
    scene: SceneObservation,
    *,
    confidence_threshold: float,
) -> PlayerSeat | None:
    if len(scene.seats) != len(PlayerSeat):
        return None
    if any(
        observation.role_confidence < confidence_threshold
        or observation.role is SeatRole.UNKNOWN
        for observation in scene.seats
    ):
        return None
    landlords = [
        observation.seat
        for observation in scene.seats
        if observation.role is SeatRole.LANDLORD
    ]
    farmers = [
        observation.seat
        for observation in scene.seats
        if observation.role is SeatRole.FARMER
    ]
    if len(landlords) != 1 or len(farmers) != 2:
        return None
    return landlords[0]


def _current_scan_payload(
    scene: SceneObservation,
    *,
    confidence_threshold: float,
    scan_source: str = "manual",
) -> tuple[_CurrentScanPayload | None, str]:
    if (
        scene.self_turn is not True
        or scene.self_turn_confidence < confidence_threshold
    ):
        return None, "请在轮到自己、出牌按钮已经显示时再扫描"

    landlord = _observed_landlord(
        scene,
        confidence_threshold=confidence_threshold,
    )
    if landlord is None:
        return None, "未稳定识别到唯一地主和两名农民"

    hand = tuple(card.rank for card in scene.self_hand)
    maximum_hand = 20 if landlord is PlayerSeat.SELF else 17
    if not hand or len(hand) > maximum_hand:
        return None, (
            f"自己的手牌张数无效：识别={len(hand)}，"
            f"角色上限={maximum_hand}"
        )
    try:
        hand_set = scene.self_hand_set
    except ValueError as exc:
        return None, f"自己的手牌不符合物理牌组：{exc}"
    if len(hand_set) != len(hand):
        return None, "自己的手牌存在重复识别"
    hand_confidence, hand_warnings = _stable_hand_confidence(
        scene.self_hand,
        confidence_threshold=confidence_threshold,
    )
    if hand_confidence < confidence_threshold:
        return None, (
            "自己的手牌置信度不足："
            f"{hand_confidence:.3f} < {confidence_threshold:.3f}"
        )

    remaining = {PlayerSeat.SELF: len(hand)}
    confidence_values = [
        hand_confidence,
        scene.self_turn_confidence,
        *(
            observation.role_confidence
            for observation in scene.seats
        ),
    ]
    for seat in (PlayerSeat.RIGHT, PlayerSeat.LEFT):
        observation = scene.seat(seat)
        if (
            observation.remaining_count is None
            or not observation.remaining_verified
            or observation.remaining_confidence < confidence_threshold
        ):
            return None, f"{seat.value} 余牌数尚未稳定确认"
        maximum = 20 if seat is landlord else 17
        if not 0 < observation.remaining_count <= maximum:
            return None, (
                f"{seat.value} 余牌数无效："
                f"{observation.remaining_count}"
            )
        remaining[seat] = observation.remaining_count
        confidence_values.append(observation.remaining_confidence)

    played_total = len(FULL_DECK) - sum(remaining.values())
    if played_total < 0:
        return None, "三家余牌合计超过 54 张"

    left = scene.seat(PlayerSeat.LEFT)
    right = scene.seat(PlayerSeat.RIGHT)
    trick_target: tuple[str, ...] = ()
    trick_leader: PlayerSeat | None = None
    consecutive_passes = 0
    inference_warnings: list[str] = []
    left_play_confidence = _current_scan_play_confidence(
        left,
        confidence_threshold=confidence_threshold,
    )
    right_play_confidence = _current_scan_play_confidence(
        right,
        confidence_threshold=confidence_threshold,
    )
    if left_play_confidence is not None:
        trick_target = tuple(card.rank for card in left.cards)
        trick_leader = PlayerSeat.LEFT
        confidence_values.append(left_play_confidence)
        if left.signal is VisualSignal.UNKNOWN:
            inference_warnings.append(
                "current_scan_accepted_left_play_from_verified_remaining"
            )
    elif right_play_confidence is not None:
        trick_target = tuple(card.rank for card in right.cards)
        trick_leader = PlayerSeat.RIGHT
        consecutive_passes = 1
        confidence_values.append(right_play_confidence)
        if right.signal is VisualSignal.UNKNOWN:
            inference_warnings.append(
                "current_scan_accepted_right_play_from_verified_remaining"
            )
        inference_warnings.append(
            "manual_scan_inferred_left_pass_after_right_play"
        )
    else:
        inference_warnings.append(
            "manual_scan_assumed_free_lead_no_visible_opponent_play"
        )

    if trick_target:
        try:
            target_play = Play.parse(trick_target)
            CardSet.parse((*hand, *trick_target))
        except ValueError as exc:
            return None, f"当前待压牌不符合物理牌组：{exc}"
        if target_play.type is PlayType.INVALID:
            return None, "当前待压牌型无法确认"
        action_confidence = min(confidence_values[-1], 1.0)
        if action_confidence < confidence_threshold:
            return None, "当前待压牌置信度不足"

    hidden_played_count = played_total - len(trick_target)
    if hidden_played_count < 0:
        return None, (
            "场上牌张数超过由余牌变化推导的已出牌总数"
        )

    confidence = min(confidence_values)
    if hidden_played_count:
        confidence *= 0.95
    if confidence < confidence_threshold:
        return None, (
            "当前场面综合置信度不足："
            f"{confidence:.3f} < {confidence_threshold:.3f}"
        )
    warnings = [
        *hand_warnings,
        *inference_warnings,
        f"{scan_source}_current_game_scan",
    ]
    if hidden_played_count:
        warnings.extend((
            f"historical_played_cards_unknown={hidden_played_count}",
            "estimated_win_rate_uses_uniform_unknown_history",
        ))
    return _CurrentScanPayload(
        landlord=landlord,
        hand=hand,
        remaining=remaining,
        trick_target=trick_target,
        trick_leader=trick_leader,
        consecutive_passes=consecutive_passes,
        hidden_played_count=hidden_played_count,
        confidence=confidence,
        warnings=tuple(dict.fromkeys(warnings)),
    ), ""


def _opening_state_payload(
    scene: SceneObservation,
    *,
    landlord: PlayerSeat,
    hand: tuple[str, ...],
    expected: dict[PlayerSeat, int],
    hand_confidence: float,
    hand_warnings: tuple[str, ...],
    confidence_threshold: float,
) -> _InitialStatePayload | None:
    if landlord is PlayerSeat.SELF:
        # A self-landlord hand has already changed after its first play, so the
        # missing cards cannot be reconstructed from the screen alone.
        return None
    active = [
        observation
        for observation in scene.seats
        if observation.signal is not VisualSignal.NEUTRAL
    ]
    if (
        len(active) != 1
        or active[0].seat is not landlord
        or active[0].signal is not VisualSignal.PLAY
        or not active[0].cards
    ):
        return None
    opening = active[0]
    opening_confidence = _action_confidence(opening)
    if opening_confidence < confidence_threshold:
        return None
    opening_count = len(opening.cards)
    derived_landlord_remaining = expected[landlord] - opening_count
    if derived_landlord_remaining <= 0:
        return None

    if Play.parse(opening.card_set.cards).type is PlayType.INVALID:
        return None

    inferred_warnings: list[str] = []
    for observation in scene.seats:
        if observation.seat is PlayerSeat.SELF:
            continue
        if observation.seat is landlord:
            if (
                observation.remaining_verified
                and observation.remaining_count is not None
                and observation.remaining_count != derived_landlord_remaining
            ):
                return None
            if observation.remaining_count != derived_landlord_remaining:
                inferred_warnings.append(
                    f"inferred_opening_{landlord.value}_remaining="
                    f"{derived_landlord_remaining}"
                )
            continue
        if observation.remaining_count != expected[observation.seat]:
            return None

    next_actor = {
        PlayerSeat.SELF: PlayerSeat.RIGHT,
        PlayerSeat.RIGHT: PlayerSeat.LEFT,
        PlayerSeat.LEFT: PlayerSeat.SELF,
    }[landlord]
    if next_actor is PlayerSeat.SELF and scene.self_turn is False:
        return None
    if next_actor is not PlayerSeat.SELF and scene.self_turn is True:
        return None

    confidence = _initial_observation_confidence(
        scene,
        hand_confidence=hand_confidence,
        include_action=opening,
    )
    if confidence < confidence_threshold:
        return None
    return _InitialStatePayload(
        landlord=landlord,
        hand=hand,
        remaining=expected,
        confidence=confidence,
        opening_cards=tuple(card.rank for card in opening.cards),
        opening_confidence=opening_confidence,
        warnings=tuple((*hand_warnings, *inferred_warnings)),
    )


def _stable_hand_confidence(
    cards: Sequence[VisualCard],
    *,
    confidence_threshold: float,
) -> tuple[float, tuple[str, ...]]:
    values = sorted(card.confidence for card in cards)
    if not values:
        return 0.0, ()
    below = [card for card in cards if card.confidence < confidence_threshold]
    if not below:
        return values[0], ()
    outlier_floor = max(0.55, confidence_threshold - 0.15)
    if (
        len(below) == 1
        and below[0].confidence >= outlier_floor
        and len(values) >= 2
    ):
        warning = (
            f"accepted_single_hand_confidence_outlier:"
            f"{below[0].rank}={below[0].confidence:.3f}"
        )
        return values[1], (warning,)
    return values[0], ()


def _initial_observation_confidence(
    scene: SceneObservation,
    *,
    hand_confidence: float,
    include_action: SeatObservation | None,
) -> float:
    confidences = [
        hand_confidence,
        *(seat.role_confidence for seat in scene.seats),
    ]
    for seat in scene.seats:
        if seat.seat is PlayerSeat.SELF:
            continue
        if seat.remaining_count is not None:
            confidences.append(seat.remaining_confidence)
    if include_action is not None:
        confidences.append(_action_confidence(include_action))
    if scene.self_turn is not None:
        confidences.append(scene.self_turn_confidence)
    positive = [value for value in confidences if value > 0]
    return min(positive) if positive else 0.0


def _initialization_message(
    initial: _InitialStatePayload,
    state: ObservableGameState,
) -> str:
    if initial.opening_cards:
        return (
            f"已由地主首手安全重建新局：地主={initial.landlord.value}，"
            f"首手={' '.join(initial.opening_cards)}，"
            f"当前行动者={state.current_actor.value}"
        )
    return (
        f"新局已初始化：地主={initial.landlord.value}，"
        f"当前行动者={state.current_actor.value}"
    )


def _seat_fingerprint(
    observation: SeatObservation,
    *,
    self_hand: Sequence[VisualCard] = (),
    self_turn: bool | None = None,
) -> tuple[object, ...]:
    return (
        observation.signal.value,
        tuple(card.rank for card in observation.cards),
        tuple(card.rank for card in self_hand),
        self_turn,
    )


def _self_hand_change(
    state: ObservableGameState,
    scene: SceneObservation,
    *,
    confidence_threshold: float,
) -> _SelfHandChange | None:
    if not scene.self_hand:
        # Once the final legal combination is played the hand ROI legitimately
        # becomes empty, so there is no remaining card confidence to
        # aggregate. This applies to pairs, triples and sequences as well as a
        # final singleton. Require a confidently inactive turn control and a
        # valid whole-hand play; an unexplained empty crop with an invalid
        # remaining combination stays a visual failure instead of ending the
        # round.
        final_play = Play.parse(state.self_hand.cards)
        if (
            scene.self_turn is False
            and scene.self_turn_confidence >= confidence_threshold
            and final_play.type is not PlayType.INVALID
        ):
            return _SelfHandChange(
                cards=state.self_hand,
                confidence=max(confidence_threshold, state.state_confidence),
            )
        return None
    if len(scene.self_hand) >= len(state.self_hand):
        return None
    observed_cards = tuple(card.rank for card in scene.self_hand)
    observed = Counter(observed_cards)
    previous = Counter(state.self_hand.cards)
    unexpected = observed - previous
    confidence, _ = _stable_hand_confidence(
        scene.self_hand,
        confidence_threshold=confidence_threshold,
    )
    if unexpected:
        return _SelfHandChange(
            cards=None,
            confidence=confidence,
            error=(
                "self hand change contains ranks outside the tracked hand: "
                + " ".join(sorted(unexpected.elements()))
            ),
        )
    removed = previous - observed
    cards = CardSet.parse(
        card
        for rank, count in removed.items()
        for card in [rank] * count
    )
    if not cards:
        return None
    if confidence < confidence_threshold:
        return _SelfHandChange(
            cards=None,
            confidence=confidence,
            error=(
                f"self hand change confidence {confidence:.3f} is below "
                f"{confidence_threshold:.3f}"
            ),
        )
    return _SelfHandChange(cards=cards, confidence=confidence)


def _remaining_confirms_play(
    state: ObservableGameState,
    observation: SeatObservation,
) -> bool:
    return bool(
        observation.signal is VisualSignal.PLAY
        and observation.remaining_verified
        and observation.remaining_count
        == state.remaining_for(observation.seat) - len(observation.cards)
    )


def _verified_low_confidence_play(
    state: ObservableGameState,
    observation: SeatObservation,
    *,
    confidence_threshold: float,
) -> float | None:
    """Calibrate a near-threshold rank with an exact verified count drop."""

    if (
        observation.signal is not VisualSignal.UNKNOWN
        or not observation.cards
        or not observation.remaining_verified
        or observation.remaining_count is None
        or observation.remaining_confidence < confidence_threshold
        or observation.remaining_count
        != state.remaining_for(observation.seat) - len(observation.cards)
    ):
        return None
    card_confidence = _action_confidence(observation)
    if card_confidence < max(0.55, confidence_threshold - 0.05):
        return None
    calibrated_floor = min(1.0, confidence_threshold + 0.05)
    return min(
        observation.remaining_confidence,
        max(card_confidence, calibrated_floor),
    )


def _current_scan_play_confidence(
    observation: SeatObservation,
    *,
    confidence_threshold: float,
) -> float | None:
    if not observation.cards:
        return None
    if observation.signal is VisualSignal.PLAY:
        return _action_confidence(observation)
    if (
        observation.signal is not VisualSignal.UNKNOWN
        or not observation.remaining_verified
        or observation.remaining_confidence < confidence_threshold
    ):
        return None
    card_confidence = _action_confidence(observation)
    if card_confidence < max(0.55, confidence_threshold - 0.05):
        return None
    calibrated_floor = min(1.0, confidence_threshold + 0.05)
    return min(
        observation.remaining_confidence,
        max(card_confidence, calibrated_floor),
    )


def _remaining_disproves_play(
    state: ObservableGameState,
    observation: SeatObservation,
) -> bool:
    return bool(
        observation.seat is not PlayerSeat.SELF
        and observation.signal is VisualSignal.PLAY
        and observation.remaining_verified
        and observation.remaining_count == state.remaining_for(observation.seat)
    )


def _can_infer_pass(
    state: ObservableGameState,
    scene: SceneObservation,
    observation: SeatObservation,
    *,
    confidence_threshold: float,
    stable_count: int,
    stability_frames: int,
) -> bool:
    # A visible card group always gets the first opportunity to become a play.
    # The self-turn indicator describes the end of the whole opponent cycle;
    # using it first can incorrectly turn a still-visible opponent play into a
    # pass while the animated remaining counter has not updated yet.
    if (
        observation.cards
        and observation.signal in {VisualSignal.PLAY, VisualSignal.UNKNOWN}
    ):
        return False
    if (
        not state.trick_target
        or not observation.remaining_verified
        or observation.remaining_count != state.remaining_for(observation.seat)
        or observation.remaining_confidence < confidence_threshold
    ):
        return False
    if observation.seat is PlayerSeat.SELF:
        try:
            if scene.self_hand_set != state.self_hand:
                return False
        except ValueError:
            return False
        # The local client does not always keep a visible "不出" label for
        # our own seat. A stable disappearance of the turn controls while the
        # complete hand remains unchanged proves a pass. self_turn participates
        # in the seat fingerprint, so a single missed-control frame cannot use
        # stability accumulated before the UI transition.
        if (
            scene.self_turn is False
            and scene.self_turn_confidence >= confidence_threshold
            and stable_count >= stability_frames
        ):
            return True
    elif scene.self_turn is True:
        return scene.self_turn_confidence >= confidence_threshold

    # A second consecutive pass clears the trick and returns the lead to its
    # last player; otherwise normal turn order continues.
    successor = (
        state.trick_leader
        if state.consecutive_passes == 1
        else state.next_player(observation.seat)
    )
    if successor is None:
        return False
    if successor is PlayerSeat.SELF:
        return bool(
            scene.self_turn is True
            and scene.self_turn_confidence >= confidence_threshold
        )
    successor_observation = scene.seat(successor)
    if _remaining_confirms_play(state, successor_observation):
        return True

    # If this would only be the first pass, the successor may also have passed
    # before the next captured frame. A verified play by the following actor
    # proves both intervening passes without guessing any cards.
    if (
        state.consecutive_passes == 0
        and successor_observation.remaining_verified
        and successor_observation.remaining_count
        == state.remaining_for(successor)
    ):
        following = state.next_player(successor)
        if following is PlayerSeat.SELF:
            return bool(
                scene.self_turn is True
                and scene.self_turn_confidence >= confidence_threshold
            )
        return _remaining_confirms_play(
            state,
            scene.seat(following),
        )
    return False


def _inferred_pass_confidence(
    scene: SceneObservation,
    observation: SeatObservation,
) -> float:
    confidences = [observation.remaining_confidence]
    if scene.self_turn is True:
        confidences.append(scene.self_turn_confidence)
    return min(value for value in confidences if value > 0)


def _action_confidence(observation: SeatObservation) -> float:
    if observation.signal is VisualSignal.PASS:
        return observation.pass_confidence
    if observation.signal is VisualSignal.PLAY:
        return min(
            (card.confidence for card in observation.cards),
            default=observation.confidence,
        )
    return observation.confidence


def _remaining_count_error(
    state: ObservableGameState,
    observation: SeatObservation,
    event: ObservedAction,
) -> str | None:
    # The current Mac client does not expose a dedicated self count in all layouts;
    # self remaining is exactly derived from the tracked hand.
    if (
        event.actor is PlayerSeat.SELF
        or observation.remaining_count is None
        or not observation.remaining_verified
    ):
        return None
    expected = state.remaining_for(event.actor) - len(event.cards)
    if observation.remaining_count != expected:
        return (
            f"{event.actor.value} remaining count mismatch: screen="
            f"{observation.remaining_count}, expected={expected}"
        )
    return None


__all__ = [
    "VisualEventTracker",
    "VisualTrackerMode",
    "VisualTrackerUpdate",
]
