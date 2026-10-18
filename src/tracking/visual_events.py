from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from src.logic.action_validation import validate_observed_action
from src.logic.rules import Play, PlayType, legal_actions
from src.state.cards import FULL_DECK, CardSet
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.game_tracker import (
    GameStateTracker,
    GameStateTransitionError,
    StateUpdateStatus,
    reduce_observed_action,
)
from src.state.observable_state import ObservableGameState
from src.vision.scene_types import (
    RoundResultObservation,
    RoundResultOutcome,
    SceneObservation,
    SeatObservation,
    SeatRole,
    VisualSignal,
)

from .observation_evidence import (
    _action_confidence as _action_confidence,
)
from .observation_evidence import (
    _initial_observation_confidence as _initial_observation_confidence,
)
from .observation_evidence import (
    _seat_fingerprint as _seat_fingerprint,
)
from .observation_evidence import (
    _stable_hand_confidence as _stable_hand_confidence,
)
from .self_terminal_evidence import SelfTerminalEvidenceGate

_RESULT_STABILITY_FRAMES = 2
_FOLLOWING_PLAY_PASS_SOURCE = "live_following_play_inferred_pass"


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
    round_result: RoundResultObservation | None = None
    completed_round_id: str | None = None

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
            "round_result": (
                self.round_result.to_log_payload()
                if self.round_result is not None
                else None
            ),
            "completed_round_id": self.completed_round_id,
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


@dataclass
class _InvalidPlayEvidence:
    context: tuple[str, int, PlayerSeat]
    fingerprint: tuple[object, ...] | None = None
    count: int = 0
    last_source_frame_id: int = -1
    last_scene_frame_id: int = -1


@dataclass(frozen=True)
class _TurnEvidence:
    """Per-frame evidence for the expected actor, gathered once per update()."""

    expected: PlayerSeat
    observation: SeatObservation
    buffered_evidence: tuple[SceneObservation, SeatObservation] | None
    buffered_pass_scene: SceneObservation | None
    buffered_observation: bool
    stable_count: int


class _SelfTurnEvidence:
    """Small, actor-scoped memory of observed control and pass transitions."""

    def __init__(self, key: tuple[object, ...]) -> None:
        self.key = key
        self.controls_seen = False
        self.last_frame_id: int | None = None
        self.hand_key: tuple[object, ...] | None = None
        self.signals: dict[PlayerSeat, VisualSignal] = {}
        self.fresh_passes: set[PlayerSeat] = set()

    def observe(
        self, state: ObservableGameState, scene: SceneObservation,
        *, confidence_threshold: float,
    ) -> None:
        if self.last_frame_id is not None and scene.frame_id <= self.last_frame_id:
            return
        self.last_frame_id = scene.frame_id
        hand_confidence, _ = _stable_hand_confidence(
            scene.self_hand, confidence_threshold=confidence_threshold,
        )
        hand_key = (
            len(scene.self_hand),
            tuple(
                (card.rank, card.confidence >= confidence_threshold)
                for card in scene.self_hand
            ),
            hand_confidence >= confidence_threshold,
        )
        if hand_key != self.hand_key:
            # A pass seen before this post-hand appeared cannot prove that
            # both opponents acted after this candidate self play.
            self.fresh_passes.clear()
            self.hand_key = hand_key
        if scene.self_turn is True and scene.self_turn_confidence >= confidence_threshold:
            self.controls_seen = True
        for observation in scene.seats:
            signal = observation.signal
            if signal is VisualSignal.UNKNOWN:
                continue
            if _action_confidence(observation) < confidence_threshold:
                continue
            if signal is VisualSignal.PASS:
                if observation.seat is not PlayerSeat.SELF and (
                    not observation.remaining_verified
                    or observation.remaining_confidence < confidence_threshold
                    or observation.remaining_count != state.remaining_for(observation.seat)
                ):
                    continue
                if self.signals.get(observation.seat) in {
                    VisualSignal.NEUTRAL, VisualSignal.PLAY,
                }:
                    self.fresh_passes.add(observation.seat)
            self.signals[observation.seat] = signal

    def opponents_completed_cycle(
        self, state: ObservableGameState, scene: SceneObservation,
        *, confidence_threshold: float,
    ) -> bool:
        return all(
            seat in self.fresh_passes
            and scene.seat(seat).signal is VisualSignal.PASS
            and scene.seat(seat).pass_confidence >= confidence_threshold
            and scene.seat(seat).remaining_verified
            and scene.seat(seat).remaining_confidence >= confidence_threshold
            and scene.seat(seat).remaining_count == state.remaining_for(seat)
            for seat in (PlayerSeat.RIGHT, PlayerSeat.LEFT)
        )


def _observe_current_self_turn(
    evidence: _SelfTurnEvidence | None,
    state: ObservableGameState,
    recent_scenes: Sequence[SceneObservation],
    *,
    min_frame_id: int | None,
    confidence_threshold: float,
) -> _SelfTurnEvidence | None:
    if state.current_actor is not PlayerSeat.SELF:
        return None
    key = (state.round_id, state.revision, state.current_actor, min_frame_id)
    if evidence is None or evidence.key != key:
        evidence = _SelfTurnEvidence(key)
    for scene in recent_scenes:
        if min_frame_id is not None and scene.frame_id < min_frame_id:
            continue
        evidence.observe(state, scene, confidence_threshold=confidence_threshold)
    return evidence


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
        self._result_stable = _StableValue()
        self._armed = {seat: False for seat in PlayerSeat}
        self._scene_buffer_capacity = max(6, 2 * stability_frames)
        self._recent_scenes: deque[SceneObservation] = deque(maxlen=self._scene_buffer_capacity)
        self._scene_evidence_start_frame_id: int | None = None
        self._self_turn_evidence: _SelfTurnEvidence | None = None
        self._actor_entry_frame_id: int | None = None
        self._self_actor_entry_hand_count: int | None = None
        self._self_actor_entry_frame_id: int | None = None
        self._uncertain_reason: str | None = None
        self._result_overlay_active = False
        self._invalid_play_evidence: _InvalidPlayEvidence | None = None
        self._self_terminal_gate = SelfTerminalEvidenceGate()

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
        if scene.round_result is not None:
            return self._update_round_result(scene)
        geometry_warnings = tuple(
            warning for warning in scene.warnings
            if warning.startswith("self_hand_geometry_unavailable:")
        )
        if geometry_warnings:
            return self._pause_unknown_hand_geometry(scene, geometry_warnings)
        self._result_stable.update(("no_round_result",))
        self._result_overlay_active = False

        initial_update = self._update_initial_scene(scene)
        if initial_update is not None:
            return initial_update
        auto_scan_update = self._update_automatic_scan(scene)
        if auto_scan_update is not None:
            return auto_scan_update
        mode_update = self._update_inactive_mode(scene)
        if mode_update is not None:
            return mode_update

        assert self._tracker is not None
        state = self._tracker.state
        # A seat can become blank before it is that seat's turn. Remember the
        # edge immediately instead of only watching the currently expected
        # actor. Otherwise two opponents acting between adjacent captures can
        # hide the blank frame for the second actor and make its new cards look
        # like a stale table prompt.
        for seat in PlayerSeat:
            if scene.seat(seat).signal is VisualSignal.NEUTRAL:
                self._armed[seat] = True

        landlord_update = self._update_landlord_change(scene, state)
        if landlord_update is not None:
            return landlord_update
        terminal_update = self._update_self_terminal(scene, state)
        if terminal_update is not None:
            return terminal_update

        self._recent_scenes.append(scene)
        self._self_turn_evidence = _observe_current_self_turn(
            self._self_turn_evidence, state, self._recent_scenes,
            min_frame_id=self._self_actor_entry_frame_id,
            confidence_threshold=self.confidence_threshold,
        )
        turn = self._collect_turn_evidence(scene, state)
        hand_change, hand_change_scene, stable_count = self._self_hand_change_evidence(
            scene, state, turn.stable_count,
        )

        # A genuine self-hand difference is a normal event and must get the
        # first opportunity to advance the reducer. The generic drift recovery
        # previously ran first and could replace a known self play plus two
        # fast opponent passes with a fresh approximate round.
        opponent_candidate = (
            turn.expected is not PlayerSeat.SELF
            and turn.observation.cards
            and _remaining_proves_play(
                state, turn.observation, confidence_threshold=self.confidence_threshold,
            )
        )
        if (
            self._invalid_play_evidence is not None
            and self._invalid_play_evidence.count > 0
        ) or opponent_candidate:
            # Give an in-flight opponent action priority over rebuilding the
            # same revision from its changed counter. Invalid ranks need fresh
            # evidence, not an approximate round that bypasses the reducer.
            # A reset counter keeps only the source watermark; it must not
            # prevent stable recovery after a real action was missed entirely.
            self._active_scan_stable = _StableValue()
        elif hand_change is None:
            recovery = self._recover_drifted_state(scene, state)
            if recovery is not None:
                return recovery

        if turn.expected is PlayerSeat.SELF and hand_change is not None:
            return self._update_self_hand_diff(
                scene, state, turn, hand_change, hand_change_scene, stable_count,
            )

        pass_update = self._update_inferred_pass(scene, state, turn, stable_count)
        if pass_update is not None:
            return pass_update

        if (
            turn.expected is PlayerSeat.SELF
            and turn.observation.signal is VisualSignal.PASS
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

        corroborated_update = self._update_corroborated_play(
            scene, state, turn, stable_count,
        )
        if corroborated_update is not None:
            return corroborated_update
        return self._update_visual_signal(scene, state, turn, stable_count)

    # ------------------------------------------------------------------
    # update() stages.  Each returns a VisualTrackerUpdate when that stage
    # decides the frame, or None to fall through to the next stage.
    # ------------------------------------------------------------------

    def _update_round_result(self, scene: SceneObservation) -> VisualTrackerUpdate:
        assert scene.round_result is not None
        self._self_terminal_gate.discard_through(scene.frame_id)
        result_fingerprint = ("round_result", scene.round_result.outcome.value)
        stable_count = self._result_stable.update(result_fingerprint)
        if stable_count < _RESULT_STABILITY_FRAMES:
            label = {
                RoundResultOutcome.VICTORY: "胜利",
                RoundResultOutcome.DEFEAT: "失败",
                RoundResultOutcome.UNKNOWN: "疑似结算界面",
            }[scene.round_result.outcome]
            return VisualTrackerUpdate(
                mode=self.mode,
                message=(
                    f"检测到{label}，正在确认 "
                    f"{stable_count}/{_RESULT_STABILITY_FRAMES}"
                ),
                state=self.state,
                warnings=scene.warnings,
            )
        return self._handle_round_result(scene.round_result)

    def _update_initial_scene(
        self, scene: SceneObservation,
    ) -> VisualTrackerUpdate | None:
        initial = _initial_state_payload(
            scene,
            confidence_threshold=self.confidence_threshold,
            allow_dealt_display=(
                self._tracker is None
                or self.mode is not VisualTrackerMode.TRACKING
            ),
        )
        if initial is None:
            self._initial_stable.update(("not_initial",))
            return None
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
            self._self_terminal_gate.discard_through(scene.frame_id)
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
        return None

    def _update_automatic_scan(
        self, scene: SceneObservation,
    ) -> VisualTrackerUpdate | None:
        if self._tracker is not None and self.mode is not VisualTrackerMode.UNCERTAIN:
            return None
        self._self_terminal_gate.discard_through(scene.frame_id)
        auto_payload, auto_error = _current_scan_payload(
            scene,
            confidence_threshold=self.confidence_threshold,
            scan_source="automatic",
        )
        if auto_payload is None:
            self._auto_scan_stable.update(("not_scannable", auto_error))
            return None
        stable_count = self._auto_scan_stable.update(auto_payload.fingerprint)
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

    def _update_inactive_mode(
        self, scene: SceneObservation,
    ) -> VisualTrackerUpdate | None:
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
            self._self_terminal_gate.discard_through(scene.frame_id)
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
        return None

    def _update_landlord_change(
        self, scene: SceneObservation, state: ObservableGameState,
    ) -> VisualTrackerUpdate | None:
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
        if observed_landlord is None or observed_landlord is state.landlord:
            return None
        reason = (
            "检测到地主位置已变化："
            f"状态={state.landlord.value}，画面={observed_landlord.value}"
        )
        self._self_terminal_gate.discard_through(scene.frame_id)
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

    def _update_self_terminal(
        self, scene: SceneObservation, state: ObservableGameState,
    ) -> VisualTrackerUpdate | None:
        watermarks = [value for value in (
            self._scene_evidence_start_frame_id, self._self_actor_entry_frame_id,
        ) if value is not None]
        table = scene.self_table

        def observe(candidate: ObservableGameState, *, commit: bool) -> ObservedAction | None:
            return self._self_terminal_gate.observe(
                state=candidate, scene=scene,
                cards=table.cards if table is not None else (),
                complete=table.complete if table is not None else False,
                empty_hand_verified=scene.self_hand_empty_verified,
                epoch=max(watermarks) if watermarks else None,
                stability_frames=self.stability_frames,
                confidence_threshold=self.confidence_threshold,
                commit=commit,
            )

        if state.current_actor is PlayerSeat.SELF:
            terminal_event = observe(state, commit=True)
            if terminal_event is None:
                return None
            return self._apply_event(scene, scene.seat(PlayerSeat.SELF), terminal_event)
        # The opponents still owed a reply to our lead, but their pass label may
        # have shown only during a self-hand crop pause and been cleared by the
        # game before geometry recovered. A blank seat with a verified unchanged
        # counter plus our complete terminal table is the pass -> finish chain;
        # the gate judges that chain on the previewed state, and the pass is
        # committed first, one event per update.
        chain = _terminal_pass_chain(scene, state, confidence_threshold=self.confidence_threshold)
        if chain is None:
            return None
        first_pass, previewed = chain
        if observe(previewed, commit=False) is None:
            return None
        return self._apply_event(scene, scene.seat(first_pass.actor), first_pass)

    def _collect_turn_evidence(
        self, scene: SceneObservation, state: ObservableGameState,
    ) -> _TurnEvidence:
        expected = state.current_actor
        current_observation = scene.seat(expected)
        invalid = self._invalid_play_evidence
        if invalid is not None:
            if invalid.context != (state.round_id, state.revision, expected):
                self._invalid_play_evidence = None
            elif scene.frame_id > invalid.last_scene_frame_id:
                if (
                    invalid.fingerprint != _invalid_play_fingerprint(
                        current_observation, self.confidence_threshold,
                    )
                    or not _usable_play_evidence(current_observation, self.confidence_threshold)
                ):
                    invalid.fingerprint = None
                    invalid.count = 0
                invalid.last_scene_frame_id = scene.frame_id
        buffered_evidence = _buffered_play_evidence(
            state,
            expected,
            self._recent_scenes,
            confidence_threshold=self.confidence_threshold,
            min_frame_id=self._actor_entry_frame_id,
        )
        buffered_pass_scene = _buffered_pass_evidence(
            state,
            expected,
            self._recent_scenes,
            confidence_threshold=self.confidence_threshold,
            min_frame_id=self._actor_entry_frame_id,
        )
        observation = (
            buffered_evidence[1]
            if buffered_evidence is not None
            else (
                buffered_pass_scene.seat(expected)
                if buffered_pass_scene is not None
                else current_observation
            )
        )
        buffered_observation = (
            buffered_evidence is not None
            or buffered_pass_scene is not None
        ) and (
            observation is not current_observation
        )
        stable_counts = {
            seat: self._seat_stable[seat].update(_seat_fingerprint(
                scene.seat(seat),
                self_hand=(
                    scene.self_hand if seat is PlayerSeat.SELF else ()
                ),
                self_turn=(
                    scene.self_turn if seat is PlayerSeat.SELF else None
                ),
            ))
            for seat in PlayerSeat
        }
        return _TurnEvidence(
            expected=expected,
            observation=observation,
            buffered_evidence=buffered_evidence,
            buffered_pass_scene=buffered_pass_scene,
            buffered_observation=buffered_observation,
            stable_count=stable_counts[expected],
        )

    def _self_hand_change_evidence(
        self,
        scene: SceneObservation,
        state: ObservableGameState,
        stable_count: int,
    ) -> tuple[_SelfHandChange | None, SceneObservation, int]:
        hand_change_scene = scene
        if state.current_actor is not PlayerSeat.SELF:
            return None, hand_change_scene, stable_count
        hand_change = _self_hand_change(
            state,
            scene,
            confidence_threshold=self.confidence_threshold,
        )
        if hand_change is not None and hand_change.error is None:
            # An older crop cannot supply a different post-hand or borrow
            # the latest scene's unrelated stable counter. Keep the latest
            # interpretation/confidence, and only consult matching evidence.
            hand_change_scene, stable_count = _current_self_hand_evidence(
                state,
                scene,
                hand_change,
                self._recent_scenes,
                confidence_threshold=self.confidence_threshold,
                min_frame_id=self._self_actor_entry_frame_id,
            )
        if hand_change is not None and self._self_actor_entry_hand_count is not None:
            successor = (
                state.trick_leader
                if state.consecutive_passes == 1
                else state.next_player(PlayerSeat.SELF)
            )
            assert successor is not None
            successor_observation = scene.seat(successor)
            target = (
                Play.parse(state.trick_target.cards)
                if state.trick_target
                else None
            )
            forced_pass = bool(
                target is not None
                and not legal_actions(
                    state.self_hand,
                    target,
                    include_pass=False,
                )
            )
            if (
                state.consecutive_passes == 0
                and self._armed[successor]
                and successor_observation.signal is VisualSignal.PASS
                and successor_observation.pass_confidence
                >= self.confidence_threshold
                and successor_observation.remaining_verified
                and successor_observation.remaining_count
                == state.remaining_for(successor)
                and (
                    forced_pass
                    or (
                        self._self_actor_entry_hand_count
                        < len(state.self_hand)
                        and self._self_actor_entry_hand_count
                        == len(scene.self_hand)
                    )
                )
                and not (
                    hand_change.error is None
                    and len(hand_change_scene.self_hand)
                    < self._self_actor_entry_hand_count
                )
            ):
                # The apparent hand decrease already existed before self
                # became current, while the next seat has since crossed from
                # neutral to an explicit pass.  This is an under-segmented
                # hand crop followed by a real self pass, not a self play.
                hand_change = None
        return hand_change, hand_change_scene, stable_count

    def _update_self_hand_diff(
        self,
        scene: SceneObservation,
        state: ObservableGameState,
        turn: _TurnEvidence,
        hand_change: _SelfHandChange,
        hand_change_scene: SceneObservation,
        stable_count: int,
    ) -> VisualTrackerUpdate:
        expected = turn.expected
        observation = turn.observation
        successor = state.next_player(PlayerSeat.SELF)
        successor_play_proves_action = _remaining_proves_play(
            state,
            hand_change_scene.seat(successor),
            confidence_threshold=self.confidence_threshold,
        )
        required = (
            1
            if (
                hand_change.error is None
                and (
                    (
                        hand_change_scene.self_turn is False
                        and hand_change_scene.self_turn_confidence
                        >= self.confidence_threshold
                    )
                    or successor_play_proves_action
                )
            )
            else min(2, self.stability_frames)
        )
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
        opponent_cycle_complete = bool(
            self._self_turn_evidence is not None
            and self._self_turn_evidence.opponents_completed_cycle(
                state, scene,
                confidence_threshold=self.confidence_threshold,
            )
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
            # Persistent old PASS labels cannot prove a new cycle;
            # do not bypass that ambiguity through automatic rescan.
            if not all(
                scene.seat(seat).signal is VisualSignal.PASS
                for seat in (PlayerSeat.RIGHT, PlayerSeat.LEFT)
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

    def _update_inferred_pass(
        self,
        scene: SceneObservation,
        state: ObservableGameState,
        turn: _TurnEvidence,
        stable_count: int,
    ) -> VisualTrackerUpdate | None:
        expected = turn.expected
        observation = turn.observation
        pass_successor = (
            state.trick_leader
            if state.consecutive_passes == 1
            else state.next_player(expected)
        )
        assert pass_successor is not None
        following_pass = _following_play_pass_evidence(
            state, self._recent_scenes,
            confidence_threshold=self.confidence_threshold,
            stability_frames=self.stability_frames,
            min_frame_id=self._actor_entry_frame_id,
            terminal_armed=self._armed.get(state.trick_leader, False),
            evidence_start_frame_id=self._scene_evidence_start_frame_id,
        )
        if _continues_following_play_pass(state) and following_pass is None:
            # The first inferred pass does not authorize another one after
            # the latest hand, count or following play stops being trustworthy.
            return VisualTrackerUpdate(
                mode=self.mode, message="等待后续出牌与完整手牌重新确认过牌链",
                state=state, warnings=scene.warnings,
            )
        buffered_successor_play = _buffered_play_evidence(
            state,
            pass_successor,
            self._recent_scenes,
            confidence_threshold=self.confidence_threshold,
            min_frame_id=self._actor_entry_frame_id,
        )
        buffered_successor_pass = _buffered_pass_evidence(
            state,
            pass_successor,
            self._recent_scenes,
            confidence_threshold=self.confidence_threshold,
            min_frame_id=self._actor_entry_frame_id,
        )
        if following_pass is None and not _can_infer_pass(
            state,
            scene,
            observation,
            confidence_threshold=self.confidence_threshold,
            stable_count=stable_count,
            stability_frames=self.stability_frames,
            successor_action_armed=(
                pass_successor in self._self_turn_evidence.fresh_passes
                if self._self_turn_evidence is not None
                else self._armed.get(pass_successor, False)
            ),
            buffered_successor_play=buffered_successor_play is not None,
            buffered_successor_pass=(
                buffered_successor_pass is not None
                and (
                    self._self_turn_evidence is None
                    or pass_successor in self._self_turn_evidence.fresh_passes
                )
            ),
            self_turn_was_active=(
                self._self_turn_evidence is not None
                and self._self_turn_evidence.controls_seen
            ),
            fresh_self_pass=(
                self._self_turn_evidence is not None
                and PlayerSeat.SELF in self._self_turn_evidence.fresh_passes
            ),
        ):
            return None
        event = ObservedAction(
            event_id=(
                f"{state.round_id}:{state.last_sequence_no + 1}:"
                f"{expected.value}"
            ),
            sequence_no=state.last_sequence_no + 1,
            actor=expected,
            cards=CardSet(()),
            confidence=(following_pass[1] if following_pass is not None
                        else _inferred_pass_confidence(scene, observation)),
            source=(_FOLLOWING_PLAY_PASS_SOURCE if following_pass is not None
                    else "live_turn_inferred_pass"),
        )
        transition_evidence = following_pass[0] if following_pass is not None else (
            buffered_successor_play[0]
            if buffered_successor_play is not None
            else buffered_successor_pass
        )
        return self._apply_event(
            scene,
            observation,
            event,
            actor_transition_scene=transition_evidence,
        )

    def _update_corroborated_play(
        self,
        scene: SceneObservation,
        state: ObservableGameState,
        turn: _TurnEvidence,
        stable_count: int,
    ) -> VisualTrackerUpdate | None:
        expected = turn.expected
        observation = turn.observation
        buffered_evidence = turn.buffered_evidence
        corroborated_confidence = _verified_low_confidence_play(
            state,
            observation,
            confidence_threshold=self.confidence_threshold,
        )
        if corroborated_confidence is None:
            return None
        if expected is not PlayerSeat.SELF and not _usable_play_evidence(
            observation, self.confidence_threshold,
        ):
            return self._pause_invalid_opponent_play(
                scene, observation, "opponent play evidence is unavailable",
                source_frame_id=(buffered_evidence[0].frame_id if buffered_evidence else scene.frame_id),
            )
        # A verified counter drop equal to the visible card count is an
        # independent second signal.  Waiting for three identical frames
        # here loses normal one-second table animations and makes the
        # reducer observe the following action under the wrong actor.
        if stable_count < 1:
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
            if expected is not PlayerSeat.SELF:
                return self._pause_invalid_opponent_play(
                    scene, observation, str(exc),
                    source_frame_id=(buffered_evidence[0].frame_id if buffered_evidence else scene.frame_id),
                )
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
            source=(
                "live_visual_remaining_correlated_buffered"
                if turn.buffered_observation
                else "live_visual_remaining_correlated"
            ),
        )
        return self._apply_event(
            scene,
            observation,
            event,
            actor_transition_scene=(
                buffered_evidence[0]
                if buffered_evidence is not None
                else None
            ),
        )

    def _update_visual_signal(
        self,
        scene: SceneObservation,
        state: ObservableGameState,
        turn: _TurnEvidence,
        stable_count: int,
    ) -> VisualTrackerUpdate:
        expected = turn.expected
        observation = turn.observation
        buffered_evidence = turn.buffered_evidence
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
        required_stability = _action_stability_frames(
            state,
            observation,
            armed=self._armed[expected],
            default=self.stability_frames,
        )
        if (
            turn.buffered_pass_scene is not None
            and observation.signal is VisualSignal.PASS
        ):
            # A neutral -> explicit pass edge plus an unchanged verified
            # counter is already two independent observations.  The notice
            # commonly survives for only one recorded frame.
            required_stability = 1
        if stable_count < required_stability:
            return VisualTrackerUpdate(
                mode=self.mode,
                message=(
                    f"稳定 {expected.value} 动作 "
                    f"{stable_count}/{required_stability}"
                ),
                state=state,
            )

        try:
            cards = observation.card_set
        except ValueError as exc:
            if expected is not PlayerSeat.SELF:
                return self._pause_invalid_opponent_play(
                    scene, observation, str(exc),
                    source_frame_id=(buffered_evidence[0].frame_id if buffered_evidence else scene.frame_id),
                )
            return self._mark_uncertain(f"self cards are invalid: {exc}")
        confidence = _action_confidence(observation)
        if not 0.0 <= confidence <= 1.0 or confidence < self.confidence_threshold:
            if expected is not PlayerSeat.SELF and observation.cards:
                return self._pause_invalid_opponent_play(
                    scene, observation, "opponent play confidence is unavailable",
                    source_frame_id=(buffered_evidence[0].frame_id if buffered_evidence else scene.frame_id),
                )
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
            source=(
                "live_visual_buffered"
                if turn.buffered_observation
                else "live_visual"
            ),
        )
        return self._apply_event(
            scene,
            observation,
            event,
            actor_transition_scene=(
                buffered_evidence[0]
                if buffered_evidence is not None
                else None
            ),
        )

    def _pause_unknown_hand_geometry(
        self,
        scene: SceneObservation,
        geometry_warnings: tuple[str, ...],
    ) -> VisualTrackerUpdate:
        # An unavailable crop is not a physical empty hand. Discard evidence
        # spanning this gap without changing the reducer or guessing a play.
        self._self_terminal_gate.discard_through(scene.frame_id)
        self._recent_scenes.clear()
        self._scene_evidence_start_frame_id = max(
            scene.frame_id + 1,
            self._scene_evidence_start_frame_id if self._scene_evidence_start_frame_id is not None
            else scene.frame_id + 1,
        )
        self._invalid_play_evidence = None
        self._self_turn_evidence = None
        self._initial_stable = _StableValue()
        self._auto_scan_stable = _StableValue()
        self._active_scan_stable = _StableValue()
        self._role_stable = _StableValue()
        self._result_stable = _StableValue()
        self._seat_stable = {seat: _StableValue() for seat in PlayerSeat}
        # Only the self-hand crop is unavailable. An opponent seat that was
        # blank on a valid frame before this gap keeps that edge: a pass shown
        # after recovery still needs fresh stable frames, but it is not a stale
        # prompt. Dropping every edge here left a recovered pass unprovable and
        # blocked the self terminal gate behind the wrong current actor.
        self._armed[PlayerSeat.SELF] = False
        self._actor_entry_frame_id = scene.frame_id + 1
        self._self_actor_entry_hand_count = None
        state = self.state
        self._self_actor_entry_frame_id = (
            scene.frame_id + 1
            if state is not None and state.current_actor is PlayerSeat.SELF
            else None
        )
        message = "手牌几何暂不可用；已清除跨帧动作证据，等待可信画面恢复"
        if state is not None and state.phase is not RoundPhase.FINISHED:
            paused = _paused_visual_update(
                state,
                message=message,
                warning=geometry_warnings[0],
                scene_warnings=scene.warnings,
            )
            return replace(paused, mode=self.mode)
        return VisualTrackerUpdate(
            mode=self.mode,
            message=message,
            state=state,
            warnings=scene.warnings,
        )

    def scan_current_scene(
        self,
        scene: SceneObservation,
    ) -> VisualTrackerUpdate:
        """Build a safe approximate state from a user-requested current scan."""

        self._self_terminal_gate.discard_through(scene.frame_id)

        geometry_warnings = tuple(
            warning for warning in scene.warnings
            if warning.startswith("self_hand_geometry_unavailable:")
        )
        if geometry_warnings:
            return self._pause_unknown_hand_geometry(scene, geometry_warnings)
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
        payload: _CurrentScanPayload,
        *,
        automatic: bool,
    ) -> VisualTrackerUpdate:
        self._self_terminal_gate.discard_through(scene.frame_id)
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
        self._invalid_play_evidence = None
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
        self._recent_scenes = deque((scene,), maxlen=self._scene_buffer_capacity)
        self._scene_evidence_start_frame_id = scene.frame_id
        self._actor_entry_frame_id = scene.frame_id
        self._self_actor_entry_hand_count = len(scene.self_hand)
        self._self_actor_entry_frame_id = scene.frame_id
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
        *,
        actor_transition_scene: SceneObservation | None = None,
    ) -> VisualTrackerUpdate:
        assert self._tracker is not None
        state = self._tracker.state
        expected = event.actor
        count_error = _remaining_count_error(state, observation, event)
        if expected is not PlayerSeat.SELF and not event.is_pass:
            # Metadata failures are not rank-recognition transients. Check
            # them before the pure reducer so they retain the hard stop.
            if (
                state.phase is not RoundPhase.PLAYING
                or expected is not state.current_actor
                or observation.seat is not expected
                or event.sequence_no != state.last_sequence_no + 1
                or event.event_id in dict(state.processed_events)
            ):
                return self._mark_uncertain("visual event metadata conflicts with state", event=event)
            error = count_error
            if not _usable_play_evidence(observation, self.confidence_threshold):
                error = error or "opponent play evidence is unavailable"
            if error is None:
                try:
                    preview = reduce_observed_action(
                        state, event, validator=validate_observed_action,
                        confidence_threshold=self.confidence_threshold,
                    )
                except GameStateTransitionError as exc:
                    error = str(exc)
                else:
                    if preview.status is not StateUpdateStatus.APPLIED:
                        error = preview.message
            if error is not None:
                return self._pause_invalid_opponent_play(
                    scene, observation, error,
                    source_frame_id=(actor_transition_scene or scene).frame_id,
                )
        if count_error is not None:
            return self._mark_uncertain(count_error, event=event)

        result = self._tracker.apply(event)
        if result.status is not StateUpdateStatus.APPLIED:
            return self._mark_uncertain(
                f"state rejected visual event: {result.message}",
                event=event,
            )

        self._armed[expected] = False
        self._self_terminal_gate.reset()
        self._invalid_play_evidence = None
        self._active_scan_stable = _StableValue()
        next_actor = result.state.current_actor
        transition_scene = actor_transition_scene or scene
        self._actor_entry_frame_id = transition_scene.frame_id
        self._self_actor_entry_hand_count = (
            len(transition_scene.self_hand)
            if next_actor is PlayerSeat.SELF
            else None
        )
        self._self_actor_entry_frame_id = (
            transition_scene.frame_id
            if next_actor is PlayerSeat.SELF
            else None
        )
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

    def _pause_invalid_opponent_play(
        self,
        scene: SceneObservation,
        observation: SeatObservation,
        reason: str,
        *,
        source_frame_id: int,
    ) -> VisualTrackerUpdate:
        assert self._tracker is not None
        state = self._tracker.state
        context = (state.round_id, state.revision, state.current_actor)
        evidence = self._invalid_play_evidence
        if evidence is None or evidence.context != context:
            evidence = _InvalidPlayEvidence(context)
            self._invalid_play_evidence = evidence
        fingerprint = _invalid_play_fingerprint(observation, self.confidence_threshold)
        if source_frame_id > evidence.last_source_frame_id:
            if _usable_play_evidence(observation, self.confidence_threshold):
                evidence.count = evidence.count + 1 if evidence.fingerprint == fingerprint else 1
                evidence.fingerprint = fingerprint
            else:
                evidence.count = 0
                evidence.fingerprint = None
            evidence.last_source_frame_id = source_frame_id
        evidence.last_scene_frame_id = max(evidence.last_scene_frame_id, scene.frame_id)
        self._active_scan_stable = _StableValue()
        warning = f"transient_invalid_opponent_play: {observation.seat.value}: {reason}"
        if evidence.count >= self.stability_frames:
            return self._mark_uncertain(warning)
        # event=None is essential: TRACKING + event is logged as an accepted
        # action by the runtime, even when the returned state is paused.
        return _paused_visual_update(
            state,
            message=(f"等待 {observation.seat.value} 非法候选视觉恢复 "
                     f"{evidence.count}/{self.stability_frames}"),
            warning=warning,
            scene_warnings=scene.warnings,
        )

    def _handle_round_result(
        self,
        result: RoundResultObservation,
    ) -> VisualTrackerUpdate:
        state = self.state
        completed_round_id = (
            state.round_id
            if state is not None and not self._result_overlay_active
            else None
        )
        label = {
            RoundResultOutcome.VICTORY: "胜利",
            RoundResultOutcome.DEFEAT: "失败",
            RoundResultOutcome.UNKNOWN: "牌局结束",
        }[result.outcome]
        self._clear_round()
        # Preserve one confirmed sample after clearing the round. A still
        # visible overlay remains recognized on the next frame, while a
        # different outcome must again pass the two-frame gate.
        self._result_stable.update(("round_result", result.outcome.value))
        self._result_overlay_active = True
        return VisualTrackerUpdate(
            mode=VisualTrackerMode.WAITING_FOR_ROUND,
            message=f"检测到结算界面：{label}；已结束当前牌局并等待下一局",
            state=None,
            warnings=(),
            round_result=result,
            completed_round_id=completed_round_id,
        )

    def handle_window_unavailable(self, reason: str) -> VisualTrackerUpdate:
        self._self_terminal_gate.reset()
        self._invalid_play_evidence = None
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
        self._self_terminal_gate.reset()
        self._invalid_play_evidence = None
        self._tracker = None
        self.mode = VisualTrackerMode.WAITING_FOR_ROUND
        self._uncertain_reason = None
        self._initial_stable = _StableValue()
        self._auto_scan_stable = _StableValue()
        self._active_scan_stable = _StableValue()
        self._role_stable = _StableValue()
        self._seat_stable = {seat: _StableValue() for seat in PlayerSeat}
        self._result_stable = _StableValue()
        self._armed = {seat: False for seat in PlayerSeat}
        self._recent_scenes = deque(maxlen=self._scene_buffer_capacity)
        self._scene_evidence_start_frame_id = None
        self._self_turn_evidence = None
        self._actor_entry_frame_id = None
        self._self_actor_entry_hand_count = None
        self._self_actor_entry_frame_id = None

    def _initialize(
        self,
        scene: SceneObservation,
        initial: _InitialStatePayload,
    ) -> VisualTrackerUpdate:
        self._self_terminal_gate.discard_through(scene.frame_id)
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
        self._invalid_play_evidence = None
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
        self._recent_scenes = deque((scene,), maxlen=self._scene_buffer_capacity)
        self._scene_evidence_start_frame_id = scene.frame_id
        self._actor_entry_frame_id = scene.frame_id
        self._self_actor_entry_hand_count = (
            len(scene.self_hand)
            if state.current_actor is PlayerSeat.SELF
            else None
        )
        self._self_actor_entry_frame_id = (
            scene.frame_id
            if state.current_actor is PlayerSeat.SELF
            else None
        )
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
        self._self_terminal_gate.reset()
        self._invalid_play_evidence = None
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
    allow_dealt_display: bool = True,
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

    # Between bidding and the landlord's first play the client shows the three
    # dealt cards at the top of the table, inside the opponents' play ROIs. A
    # seat whose verified counter still equals its deal cannot have played, so
    # such cards are the dealt-cards display, not an action to rebuild from.
    # The caller disables this while a round is actively tracked: there the
    # same frame is an opponent play whose counter has not yet decremented,
    # which the action path keeps waiting for instead of re-dealing.
    dealt_display_only = allow_dealt_display and scene.seat(
        PlayerSeat.SELF,
    ).signal is VisualSignal.NEUTRAL and all(
        observation.signal is VisualSignal.NEUTRAL
        or (
            observation.remaining_verified
            and observation.remaining_count == expected[observation.seat]
        )
        for observation in scene.seats
        if observation.seat is not PlayerSeat.SELF
    )
    if any(
        observation.signal is not VisualSignal.NEUTRAL
        for observation in scene.seats
    ) and not dealt_display_only:
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

    def clear_confidence(observation: SeatObservation) -> float | None:
        # An unresolved candidate (including an empty UNKNOWN) is not proof
        # that this seat has no current play. Keep supported UNKNOWN targets.
        if observation.cards:
            return None
        if observation.signal is VisualSignal.NEUTRAL:
            value = observation.confidence
        elif observation.signal is VisualSignal.PASS:
            value = observation.pass_confidence
        else:
            return None
        # The closed range also rejects NaN and either infinity.
        if not 0.0 <= value <= 1.0 or value < confidence_threshold:
            return None
        return value

    if left_play_confidence is not None:
        trick_target = tuple(card.rank for card in left.cards)
        trick_leader = PlayerSeat.LEFT
        confidence_values.append(left_play_confidence)
        if left.signal is VisualSignal.UNKNOWN:
            inference_warnings.append(
                "current_scan_accepted_left_play_from_verified_remaining"
            )
    elif right_play_confidence is not None:
        left_clear_confidence = clear_confidence(left)
        if left_clear_confidence is None:
            return None, "left 场上牌尚未确认为空或不出，不能推断已过牌"
        trick_target = tuple(card.rank for card in right.cards)
        trick_leader = PlayerSeat.RIGHT
        consecutive_passes = 1
        confidence_values.append(left_clear_confidence)
        confidence_values.append(right_play_confidence)
        if right.signal is VisualSignal.UNKNOWN:
            inference_warnings.append(
                "current_scan_accepted_right_play_from_verified_remaining"
            )
        inference_warnings.append(
            "manual_scan_inferred_left_pass_after_right_play"
        )
    else:
        for observation in (left, right):
            evidence_confidence = clear_confidence(observation)
            if evidence_confidence is None:
                return None, (
                    f"{observation.seat.value} 场上牌尚未确认为空或不出，"
                    "不能确认自由出牌"
                )
            confidence_values.append(evidence_confidence)
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




def _self_hand_evidence_key(
    state: ObservableGameState,
    scene: SceneObservation,
    change: _SelfHandChange,
    *,
    confidence_threshold: float,
    min_frame_id: int | None,
) -> tuple[object, ...]:
    assert change.cards is not None and change.error is None
    post_hand = Counter(state.self_hand.cards) - Counter(change.cards.cards)
    return (
        state.round_id,
        state.revision,
        state.current_actor,
        min_frame_id,
        state.self_hand.cards,
        change.cards.cards,
        CardSet.parse(post_hand.elements()).cards,
        len(scene.self_hand),
        # Every physical slot participates, including low-confidence slots.
        # Compare confidence eligibility, not jittering floating-point values;
        # _self_hand_change still enforces the existing aggregate/outlier rule.
        tuple(
            (card.rank, card.confidence >= confidence_threshold)
            for card in scene.self_hand
        ),
        scene.seat(PlayerSeat.SELF).signal,
        tuple(card.rank for card in scene.seat(PlayerSeat.SELF).cards),
        scene.self_turn,
        scene.self_turn_confidence >= confidence_threshold,
    )


def _current_self_hand_evidence(
    state: ObservableGameState,
    latest: SceneObservation,
    latest_change: _SelfHandChange,
    recent_scenes: Sequence[SceneObservation],
    *,
    confidence_threshold: float,
    min_frame_id: int | None,
) -> tuple[SceneObservation, int]:
    """Count the contiguous matching suffix, anchored to a valid latest hand.

    Historical evidence is usable only for this exact current interpretation.
    A restored slot, changed rank, ineligible frame or actor boundary breaks
    the suffix. The latest action and confidence are never replaced.
    """
    expected_key = _self_hand_evidence_key(
        state, latest, latest_change,
        confidence_threshold=confidence_threshold,
        min_frame_id=min_frame_id,
    )
    evidence_scene = latest
    support_count = 0
    seen_frame_ids: set[int] = set()
    successor = state.next_player(PlayerSeat.SELF)
    for candidate in reversed(recent_scenes):
        if min_frame_id is not None and candidate.frame_id < min_frame_id:
            break
        if candidate.frame_id in seen_frame_ids:
            continue
        seen_frame_ids.add(candidate.frame_id)
        change = (
            latest_change
            if candidate is latest
            else _self_hand_change(
                state, candidate,
                confidence_threshold=confidence_threshold,
            )
        )
        if change is None or change.error is not None:
            break
        candidate_key = _self_hand_evidence_key(
            state, candidate, change,
            confidence_threshold=confidence_threshold,
            min_frame_id=min_frame_id,
        )
        if candidate_key != expected_key:
            break
        support_count += 1
        # Preserve short-lived successor-play corroboration, but only after
        # matching the entire post-hand evidence to the current scene.
        if (
            not _remaining_proves_play(
                state, evidence_scene.seat(successor),
                confidence_threshold=confidence_threshold,
            )
            and _remaining_proves_play(
                state, candidate.seat(successor),
                confidence_threshold=confidence_threshold,
            )
        ):
            evidence_scene = candidate
    return evidence_scene, support_count


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
    confidence, _ = _stable_hand_confidence(
        scene.self_hand,
        confidence_threshold=confidence_threshold,
    )
    removed_count = len(state.self_hand) - len(scene.self_hand)
    successor = state.next_player(PlayerSeat.SELF)
    successor_play_proves_action = _remaining_proves_play(
        state,
        scene.seat(successor),
        confidence_threshold=confidence_threshold,
    )
    inferred = _infer_self_play_from_legal_actions(
        state,
        observed_cards,
        removed_count=removed_count,
        max_corrections=(3 if successor_play_proves_action else 1),
    )
    if inferred is None:
        unexpected = observed - previous
        detail = (
            " ".join(sorted(unexpected.elements()))
            if unexpected
            else "no unique legal hand-difference interpretation"
        )
        return _SelfHandChange(
            cards=None,
            confidence=confidence,
            error=(
                "self hand change contains ranks outside the tracked hand: "
                + detail
            ),
        )
    if confidence < confidence_threshold:
        return _SelfHandChange(
            cards=None,
            confidence=confidence,
            error=(
                f"self hand change confidence {confidence:.3f} is below "
                f"{confidence_threshold:.3f}"
            ),
        )
    return _SelfHandChange(cards=inferred, confidence=confidence)


def _infer_self_play_from_legal_actions(
    state: ObservableGameState,
    observed_cards: tuple[str, ...],
    *,
    removed_count: int,
    max_corrections: int,
) -> CardSet | None:
    """Resolve a hand difference against the legal-action set.

    A selected/moving hand occasionally gives one crop the neighbouring
    rank.  The previous implementation treated that single classifier error
    as permanent state drift.  Here the physical previous hand, observed hand
    size and current trick jointly constrain the answer.  Normally only one
    substitution is allowed; a verified next-player action permits a few more
    transient glyph errors, still only when one legal action is the unique
    best fit.
    """

    if removed_count <= 0:
        return None
    observed = Counter(observed_cards)
    previous = Counter(state.self_hand.cards)
    target = Play.parse(state.trick_target.cards) if state.trick_target else None
    candidates: list[tuple[int, int, CardSet]] = []
    for play in legal_actions(state.self_hand, target, include_pass=False):
        if len(play.cards) != removed_count:
            continue
        remaining = previous - Counter(play.cards)
        # Both counters have the same total.  The positive difference counts
        # the number of rank substitutions needed to explain the observation.
        substitutions = sum((observed - remaining).values())
        remaining_cards = CardSet.parse(
            rank
            for rank, count in remaining.items()
            for _ in range(count)
        ).cards
        ordered_mismatches = min(
            sum(
                actual != expected
                for actual, expected in zip(
                    observed_cards,
                    expected_order,
                    strict=True,
                )
            )
            for expected_order in (
                remaining_cards,
                tuple(reversed(remaining_cards)),
            )
        )
        candidates.append((
            substitutions,
            ordered_mismatches,
            CardSet.parse(play.cards),
        ))
    if not candidates:
        return None
    candidates = list({
        cards.cards: (substitutions, ordered, cards)
        for substitutions, ordered, cards in candidates
    }.values())
    candidates.sort(key=lambda item: (item[0], item[1], item[2].cards))
    best_score = candidates[0][:2]
    best = [cards for error, ordered, cards in candidates if (error, ordered) == best_score]
    if (
        best_score[0] > max_corrections
        or best_score[1] > max_corrections
        or len(best) != 1
    ):
        return None
    return best[0]


def _invalid_play_fingerprint(
    observation: SeatObservation, confidence_threshold: float,
) -> tuple[object, ...]:
    return (
        observation.seat,
        observation.signal,
        tuple(sorted(card.rank for card in observation.cards)),
        observation.remaining_count,
        observation.remaining_verified,
        0.0 <= observation.remaining_confidence <= 1.0
        and observation.remaining_confidence >= confidence_threshold,
    )


def _usable_play_evidence(observation: SeatObservation, threshold: float) -> bool:
    if not observation.cards or observation.signal not in (VisualSignal.PLAY, VisualSignal.UNKNOWN):
        return False
    scores = (observation.confidence, *(card.confidence for card in observation.cards))
    if any(not 0.0 <= score <= 1.0 for score in scores):
        return False
    if observation.signal is VisualSignal.UNKNOWN:
        return (
            min(scores) >= max(0.55, threshold - 0.05)
            and observation.remaining_verified
            and 0.0 <= observation.remaining_confidence <= 1.0
            and observation.remaining_confidence >= threshold
        )
    return min(card.confidence for card in observation.cards) >= threshold


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


def _buffered_play_evidence(
    state: ObservableGameState,
    seat: PlayerSeat,
    scenes: Sequence[SceneObservation],
    *,
    confidence_threshold: float,
    min_frame_id: int | None = None,
) -> tuple[SceneObservation, SeatObservation] | None:
    """Recover a short-lived play captured beside the preceding action.

    The game may advance twice between processed frames.  A later actor's
    cards can therefore be visible in the same image in which the reducer is
    still committing the preceding actor.  The verified post-play counter is
    monotonic and makes such a cached observation unambiguous once that actor
    becomes current.
    """

    if seat is PlayerSeat.SELF:
        return None
    for scene in reversed(scenes):
        if min_frame_id is not None and scene.frame_id < min_frame_id:
            continue
        observation = scene.seat(seat)
        if _remaining_confirms_play(state, observation):
            return scene, observation
        if _terminal_play_confirms_round(state, observation):
            return scene, observation
        if _verified_low_confidence_play(
            state,
            observation,
            confidence_threshold=confidence_threshold,
        ) is not None:
            return scene, observation
    return None


def _buffered_pass_evidence(
    state: ObservableGameState,
    seat: PlayerSeat,
    scenes: Sequence[SceneObservation],
    *,
    confidence_threshold: float,
    min_frame_id: int | None = None,
) -> SceneObservation | None:
    previous: SeatObservation | None = None
    for scene in scenes:
        observation = scene.seat(seat)
        if (
            (min_frame_id is None or scene.frame_id >= min_frame_id)
            and previous is not None
            and previous.signal is VisualSignal.NEUTRAL
            and observation.signal is VisualSignal.PASS
            and observation.pass_confidence >= confidence_threshold
            and observation.remaining_verified
            and observation.remaining_count == state.remaining_for(seat)
        ):
            return scene
        previous = observation
    return None


def _action_stability_frames(
    state: ObservableGameState,
    observation: SeatObservation,
    *,
    armed: bool,
    default: int,
) -> int:
    """Use one frame only when the action has an independent visual proof."""

    if _remaining_confirms_play(state, observation):
        return 1
    if armed and _terminal_play_confirms_round(state, observation):
        return 1
    if (
        armed
        and observation.signal is VisualSignal.PASS
        and observation.remaining_verified
        and observation.remaining_count == state.remaining_for(observation.seat)
    ):
        return min(2, default)
    return default


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


def _remaining_proves_play(
    state: ObservableGameState,
    observation: SeatObservation,
    *,
    confidence_threshold: float,
) -> bool:
    return bool(
        _remaining_confirms_play(state, observation)
        or _terminal_play_confirms_round(state, observation)
        or _verified_low_confidence_play(
            state,
            observation,
            confidence_threshold=confidence_threshold,
        ) is not None
    )


def _terminal_play_confirms_round(
    state: ObservableGameState,
    observation: SeatObservation,
) -> bool:
    """Accept the visible last hand when the counter disappears at zero.

    The client removes an opponent's yellow counter at the same moment as the
    final card animation.  The previously tracked count is still exact: if the
    player had N cards and the newly visible legal group contains N cards, the
    action itself proves the zero transition.  Requiring an unavailable or
    unverified post-action counter prevents this path from weakening ordinary
    mid-round counter checks.
    """

    return bool(
        observation.seat is not PlayerSeat.SELF
        and observation.signal is VisualSignal.PLAY
        and observation.cards
        and state.remaining_for(observation.seat) == len(observation.cards)
        and (
            observation.remaining_count is None
            or not observation.remaining_verified
        )
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


def _continues_following_play_pass(state: ObservableGameState) -> bool:
    return bool(
        state.current_actor is PlayerSeat.SELF
        and state.consecutive_passes == 1
        and state.history
        and state.history[-1].source == _FOLLOWING_PLAY_PASS_SOURCE
    )


def _following_play_pass_evidence(
    state: ObservableGameState,
    scenes: Sequence[SceneObservation],
    *,
    confidence_threshold: float,
    stability_frames: int,
    min_frame_id: int | None,
    terminal_armed: bool,
    evidence_start_frame_id: int | None = None,
) -> tuple[SceneObservation, float] | None:
    """Prove the missing opponent -> self passes from a new leader play.

    Use only a fresh, contiguous suffix after the actor/geometry watermark.
    Preflight the entire chain before emitting its first event; never borrow
    a control hint, stale PASS label or a state-derived visual zero.
    """
    continuing = _continues_following_play_pass(state)
    if (
        state.phase is not RoundPhase.PLAYING
        or not state.trick_target
        or state.trick_leader in {None, PlayerSeat.SELF}
        or not (
            (state.consecutive_passes == 0
             and state.current_actor is not PlayerSeat.SELF
             and state.next_player(state.current_actor) is PlayerSeat.SELF)
            or continuing
        )
    ):
        return None
    if continuing and (
        state.history[-1].sequence_no != state.last_sequence_no
        or state.history[-1].event_id
        != f"{state.round_id}:{state.last_sequence_no}:{state.history[-1].actor.value}"
    ):
        return None
    following = state.trick_leader
    passing_opponent = state.next_player(following)
    if state.next_player(passing_opponent) is not PlayerSeat.SELF:
        return None
    recent = list(scenes)[-stability_frames:]
    if len(recent) < stability_frames:
        return None
    if continuing and min_frame_id is not None and recent[-1].frame_id <= min_frame_id:
        return None
    # The legacy armed bit accepts any neutral signal. This stricter terminal
    # proof needs an actually trusted blank/counter before the new card group.
    terminal_blank_confidence = max((
        min(item.seat(following).confidence, item.seat(following).remaining_confidence)
        for item in scenes
        if item.frame_id < recent[0].frame_id
        and (evidence_start_frame_id is None or item.frame_id >= evidence_start_frame_id)
        and item.seat(following).signal is VisualSignal.NEUTRAL
        and not item.seat(following).cards
        and confidence_threshold <= item.seat(following).confidence <= 1.0
        and item.seat(following).remaining_verified
        and item.seat(following).remaining_count == state.remaining_for(following)
        and confidence_threshold <= item.seat(following).remaining_confidence <= 1.0
    ), default=0.0)
    confidence = 1.0
    previous_frame: int | None = None
    candidate: CardSet | None = None
    for scene in recent:
        if (
            (min_frame_id is not None and scene.frame_id < min_frame_id)
            or (evidence_start_frame_id is not None and scene.frame_id < evidence_start_frame_id)
            or (previous_frame is not None and scene.frame_id <= previous_frame)
            or scene.self_turn is True
            or scene.round_result is not None
            or any(warning.startswith("self_hand_geometry_unavailable:") for warning in scene.warnings)
            or not all(confidence_threshold <= seat.role_confidence <= 1.0 for seat in scene.seats)
            or _observed_landlord(scene, confidence_threshold=confidence_threshold) is not state.landlord
            or not all(0.0 <= card.confidence <= 1.0 for card in scene.self_hand)
            or not scene.seat(PlayerSeat.SELF).remaining_verified
            or scene.seat(PlayerSeat.SELF).remaining_count != len(state.self_hand)
        ):
            return None
        previous_frame = scene.frame_id
        hand_confidence, _ = _stable_hand_confidence(
            scene.self_hand, confidence_threshold=confidence_threshold,
        )
        try:
            if scene.self_hand_set != state.self_hand or hand_confidence < confidence_threshold:
                return None
            play = scene.seat(following)
            cards = play.card_set
        except ValueError:
            return None
        opponent = scene.seat(passing_opponent)
        if (
            opponent.signal not in {VisualSignal.NEUTRAL, VisualSignal.PASS}
            or opponent.cards
            or not opponent.remaining_verified
            or opponent.remaining_count != state.remaining_for(passing_opponent)
            or not confidence_threshold <= opponent.remaining_confidence <= 1.0
            or not confidence_threshold <= _action_confidence(opponent) <= 1.0
            or scene.seat(PlayerSeat.SELF).cards
            or play.signal is not VisualSignal.PLAY
            or not cards
            or not all(confidence_threshold <= card.confidence <= 1.0 for card in play.cards)
            or (candidate is not None and cards != candidate)
        ):
            return None
        count_proven = bool(
            _remaining_confirms_play(state, play)
            and confidence_threshold <= play.remaining_confidence <= 1.0
        )
        terminal_proven = bool(
            terminal_armed and terminal_blank_confidence >= confidence_threshold
            and _terminal_play_confirms_round(state, play)
        )
        if not count_proven and not terminal_proven:
            return None
        candidate = cards
        confidence = min(confidence, hand_confidence, opponent.remaining_confidence,
                         _action_confidence(opponent), _action_confidence(play),
                         play.remaining_confidence if count_proven else terminal_blank_confidence,
                         *(seat.role_confidence for seat in scene.seats))
    assert candidate is not None
    preview = state
    # Reuse the real reducer for sequence, deck conservation and legal lead.
    # No mutable tracker/pending queue is involved in this preflight.
    for cards in ([CardSet(())] if continuing else [CardSet(()), CardSet(())]) + [candidate]:
        event = ObservedAction(
            event_id=f"{preview.round_id}:{preview.last_sequence_no + 1}:{preview.current_actor.value}",
            sequence_no=preview.last_sequence_no + 1, actor=preview.current_actor,
            cards=cards, confidence=confidence, source=_FOLLOWING_PLAY_PASS_SOURCE,
        )
        try:
            result = reduce_observed_action(
                preview, event, validator=validate_observed_action,
                confidence_threshold=confidence_threshold,
            )
        except ValueError:
            return None
        if result.status is not StateUpdateStatus.APPLIED:
            return None
        preview = result.state
    # Advancing the actor at the actual commit frame prevents a replayed
    # pre-commit suffix from authorizing the second pass a second time.
    return recent[-1], confidence


def _can_infer_pass(
    state: ObservableGameState,
    scene: SceneObservation,
    observation: SeatObservation,
    *,
    confidence_threshold: float,
    stable_count: int,
    stability_frames: int,
    successor_action_armed: bool = False,
    buffered_successor_play: bool = False,
    buffered_successor_pass: bool = False,
    self_turn_was_active: bool = False,
    fresh_self_pass: bool = False,
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
    if not state.trick_target:
        return False
    if (
        observation.seat is not PlayerSeat.SELF
        and (
            not observation.remaining_verified
            or observation.remaining_count
            != state.remaining_for(observation.seat)
            or observation.remaining_confidence < confidence_threshold
        )
    ):
        return False
    if observation.seat is PlayerSeat.SELF:
        successor = (
            state.trick_leader
            if state.consecutive_passes == 1
            else state.next_player(PlayerSeat.SELF)
        )
        successor_play_proves_pass = bool(
            successor is not None
            and successor is not PlayerSeat.SELF
            and (
                _remaining_proves_play(
                    state,
                    scene.seat(successor),
                    confidence_threshold=confidence_threshold,
                )
                or (
                    state.consecutive_passes == 0
                    and scene.seat(successor).remaining_verified
                    and scene.seat(successor).remaining_count
                    == state.remaining_for(successor)
                    and _remaining_proves_play(
                        state,
                        scene.seat(state.next_player(successor)),
                        confidence_threshold=confidence_threshold,
                    )
                )
            )
        )
        successor_observation = (
            scene.seat(successor)
            if successor is not None and successor is not PlayerSeat.SELF
            else None
        )
        successor_pass_proves_pass = bool(
            state.consecutive_passes == 0
            and (successor_action_armed or buffered_successor_pass)
            and successor_observation is not None
            and (
                buffered_successor_pass
                or (
                    successor_observation.signal is VisualSignal.PASS
                    and successor_observation.pass_confidence
                    >= confidence_threshold
                    and successor_observation.remaining_verified
                    and successor_observation.remaining_count
                    == state.remaining_for(successor_observation.seat)
                )
            )
        )
        exact_hand = False
        try:
            exact_hand = scene.self_hand_set == state.self_hand
            same_physical_count = len(scene.self_hand) == len(state.self_hand)
            if not exact_hand and not (
                (
                    same_physical_count
                    and successor_play_proves_pass
                )
                or successor_pass_proves_pass
            ):
                return False
        except ValueError:
            if not (
                (
                    len(scene.self_hand) == len(state.self_hand)
                    and successor_play_proves_pass
                )
                or successor_pass_proves_pass
            ):
                return False
        hand_confidence, _ = _stable_hand_confidence(
            scene.self_hand, confidence_threshold=confidence_threshold,
        )
        # Continuous absence does not prove disappearance. Require controls
        # observed in this actor epoch (or a fresh explicit self-pass edge),
        # plus a stable inactive hint and a trustworthy unchanged whole hand.
        # A strict successor action remains independent corroborating evidence.
        if (
            scene.self_turn is False
            and scene.self_turn_confidence >= confidence_threshold
            and (
                (
                    stable_count >= stability_frames
                    and (self_turn_was_active or fresh_self_pass)
                    and exact_hand
                    and hand_confidence >= confidence_threshold
                )
                or successor_play_proves_pass
                or successor_pass_proves_pass
            )
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
    if buffered_successor_play:
        return True
    if buffered_successor_pass and state.consecutive_passes == 0:
        return True
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


_TERMINAL_PASS_SOURCE = "live_terminal_inferred_pass"


def _terminal_pass_chain(
    scene: SceneObservation,
    state: ObservableGameState,
    *,
    confidence_threshold: float,
) -> tuple[ObservedAction, ObservableGameState] | None:
    """Pass events for the blank, unchanged opponents owed after our lead.

    Returns the first pass and the state previewed through every pass up to
    the point where SELF is current again, or None when any owed seat shows
    cards, an unverified or changed counter, or the reducer rejects a pass.
    """
    if (
        state.phase is not RoundPhase.PLAYING
        or state.trick_leader is not PlayerSeat.SELF
        or not state.trick_target.cards
        or state.current_actor is PlayerSeat.SELF
    ):
        return None
    previewed = state
    first: ObservedAction | None = None
    for _ in range(len(PlayerSeat) - 1):
        actor = previewed.current_actor
        if actor is PlayerSeat.SELF:
            break
        observation = scene.seat(actor)
        if (
            observation.signal not in {VisualSignal.NEUTRAL, VisualSignal.PASS}
            or observation.cards
            or not observation.remaining_verified
            or observation.remaining_count != previewed.remaining_for(actor)
        ):
            return None
        confidence = max(
            observation.pass_confidence,
            min(observation.confidence, observation.remaining_confidence),
        )
        if not 0.0 <= confidence <= 1.0:
            return None
        event = ObservedAction(
            event_id=f"{previewed.round_id}:{previewed.last_sequence_no + 1}:{actor.value}",
            sequence_no=previewed.last_sequence_no + 1,
            actor=actor,
            cards=CardSet(()),
            confidence=confidence,
            source=_TERMINAL_PASS_SOURCE,
        )
        try:
            preview = reduce_observed_action(
                previewed, event, validator=validate_observed_action,
                confidence_threshold=confidence_threshold,
            )
        except GameStateTransitionError:
            return None
        if preview.status is not StateUpdateStatus.APPLIED:
            return None
        first = first or event
        previewed = preview.state
    if first is None or previewed.current_actor is not PlayerSeat.SELF:
        return None
    return first, previewed


def _inferred_pass_confidence(
    scene: SceneObservation,
    observation: SeatObservation,
) -> float:
    if observation.seat is PlayerSeat.SELF:
        # A hand-rank outlier must not lower a pass that is independently
        # proved by the turn returning to us or by a verified opponent action.
        # The physical hand-count equality was already checked by
        # _can_infer_pass.
        proof_confidences: list[float] = []
        if scene.self_turn is not None:
            proof_confidences.append(scene.self_turn_confidence)
        for seat in scene.seats:
            if seat.seat is PlayerSeat.SELF:
                continue
            if seat.signal in {VisualSignal.PLAY, VisualSignal.PASS}:
                proof_confidences.append(
                    min(
                        _action_confidence(seat),
                        seat.remaining_confidence,
                    )
                )
        if proof_confidences:
            return max(proof_confidences)
    confidences = [observation.remaining_confidence]
    if scene.self_turn is True:
        confidences.append(scene.self_turn_confidence)
    return min(value for value in confidences if value > 0)




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
