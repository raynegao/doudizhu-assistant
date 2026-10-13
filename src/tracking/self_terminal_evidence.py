"""Strict, state-bound evidence for an independently visible self last play."""

from __future__ import annotations

import math

from src.logic.action_validation import validate_observed_action
from src.state.cards import CardSet
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.game_tracker import StateUpdateStatus, reduce_observed_action
from src.state.observable_state import ObservableGameState
from src.vision.scene_types import SceneObservation, VisualCard


class SelfTerminalEvidenceGate:
    """Accumulate consecutive observations, never manufacture missing card slots.

    ``epoch`` is a minimum source frame ID, not a counter. Resetting interrupts
    support but preserves source/epoch watermarks and the last emitted revision.
    A genuinely restarted frame stream therefore needs a new gate instance.
    """

    def __init__(self) -> None:
        self._last_frame_id = -1
        self._epoch_floor: int | None = None
        self._key: tuple[object, ...] | None = None
        self._support = 0
        self._confidence = 1.0
        self._emitted_revision: tuple[str, int] | None = None

    def reset(self) -> None:
        """Forget candidate support, without making old evidence fresh again."""
        self._key = None
        self._support = 0
        self._confidence = 1.0

    def discard_through(self, frame_id: int) -> None:
        """Pause paths must also reject later re-delivery of the skipped frame."""
        if isinstance(frame_id, int) and not isinstance(frame_id, bool):
            self._last_frame_id = max(self._last_frame_id, frame_id)
        self.reset()

    def observe(
        self,
        *,
        state: ObservableGameState,
        scene: SceneObservation,
        cards: tuple[VisualCard, ...],
        complete: bool,
        empty_hand_verified: bool,
        epoch: int | None,
        stability_frames: int,
        confidence_threshold: float,
    ) -> ObservedAction | None:
        if (
            isinstance(stability_frames, bool)
            or not isinstance(stability_frames, int)
            or stability_frames < 1
            or isinstance(confidence_threshold, bool)
            or not math.isfinite(confidence_threshold)
            or not 0.0 <= confidence_threshold <= 1.0
            or (epoch is not None and (
                isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
            ))
        ):
            self.reset()
            raise ValueError("invalid self-terminal stability, confidence, or epoch setting")

        frame_id = scene.frame_id
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            self.reset()
            return None
        fresh_frame = frame_id > self._last_frame_id
        self._last_frame_id = max(self._last_frame_id, frame_id)
        stale_epoch = self._epoch_floor is not None and (
            epoch is None or epoch < self._epoch_floor
        )
        if epoch is not None:
            self._epoch_floor = max(self._epoch_floor or 0, epoch)
        if (
            not fresh_frame
            or stale_epoch
            or (self._epoch_floor is not None and frame_id < self._epoch_floor)
        ):
            self.reset()
            return None

        if (
            state.phase is not RoundPhase.PLAYING
            or state.current_actor is not PlayerSeat.SELF
            or not state._turn_is_consistent()
            or not state.self_hand
            or scene.self_hand
            or complete is not True
            or empty_hand_verified is not True
            or scene.self_turn is True
            or not cards
            or any(warning.startswith("self_hand_geometry_unavailable:")
                   for warning in scene.warnings)
            or any(
                isinstance(card.confidence, bool)
                or not math.isfinite(card.confidence)
                or not confidence_threshold <= card.confidence <= 1.0
                for card in cards
            )
            or self._emitted_revision == (state.round_id, state.revision)
        ):
            self.reset()
            return None
        try:
            card_set = CardSet.parse(card.rank for card in cards)
        except ValueError:
            self.reset()
            return None
        if card_set != state.self_hand:
            self.reset()
            return None

        # Keep slot order and geometry; sorting only for multiset equality would
        # otherwise let a swapped/unstable layout borrow previous frame support.
        key = (
            state.round_id, state.revision, state.current_actor, epoch,
            state.self_hand.cards, state.last_sequence_no, state.trick_target.cards,
            state.trick_leader, state.consecutive_passes, state.turn_order,
            tuple((card.rank, card.box) for card in cards),
            complete, empty_hand_verified, scene.self_turn,
            stability_frames, confidence_threshold,
        )
        if key != self._key:
            self.reset()
            self._key = key
        self._confidence = min(self._confidence, *(card.confidence for card in cards))
        event = ObservedAction(
            event_id=f"{state.round_id}:{state.revision}:self-table-terminal:{frame_id}",
            sequence_no=state.last_sequence_no + 1,
            actor=PlayerSeat.SELF,
            cards=card_set,
            confidence=self._confidence,
            source="live_self_table_terminal",
        )
        try:
            preview = reduce_observed_action(
                state, event, validator=validate_observed_action,
                confidence_threshold=confidence_threshold,
            )
        except ValueError:
            self.reset()
            return None
        if (
            preview.status is not StateUpdateStatus.APPLIED
            or preview.state.phase is not RoundPhase.FINISHED
            or preview.state.winner is not PlayerSeat.SELF
        ):
            self.reset()
            return None
        self._support += 1
        if self._support < stability_frames:
            return None
        self._emitted_revision = (state.round_id, state.revision)
        self.reset()
        return event


__all__ = ["SelfTerminalEvidenceGate"]
