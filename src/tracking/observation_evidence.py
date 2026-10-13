"""Pure confidence and fingerprint helpers used by visual event tracking."""

from __future__ import annotations

from collections.abc import Sequence

from src.state.events import PlayerSeat
from src.vision.scene_types import SceneObservation, SeatObservation, VisualCard, VisualSignal


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


def _action_confidence(observation: SeatObservation) -> float:
    if observation.signal is VisualSignal.PASS:
        return observation.pass_confidence
    if observation.signal is VisualSignal.PLAY:
        return min(
            (card.confidence for card in observation.cards),
            default=observation.confidence,
        )
    return observation.confidence

