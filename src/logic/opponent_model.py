from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

from src.logic.rules import Play, legal_actions
from src.state.cards import RANKS, CardSet, sort_cards
from src.state.events import PlayerSeat
from src.state.observable_state import ObservableGameState


class OpponentModelError(ValueError):
    pass


@dataclass(frozen=True)
class OpponentDeal:
    hands: tuple[tuple[PlayerSeat, CardSet], ...]

    def hand_for(self, player: PlayerSeat) -> CardSet:
        for seat, hand in self.hands:
            if seat is player:
                return hand
        raise KeyError(player)

    def to_log_payload(self) -> dict[str, list[str]]:
        return {seat.value: hand.to_list() for seat, hand in self.hands}


@dataclass(frozen=True)
class OpponentEstimate:
    sample_count: int
    rank_presence_probability: tuple[
        tuple[PlayerSeat, tuple[tuple[str, float], ...]], ...
    ]
    bomb_probability: tuple[tuple[PlayerSeat, float], ...]
    rocket_probability: tuple[tuple[PlayerSeat, float], ...]
    can_beat_probability: tuple[tuple[PlayerSeat, float], ...]
    model_version: str = "uniform-remaining-cards-v1"

    def to_log_payload(self) -> dict[str, object]:
        return {
            "model_version": self.model_version,
            "sample_count": self.sample_count,
            "rank_presence_probability": {
                seat.value: {rank: probability for rank, probability in values}
                for seat, values in self.rank_presence_probability
            },
            "bomb_probability": {
                seat.value: probability for seat, probability in self.bomb_probability
            },
            "rocket_probability": {
                seat.value: probability for seat, probability in self.rocket_probability
            },
            "can_beat_probability": {
                seat.value: probability for seat, probability in self.can_beat_probability
            },
        }


class UniformOpponentModel:
    model_version = "uniform-remaining-cards-v1"

    def validate_state(self, state: ObservableGameState) -> None:
        if not state.decision_ready:
            raise OpponentModelError("observable state is not decision-ready")
        opponents = [seat for seat in state.turn_order if seat is not PlayerSeat.SELF]
        required = sum(state.remaining_for(seat) for seat in opponents)
        available = len(state.unknown_cards)
        if available != required:
            raise OpponentModelError(
                f"unknown card count mismatch: available={available}, opponent_remaining={required}"
            )

    def sample(self, state: ObservableGameState, rng: random.Random) -> OpponentDeal:
        self.validate_state(state)
        pool = list(state.unknown_cards.cards)
        rng.shuffle(pool)
        offset = 0
        hands: list[tuple[PlayerSeat, CardSet]] = []
        for seat in state.turn_order:
            if seat is PlayerSeat.SELF:
                continue
            count = state.remaining_for(seat)
            cards = CardSet(sort_cards(pool[offset : offset + count]))
            hands.append((seat, cards))
            offset += count
        if offset != len(pool):
            raise OpponentModelError("opponent deal did not consume the full unknown card pool")
        return OpponentDeal(tuple(hands))

    def sample_many(
        self,
        state: ObservableGameState,
        *,
        count: int,
        seed: int,
    ) -> tuple[OpponentDeal, ...]:
        if count <= 0:
            raise ValueError("sample count must be positive")
        rng = random.Random(seed)
        return tuple(self.sample(state, rng) for _ in range(count))

    def summarize(
        self,
        state: ObservableGameState,
        deals: tuple[OpponentDeal, ...],
    ) -> OpponentEstimate:
        if not deals:
            raise ValueError("at least one opponent deal is required")
        previous = Play.parse(state.trick_target.cards)
        seats = tuple(seat for seat in state.turn_order if seat is not PlayerSeat.SELF)
        rank_probabilities: list[tuple[PlayerSeat, tuple[tuple[str, float], ...]]] = []
        bomb_probabilities: list[tuple[PlayerSeat, float]] = []
        rocket_probabilities: list[tuple[PlayerSeat, float]] = []
        beat_probabilities: list[tuple[PlayerSeat, float]] = []

        for seat in seats:
            hands = [deal.hand_for(seat) for deal in deals]
            rank_probabilities.append(
                (
                    seat,
                    tuple(
                        (
                            rank,
                            sum(rank in hand.counts for hand in hands) / len(hands),
                        )
                        for rank in RANKS
                    ),
                )
            )
            bomb_probabilities.append(
                (
                    seat,
                    sum(any(count == 4 for count in hand.counts.values()) for hand in hands)
                    / len(hands),
                )
            )
            rocket_probabilities.append(
                (
                    seat,
                    sum({"SJ", "BJ"} <= set(hand.cards) for hand in hands) / len(hands),
                )
            )
            beat_probabilities.append(
                (
                    seat,
                    sum(
                        any(not action.is_pass for action in legal_actions(hand, previous))
                        for hand in hands
                    )
                    / len(hands),
                )
            )

        return OpponentEstimate(
            sample_count=len(deals),
            rank_presence_probability=tuple(rank_probabilities),
            bomb_probability=tuple(bomb_probabilities),
            rocket_probability=tuple(rocket_probabilities),
            can_beat_probability=tuple(beat_probabilities),
        )


__all__ = [
    "OpponentDeal",
    "OpponentEstimate",
    "OpponentModelError",
    "UniformOpponentModel",
]
