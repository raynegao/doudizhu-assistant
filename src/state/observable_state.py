from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from .cards import FULL_DECK, CardParseError, CardSet, parse_cards, sort_cards, validate_card_counts
from .events import DEFAULT_TURN_ORDER, ObservedAction, PlayerSeat, RoundPhase
from .game_state import GameStateSnapshot


def _normalize_remaining(
    hand: CardSet,
    landlord: PlayerSeat,
    values: Mapping[PlayerSeat | str, int] | None,
) -> tuple[tuple[PlayerSeat, int], ...]:
    defaults = {
        seat: (20 if seat is landlord else 17)
        for seat in PlayerSeat
    }
    defaults[PlayerSeat.SELF] = len(hand)
    if values:
        for seat, count in values.items():
            defaults[PlayerSeat(seat)] = int(count)
    return tuple((seat, defaults[seat]) for seat in PlayerSeat)


def _append_missing_cards(base: tuple[str, ...], required: tuple[str, ...]) -> tuple[str, ...]:
    missing = Counter(required) - Counter(base)
    additions = tuple(card for rank in missing for card in [rank] * missing[rank])
    return sort_cards((*base, *additions))


@dataclass(frozen=True)
class ObservableGameState:
    round_id: str
    revision: int
    phase: RoundPhase
    landlord: PlayerSeat
    turn_order: tuple[PlayerSeat, ...]
    self_hand: CardSet
    remaining_cards: tuple[tuple[PlayerSeat, int], ...]
    current_actor: PlayerSeat
    trick_target: CardSet = CardSet(())
    trick_leader: PlayerSeat | None = None
    consecutive_passes: int = 0
    played_cards: tuple[str, ...] = ()
    history: tuple[ObservedAction, ...] = ()
    processed_events: tuple[tuple[str, str], ...] = ()
    last_sequence_no: int = 0
    state_confidence: float = 1.0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.turn_order) != 3 or set(self.turn_order) != set(PlayerSeat):
            raise ValueError("turn_order must contain self, left, and right exactly once")
        if self.current_actor not in self.turn_order:
            raise ValueError("current_actor must be present in turn_order")
        if not 0.0 <= self.state_confidence <= 1.0:
            raise ValueError("state_confidence must be between 0 and 1")
        if self.consecutive_passes not in {0, 1}:
            raise ValueError("consecutive_passes must be 0 or 1 in a valid active trick")
        if self.trick_target and self.trick_leader is None:
            raise ValueError("trick_leader is required when trick_target is not pass")
        if not self.trick_target and self.consecutive_passes:
            raise ValueError("cannot retain passes without an active trick target")

        remaining = dict(self.remaining_cards)
        if set(remaining) != set(PlayerSeat):
            raise ValueError("remaining_cards must contain all three players")
        if any(count < 0 for count in remaining.values()):
            raise ValueError("remaining card counts cannot be negative")
        for player, count in remaining.items():
            maximum = 20 if player is self.landlord else 17
            if count > maximum:
                raise ValueError(
                    f"{player.value} remaining count exceeds role maximum {maximum}: {count}"
                )
        winners = [player for player, count in remaining.items() if count == 0]
        if len(winners) > 1:
            raise ValueError("at most one player can have zero remaining cards")
        if winners and self.phase is not RoundPhase.FINISHED:
            raise ValueError("a zero remaining count requires phase=finished")
        if self.phase is RoundPhase.FINISHED and not winners:
            raise ValueError("phase=finished requires exactly one winner")
        if remaining[PlayerSeat.SELF] != len(self.self_hand):
            raise ValueError("self remaining count must equal current hand size")

        validate_card_counts((*self.self_hand.cards, *self.played_cards))
        if self.trick_target and not (
            Counter(self.trick_target.cards) <= Counter(self.played_cards)
        ):
            raise ValueError("trick_target must already be included in played_cards")
        if sum(remaining.values()) + len(self.played_cards) > len(FULL_DECK):
            raise ValueError("remaining cards plus played cards exceed the 54-card deck")
        if not self.trick_target and self.trick_leader is not None:
            raise ValueError("trick_leader must be empty when there is no trick target")

    @classmethod
    def from_inputs(
        cls,
        hand: str | Iterable[str],
        *,
        round_id: str = "manual-round",
        landlord: PlayerSeat | str = PlayerSeat.SELF,
        current_actor: PlayerSeat | str = PlayerSeat.SELF,
        turn_order: Iterable[PlayerSeat | str] = DEFAULT_TURN_ORDER,
        remaining_cards: Mapping[PlayerSeat | str, int] | None = None,
        played_cards: str | Iterable[str] | None = None,
        last_play: str | Iterable[str] | None = None,
        last_player: PlayerSeat | str | None = None,
        consecutive_passes: int = 0,
        state_confidence: float = 1.0,
    ) -> "ObservableGameState":
        hand_set = CardSet.parse(hand)
        target = CardSet.parse(last_play)
        known_played = parse_cards(played_cards)
        if target:
            known_played = _append_missing_cards(known_played, target.cards)
        validate_card_counts((*hand_set.cards, *known_played))

        landlord_seat = PlayerSeat(landlord)
        remaining = _normalize_remaining(hand_set, landlord_seat, remaining_cards)
        warnings: list[str] = []
        accounted = sum(dict(remaining).values()) + len(known_played)
        if accounted != len(FULL_DECK):
            warnings.append(
                f"incomplete deck accounting: remaining+played={accounted}, expected={len(FULL_DECK)}"
            )

        winners = [seat for seat, count in remaining if count == 0]
        if len(winners) > 1:
            raise ValueError("at most one player can have zero remaining cards")
        phase = RoundPhase.FINISHED if winners else RoundPhase.PLAYING
        actor = winners[0] if winners else PlayerSeat(current_actor)
        return cls(
            round_id=round_id,
            revision=0,
            phase=phase,
            landlord=landlord_seat,
            turn_order=tuple(PlayerSeat(seat) for seat in turn_order),
            self_hand=hand_set,
            remaining_cards=remaining,
            current_actor=actor,
            trick_target=target,
            trick_leader=PlayerSeat(last_player) if last_player is not None else None,
            consecutive_passes=consecutive_passes,
            played_cards=known_played,
            state_confidence=state_confidence,
            warnings=tuple(warnings),
        )

    @property
    def remaining_by_player(self) -> dict[PlayerSeat, int]:
        return dict(self.remaining_cards)

    def remaining_for(self, player: PlayerSeat) -> int:
        return self.remaining_by_player[player]

    @property
    def unknown_cards(self) -> CardSet:
        pool = Counter(FULL_DECK)
        pool.subtract(self.self_hand.cards)
        pool.subtract(self.played_cards)
        if any(count < 0 for count in pool.values()):
            raise CardParseError("known cards exceed the physical deck")
        cards = tuple(card for rank in pool for card in [rank] * pool[rank])
        return CardSet(sort_cards(cards))

    @property
    def decision_ready(self) -> bool:
        opponent_count = sum(
            count
            for player, count in self.remaining_cards
            if player is not PlayerSeat.SELF
        )
        return (
            self.phase is RoundPhase.PLAYING
            and self.winner is None
            and self.current_actor is PlayerSeat.SELF
            and self.state_confidence >= 0.70
            and len(self.unknown_cards) == opponent_count
            and self._turn_is_consistent()
        )

    def _turn_is_consistent(self) -> bool:
        if not self.trick_target:
            return self.trick_leader is None and self.consecutive_passes == 0
        if self.trick_leader is None:
            return False
        expected = self.next_player(self.trick_leader)
        for _ in range(self.consecutive_passes):
            expected = self.next_player(expected)
        return self.current_actor is expected

    @property
    def winner(self) -> PlayerSeat | None:
        for player, count in self.remaining_cards:
            if count == 0:
                return player
        return None

    def next_player(self, player: PlayerSeat) -> PlayerSeat:
        index = self.turn_order.index(player)
        return self.turn_order[(index + 1) % len(self.turn_order)]

    def to_decision_snapshot(self) -> GameStateSnapshot:
        return GameStateSnapshot(hand=self.self_hand, last_play=self.trick_target)

    def to_log_payload(self) -> dict[str, object]:
        return {
            "round_id": self.round_id,
            "revision": self.revision,
            "phase": self.phase.value,
            "landlord": self.landlord.value,
            "turn_order": [seat.value for seat in self.turn_order],
            "current_actor": self.current_actor.value,
            "self_hand": self.self_hand.to_list(),
            "remaining_cards": {
                player.value: count for player, count in self.remaining_cards
            },
            "trick_target": self.trick_target.to_list(),
            "trick_leader": self.trick_leader.value if self.trick_leader else None,
            "consecutive_passes": self.consecutive_passes,
            "played_cards": list(self.played_cards),
            "unknown_card_count": len(self.unknown_cards),
            "state_confidence": self.state_confidence,
            "decision_ready": self.decision_ready,
            "warnings": list(self.warnings),
        }


__all__ = ["ObservableGameState"]
