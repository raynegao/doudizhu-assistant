from __future__ import annotations

from dataclasses import dataclass

from .cards import CardSet


@dataclass(frozen=True)
class GameStateSnapshot:
    hand: CardSet
    last_play: CardSet = CardSet(())

    @classmethod
    def from_inputs(cls, hand: str, last_play: str | None = None) -> "GameStateSnapshot":
        return cls(hand=CardSet.parse(hand), last_play=CardSet.parse(last_play))

    def to_log_payload(self) -> dict[str, list[str]]:
        return {
            "input_cards": self.hand.to_list(),
            "last_play": self.last_play.to_list(),
        }


__all__ = ["GameStateSnapshot"]
