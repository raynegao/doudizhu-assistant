from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping

from .cards import CardSet


class PlayerSeat(str, Enum):
    SELF = "self"
    LEFT = "left"
    RIGHT = "right"


class RoundPhase(str, Enum):
    PLAYING = "playing"
    FINISHED = "finished"
    UNCERTAIN = "uncertain"


DEFAULT_TURN_ORDER: tuple[PlayerSeat, ...] = (
    PlayerSeat.SELF,
    PlayerSeat.RIGHT,
    PlayerSeat.LEFT,
)


@dataclass(frozen=True)
class ObservedAction:
    event_id: str
    sequence_no: int
    actor: PlayerSeat
    cards: CardSet
    confidence: float = 1.0
    source: str = "manual"

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id cannot be empty")
        if self.sequence_no <= 0:
            raise ValueError("sequence_no must be positive")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be a finite value between 0 and 1")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ObservedAction":
        try:
            event_type = str(payload.get("event", "play_observed"))
            if event_type not in {"play_observed", "pass_observed"}:
                raise ValueError(f"unsupported game event: {event_type}")
            event_id = payload["event_id"]
            actor = payload["actor"]
            if not isinstance(event_id, str):
                raise TypeError("event_id must be a string")
            if not isinstance(actor, str):
                raise TypeError("actor must be a string")
            raw_cards = payload.get("cards", ())
            if not isinstance(raw_cards, (str, list, tuple)):
                raise TypeError("cards must be a string or array")
            cards = CardSet.parse(raw_cards)
            if event_type == "play_observed" and not cards:
                raise ValueError("play_observed requires at least one card")
            if event_type == "pass_observed" and cards:
                raise ValueError("pass_observed cannot contain cards")
            return cls(
                event_id=event_id,
                sequence_no=int(payload["sequence_no"]),
                actor=PlayerSeat(actor),
                cards=cards,
                confidence=float(payload.get("confidence", 1.0)),
                source=str(payload.get("source", "manual")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid observed action payload: {exc}") from exc

    @property
    def is_pass(self) -> bool:
        return not self.cards

    @property
    def fingerprint(self) -> str:
        return repr(
            (
                self.sequence_no,
                self.actor.value,
                self.cards.cards,
                round(self.confidence, 6),
                self.source,
            )
        )

    def to_log_payload(self) -> dict[str, object]:
        return {
            "event": "pass_observed" if self.is_pass else "play_observed",
            "event_id": self.event_id,
            "sequence_no": self.sequence_no,
            "actor": self.actor.value,
            "cards": self.cards.to_list(),
            "confidence": self.confidence,
            "source": self.source,
        }


__all__ = [
    "DEFAULT_TURN_ORDER",
    "ObservedAction",
    "PlayerSeat",
    "RoundPhase",
]
