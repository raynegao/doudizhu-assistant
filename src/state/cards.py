from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable


RANKS: tuple[str, ...] = (
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "J",
    "Q",
    "K",
    "A",
    "2",
    "SJ",
    "BJ",
)

RANK_VALUE: dict[str, int] = {rank: index for index, rank in enumerate(RANKS)}
CHAIN_RANKS: tuple[str, ...] = RANKS[:12]
FULL_DECK: tuple[str, ...] = tuple(
    card
    for rank in RANKS
    for card in ([rank] if rank in {"SJ", "BJ"} else [rank] * 4)
)

_ALIASES: dict[str, str] = {
    "T": "10",
    "10": "10",
    "J": "J",
    "Q": "Q",
    "K": "K",
    "A": "A",
    "X": "SJ",
    "SJ": "SJ",
    "SMALLJOKER": "SJ",
    "SMALL": "SJ",
    "小王": "SJ",
    "D": "BJ",
    "BJ": "BJ",
    "RJ": "BJ",
    "BLACKJOKER": "BJ",
    "REDJOKER": "BJ",
    "BIGJOKER": "BJ",
    "BIG": "BJ",
    "大王": "BJ",
}

_TOKEN_RE = re.compile(
    r"小王|大王|BLACKJOKER|REDJOKER|SMALLJOKER|BIGJOKER|BJ|RJ|SJ|10|[2-9JQKATXD]",
    re.IGNORECASE,
)


class CardParseError(ValueError):
    pass


def normalize_rank(token: str) -> str:
    value = token.strip()
    if not value:
        raise CardParseError("empty card token")
    upper = value.upper()
    if upper in RANK_VALUE:
        return upper
    if value in _ALIASES:
        return _ALIASES[value]
    if upper in _ALIASES:
        return _ALIASES[upper]
    raise CardParseError(f"unknown card rank: {token}")


def sort_cards(cards: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted((normalize_rank(card) for card in cards), key=RANK_VALUE.__getitem__))


def parse_cards(text: str | Iterable[str] | None) -> tuple[str, ...]:
    if text is None:
        return ()
    if isinstance(text, str):
        stripped = text.strip()
        if not stripped:
            return ()
        if re.search(r"[\s,，;；]+", stripped):
            raw_tokens = [token for token in re.split(r"[\s,，;；]+", stripped) if token]
            return sort_cards(raw_tokens)
        matches = list(_TOKEN_RE.finditer(stripped))
        if not matches or "".join(match.group(0) for match in matches) != stripped:
            raise CardParseError(f"cannot parse cards: {text}")
        return sort_cards(match.group(0) for match in matches)
    return sort_cards(text)


def validate_card_counts(cards: Iterable[str]) -> None:
    counts = Counter(sort_cards(cards))
    for rank, count in counts.items():
        max_count = 1 if rank in {"SJ", "BJ"} else 4
        if count > max_count:
            raise CardParseError(f"too many {rank} cards: {count}")


@dataclass(frozen=True)
class CardSet:
    cards: tuple[str, ...]

    @classmethod
    def parse(cls, value: str | Iterable[str] | None) -> "CardSet":
        cards = parse_cards(value)
        validate_card_counts(cards)
        return cls(cards)

    @property
    def counts(self) -> Counter[str]:
        return Counter(self.cards)

    def to_list(self) -> list[str]:
        return list(self.cards)

    def __bool__(self) -> bool:
        return bool(self.cards)

    def __len__(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        return " ".join(self.cards) if self.cards else "pass"


__all__ = [
    "CHAIN_RANKS",
    "CardParseError",
    "CardSet",
    "FULL_DECK",
    "RANKS",
    "RANK_VALUE",
    "normalize_rank",
    "parse_cards",
    "sort_cards",
    "validate_card_counts",
]
