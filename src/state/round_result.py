"""Pure result semantics; a farmer's teammate winning is also a self victory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SEATS = {"self", "right", "left"}


@dataclass(frozen=True)
class RoundResultFacts:
    landlord: str | None
    winner: str | None
    winning_side: str | None
    outcome: str
    team_verified: bool


def _optional_seat(raw: object, name: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or raw not in SEATS:
        raise ValueError(f"round result {name} is invalid")
    return raw


def parse_round_result(
    row: Mapping[str, object],
    *,
    own_landlord_context: str | None = None,
) -> RoundResultFacts:
    """Validate known facts without treating unknowns as matching wildcards.

    Context must come from this same evidence stream and round, never from the
    other side of an evaluation comparison.
    """
    if "winner" not in row or "outcome" not in row:
        raise ValueError("round result winner/outcome fields are required")
    winner = _optional_seat(row["winner"], "winner")
    landlord = _optional_seat(row.get("landlord"), "landlord")
    context = _optional_seat(own_landlord_context, "landlord context")
    if landlord is not None and context is not None and landlord != context:
        raise ValueError("round result contradicts its own role context")
    landlord = landlord if landlord is not None else context
    outcome = row["outcome"]
    if not isinstance(outcome, str) or outcome not in {"victory", "defeat", "unknown"}:
        raise ValueError("round result outcome is invalid")
    side = row.get("winning_side")
    if "winning_side" in row and (
        not isinstance(side, str) or side not in {"landlord", "farmers", "unknown"}
    ):
        raise ValueError("round result winning_side is invalid")
    known_side = side if isinstance(side, str) and side != "unknown" else None
    if winner == "self" and outcome == "defeat":
        raise ValueError("self emptied hand but result claims defeat")
    winner_side = (
        ("landlord" if winner == landlord else "farmers")
        if winner is not None and landlord is not None else None
    )
    if known_side is not None and winner_side is not None and known_side != winner_side:
        raise ValueError("winner contradicts landlord/winning_side")
    if landlord is None or outcome == "unknown":
        return RoundResultFacts(landlord, winner, known_side, outcome, False)
    landlord_won = (outcome == "victory") == (landlord == "self")
    resolved_side = "landlord" if landlord_won else "farmers"
    if known_side is not None and known_side != resolved_side:
        raise ValueError("winning_side contradicts landlord/outcome")
    if winner_side is not None and winner_side != resolved_side:
        raise ValueError("winner contradicts landlord/outcome")
    return RoundResultFacts(landlord, winner, resolved_side, outcome, True)


def infer_opening_landlord(
    actions: Sequence[Mapping[str, object]],
    scenes: Sequence[Mapping[str, object]],
) -> str | None:
    """Recover a legacy role only from its own first action/remaining labels."""
    if not actions or not scenes:
        return None
    first = actions[0]
    opening = next((row for row in scenes if row.get("after_sequence_no") == 1), None)
    if first.get("sequence_no") != 1 or opening is None:
        return None
    remaining = opening.get("remaining")
    if not isinstance(remaining, Mapping) or set(remaining) != SEATS:
        return None
    before: dict[str, int] = {}
    for seat in SEATS:
        count = remaining[seat]
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 20:
            return None
        before[seat] = count
    cards, actor = first.get("cards"), first.get("actor")
    if not isinstance(cards, list) or not isinstance(actor, str) or actor not in SEATS:
        return None
    if first.get("event") != "play_observed" or not cards:
        return None
    before[actor] += len(cards)
    if sorted(before.values()) != [17, 17, 20]:
        return None
    return next(seat for seat, count in before.items() if count == 20)
