from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .events import DEFAULT_TURN_ORDER, ObservedAction
from .game_tracker import (
    ActionValidator,
    GameStateTracker,
    GameStateTransitionError,
    StateUpdateStatus,
)
from .observable_state import ObservableGameState


@dataclass(frozen=True)
class ReplayLoadResult:
    state: ObservableGameState
    warnings: tuple[str, ...]
    event_count: int


def load_event_replay(
    path: Path,
    *,
    validator: ActionValidator,
    confidence_threshold: float = 0.70,
) -> ReplayLoadResult:
    payloads: list[Mapping[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"event replay line {line_number} must be a JSON object")
        payloads.append(payload)

    if not payloads or payloads[0].get("event") != "game_started":
        raise ValueError("events file must begin with a game_started object")
    start = payloads[0]
    remaining_payload = start.get("remaining_cards")
    if not isinstance(remaining_payload, dict):
        raise ValueError("game_started.remaining_cards must be an object")
    hand = _card_input(start.get("hand", ()), field="hand")
    turn_order = _string_sequence(
        start.get(
            "turn_order",
            [seat.value for seat in DEFAULT_TURN_ORDER],
        ),
        field="turn_order",
    )
    played_cards = _card_input(
        start.get("played_cards", ()),
        field="played_cards",
    )
    last_play = _card_input(start.get("last_play", ()), field="last_play")
    raw_last_player = start.get("last_player")
    if raw_last_player is not None and not isinstance(raw_last_player, str):
        raise ValueError("game_started.last_player must be a string or null")

    state = ObservableGameState.from_inputs(
        hand,
        round_id=str(start.get("round_id", path.stem)),
        landlord=str(start.get("landlord", "self")),
        current_actor=str(start.get("current_actor", "self")),
        turn_order=turn_order,
        remaining_cards={
            str(seat): int(str(count))
            for seat, count in remaining_payload.items()
        },
        played_cards=played_cards,
        hidden_played_count=int(str(start.get("hidden_played_count", 0))),
        last_play=last_play,
        last_player=raw_last_player,
        consecutive_passes=int(str(start.get("consecutive_passes", 0))),
        state_confidence=float(str(start.get("state_confidence", 1.0))),
    )
    tracker = GameStateTracker(
        state,
        validator=validator,
        confidence_threshold=confidence_threshold,
    )
    warnings: list[str] = []
    for payload in payloads[1:]:
        event = ObservedAction.from_payload(payload)
        result = tracker.apply(event)
        if result.status is StateUpdateStatus.REJECTED:
            raise GameStateTransitionError(result.message)
        if result.status in {StateUpdateStatus.DEFERRED, StateUpdateStatus.DUPLICATE}:
            warnings.append(f"{result.status.value}: {event.event_id}: {result.message}")
            warnings.extend(result.warnings)

    return ReplayLoadResult(
        state=tracker.state,
        warnings=tuple(dict.fromkeys(warnings)),
        event_count=len(payloads) - 1,
    )


def _card_input(value: object, *, field: str) -> str | tuple[str, ...]:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return tuple(value)
    raise ValueError(f"game_started.{field} must be a string or string array")


def _string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return tuple(value)
    raise ValueError(f"game_started.{field} must be a string array")


__all__ = ["ReplayLoadResult", "load_event_replay"]
