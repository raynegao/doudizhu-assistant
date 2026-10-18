"""Self-play arena: measure decision agents under the full rule engine.

Every game is played on three ``GameStateTracker`` instances, one per seat,
fed the same actions from each seat's own perspective. Agents therefore
decide on the production ``ObservableGameState`` (their own hand, counters,
the trick, the public history) and never see the other hands, while the
arena keeps the true hands and checks every action against them. Deals are
seeded, agents are deterministic given the game seed, and the Monte Carlo
agent runs without a wall-clock budget so results are reproducible.

Pure logic: no screen, no GUI, no model files.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

from src.logic.action_validation import validate_observed_action
from src.logic.decision import recommend_action
from src.logic.monte_carlo import (
    HeuristicRolloutPolicy,
    MonteCarloSettings,
    recommend_phase4,
)
from src.logic.rollout_policies import ShapedRolloutPolicy
from src.logic.rules import Play, legal_actions
from src.state.cards import FULL_DECK, CardSet, sort_cards
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.game_tracker import GameStateTracker, StateUpdateStatus
from src.state.observable_state import ObservableGameState

ARENA_SCHEMA_VERSION = "decision-arena-v1"
SEATS = 3
# Absolute seats play in increasing order; from seat p the next seat is its
# RIGHT and the one after that its LEFT, matching DEFAULT_TURN_ORDER.
_RELATIVE = (PlayerSeat.SELF, PlayerSeat.RIGHT, PlayerSeat.LEFT)
MAX_PLIES = 300


class ArenaError(RuntimeError):
    """An agent broke the rules or a tracker disagreed with the true hands."""


def relative_seat(absolute: int, perspective: int) -> PlayerSeat:
    return _RELATIVE[(absolute - perspective) % SEATS]


class ArenaAgent(Protocol):
    @property
    def name(self) -> str: ...

    def choose(self, state: ObservableGameState, rng: random.Random) -> Play: ...


@dataclass(frozen=True)
class RandomAgent:
    """Uniform over the legal non-pass actions; passes only when forced."""

    name: str = "random"

    def choose(self, state: ObservableGameState, rng: random.Random) -> Play:
        actions = legal_actions(state.self_hand, Play.parse(state.trick_target.cards))
        playable = [action for action in actions if not action.is_pass]
        return rng.choice(playable) if playable else Play.parse(())


@dataclass(frozen=True)
class RuleAgent:
    """The Phase 1 deterministic recommender (lowest-cost legal action)."""

    name: str = "rule"

    def choose(self, state: ObservableGameState, rng: random.Random) -> Play:
        return recommend_action(state.to_decision_snapshot()).action


@dataclass(frozen=True)
class HeuristicAgent:
    """The Monte Carlo rollout policy used directly as a player."""

    name: str = "heuristic"

    def choose(self, state: ObservableGameState, rng: random.Random) -> Play:
        return HeuristicRolloutPolicy().choose_action(
            player=PlayerSeat.SELF,
            hand=state.self_hand,
            previous=Play.parse(state.trick_target.cards),
            last_player=state.trick_leader,
            landlord=state.landlord,
            remaining_cards=state.remaining_by_player,
        )


@dataclass(frozen=True)
class ShapedAgent:
    """The shaped rollout policy used directly as a player."""

    name: str = "shaped"

    def choose(self, state: ObservableGameState, rng: random.Random) -> Play:
        return ShapedRolloutPolicy().choose_action(
            player=PlayerSeat.SELF,
            hand=state.self_hand,
            previous=Play.parse(state.trick_target.cards),
            last_player=state.trick_leader,
            landlord=state.landlord,
            remaining_cards=state.remaining_by_player,
        )


@dataclass(frozen=True)
class MonteCarloAgent:
    """The Phase 4/6 recommender; the seed of every decision comes from the game rng.

    ``diagnostics`` collects, per decision, whether the top two confidence
    intervals overlapped and how wide the chosen action's interval was.
    """

    settings: MonteCarloSettings = field(
        default_factory=lambda: MonteCarloSettings(
            simulations=128, max_depth=60, time_budget_ms=0, max_candidates=20,
        ),
    )
    name: str = "monte_carlo"
    diagnostics: list[dict[str, float]] = field(default_factory=list, compare=False)

    def choose(self, state: ObservableGameState, rng: random.Random) -> Play:
        seed = rng.randrange(1, 2**31)
        result = recommend_phase4(state, replace(self.settings, seed=seed))
        best = result.rankings[0]
        low, high = best.win_rate_ci95_low, best.win_rate_ci95_high
        self.diagnostics.append({
            "overlap": float("top_action_confidence_intervals_overlap" in result.warnings),
            "ci_width": (high - low) if low is not None and high is not None else 1.0,
            "samples": float(best.simulations),
            "candidates": float(result.all_candidate_count),
        })
        return result.action


@dataclass(frozen=True)
class AgentSpec:
    """Picklable description of an agent so worker processes can rebuild it."""

    kind: str
    settings: Mapping[str, int | bool | str] = field(default_factory=dict)

    def build(self) -> ArenaAgent:
        if self.kind == "random":
            return RandomAgent()
        if self.kind == "rule":
            return RuleAgent()
        if self.kind == "heuristic":
            return HeuristicAgent()
        if self.kind == "shaped":
            return ShapedAgent()
        if self.kind == "monte_carlo":
            defaults = MonteCarloAgent().settings
            overrides = cast(dict[str, Any], dict(self.settings))
            return MonteCarloAgent(settings=replace(defaults, **overrides))
        raise ValueError(f"unknown arena agent kind: {self.kind}")

    def to_payload(self) -> dict[str, object]:
        return {"kind": self.kind, "settings": dict(self.settings)}


@dataclass(frozen=True)
class ArenaDeal:
    hands: tuple[CardSet, CardSet, CardSet]
    landlord: int

    def __post_init__(self) -> None:
        if self.landlord not in range(SEATS):
            raise ValueError("landlord must be an absolute seat index")
        counts = [len(hand) for hand in self.hands]
        if counts[self.landlord] != 20 or sum(counts) != 54:
            raise ValueError("a deal is 17/17/17 plus three cards for the landlord")


def deal_cards(rng: random.Random, landlord: int) -> ArenaDeal:
    deck = list(FULL_DECK)
    rng.shuffle(deck)
    hands = [deck[0:17], deck[17:34], deck[34:51]]
    hands[landlord] = hands[landlord] + deck[51:54]
    return ArenaDeal(
        hands=tuple(CardSet(sort_cards(hand)) for hand in hands),  # type: ignore[arg-type]
        landlord=landlord,
    )


@dataclass(frozen=True)
class GameRecord:
    game_id: str
    landlord: int
    winner: int
    landlord_won: bool
    plies: int
    agents: tuple[str, str, str]
    actions: tuple[tuple[int, tuple[str, ...]], ...]
    decision_ms: tuple[float, float, float]
    decisions: tuple[int, int, int]
    # Per seat: Monte Carlo diagnostics summed over that seat's decisions
    # (overlap count, total CI width, total samples); zeros for other agents.
    diagnostics: tuple[dict[str, float], dict[str, float], dict[str, float]] = (
        {}, {}, {},
    )

    def to_payload(self) -> dict[str, object]:
        return {
            "game_id": self.game_id,
            "landlord": self.landlord,
            "winner": self.winner,
            "landlord_won": self.landlord_won,
            "plies": self.plies,
            "agents": list(self.agents),
            "actions": [[seat, list(cards)] for seat, cards in self.actions],
            "decision_ms": [round(value, 3) for value in self.decision_ms],
            "decisions": list(self.decisions),
            "diagnostics": [
                {key: round(value, 6) for key, value in item.items()} for item in self.diagnostics
            ],
        }


def _perspective_state(deal: ArenaDeal, perspective: int, game_id: str) -> ObservableGameState:
    return ObservableGameState.from_inputs(
        deal.hands[perspective].cards,
        round_id=f"{game_id}:{perspective}",
        landlord=relative_seat(deal.landlord, perspective),
        current_actor=relative_seat(deal.landlord, perspective),
        remaining_cards={
            relative_seat(seat, perspective): len(deal.hands[seat]) for seat in range(SEATS)
        },
    )


def play_game(
    deal: ArenaDeal,
    agents: Sequence[ArenaAgent],
    *,
    rng: random.Random,
    game_id: str = "arena",
    clock=time.perf_counter,
) -> GameRecord:
    if len(agents) != SEATS:
        raise ValueError("exactly three agents are required")
    trackers = [
        GameStateTracker(_perspective_state(deal, seat, game_id), validator=validate_observed_action)
        for seat in range(SEATS)
    ]
    hands = [hand for hand in deal.hands]
    current = deal.landlord
    actions: list[tuple[int, tuple[str, ...]]] = []
    decision_ms = [0.0, 0.0, 0.0]
    decisions = [0, 0, 0]
    diagnostics: list[dict[str, float]] = [{}, {}, {}]
    for agent in agents:
        collected = getattr(agent, "diagnostics", None)
        if isinstance(collected, list):
            collected.clear()
    winner: int | None = None
    for sequence_no in range(1, MAX_PLIES + 1):
        state = trackers[current].state
        if state.phase is not RoundPhase.PLAYING or state.current_actor is not PlayerSeat.SELF:
            raise ArenaError(f"seat {current} is not to act in its own tracker: {state.phase.value}")
        if not state.decision_ready:
            raise ArenaError(f"seat {current} state is not decision-ready: {state.warnings}")
        previous = Play.parse(state.trick_target.cards)
        started = clock()
        play = agents[current].choose(state, rng)
        decision_ms[current] += (clock() - started) * 1000
        decisions[current] += 1
        if play not in legal_actions(hands[current], previous):
            raise ArenaError(
                f"{agents[current].name} at seat {current} returned an illegal action {play} "
                f"against {previous} with hand {hands[current]}"
            )
        cards = CardSet(play.cards)
        for perspective, tracker in enumerate(trackers):
            event = ObservedAction(
                event_id=f"{game_id}:{sequence_no}",
                sequence_no=sequence_no,
                actor=relative_seat(current, perspective),
                cards=cards,
                source="arena",
            )
            result = tracker.apply(event)
            if result.status is not StateUpdateStatus.APPLIED:
                raise ArenaError(
                    f"tracker {perspective} rejected action {sequence_no}: {result.message}"
                )
        if not play.is_pass:
            counts = hands[current].counts
            counts.subtract(play.cards)
            hands[current] = CardSet(
                sort_cards(card for rank, count in counts.items() for card in [rank] * count)
            )
        actions.append((current, tuple(play.cards)))
        for perspective, tracker in enumerate(trackers):
            observed = tracker.state.remaining_by_player
            for seat in range(SEATS):
                if observed[relative_seat(seat, perspective)] != len(hands[seat]):
                    raise ArenaError(
                        f"tracker {perspective} counts {observed} disagree with the true hands"
                    )
        if not hands[current]:
            winner = current
            break
        current = (current + 1) % SEATS
    if winner is None:
        raise ArenaError(f"game {game_id} exceeded {MAX_PLIES} plies")
    for seat, agent in enumerate(agents):
        collected = getattr(agent, "diagnostics", None)
        if isinstance(collected, list) and collected:
            diagnostics[seat] = {
                "decisions": float(len(collected)),
                "overlap": sum(item["overlap"] for item in collected),
                "ci_width": sum(item["ci_width"] for item in collected),
                "samples": sum(item["samples"] for item in collected),
                "candidates": sum(item["candidates"] for item in collected),
            }
            collected.clear()
    for perspective, tracker in enumerate(trackers):
        state = tracker.state
        if state.phase is not RoundPhase.FINISHED or state.winner is not relative_seat(winner, perspective):
            raise ArenaError(f"tracker {perspective} did not finish on the true winner")
    return GameRecord(
        game_id=game_id,
        landlord=deal.landlord,
        winner=winner,
        landlord_won=winner == deal.landlord,
        plies=len(actions),
        agents=tuple(agent.name for agent in agents),  # type: ignore[arg-type]
        actions=tuple(actions),
        decision_ms=tuple(decision_ms),  # type: ignore[arg-type]
        decisions=tuple(decisions),  # type: ignore[arg-type]
        diagnostics=tuple(diagnostics),  # type: ignore[arg-type]
    )


# Configurations played on every deal. The reference game pits the opponent
# against itself so the subject's landlord and farmer results can be paired
# per deal against the same opponents on the same cards.
CONFIGURATIONS = (
    "subject_landlord_vs_opponent_farmers",
    "opponent_landlord_vs_subject_farmers",
    "opponent_landlord_vs_opponent_farmers",
)


@dataclass(frozen=True)
class MatchSettings:
    games: int = 20
    seed: int = 20261018
    subject: AgentSpec = AgentSpec("monte_carlo")
    opponent: AgentSpec = AgentSpec("heuristic")
    workers: int = 1

    def __post_init__(self) -> None:
        if self.games <= 0 or self.workers <= 0:
            raise ValueError("games and workers must be positive")

    def to_payload(self) -> dict[str, object]:
        return {
            "games": self.games,
            "seed": self.seed,
            "subject": self.subject.to_payload(),
            "opponent": self.opponent.to_payload(),
            "workers": self.workers,
        }


def _game_seed(seed: int, index: int, configuration: str) -> int:
    return random.Random((seed, index, configuration).__hash__() & 0x7FFFFFFF).randrange(1, 2**31)


def play_deal(settings: MatchSettings, index: int) -> dict[str, GameRecord]:
    """Play the three configurations of deal ``index``; safe to call in a worker."""
    landlord = index % SEATS
    deal = deal_cards(random.Random((settings.seed, index).__hash__() & 0x7FFFFFFF), landlord)
    subject = settings.subject.build()
    opponent = settings.opponent.build()
    records: dict[str, GameRecord] = {}
    for configuration in CONFIGURATIONS:
        if configuration == "subject_landlord_vs_opponent_farmers":
            agents = [subject if seat == landlord else opponent for seat in range(SEATS)]
        elif configuration == "opponent_landlord_vs_subject_farmers":
            agents = [opponent if seat == landlord else subject for seat in range(SEATS)]
        else:
            agents = [opponent, opponent, opponent]
        records[configuration] = play_game(
            deal,
            agents,
            rng=random.Random(_game_seed(settings.seed, index, configuration)),
            game_id=f"deal-{index:04d}:{configuration}",
        )
    return records


def wilson_interval(wins: int, count: int, z: float = 1.959964) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 1.0
    mean = wins / count
    denominator = 1 + z * z / count
    centre = (mean + z * z / (2 * count)) / denominator
    half = z * math.sqrt(mean * (1 - mean) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _paired(differences: Sequence[int]) -> dict[str, object]:
    count = len(differences)
    mean = sum(differences) / count if count else 0.0
    if count >= 2:
        variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
        half = 1.959964 * math.sqrt(variance / count)
    else:
        half = 1.0
    return {
        "n": count,
        "mean": round(mean, 6),
        "ci95": [round(max(-1.0, mean - half), 6), round(min(1.0, mean + half), 6)],
        "wins_gained": sum(1 for value in differences if value > 0),
        "wins_lost": sum(1 for value in differences if value < 0),
    }


def summarize(settings: MatchSettings, deals: Sequence[Mapping[str, GameRecord]]) -> dict[str, object]:
    configurations: dict[str, object] = {}
    for configuration in CONFIGURATIONS:
        records = [deal[configuration] for deal in deals]
        count = len(records)
        landlord_wins = sum(record.landlord_won for record in records)
        low, high = wilson_interval(landlord_wins, count)
        subject_seats = {
            "subject_landlord_vs_opponent_farmers": lambda record: (record.landlord,),
            "opponent_landlord_vs_subject_farmers": lambda record: tuple(
                seat for seat in range(SEATS) if seat != record.landlord
            ),
            "opponent_landlord_vs_opponent_farmers": lambda _record: (),
        }[configuration]
        subject_ms = [
            record.decision_ms[seat] / max(1, record.decisions[seat])
            for record in records
            for seat in subject_seats(record)
        ]
        diagnostics = [
            record.diagnostics[seat]
            for record in records
            for seat in subject_seats(record)
            if record.diagnostics[seat]
        ]
        decided = sum(item["decisions"] for item in diagnostics)
        configurations[configuration] = {
            "games": count,
            "landlord_wins": landlord_wins,
            "landlord_win_rate": round(landlord_wins / count, 6) if count else None,
            "landlord_win_rate_ci95": [round(low, 6), round(high, 6)],
            "farmers_win_rate": round(1 - landlord_wins / count, 6) if count else None,
            "average_plies": round(sum(record.plies for record in records) / count, 3) if count else None,
            "subject_mean_decision_ms": (
                round(sum(subject_ms) / len(subject_ms), 3) if subject_ms else None
            ),
            "subject_monte_carlo": (
                {
                    "decisions": int(decided),
                    "top2_ci_overlap_rate": round(
                        sum(item["overlap"] for item in diagnostics) / decided, 6,
                    ),
                    "mean_best_ci_width": round(
                        sum(item["ci_width"] for item in diagnostics) / decided, 6,
                    ),
                    "mean_best_samples": round(
                        sum(item["samples"] for item in diagnostics) / decided, 3,
                    ),
                    "mean_legal_candidates": round(
                        sum(item["candidates"] for item in diagnostics) / decided, 3,
                    ),
                }
                if decided else None
            ),
        }
    landlord_pairs = [
        int(deal["subject_landlord_vs_opponent_farmers"].landlord_won)
        - int(deal["opponent_landlord_vs_opponent_farmers"].landlord_won)
        for deal in deals
    ]
    farmer_pairs = [
        int(not deal["opponent_landlord_vs_subject_farmers"].landlord_won)
        - int(not deal["opponent_landlord_vs_opponent_farmers"].landlord_won)
        for deal in deals
    ]
    return {
        "schema_version": ARENA_SCHEMA_VERSION,
        "settings": settings.to_payload(),
        "configurations": configurations,
        "paired_vs_reference": {
            "subject_as_landlord": _paired(landlord_pairs),
            "subject_as_farmers": _paired(farmer_pairs),
        },
        "limitations": [
            "landlord is assigned by rotation, no bidding model",
            "win/loss only, no bomb or spring multipliers",
            "deterministic agents given the seed; Monte Carlo runs without a wall-clock budget",
        ],
    }


def run_match(settings: MatchSettings) -> tuple[dict[str, object], list[dict[str, GameRecord]]]:
    indices = range(settings.games)
    if settings.workers == 1:
        deals = [play_deal(settings, index) for index in indices]
    else:
        with ProcessPoolExecutor(max_workers=settings.workers) as pool:
            deals = list(pool.map(play_deal, [settings] * settings.games, indices))
    return summarize(settings, deals), deals


__all__ = [
    "ARENA_SCHEMA_VERSION",
    "AgentSpec",
    "ArenaAgent",
    "ArenaDeal",
    "ArenaError",
    "CONFIGURATIONS",
    "GameRecord",
    "HeuristicAgent",
    "MatchSettings",
    "MonteCarloAgent",
    "RandomAgent",
    "RuleAgent",
    "ShapedAgent",
    "deal_cards",
    "play_deal",
    "play_game",
    "relative_seat",
    "run_match",
    "summarize",
    "wilson_interval",
]
