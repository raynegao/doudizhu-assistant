"""Rollout policies for Monte Carlo simulation and a greedy hand-shape estimate.

``hands_to_finish`` approximates the number of plays a hand still needs: it
greedily peels off the rocket, bombs, the longest planes, straights and pair
straights, then triples, pairs and singles, letting triples and planes absorb
the kickers they are allowed to carry. It is a cheap proxy for hand shape,
not a minimal decomposition, and it is used only to rank candidate plays
inside rollouts and in the direct ``ShapedRolloutPolicy`` player.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from src.logic.rules import Play, PlayType, legal_actions
from src.state.cards import CHAIN_RANKS, RANK_VALUE, CardSet, sort_cards
from src.state.events import PlayerSeat

ROLLOUT_POLICIES = ("heuristic", "shaped")


def _chain_runs(counts: Counter[str], minimum_count: int, minimum_length: int) -> list[list[str]]:
    runs: list[list[str]] = []
    current: list[str] = []
    for rank in CHAIN_RANKS:
        if counts[rank] >= minimum_count:
            current.append(rank)
        else:
            if len(current) >= minimum_length:
                runs.append(current)
            current = []
    if len(current) >= minimum_length:
        runs.append(current)
    return runs


def hands_to_finish(cards: Sequence[str]) -> int:
    counts = Counter(cards)
    hands = 0
    if counts["SJ"] and counts["BJ"]:
        counts["SJ"] -= 1
        counts["BJ"] -= 1
        hands += 1
    for rank in list(counts):
        if counts[rank] == 4:
            counts[rank] = 0
            hands += 1
    kicker_slots = 0
    while True:
        runs = _chain_runs(counts, 3, 2)
        if not runs:
            break
        run = max(runs, key=len)
        for rank in run:
            counts[rank] -= 3
        hands += 1
        kicker_slots += len(run)
    while True:
        runs = _chain_runs(counts, 1, 5)
        if not runs:
            break
        run = max(runs, key=len)
        for rank in run:
            counts[rank] -= 1
        hands += 1
    while True:
        runs = _chain_runs(counts, 2, 3)
        if not runs:
            break
        run = max(runs, key=len)
        for rank in run:
            counts[rank] -= 2
        hands += 1
    for rank in list(counts):
        if counts[rank] == 3:
            counts[rank] = 0
            hands += 1
            kicker_slots += 1
    singles = sorted((rank for rank, count in counts.items() if count == 1), key=RANK_VALUE.__getitem__)
    pairs = sorted((rank for rank, count in counts.items() if count == 2), key=RANK_VALUE.__getitem__)
    # Triples and planes may carry one single (or pair) per triple; absorb the
    # smallest leftovers first, singles before pairs.
    absorbed = min(kicker_slots, len(singles))
    singles = singles[absorbed:]
    kicker_slots -= absorbed
    absorbed = min(kicker_slots, len(pairs))
    pairs = pairs[absorbed:]
    return hands + len(singles) + len(pairs)


def _same_team(left: PlayerSeat, right: PlayerSeat, landlord: PlayerSeat) -> bool:
    return (left is landlord) == (right is landlord)


def _control_assets(counts: Counter[str]) -> frozenset[str]:
    assets = {f"bomb:{rank}" for rank, count in counts.items() if count == 4}
    if counts["SJ"] and counts["BJ"]:
        assets.add("rocket")
    return frozenset(assets)


def _stable_key(play: Play) -> tuple[int, int, int, tuple[int, ...]]:
    return (
        list(PlayType).index(play.type),
        len(play.cards),
        RANK_VALUE[play.main_rank] if play.main_rank else -1,
        tuple(RANK_VALUE[card] for card in play.cards),
    )


class ShapedRolloutPolicy:
    """Play the move that keeps the hand closest to finishing.

    ``urgent_threshold``: an enemy holding this many cards or fewer makes every
    beat (bombs included) worth taking and changes the lead choice; 3 measured
    best in paired self-play against the heuristic and rule baselines (2 and 4
    were within noise, 2 slightly worse).

    Leading: the action whose residual needs the fewest hands, preferring small
    main ranks and avoiding bombs/rocket unless an enemy is about to finish;
    singles are avoided against a one-card enemy and pairs against a two-card
    enemy. Following an enemy: the smallest sufficient action that does not
    break the hand's shape, otherwise pass; bombs only when urgent. Following
    the partner: pass unless urgent or the hand can be finished.
    """

    def __init__(self, *, urgent_threshold: int = 3) -> None:
        if urgent_threshold < 1:
            raise ValueError("urgent_threshold must be positive")
        self.urgent_threshold = urgent_threshold

    def choose_action(
        self,
        *,
        player: PlayerSeat,
        hand: CardSet,
        previous: Play,
        last_player: PlayerSeat | None,
        landlord: PlayerSeat,
        remaining_cards: Mapping[PlayerSeat, int],
    ) -> Play:
        actions = legal_actions(hand, previous)
        non_pass = [action for action in actions if not action.is_pass]
        if not non_pass:
            return Play.parse(())
        exact = [action for action in non_pass if len(action.cards) == len(hand)]
        if exact:
            return min(exact, key=_stable_key)
        pass_action = next((action for action in actions if action.is_pass), None)
        enemy_counts = [
            count for seat, count in remaining_cards.items()
            if not _same_team(player, seat, landlord)
        ]
        enemy_min = min(enemy_counts) if enemy_counts else 99
        urgent = enemy_min <= self.urgent_threshold
        if (
            pass_action is not None
            and last_player is not None
            and last_player is not player
            and _same_team(player, last_player, landlord)
            and not urgent
        ):
            return pass_action

        leading = not previous.cards
        hand_counts = hand.counts
        base = hands_to_finish(hand.cards)
        assets_before = _control_assets(hand_counts)
        best: tuple[tuple[float, int, int, int, tuple[int, ...]], Play, int, bool] | None = None
        for action in non_pass:
            residual_counts = Counter(hand_counts)
            residual_counts.subtract(action.cards)
            residual = sort_cards(
                card for rank, count in residual_counts.items() for card in [rank] * count
            )
            remaining_hands = hands_to_finish(residual)
            spends_control = bool(assets_before - _control_assets(residual_counts))
            main = RANK_VALUE[action.main_rank] if action.main_rank else 0
            cost = 10.0 * remaining_hands
            if spends_control and not urgent:
                cost += 8.0 if leading else 6.0
            if leading:
                if enemy_min == 1 and action.type is PlayType.SINGLE:
                    cost += 6.0 - 0.30 * main  # if a single must go, the biggest one
                elif enemy_min == 2 and action.type is PlayType.PAIR:
                    cost += 4.0 - 0.30 * main
                else:
                    cost += 0.15 * main
            else:
                cost += 0.15 * main
            key = (cost, *_stable_key(action))
            if best is None or key < best[0]:
                best = (key, action, remaining_hands, spends_control)
        assert best is not None
        _, action, remaining_hands, spends_control = best
        if (
            not leading
            and pass_action is not None
            and not urgent
            and (remaining_hands > base or spends_control)
        ):
            # Beating would leave a worse shape than the current hand (for
            # example splitting a straight) or spend a bomb on an ordinary
            # trick; let it go and keep the hand. Breaking a pair to beat a
            # single is neutral in hands and is taken for tempo.
            return pass_action
        return action


def build_rollout_policy(name: str):
    if name == "heuristic":
        from src.logic.monte_carlo import HeuristicRolloutPolicy

        return HeuristicRolloutPolicy()
    if name == "shaped":
        return ShapedRolloutPolicy()
    raise ValueError(f"rollout_policy must be one of {ROLLOUT_POLICIES}")


__all__ = ["ROLLOUT_POLICIES", "ShapedRolloutPolicy", "build_rollout_policy", "hands_to_finish"]
