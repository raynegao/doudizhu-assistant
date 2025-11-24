import unittest
from collections import Counter
from unittest.mock import patch

from game_logic.cards import Card
from game_logic.simulator import _choose_play, _remove_known_cards, estimate_win_rate
from game_logic.state_parser import GameState


class TestSimulatorHelpers(unittest.TestCase):
    def test_remove_known_cards(self) -> None:
        state = GameState(
            my_hand=[Card("3"), Card("joker_big")],
            last_play=[Card("4")],
            history=[[Card("5"), Card("5")]],
        )
        pool = _remove_known_cards([Card(r) for r in ["3"] * 4 + ["4"] * 4 + ["5"] * 4], state)
        counts = Counter([c.rank for c in pool])
        self.assertEqual(counts["3"], 3)
        self.assertEqual(counts["4"], 3)
        self.assertEqual(counts["5"], 2)

    def test_choose_play_prefers_smallest(self) -> None:
        prev: list[Card] = []
        legal = [[Card("3")], [Card("4")], [Card("5"), Card("5")]]
        picked = _choose_play(prev, legal)
        self.assertEqual(picked, [Card("3")])

        prev = [Card("4")]
        legal = [[Card("3")], [Card("6")], [Card("7"), Card("7")]]
        picked = _choose_play(prev, legal)
        self.assertEqual(picked, [Card("6")])

    def test_choose_play_avoids_bomb_if_other_option(self) -> None:
        prev = [Card("8")]
        legal = [[Card("9")], [Card("10")] * 4]  # single 和炸弹
        picked = _choose_play(prev, legal)
        self.assertEqual(picked, [Card("9")])


class TestEstimateWinRate(unittest.TestCase):
    def test_estimate_win_rate_with_mock(self) -> None:
        state = GameState(my_hand=[Card("3")])
        with patch("game_logic.simulator.simulate_round", side_effect=[True, False, True, False]):
            win_rate = estimate_win_rate(state, num_samples=4)
        self.assertAlmostEqual(win_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
