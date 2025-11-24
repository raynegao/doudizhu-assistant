import unittest

from game_logic.cards import RANK_TO_ID
from game_logic.state_parser import GameState, group_detections_by_row, parse_game_state


class TestStateParser(unittest.TestCase):
    def test_group_rows_and_parse(self) -> None:
        # 两行：上方 1 张，对方；下方 2 张，自己
        detections = [
            (200.0, 500.0, 50.0, 30.0, RANK_TO_ID["3"], 0.9),
            (120.0, 120.0, 50.0, 30.0, RANK_TO_ID["4"], 0.8),
            (260.0, 500.0, 50.0, 30.0, RANK_TO_ID["5"], 0.85),
        ]

        rows = group_detections_by_row(detections)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 1)
        self.assertEqual(len(rows[1]), 2)

        state = parse_game_state(detections)
        self.assertIsInstance(state, GameState)
        self.assertEqual(len(state.my_hand), 2)
        self.assertEqual(state.left_opponent_count, 1)
        self.assertEqual(state.right_opponent_count, 0)


if __name__ == "__main__":
    unittest.main()
