import unittest

from game_logic.cards import Card
from game_logic.doudizhu_rules import HandType, can_beat, classify_hand, generate_all_legal_hands


class TestRulesClassification(unittest.TestCase):
    def test_basic_types(self) -> None:
        self.assertEqual(classify_hand([Card("3")]), HandType.SINGLE)
        self.assertEqual(classify_hand([Card("3"), Card("3")]), HandType.PAIR)
        self.assertEqual(classify_hand([Card("5"), Card("5"), Card("5")]), HandType.TRIPLE)
        self.assertEqual(
            classify_hand([Card("5"), Card("5"), Card("5"), Card("6")]), HandType.TRIPLE_WITH_SINGLE
        )
        self.assertEqual(
            classify_hand([Card("5"), Card("5"), Card("5"), Card("6"), Card("6")]), HandType.TRIPLE_WITH_PAIR
        )

    def test_rocket_and_bomb(self) -> None:
        self.assertEqual(classify_hand([Card("joker_small"), Card("joker_big")]), HandType.ROCKET)
        self.assertEqual(
            classify_hand([Card("7"), Card("7"), Card("7"), Card("7")]),
            HandType.BOMB,
        )

    def test_sequences(self) -> None:
        straight = [Card(r) for r in ["3", "4", "5", "6", "7"]]
        self.assertEqual(classify_hand(straight), HandType.STRAIGHT)

        double_seq = [Card(r) for r in ["3", "3", "4", "4", "5", "5", "6", "6"]]
        self.assertEqual(classify_hand(double_seq), HandType.DOUBLE_SEQUENCE)

        triple_chain = [Card(r) for r in ["3", "3", "3", "4", "4", "4"]]
        self.assertEqual(classify_hand(triple_chain), HandType.AIRPLANE)

        airplane_wings = [Card(r) for r in ["3", "3", "3", "4", "4", "4", "6", "7"]]
        self.assertEqual(classify_hand(airplane_wings), HandType.AIRPLANE_WITH_WINGS)

    def test_invalid_straight_with_two(self) -> None:
        invalid = [Card(r) for r in ["10", "J", "Q", "K", "A", "2"]]
        self.assertEqual(classify_hand(invalid), HandType.INVALID)


class TestRulesComparison(unittest.TestCase):
    def test_priority(self) -> None:
        prev = [Card("5")]
        curr = [Card("6")]
        self.assertTrue(can_beat(prev, curr))

        prev_bomb = [Card("7")] * 4
        curr_triple = [Card("8")] * 3
        self.assertFalse(can_beat(prev_bomb, curr_triple))

        self.assertTrue(can_beat(prev_triple := [Card("9")] * 3, curr_bomb := [Card("10")] * 4))

        rocket = [Card("joker_small"), Card("joker_big")]
        self.assertTrue(can_beat(prev_bomb, rocket))

    def test_generate_filtered_by_prev(self) -> None:
        hand = [
            Card("3"),
            Card("4"),
            Card("5"),
            Card("6"),
            Card("6"),
            Card("joker_small"),
            Card("joker_big"),
        ]
        prev = [Card("5")]
        legal = generate_all_legal_hands(hand, prev)
        # 最小可压单牌
        self.assertIn([Card("6")], legal)
        # 火箭应始终可出
        self.assertIn([Card("joker_small"), Card("joker_big")], legal)


if __name__ == "__main__":
    unittest.main()
