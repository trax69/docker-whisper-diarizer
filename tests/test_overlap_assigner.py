"""P1 – assign each word to the speaker with the largest temporal overlap."""
import unittest

from speaker_alignment import OverlapSpeakerAssigner
from tests.helpers import speakers, turn, word


class OverlapSpeakerAssignerTest(unittest.TestCase):
    def setUp(self):
        self.assigner = OverlapSpeakerAssigner()

    def test_assigns_speaker_of_the_containing_turn(self):
        turns = [turn(0, 5, "A"), turn(5, 10, "B")]
        result = self.assigner.assign([word(1, 2), word(6, 7)], turns)
        self.assertEqual(speakers(result), ["A", "B"])

    def test_picks_largest_overlap_instead_of_first_listed_turn(self):
        # Midpoint 5.0 lies inside both turns; the old code returned A (listed first).
        turns = [turn(0, 5, "A"), turn(4, 10, "B")]
        result = self.assigner.assign([word(4.0, 6.0)], turns)
        self.assertEqual(speakers(result), ["B"])

    def test_overlap_accumulates_across_turns_of_the_same_speaker(self):
        turns = [turn(0, 1, "A"), turn(1, 1.8, "B"), turn(1.8, 3, "A")]
        result = self.assigner.assign([word(0.5, 2.5)], turns)  # A=1.2 s, B=0.8 s
        self.assertEqual(speakers(result), ["A"])

    def test_word_in_a_gap_goes_to_the_nearest_turn_not_the_previous_one(self):
        turns = [turn(0, 5, "A"), turn(10, 15, "B")]
        result = self.assigner.assign([word(9.5, 9.8)], turns)
        self.assertEqual(speakers(result), ["B"])

    def test_word_in_a_gap_closer_to_the_previous_turn(self):
        turns = [turn(0, 5, "A"), turn(10, 15, "B")]
        result = self.assigner.assign([word(5.1, 5.3)], turns)
        self.assertEqual(speakers(result), ["A"])

    def test_word_after_the_last_turn_goes_to_the_last_turn(self):
        turns = [turn(0, 5, "A"), turn(6, 8, "B")]
        result = self.assigner.assign([word(20, 21)], turns)
        self.assertEqual(speakers(result), ["B"])

    def test_word_before_the_first_turn_goes_to_the_first_turn(self):
        turns = [turn(3, 5, "A"), turn(6, 8, "B")]
        result = self.assigner.assign([word(0, 0.5)], turns)
        self.assertEqual(speakers(result), ["A"])

    def test_unsorted_turns_are_handled(self):
        turns = [turn(10, 15, "B"), turn(0, 5, "A")]
        result = self.assigner.assign([word(1, 2), word(9.5, 9.8)], turns)
        self.assertEqual(speakers(result), ["A", "B"])

    def test_long_turn_starting_early_is_still_found(self):
        turns = [turn(0, 100, "A"), turn(10, 11, "B"), turn(20, 21, "B")]
        result = self.assigner.assign([word(50, 51)], turns)
        self.assertEqual(speakers(result), ["A"])

    def test_zero_length_word_inside_a_turn(self):
        turns = [turn(0, 5, "A"), turn(5, 10, "B")]
        result = self.assigner.assign([word(6, 6)], turns)
        self.assertEqual(speakers(result), ["B"])

    def test_without_turns_every_word_gets_the_default_speaker(self):
        result = self.assigner.assign([word(0, 1), word(1, 2)], [])
        self.assertEqual(speakers(result), ["SPEAKER_00", "SPEAKER_00"])

    def test_keeps_text_and_timestamps(self):
        result = self.assigner.assign([word(1, 2, "hola")], [turn(0, 5, "A")])
        self.assertEqual((result[0].start, result[0].end, result[0].word), (1, 2, " hola"))

    def test_does_not_modify_the_input_words(self):
        words = [word(1, 2)]
        self.assigner.assign(words, [turn(0, 5, "A")])
        self.assertEqual(speakers(words), ["UNKNOWN"])


if __name__ == "__main__":
    unittest.main()
