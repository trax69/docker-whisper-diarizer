"""P2 – absorb speaker turns that are too short to be a real intervention."""
import unittest

from speaker_alignment import ShortTurnSmoother
from tests.helpers import run_of, speakers, word


class ShortTurnSmootherTest(unittest.TestCase):
    def setUp(self):
        self.smoother = ShortTurnSmoother(min_turn_words=3, min_turn_sec=1.5)

    def test_single_word_turn_between_same_speaker_is_absorbed(self):
        words = run_of("A", 0.0, 6) + [word(3.1, 3.4, "pues", "B")] + run_of("A", 3.5, 6)
        self.assertEqual(set(speakers(self.smoother.process(words))), {"A"})

    def test_long_enough_turn_is_preserved(self):
        words = run_of("A", 0.0, 6) + run_of("B", 3.1, 6) + run_of("A", 6.2, 6)
        result = self.smoother.process(words)
        self.assertEqual(speakers(result), ["A"] * 6 + ["B"] * 6 + ["A"] * 6)

    def test_turn_with_enough_words_but_too_short_in_time_is_absorbed(self):
        words = run_of("A", 0.0, 6) + run_of("B", 3.1, 3, dur=0.3) + run_of("A", 4.1, 6)
        self.assertEqual(set(speakers(self.smoother.process(words))), {"A"})

    def test_turn_long_in_time_but_with_too_few_words_is_absorbed(self):
        words = run_of("A", 0.0, 6) + run_of("B", 3.1, 2, dur=1.0) + run_of("A", 5.2, 6)
        self.assertEqual(set(speakers(self.smoother.process(words))), {"A"})

    def test_short_turn_at_the_start_goes_to_the_next_speaker(self):
        words = [word(0.0, 0.3, speaker="B")] + run_of("A", 0.5, 6)
        self.assertEqual(set(speakers(self.smoother.process(words))), {"A"})

    def test_short_turn_at_the_end_goes_to_the_previous_speaker(self):
        words = run_of("A", 0.0, 6) + [word(3.2, 3.4, speaker="B")]
        self.assertEqual(set(speakers(self.smoother.process(words))), {"A"})

    def test_short_turn_between_different_speakers_goes_to_the_closest_in_time(self):
        words = run_of("A", 0.0, 6) + [word(5.0, 5.3, speaker="B")] + run_of("C", 5.4, 6)
        result = self.smoother.process(words)
        self.assertEqual(speakers(result), ["A"] * 6 + ["C"] * 7)

    def test_consecutive_short_turns_are_resolved(self):
        words = (run_of("A", 0.0, 6)
                 + [word(3.1, 3.3, speaker="B"), word(3.4, 3.6, speaker="C")]
                 + run_of("A", 3.7, 6))
        self.assertEqual(set(speakers(self.smoother.process(words))), {"A"})

    def test_single_turn_input_is_unchanged_even_if_short(self):
        words = [word(0.0, 0.2, speaker="A")]
        self.assertEqual(speakers(self.smoother.process(words)), ["A"])

    def test_zero_thresholds_disable_smoothing(self):
        smoother = ShortTurnSmoother(min_turn_words=0, min_turn_sec=0)
        words = run_of("A", 0.0, 6) + [word(3.1, 3.4, speaker="B")] + run_of("A", 3.5, 6)
        self.assertEqual(speakers(smoother.process(words)), speakers(words))

    def test_empty_input(self):
        self.assertEqual(self.smoother.process([]), [])

    def test_does_not_modify_the_input_words(self):
        words = run_of("A", 0.0, 6) + [word(3.1, 3.4, speaker="B")] + run_of("A", 3.5, 6)
        before = speakers(words)
        self.smoother.process(words)
        self.assertEqual(speakers(words), before)

    def test_negative_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            ShortTurnSmoother(min_turn_words=-1, min_turn_sec=1)
        with self.assertRaises(ValueError):
            ShortTurnSmoother(min_turn_words=1, min_turn_sec=-0.5)

    def test_fragments_separated_by_a_long_silence_are_judged_one_by_one(self):
        # Mirrors 631009 @3603s: "aquí." … 4 s of silence … "Sí, pequeño HTML.", both
        # tagged as another speaker. Glued together they look like a 5.7 s turn;
        # each one alone is short and must be absorbed.
        words = (run_of("A", 0.0, 6) + [word(3.0, 3.3, "aquí.", "B")]
                 + run_of("B", 7.3, 3, dur=0.45) + run_of("A", 9.0, 6))
        self.assertEqual(set(speakers(self.smoother.process(words))), {"A"})

    def test_real_turn_after_a_long_silence_is_preserved(self):
        # Mirrors 631009 @1371s: a stray "A" is absorbed, but the 11-word sentence
        # that follows 7 s later stands on its own.
        sentence = run_of("B", 10.0, 11, dur=0.25)
        words = run_of("A", 0.0, 6) + [word(3.2, 3.5, "A", "B")] + sentence + run_of("A", 14.0, 6)
        result = self.smoother.process(words)
        self.assertEqual(speakers(result), ["A"] * 7 + ["B"] * 11 + ["A"] * 6)

    def test_isolated_short_turn_between_silences_goes_to_the_closest(self):
        # Silences on both sides (2.5 s before, 3.5 s after) → closest is A.
        words = run_of("A", 0.0, 6) + [word(5.5, 5.7, speaker="B")] + run_of("C", 9.2, 6)
        result = self.smoother.process(words)
        self.assertEqual(speakers(result), ["A"] * 7 + ["C"] * 6)

    def test_same_speaker_fragments_split_by_a_long_pause_are_judged_separately(self):
        # B says 6 words, pauses 3 s, then 1 stray word right before A resumes:
        # the first block is a real turn, the stray word goes to the closest (A).
        words = (run_of("A", 0.0, 6) + run_of("B", 3.1, 6)
                 + [word(9.1, 9.3, speaker="B")] + run_of("A", 9.4, 6))
        result = self.smoother.process(words)
        self.assertEqual(speakers(result), ["A"] * 6 + ["B"] * 6 + ["A"] * 7)

    def test_absorbed_words_do_not_make_a_false_turn_long_enough(self):
        # Mirrors 631009 @5208s: A "…un" | B "poquito, creo" (2 words) | A "vale,"
        # (closest to B) | 5 s silence | A "para hablar del DOM…".
        # "vale," is absorbed by B, but B must still be judged by its own 2 words,
        # so it ends up absorbed by A (A on both sides).
        words = (run_of("A", 0.0, 10) + run_of("B", 5.0, 2, dur=0.95)
                 + [word(7.82, 8.46, "vale,", "A")] + run_of("A", 13.56, 10))
        self.assertEqual(set(speakers(self.smoother.process(words))), {"A"})

    def test_non_positive_pause_is_rejected(self):
        with self.assertRaises(ValueError):
            ShortTurnSmoother(min_turn_words=3, min_turn_sec=1.5, pause_split_sec=0)

    def test_handles_many_alternating_turns_quickly(self):
        words = []
        for i in range(3000):
            words += run_of("A", i * 4.0, 6) + [word(i * 4.0 + 3.1, i * 4.0 + 3.4, speaker="B")]
        result = self.smoother.process(words)
        self.assertEqual(set(speakers(result)), {"A"})


if __name__ == "__main__":
    unittest.main()
