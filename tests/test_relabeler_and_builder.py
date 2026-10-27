"""P5 – relabel speakers by speaking time, and words → output segments."""
import unittest

from speaker_alignment import SegmentBuilder, SpeakerRelabeler
from transcript_models import Segment
from tests.helpers import run_of, speakers, word


class SpeakerRelabelerTest(unittest.TestCase):
    def test_speaker_with_most_speech_becomes_speaker_00(self):
        words = [word(0, 1, speaker="SPEAKER_00")] + run_of("SPEAKER_01", 1, 10)
        result = SpeakerRelabeler().process(words)
        self.assertEqual(speakers(result), ["SPEAKER_01"] + ["SPEAKER_00"] * 10)

    def test_labels_follow_descending_speaking_time(self):
        words = run_of("x", 0, 2) + run_of("y", 1, 6) + run_of("z", 4, 4)
        result = SpeakerRelabeler().process(words)
        self.assertEqual(
            speakers(result),
            ["SPEAKER_02"] * 2 + ["SPEAKER_00"] * 6 + ["SPEAKER_01"] * 4,
        )

    def test_main_speaker_label_replaces_the_top_speaker_only(self):
        words = [word(0, 1, speaker="X")] + run_of("Y", 1, 10)
        result = SpeakerRelabeler(main_speaker_label="PROFESOR").process(words)
        self.assertEqual(speakers(result), ["SPEAKER_01"] + ["PROFESOR"] * 10)

    def test_ties_are_broken_by_first_appearance(self):
        words = [word(0, 1, speaker="B"), word(1, 2, speaker="A")]
        result = SpeakerRelabeler().process(words)
        self.assertEqual(speakers(result), ["SPEAKER_00", "SPEAKER_01"])

    def test_does_not_modify_the_input_words(self):
        words = [word(0, 1, speaker="B")]
        SpeakerRelabeler().process(words)
        self.assertEqual(speakers(words), ["B"])

    def test_empty_input(self):
        self.assertEqual(SpeakerRelabeler().process([]), [])


class SegmentBuilderTest(unittest.TestCase):
    def setUp(self):
        self.builder = SegmentBuilder(pause_split_sec=2.0)

    def test_merges_consecutive_words_of_the_same_speaker(self):
        words = [word(0, 1, "Hola", "A"), word(1, 2, "mundo", "A")]
        self.assertEqual(self.builder.build(words), [Segment(0, 2, "Hola mundo", "A")])

    def test_splits_on_speaker_change(self):
        words = [word(0, 1, "Hola", "A"), word(1, 2, "Buenas", "B")]
        self.assertEqual(
            self.builder.build(words),
            [Segment(0, 1, "Hola", "A"), Segment(1, 2, "Buenas", "B")],
        )

    def test_splits_on_a_long_pause(self):
        words = [word(0, 1, "Hola", "A"), word(3.5, 4, "otra", "A")]
        self.assertEqual(len(self.builder.build(words)), 2)

    def test_pause_equal_to_the_threshold_does_not_split(self):
        words = [word(0, 1, "Hola", "A"), word(3.0, 4, "sigo", "A")]
        self.assertEqual(len(self.builder.build(words)), 1)

    def test_blank_words_do_not_produce_segments(self):
        words = [word(0, 1, " ", "A")]
        self.assertEqual(self.builder.build(words), [])

    def test_empty_input(self):
        self.assertEqual(self.builder.build([]), [])

    def test_non_positive_pause_is_rejected(self):
        with self.assertRaises(ValueError):
            SegmentBuilder(pause_split_sec=0)


if __name__ == "__main__":
    unittest.main()
