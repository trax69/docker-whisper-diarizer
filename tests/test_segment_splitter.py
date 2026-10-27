"""E1 – split long same-speaker paragraphs at sentence boundaries."""
import unittest

from speaker_alignment import MaxDurationSplitter, NoSplit, SegmentBuilder
from transcript_models import Segment
from tests.helpers import word


def sentence_words(start, texts, dur=1.0, speaker="A"):
    """One word per text, `dur` seconds each, back to back from `start`."""
    return [word(start + i * dur, start + (i + 1) * dur, t, speaker) for i, t in enumerate(texts)]


class NoSplitTest(unittest.TestCase):
    def test_never_splits(self):
        self.assertFalse(NoSplit().should_split(Segment(0, 0, ""), word(999, 1000, "fin.")))


class MaxDurationSplitterTest(unittest.TestCase):
    def setUp(self):
        self.splitter = MaxDurationSplitter(max_segment_sec=30)

    def segment(self, start=0.0):
        return Segment(start, start, "")

    def test_does_not_split_before_the_threshold_even_at_a_full_stop(self):
        self.assertFalse(self.splitter.should_split(self.segment(), word(28, 29, "fin.")))

    def test_splits_at_the_first_sentence_end_after_the_threshold(self):
        for text in ("fin.", "¿vale?", "genial!", "bueno…"):
            with self.subTest(text=text):
                self.assertTrue(self.splitter.should_split(self.segment(), word(30, 31, text)))

    def test_does_not_split_at_a_comma_before_twice_the_threshold(self):
        self.assertFalse(self.splitter.should_split(self.segment(), word(45, 46, "bueno,")))

    def test_splits_at_a_soft_boundary_after_twice_the_threshold(self):
        for text in ("bueno,", "ejemplo;", "es:"):
            with self.subTest(text=text):
                self.assertTrue(self.splitter.should_split(self.segment(), word(59, 60, text)))

    def test_forces_a_split_after_three_times_the_threshold(self):
        self.assertTrue(self.splitter.should_split(self.segment(), word(89, 90, "y")))
        self.assertFalse(self.splitter.should_split(self.segment(), word(88, 89, "y")))

    def test_duration_is_measured_from_the_segment_start(self):
        self.assertFalse(self.splitter.should_split(self.segment(start=100), word(120, 121, "fin.")))
        self.assertTrue(self.splitter.should_split(self.segment(start=100), word(130, 131, "fin.")))

    def test_trailing_quotes_or_brackets_do_not_hide_the_punctuation(self):
        self.assertTrue(self.splitter.should_split(self.segment(), word(30, 31, 'dijo "vale."')))
        self.assertTrue(self.splitter.should_split(self.segment(), word(30, 31, "(fin.)")))

    def test_zero_or_negative_threshold_is_rejected(self):
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MaxDurationSplitter(max_segment_sec=value)


class SegmentBuilderWithSplitterTest(unittest.TestCase):
    def test_long_monologue_is_split_at_sentence_ends(self):
        texts = ["palabra"] * 34 + ["fin."] + ["palabra"] * 10 + ["final."]
        builder = SegmentBuilder(pause_split_sec=2.0, splitter=MaxDurationSplitter(30))

        segments = builder.build(sentence_words(0.0, texts))

        self.assertEqual(len(segments), 2)
        self.assertTrue(segments[0].text.endswith("fin."))
        self.assertEqual((segments[0].start, segments[0].end), (0.0, 35.0))
        self.assertEqual((segments[1].start, segments[1].end), (35.0, 46.0))

    def test_no_text_is_lost_or_duplicated(self):
        texts = [f"p{i}." if i % 7 == 0 else f"p{i}" for i in range(300)]
        builder = SegmentBuilder(pause_split_sec=2.0, splitter=MaxDurationSplitter(30))

        segments = builder.build(sentence_words(0.0, texts))

        self.assertEqual(" ".join(s.text for s in segments), " ".join(texts))
        self.assertTrue(all(s.end - s.start <= 90 for s in segments))

    def test_text_without_punctuation_is_force_split(self):
        builder = SegmentBuilder(pause_split_sec=2.0, splitter=MaxDurationSplitter(30))
        segments = builder.build(sentence_words(0.0, ["palabra"] * 200))
        self.assertEqual([round(s.end - s.start) for s in segments], [90, 90, 20])

    def test_last_word_never_leaves_an_empty_segment(self):
        builder = SegmentBuilder(pause_split_sec=2.0, splitter=MaxDurationSplitter(30))
        segments = builder.build(sentence_words(0.0, ["palabra"] * 30 + ["fin."]))
        self.assertEqual(len(segments), 1)

    def test_speaker_change_and_pauses_still_split(self):
        words = (sentence_words(0.0, ["hola", "qué", "tal."], speaker="A")
                 + sentence_words(3.0, ["bien."], speaker="B")
                 + sentence_words(10.0, ["sigo."], speaker="B"))
        builder = SegmentBuilder(pause_split_sec=2.0, splitter=MaxDurationSplitter(30))
        self.assertEqual([s.text for s in builder.build(words)], ["hola qué tal.", "bien.", "sigo."])

    def test_default_builder_does_not_split_by_duration(self):
        segments = SegmentBuilder(pause_split_sec=2.0).build(sentence_words(0.0, ["fin."] * 200))
        self.assertEqual(len(segments), 1)


if __name__ == "__main__":
    unittest.main()
