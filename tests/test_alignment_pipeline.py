"""P6 – composed alignment pipeline (assign → post-process → build segments)."""
import unittest
from dataclasses import replace

from speaker_alignment import (
    AlignmentConfig,
    SegmentBuilder,
    SpeakerAlignmentPipeline,
    build_alignment_pipeline,
)
from transcript_models import Segment
from tests.helpers import turn, unlabelled_run, word


class _FixedAssigner:
    def assign(self, words, turns):
        return [replace(w, speaker="S") for w in words]


class _TagProcessor:
    def __init__(self, tag, log):
        self._tag, self._log = tag, log

    def process(self, words):
        self._log.append(self._tag)
        return [replace(w, speaker=w.speaker + self._tag) for w in words]


class SpeakerAlignmentPipelineTest(unittest.TestCase):
    def test_applies_post_processors_in_order(self):
        log = []
        pipeline = SpeakerAlignmentPipeline(
            assigner=_FixedAssigner(),
            post_processors=[_TagProcessor("1", log), _TagProcessor("2", log)],
            segment_builder=SegmentBuilder(),
        )
        segments = pipeline.run([word(0, 1, "hola")], [])
        self.assertEqual(log, ["1", "2"])
        self.assertEqual(segments, [Segment(0, 1, "hola", "S12")])

    def test_empty_words_give_no_segments(self):
        self.assertEqual(build_alignment_pipeline(AlignmentConfig()).run([], []), [])


class BuildAlignmentPipelineTest(unittest.TestCase):
    def test_spurious_one_word_turn_is_removed(self):
        # Mirrors "[17.30s - 17.86s] SPEAKER_01: pues" in the real transcript.
        words = (unlabelled_run(2.0, 20) + [word(17.30, 17.86, "pues")]
                 + unlabelled_run(17.9, 20))
        turns = [turn(2.0, 17.2, "SPEAKER_00"), turn(17.25, 17.9, "SPEAKER_01"),
                 turn(17.95, 33.0, "SPEAKER_00")]
        pipeline = build_alignment_pipeline(AlignmentConfig(main_speaker_label="PROFESOR"))

        segments = pipeline.run(words, turns)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].speaker, "PROFESOR")
        self.assertIn("pues", segments[0].text)

    def test_real_student_question_is_preserved(self):
        words = (unlabelled_run(0.0, 13) + unlabelled_run(11.0, 8, dur=0.4)
                 + unlabelled_run(15.0, 13))
        turns = [turn(0.0, 10.2, "SPEAKER_01"), turn(10.9, 14.4, "SPEAKER_00"),
                 turn(14.8, 25.0, "SPEAKER_01")]

        segments = build_alignment_pipeline(AlignmentConfig()).run(words, turns)

        self.assertEqual([s.speaker for s in segments],
                         ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])

    def test_smoothing_can_be_disabled(self):
        words = (unlabelled_run(2.0, 20) + [word(17.30, 17.86, "pues")]
                 + unlabelled_run(17.9, 20))
        turns = [turn(2.0, 17.2, "A"), turn(17.25, 17.9, "B"), turn(17.95, 33.0, "A")]
        config = AlignmentConfig(min_turn_words=0, min_turn_sec=0)

        segments = build_alignment_pipeline(config).run(words, turns)

        self.assertEqual([s.speaker for s in segments],
                         ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])

    def test_without_diarization_everything_is_a_single_speaker(self):
        segments = build_alignment_pipeline(AlignmentConfig()).run(unlabelled_run(0.0, 5), [])
        self.assertEqual([s.speaker for s in segments], ["SPEAKER_00"])

    def test_invalid_config_fails_fast(self):
        with self.assertRaises(ValueError):
            build_alignment_pipeline(AlignmentConfig(min_turn_sec=-1))

    def test_default_config_values(self):
        config = AlignmentConfig()
        self.assertEqual(config.min_turn_words, 3)
        self.assertEqual(config.min_turn_sec, 1.5)
        self.assertIsNone(config.main_speaker_label)
        self.assertEqual(config.pause_split_sec, 2.0)
        self.assertEqual(config.max_segment_sec, 30.0)

    def test_long_paragraph_is_split_by_default(self):
        texts = [f"p{i}." if i % 10 == 9 else f"p{i}" for i in range(480)]  # 6 min monologue
        words = [word(i * 0.75, (i + 1) * 0.75, t) for i, t in enumerate(texts)]

        segments = build_alignment_pipeline(AlignmentConfig()).run(words, [])

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(s.end - s.start <= 90 for s in segments))
        self.assertEqual(" ".join(s.text for s in segments), " ".join(texts))

    def test_max_segment_zero_disables_splitting(self):
        words = [word(i * 0.75, (i + 1) * 0.75, "fin.") for i in range(480)]
        segments = build_alignment_pipeline(AlignmentConfig(max_segment_sec=0)).run(words, [])
        self.assertEqual(len(segments), 1)

    def test_negative_max_segment_fails_fast(self):
        with self.assertRaises(ValueError):
            build_alignment_pipeline(AlignmentConfig(max_segment_sec=-5))


if __name__ == "__main__":
    unittest.main()
