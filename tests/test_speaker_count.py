"""P4 – speaker-count hints forwarded to pyannote."""
import unittest

from speaker_count import SpeakerCountHint


class SpeakerCountHintTest(unittest.TestCase):
    def test_empty_hint_produces_no_kwargs(self):
        self.assertEqual(SpeakerCountHint().to_pipeline_kwargs(), {})

    def test_exact_number_of_speakers(self):
        self.assertEqual(SpeakerCountHint(num_speakers=2).to_pipeline_kwargs(),
                         {"num_speakers": 2})

    def test_min_and_max_speakers(self):
        hint = SpeakerCountHint(min_speakers=1, max_speakers=3)
        self.assertEqual(hint.to_pipeline_kwargs(), {"min_speakers": 1, "max_speakers": 3})

    def test_only_max_speakers(self):
        self.assertEqual(SpeakerCountHint(max_speakers=4).to_pipeline_kwargs(),
                         {"max_speakers": 4})

    def test_exact_number_takes_precedence_over_range(self):
        hint = SpeakerCountHint(num_speakers=2, min_speakers=1, max_speakers=3)
        self.assertEqual(hint.to_pipeline_kwargs(), {"num_speakers": 2})

    def test_non_positive_values_are_rejected(self):
        for kwargs in ({"num_speakers": 0}, {"min_speakers": 0}, {"max_speakers": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpeakerCountHint(**kwargs)

    def test_min_greater_than_max_is_rejected(self):
        with self.assertRaises(ValueError):
            SpeakerCountHint(min_speakers=3, max_speakers=2)

    def test_exact_number_outside_the_range_is_rejected(self):
        with self.assertRaises(ValueError):
            SpeakerCountHint(num_speakers=5, max_speakers=3)

    def test_describe_is_human_readable(self):
        self.assertEqual(SpeakerCountHint().describe(), "auto")
        self.assertEqual(SpeakerCountHint(num_speakers=2).describe(), "num_speakers=2")
        self.assertEqual(SpeakerCountHint(min_speakers=1, max_speakers=3).describe(),
                         "min_speakers=1, max_speakers=3")


if __name__ == "__main__":
    unittest.main()
