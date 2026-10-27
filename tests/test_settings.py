"""P4 – daemon configuration read from environment variables."""
import unittest

from settings import alignment_config_from_env, speaker_count_hint_from_env
from speaker_alignment import AlignmentConfig
from speaker_count import SpeakerCountHint


class SpeakerCountHintFromEnvTest(unittest.TestCase):
    def test_missing_variables_mean_auto_detection(self):
        self.assertEqual(speaker_count_hint_from_env({}), SpeakerCountHint())

    def test_empty_strings_mean_auto_detection(self):
        env = {"NUM_SPEAKERS": "", "MIN_SPEAKERS": " ", "MAX_SPEAKERS": ""}
        self.assertEqual(speaker_count_hint_from_env(env), SpeakerCountHint())

    def test_reads_all_variables(self):
        env = {"MIN_SPEAKERS": "1", "MAX_SPEAKERS": "4"}
        self.assertEqual(speaker_count_hint_from_env(env),
                         SpeakerCountHint(min_speakers=1, max_speakers=4))

    def test_reads_exact_number(self):
        self.assertEqual(speaker_count_hint_from_env({"NUM_SPEAKERS": "2"}),
                         SpeakerCountHint(num_speakers=2))

    def test_invalid_number_names_the_variable(self):
        with self.assertRaisesRegex(ValueError, "MAX_SPEAKERS"):
            speaker_count_hint_from_env({"MAX_SPEAKERS": "dos"})

    def test_inconsistent_values_are_rejected(self):
        with self.assertRaises(ValueError):
            speaker_count_hint_from_env({"MIN_SPEAKERS": "3", "MAX_SPEAKERS": "2"})


class AlignmentConfigFromEnvTest(unittest.TestCase):
    def test_defaults_when_nothing_is_set(self):
        self.assertEqual(alignment_config_from_env({}), AlignmentConfig())

    def test_reads_all_variables(self):
        env = {"MIN_TURN_WORDS": "5", "MIN_TURN_SEC": "2.5",
               "MAIN_SPEAKER_LABEL": "PROFESOR", "PAUSE_SPLIT_SEC": "3"}
        self.assertEqual(
            alignment_config_from_env(env),
            AlignmentConfig(min_turn_words=5, min_turn_sec=2.5,
                            main_speaker_label="PROFESOR", pause_split_sec=3.0),
        )

    def test_decimal_comma_is_accepted(self):
        self.assertEqual(alignment_config_from_env({"MIN_TURN_SEC": "1,5"}).min_turn_sec, 1.5)

    def test_blank_label_means_no_label(self):
        self.assertIsNone(alignment_config_from_env({"MAIN_SPEAKER_LABEL": "  "}).main_speaker_label)

    def test_label_is_trimmed(self):
        config = alignment_config_from_env({"MAIN_SPEAKER_LABEL": " PROFESOR "})
        self.assertEqual(config.main_speaker_label, "PROFESOR")

    def test_invalid_number_names_the_variable(self):
        with self.assertRaisesRegex(ValueError, "MIN_TURN_SEC"):
            alignment_config_from_env({"MIN_TURN_SEC": "mucho"})
        with self.assertRaisesRegex(ValueError, "MIN_TURN_WORDS"):
            alignment_config_from_env({"MIN_TURN_WORDS": "1.5"})


if __name__ == "__main__":
    unittest.main()
