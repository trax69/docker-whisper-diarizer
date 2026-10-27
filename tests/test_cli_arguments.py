"""P4 – speaker-count / smoothing flags of the one-shot CLI (transcribir.py).

transcribir.py imports torch and faster_whisper at module level, so these tests
only run where those are installed (i.e. inside the Docker image).
"""
import contextlib
import importlib.util
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HEAVY_DEPS = all(importlib.util.find_spec(m) for m in ("torch", "faster_whisper", "tqdm"))


def _load_cli():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import transcribir
    return transcribir


@unittest.skipUnless(_HEAVY_DEPS, "torch/faster_whisper not installed (run inside the image)")
class CliArgumentsTest(unittest.TestCase):
    def setUp(self):
        self.cli = _load_cli()

    def test_defaults(self):
        config = self.cli.parse_arguments(["-i", "in.mp4", "-o", "out.txt"])
        self.assertEqual(config.speaker_hint, self.cli.SpeakerCountHint())
        self.assertEqual(config.alignment, self.cli.AlignmentConfig())

    def test_speaker_range_and_smoothing_flags(self):
        config = self.cli.parse_arguments([
            "-i", "in.mp4", "-o", "out.txt", "--min-speakers", "1", "--max-speakers", "3",
            "--min-turn-words", "4", "--min-turn-sec", "2", "--main-speaker-label", "PROFESOR",
        ])
        self.assertEqual(config.speaker_hint,
                         self.cli.SpeakerCountHint(min_speakers=1, max_speakers=3))
        self.assertEqual(config.alignment, self.cli.AlignmentConfig(
            min_turn_words=4, min_turn_sec=2.0, main_speaker_label="PROFESOR"))

    def test_max_segment_flag(self):
        config = self.cli.parse_arguments(["-i", "a", "-o", "b", "--max-segment-sec", "45"])
        self.assertEqual(config.alignment.max_segment_sec, 45.0)
        config = self.cli.parse_arguments(["-i", "a", "-o", "b", "--max-segment-sec", "0"])
        self.assertEqual(config.alignment.max_segment_sec, 0.0)

    def test_negative_max_segment_is_rejected(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.cli.parse_arguments(["-i", "a", "-o", "b", "--max-segment-sec", "-1"])

    def test_exact_speakers_flag_is_kept(self):
        config = self.cli.parse_arguments(["-i", "a", "-o", "b", "-n", "2"])
        self.assertEqual(config.speaker_hint, self.cli.SpeakerCountHint(num_speakers=2))

    def test_inconsistent_range_is_rejected(self):
        stderr = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(stderr):
            self.cli.parse_arguments(["-i", "a", "-o", "b", "--min-speakers", "3",
                                      "--max-speakers", "2"])
        self.assertIn("transcribir.py: error: min_speakers (3)", stderr.getvalue())

    def test_negative_smoothing_threshold_is_rejected(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.cli.parse_arguments(["-i", "a", "-o", "b", "--min-turn-sec", "-1"])


if __name__ == "__main__":
    unittest.main()
