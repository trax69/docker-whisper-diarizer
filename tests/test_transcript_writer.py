"""Shared output format (used by both the daemon and the CLI)."""
import io
import unittest

from transcript_models import Segment
from transcript_writer import format_segment, write_segments


class TranscriptWriterTest(unittest.TestCase):
    def test_format_keeps_the_existing_line_format(self):
        self.assertEqual(format_segment(Segment(2.16, 17.3, " Hola ", "SPEAKER_00")),
                         "[2.16s - 17.30s] SPEAKER_00: Hola")

    def test_write_segments_writes_one_line_per_segment(self):
        buffer = io.StringIO()
        write_segments(buffer, [Segment(0, 1, "a", "X"), Segment(1, 2, "b", "Y")])
        self.assertEqual(buffer.getvalue(), "[0.00s - 1.00s] X: a\n[1.00s - 2.00s] Y: b\n")


if __name__ == "__main__":
    unittest.main()
