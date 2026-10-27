"""Single source of truth for the transcript line format (daemon and CLI)."""
from typing import Iterable, TextIO

from transcript_models import Segment


def format_segment(segment: Segment) -> str:
    return f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.speaker}: {segment.text.strip()}"


def write_segments(stream: TextIO, segments: Iterable[Segment]) -> None:
    for segment in segments:
        stream.write(format_segment(segment) + "\n")
