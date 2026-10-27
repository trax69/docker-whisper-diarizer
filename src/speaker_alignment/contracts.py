"""Abstractions the alignment pipeline depends on (Dependency Inversion).

Any object with the right method can be plugged in; no inheritance needed.
"""
from typing import List, Protocol, Sequence

from transcript_models import Segment, SpeakerTurn, WordToken


class SpeakerAssigner(Protocol):
    """Gives every word a speaker label using the diarization turns."""

    def assign(self, words: Sequence[WordToken], turns: Sequence[SpeakerTurn]) -> List[WordToken]:
        ...


class WordPostProcessor(Protocol):
    """Transforms already-labelled words (smoothing, relabelling…). Must not mutate its input."""

    def process(self, words: Sequence[WordToken]) -> List[WordToken]:
        ...


class SegmentAssembler(Protocol):
    """Groups labelled words into transcript segments."""

    def build(self, words: Sequence[WordToken]) -> List[Segment]:
        ...


class TranscriptAligner(Protocol):
    """End-to-end: Whisper words + diarization turns → transcript segments."""

    def run(self, words: Sequence[WordToken], turns: Sequence[SpeakerTurn]) -> List[Segment]:
        ...
