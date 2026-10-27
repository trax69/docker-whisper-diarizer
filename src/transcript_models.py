"""Plain data structures shared by the daemon, the CLI and the alignment pipeline.

Kept free of heavy dependencies (torch, pyannote, faster-whisper) so every
module that uses them can be unit-tested without a GPU.
"""
from dataclasses import dataclass


@dataclass
class WordToken:
    """A word produced by Whisper with its timestamps and, once assigned, its speaker."""
    start: float
    end: float
    word: str
    speaker: str = "UNKNOWN"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Segment:
    """A line of the final transcript: consecutive words of one speaker."""
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"


@dataclass(frozen=True)
class SpeakerTurn:
    """A diarization turn: `speaker` is talking between `start` and `end` (seconds)."""
    start: float
    end: float
    speaker: str
