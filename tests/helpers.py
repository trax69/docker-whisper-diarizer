"""Small builders shared by the test modules."""
from typing import List

from transcript_models import SpeakerTurn, WordToken


def word(start: float, end: float, text: str = "palabra", speaker: str = "UNKNOWN") -> WordToken:
    """Whisper emits words with a leading space, so the helper does too."""
    return WordToken(start, end, f" {text}", speaker)


def turn(start: float, end: float, speaker: str) -> SpeakerTurn:
    return SpeakerTurn(start, end, speaker)


def run_of(speaker: str, start: float, count: int, dur: float = 0.5) -> List[WordToken]:
    """`count` consecutive words of `dur` seconds each, already labelled."""
    return [word(start + i * dur, start + (i + 1) * dur, speaker=speaker) for i in range(count)]


def unlabelled_run(start: float, count: int, dur: float = 0.75) -> List[WordToken]:
    """`count` consecutive words without speaker (as they come from Whisper)."""
    return [word(start + i * dur, start + (i + 1) * dur) for i in range(count)]


def speakers(words: List[WordToken]) -> List[str]:
    return [w.speaker for w in words]
