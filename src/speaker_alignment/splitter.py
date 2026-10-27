"""E1 – policies deciding where a long same-speaker paragraph is split.

The SegmentBuilder asks the policy after appending each word; returning True
closes the current segment *after* that word, so no text is lost or repeated.
"""
import re

from transcript_models import Segment, WordToken

_TRAILING_WRAPPERS = "\"'»”’)]}"
_SENTENCE_END = re.compile(r"[.?!…]$")
_SOFT_BOUNDARY = re.compile(r"[,;:]$")


def _ending(word: WordToken) -> str:
    """Word text without trailing quotes/brackets, so 'vale."' ends in '.'."""
    return word.word.strip().rstrip(_TRAILING_WRAPPERS)


class NoSplit:
    """Never splits by duration (legacy behaviour)."""

    def should_split(self, segment: Segment, last_word: WordToken) -> bool:
        return False


class MaxDurationSplitter:
    """Splits paragraphs longer than ``max_segment_sec`` at natural boundaries.

    Measured from the segment start to the end of the word just appended:

    * ``≥ T``  → split after a sentence end (``.`` ``?`` ``!`` ``…``);
    * ``≥ 2T`` → also after a soft boundary (``,`` ``;`` ``:``);
    * ``≥ 3T`` → split anyway (text without punctuation).
    """

    def __init__(self, max_segment_sec: float = 30.0):
        if max_segment_sec <= 0:
            raise ValueError(f"max_segment_sec must be > 0 (got {max_segment_sec})")
        self._limit = max_segment_sec

    def should_split(self, segment: Segment, last_word: WordToken) -> bool:
        duration = last_word.end - segment.start
        if duration < self._limit:
            return False
        if duration >= 3 * self._limit:
            return True
        ending = _ending(last_word)
        if _SENTENCE_END.search(ending):
            return True
        return duration >= 2 * self._limit and bool(_SOFT_BOUNDARY.search(ending))
