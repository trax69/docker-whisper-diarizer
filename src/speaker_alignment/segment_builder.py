"""Groups labelled words into transcript segments."""
from typing import List, Sequence

from transcript_models import Segment, WordToken


class SegmentBuilder:
    """Joins consecutive words of the same speaker into one segment.

    A new segment starts when the speaker changes or when the silence between
    two words is longer than ``pause_split_sec``.
    """

    def __init__(self, pause_split_sec: float = 2.0):
        if pause_split_sec <= 0:
            raise ValueError(f"pause_split_sec must be > 0 (got {pause_split_sec})")
        self._pause_split_sec = pause_split_sec

    def build(self, words: Sequence[WordToken]) -> List[Segment]:
        result: List[Segment] = []
        if not words:
            return result

        current = Segment(words[0].start, words[0].end, words[0].word, words[0].speaker)
        for w in words[1:]:
            pause = w.start - current.end
            if w.speaker == current.speaker and pause <= self._pause_split_sec:
                current.text += w.word
                current.end = w.end
            else:
                self._flush(current, result)
                current = Segment(w.start, w.end, w.word, w.speaker)
        self._flush(current, result)
        return result

    @staticmethod
    def _flush(segment: Segment, result: List[Segment]) -> None:
        text = segment.text.strip()
        if text:
            result.append(Segment(segment.start, segment.end, text, segment.speaker))
