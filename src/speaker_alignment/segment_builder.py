"""Groups labelled words into transcript segments."""
from typing import List, Optional, Sequence

from transcript_models import Segment, WordToken

from .contracts import SegmentSplitter
from .splitter import NoSplit


class SegmentBuilder:
    """Joins consecutive words of the same speaker into one segment.

    A new segment starts when the speaker changes, when the silence between
    two words is longer than ``pause_split_sec``, or when the injected
    ``splitter`` says so (e.g. long paragraphs at a sentence end).
    """

    def __init__(self, pause_split_sec: float = 2.0,
                 splitter: Optional[SegmentSplitter] = None):
        if pause_split_sec <= 0:
            raise ValueError(f"pause_split_sec must be > 0 (got {pause_split_sec})")
        self._pause_split_sec = pause_split_sec
        self._splitter = splitter or NoSplit()

    def build(self, words: Sequence[WordToken]) -> List[Segment]:
        result: List[Segment] = []
        current: Optional[Segment] = None
        for w in words:
            if current is not None and self._continues(current, w):
                current.text += w.word
                current.end = w.end
            else:
                self._flush(current, result)
                current = Segment(w.start, w.end, w.word, w.speaker)
            if self._splitter.should_split(current, w):
                self._flush(current, result)
                current = None
        self._flush(current, result)
        return result

    def _continues(self, current: Segment, w: WordToken) -> bool:
        return w.speaker == current.speaker and (w.start - current.end) <= self._pause_split_sec

    @staticmethod
    def _flush(segment: Optional[Segment], result: List[Segment]) -> None:
        if segment is None:
            return
        text = segment.text.strip()
        if text:
            result.append(Segment(segment.start, segment.end, text, segment.speaker))
