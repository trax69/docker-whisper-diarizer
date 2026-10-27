"""P1 – word → speaker assignment based on temporal overlap."""
from bisect import bisect_left, bisect_right
from dataclasses import replace
from typing import Dict, List, Sequence

from transcript_models import SpeakerTurn, WordToken

DEFAULT_SPEAKER = "SPEAKER_00"


class _TurnIndex:
    """Turns sorted by start, plus a running maximum of their end times.

    The running maximum lets us stop scanning backwards as soon as no earlier
    turn can reach the word, even when a very long turn started much earlier.
    """

    def __init__(self, turns: Sequence[SpeakerTurn]):
        self.turns = sorted(turns, key=lambda t: (t.start, t.end))
        self.starts = [t.start for t in self.turns]
        self.max_end_idx: List[int] = []
        best = 0
        for i, t in enumerate(self.turns):
            if t.end > self.turns[best].end:
                best = i
            self.max_end_idx.append(best)

    def max_end_before(self, i: int) -> SpeakerTurn:
        return self.turns[self.max_end_idx[i]]


class OverlapSpeakerAssigner:
    """Gives each word the speaker that overlaps it the most in time.

    * Overlap is accumulated per speaker (a word spanning A-B-A goes to the
      speaker with the most total overlap, not to the first turn listed).
    * Words that fall in a gap with no overlapping turn go to the *nearest*
      turn, before or after (the old code always took the previous one).
    """

    def __init__(self, default_speaker: str = DEFAULT_SPEAKER):
        self._default_speaker = default_speaker

    def assign(self, words: Sequence[WordToken], turns: Sequence[SpeakerTurn]) -> List[WordToken]:
        if not turns:
            return [replace(w, speaker=self._default_speaker) for w in words]
        index = _TurnIndex(turns)
        return [replace(w, speaker=self._speaker_for(w, index)) for w in words]

    def _speaker_for(self, word: WordToken, index: _TurnIndex) -> str:
        return self._by_overlap(word, index) or self._nearest(word, index)

    @staticmethod
    def _by_overlap(word: WordToken, index: _TurnIndex):
        overlaps: Dict[str, float] = {}
        i = bisect_left(index.starts, word.end) - 1  # turns starting before the word ends
        while i >= 0 and index.max_end_before(i).end > word.start:
            t = index.turns[i]
            overlap = min(word.end, t.end) - max(word.start, t.start)
            if overlap > 0:
                overlaps[t.speaker] = overlaps.get(t.speaker, 0.0) + overlap
            i -= 1
        if not overlaps:
            return None
        return max(overlaps.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _nearest(word: WordToken, index: _TurnIndex) -> str:
        def distance(t: SpeakerTurn) -> float:
            return max(0.0, t.start - word.end, word.start - t.end)

        candidates = []
        hi = bisect_right(index.starts, word.end)
        if hi > 0:
            candidates.append(index.max_end_before(hi - 1))  # latest-ending turn so far
        if hi < len(index.turns):
            candidates.append(index.turns[hi])               # first turn after the word
        return min(candidates, key=distance).speaker
