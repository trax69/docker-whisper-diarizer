"""P2 – absorb speaker turns too short to be a real intervention."""
import heapq
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from transcript_models import WordToken


class _Run:
    """A block of consecutive words with the same speaker (doubly linked).

    ``evidence`` keeps, per *original* diarization label, how many words the
    run contains and the time they span. A run is judged only by the evidence
    for its current speaker, so absorbing someone else's stray word never makes
    a dubious turn look long enough to survive.
    """
    __slots__ = ("speaker", "first", "last", "start", "end", "evidence",
                 "prev", "next", "alive", "version")

    def __init__(self, speaker: str, first: int, word: WordToken):
        self.speaker = speaker
        self.first = self.last = first
        self.start, self.end = word.start, word.end
        self.evidence: Dict[str, List[float]] = {speaker: [1, word.start, word.end]}
        self.prev: Optional["_Run"] = None
        self.next: Optional["_Run"] = None
        self.alive = True
        self.version = 0

    @property
    def own_count(self) -> int:
        return int(self.evidence[self.speaker][0]) if self.speaker in self.evidence else 0

    @property
    def own_span(self) -> float:
        if self.speaker not in self.evidence:
            return 0.0
        _, start, end = self.evidence[self.speaker]
        return max(0.0, end - start)

    def add_word(self, index: int, word: WordToken) -> None:
        """Extend the run with the next word (same original speaker)."""
        self.last, self.end = index, max(self.end, word.end)
        own = self.evidence[self.speaker]
        own[0] += 1
        own[2] = max(own[2], word.end)

    def absorb_right(self, other: "_Run") -> None:
        """Append the adjacent run `other` (which must be self.next) to this run."""
        self.last, self.end = other.last, max(self.end, other.end)
        for speaker, (count, start, end) in other.evidence.items():
            mine = self.evidence.setdefault(speaker, [0, start, end])
            mine[0] += count
            mine[1], mine[2] = min(mine[1], start), max(mine[2], end)
        self.next = other.next
        if other.next is not None:
            other.next.prev = self
        other.alive = False
        self.version += 1


class ShortTurnSmoother:
    """Relabels speaker turns that are too short to be trusted.

    A *turn* is a block of consecutive words of one speaker with no silence
    longer than ``pause_split_sec`` (the same rule that splits output lines),
    so stray fragments separated by long silences are judged one by one.

    A turn is *short* when it has fewer than ``min_turn_words`` words **or**
    lasts less than ``min_turn_sec`` seconds. Short turns are merged, shortest
    first, into a neighbouring turn:

    1. both neighbours are the same speaker → that speaker (the common case:
       a stray "pues" inside the teacher's sentence);
    2. only one neighbour (start/end of the file) → that neighbour;
    3. otherwise → the neighbour closest in time (ties go to the previous one).

    Set both thresholds to 0 to disable smoothing.
    """

    def __init__(self, min_turn_words: int = 3, min_turn_sec: float = 1.5,
                 pause_split_sec: float = 2.0):
        if min_turn_words < 0:
            raise ValueError(f"min_turn_words must be >= 0 (got {min_turn_words})")
        if min_turn_sec < 0:
            raise ValueError(f"min_turn_sec must be >= 0 (got {min_turn_sec})")
        if pause_split_sec <= 0:
            raise ValueError(f"pause_split_sec must be > 0 (got {pause_split_sec})")
        self._min_words = min_turn_words
        self._min_sec = min_turn_sec
        self._pause_split_sec = pause_split_sec

    def process(self, words: Sequence[WordToken]) -> List[WordToken]:
        if not words:
            return []
        head = self._build_runs(words)
        self._smooth(head)
        return self._relabel(words, head)

    # ------------------------------------------------------------------

    def _is_short(self, run: _Run) -> bool:
        return run.own_count < self._min_words or run.own_span < self._min_sec

    def _build_runs(self, words: Sequence[WordToken]) -> _Run:
        head = tail = _Run(words[0].speaker, 0, words[0])
        for i in range(1, len(words)):
            w = words[i]
            if w.speaker == tail.speaker and (w.start - tail.end) <= self._pause_split_sec:
                tail.add_word(i, w)
            else:
                run = _Run(w.speaker, i, w)
                run.prev, tail.next = tail, run
                tail = run
        return head

    def _smooth(self, head: _Run) -> None:
        heap: list = []
        seq = 0

        def push(run: _Run) -> None:
            nonlocal seq
            if self._is_short(run):
                heapq.heappush(heap, (run.own_count, run.own_span, seq, run.version, run))
                seq += 1

        run: Optional[_Run] = head
        while run is not None:
            push(run)
            run = run.next

        while heap:
            _, _, _, version, run = heapq.heappop(heap)
            if not run.alive or run.version != version or not self._is_short(run):
                continue
            if run.prev is None and run.next is None:
                return  # a single turn left: nothing to merge with
            push(self._merge_into_neighbour(run))

    @staticmethod
    def _merge_into_neighbour(run: _Run) -> _Run:
        prev, nxt = run.prev, run.next
        if prev is not None and nxt is not None and prev.speaker == nxt.speaker:  # rule 1
            prev.absorb_right(run)
            prev.absorb_right(nxt)
            return prev
        if nxt is None or (prev is not None and                                    # rules 2-3
                           (run.start - prev.end) <= (nxt.start - run.end)):
            return ShortTurnSmoother._merge_left(run)
        return ShortTurnSmoother._merge_right(run)

    @staticmethod
    def _merge_left(run: _Run) -> _Run:
        prev = run.prev
        prev.absorb_right(run)
        return prev

    @staticmethod
    def _merge_right(run: _Run) -> _Run:
        run.speaker = run.next.speaker
        run.absorb_right(run.next)
        return run

    @staticmethod
    def _relabel(words: Sequence[WordToken], head: _Run) -> List[WordToken]:
        result: List[WordToken] = []
        run: Optional[_Run] = head
        while run is not None:
            result.extend(replace(words[i], speaker=run.speaker)
                          for i in range(run.first, run.last + 1))
            run = run.next
        return result
