"""P5 – stable speaker labels ordered by speaking time."""
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from transcript_models import WordToken


class SpeakerRelabeler:
    """Renames speakers so ``SPEAKER_00`` is always whoever speaks the most.

    pyannote's labels are arbitrary (the teacher may come out as SPEAKER_03).
    Ranking by speaking time makes outputs consistent across files. Ties are
    broken by first appearance. Optionally the top speaker gets a custom label
    (e.g. ``PROFESOR``); the rest keep ``SPEAKER_NN`` with their rank.
    """

    def __init__(self, main_speaker_label: Optional[str] = None):
        self._main_label = main_speaker_label

    def process(self, words: Sequence[WordToken]) -> List[WordToken]:
        mapping = self._mapping(words)
        return [replace(w, speaker=mapping[w.speaker]) for w in words]

    def _mapping(self, words: Sequence[WordToken]) -> Dict[str, str]:
        talk_time: Dict[str, float] = {}
        for w in words:  # dict keeps insertion order → first appearance
            talk_time[w.speaker] = talk_time.get(w.speaker, 0.0) + w.duration
        ranked = sorted(talk_time, key=lambda spk: -talk_time[spk])  # stable sort
        mapping = {spk: f"SPEAKER_{rank:02d}" for rank, spk in enumerate(ranked)}
        if ranked and self._main_label:
            mapping[ranked[0]] = self._main_label
        return mapping
