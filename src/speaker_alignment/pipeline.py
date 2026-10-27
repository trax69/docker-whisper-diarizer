"""P6 – composes assignment, post-processing and segment building."""
from dataclasses import dataclass
from typing import List, Optional, Sequence

from transcript_models import Segment, SpeakerTurn, WordToken

from .assigner import OverlapSpeakerAssigner
from .contracts import SegmentAssembler, SpeakerAssigner, WordPostProcessor
from .relabeler import SpeakerRelabeler
from .segment_builder import SegmentBuilder
from .smoother import ShortTurnSmoother


@dataclass(frozen=True)
class AlignmentConfig:
    """Tuning knobs for turning words + diarization turns into a transcript.

    * ``min_turn_words`` / ``min_turn_sec`` – a speaker turn with fewer words
      **or** shorter than this is absorbed by its neighbours (0 and 0 disable it).
    * ``main_speaker_label`` – optional label for whoever speaks most (e.g. PROFESOR).
    * ``pause_split_sec`` – silence that starts a new line for the same speaker.
    """
    min_turn_words: int = 3
    min_turn_sec: float = 1.5
    main_speaker_label: Optional[str] = None
    pause_split_sec: float = 2.0


class SpeakerAlignmentPipeline:
    """assign speakers → post-process words (in order) → build segments."""

    def __init__(
        self,
        assigner: SpeakerAssigner,
        post_processors: Sequence[WordPostProcessor],
        segment_builder: SegmentAssembler,
    ):
        self._assigner = assigner
        self._post_processors = tuple(post_processors)
        self._segment_builder = segment_builder

    def run(self, words: Sequence[WordToken], turns: Sequence[SpeakerTurn]) -> List[Segment]:
        labelled = self._assigner.assign(words, turns)
        for processor in self._post_processors:
            labelled = processor.process(labelled)
        return self._segment_builder.build(labelled)


def build_alignment_pipeline(config: AlignmentConfig) -> SpeakerAlignmentPipeline:
    """Default wiring used by both the daemon and the CLI."""
    return SpeakerAlignmentPipeline(
        assigner=OverlapSpeakerAssigner(),
        post_processors=[
            ShortTurnSmoother(config.min_turn_words, config.min_turn_sec,
                              config.pause_split_sec),
            SpeakerRelabeler(config.main_speaker_label),
        ],
        segment_builder=SegmentBuilder(config.pause_split_sec),
    )
