"""Word-level speaker alignment: Whisper words + pyannote turns → transcript lines.

Pure Python (no torch / pyannote / faster-whisper) so it is fast to unit-test
and shared by the daemon (``src/engine.py``) and the CLI (``transcribir.py``).
"""
from .assigner import DEFAULT_SPEAKER, OverlapSpeakerAssigner
from .contracts import (
    SegmentAssembler,
    SegmentSplitter,
    SpeakerAssigner,
    TranscriptAligner,
    WordPostProcessor,
)
from .pipeline import AlignmentConfig, SpeakerAlignmentPipeline, build_alignment_pipeline
from .relabeler import SpeakerRelabeler
from .segment_builder import SegmentBuilder
from .smoother import ShortTurnSmoother
from .splitter import MaxDurationSplitter, NoSplit

__all__ = [
    "AlignmentConfig",
    "DEFAULT_SPEAKER",
    "MaxDurationSplitter",
    "NoSplit",
    "OverlapSpeakerAssigner",
    "SegmentAssembler",
    "SegmentBuilder",
    "SegmentSplitter",
    "ShortTurnSmoother",
    "SpeakerAlignmentPipeline",
    "SpeakerAssigner",
    "SpeakerRelabeler",
    "TranscriptAligner",
    "WordPostProcessor",
    "build_alignment_pipeline",
]
