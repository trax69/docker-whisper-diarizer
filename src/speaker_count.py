"""Hints about how many speakers pyannote should look for."""
from dataclasses import dataclass
from typing import Dict, Optional

_FIELDS = ("num_speakers", "min_speakers", "max_speakers")


@dataclass(frozen=True)
class SpeakerCountHint:
    """Optional constraints forwarded to the pyannote diarization pipeline.

    * ``num_speakers`` – exact number of speakers (takes precedence).
    * ``min_speakers`` / ``max_speakers`` – range used when the exact number is unknown.

    All ``None`` means "let pyannote estimate it".
    """
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    def __post_init__(self) -> None:
        for name in _FIELDS:
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 (got {value})")

        low, high, exact = self.min_speakers, self.max_speakers, self.num_speakers
        if low is not None and high is not None and low > high:
            raise ValueError(f"min_speakers ({low}) cannot be greater than max_speakers ({high})")
        if exact is not None and ((low is not None and exact < low)
                                  or (high is not None and exact > high)):
            raise ValueError(
                f"num_speakers ({exact}) is outside the range "
                f"[{low if low is not None else 1}, {high if high is not None else '∞'}]"
            )

    def to_pipeline_kwargs(self) -> Dict[str, int]:
        """Keyword arguments for ``pyannote.audio.Pipeline.__call__``."""
        if self.num_speakers is not None:
            return {"num_speakers": self.num_speakers}
        kwargs: Dict[str, int] = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers
        return kwargs

    def describe(self) -> str:
        """Short human-readable form for logs."""
        kwargs = self.to_pipeline_kwargs()
        return ", ".join(f"{key}={value}" for key, value in kwargs.items()) or "auto"
