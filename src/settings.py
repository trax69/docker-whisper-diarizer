"""Daemon configuration read from environment variables (see docker-compose.yml).

Blank or missing variables fall back to the defaults, so compose entries like
``MIN_SPEAKERS=${MIN_SPEAKERS:-}`` are safe when the value is not set in .env.
"""
from typing import Callable, Mapping, Optional, TypeVar

from speaker_alignment import AlignmentConfig
from speaker_count import SpeakerCountHint

T = TypeVar("T")


def _raw(env: Mapping[str, str], name: str) -> Optional[str]:
    value = env.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _parse(env: Mapping[str, str], name: str, convert: Callable[[str], T], default: T) -> T:
    raw = _raw(env, name)
    if raw is None:
        return default
    try:
        return convert(raw)
    except ValueError:
        raise ValueError(f"Invalid value for {name}: {raw!r}") from None


def _to_float(raw: str) -> float:
    return float(raw.replace(",", "."))  # accept Spanish decimal comma: 1,5


def speaker_count_hint_from_env(env: Mapping[str, str]) -> SpeakerCountHint:
    """NUM_SPEAKERS, MIN_SPEAKERS, MAX_SPEAKERS → SpeakerCountHint (blank = auto)."""
    return SpeakerCountHint(
        num_speakers=_parse(env, "NUM_SPEAKERS", int, None),
        min_speakers=_parse(env, "MIN_SPEAKERS", int, None),
        max_speakers=_parse(env, "MAX_SPEAKERS", int, None),
    )


def alignment_config_from_env(env: Mapping[str, str]) -> AlignmentConfig:
    """MIN_TURN_WORDS, MIN_TURN_SEC, MAIN_SPEAKER_LABEL, PAUSE_SPLIT_SEC → AlignmentConfig."""
    defaults = AlignmentConfig()
    return AlignmentConfig(
        min_turn_words=_parse(env, "MIN_TURN_WORDS", int, defaults.min_turn_words),
        min_turn_sec=_parse(env, "MIN_TURN_SEC", _to_float, defaults.min_turn_sec),
        main_speaker_label=_raw(env, "MAIN_SPEAKER_LABEL"),
        pause_split_sec=_parse(env, "PAUSE_SPLIT_SEC", _to_float, defaults.pause_split_sec),
    )
