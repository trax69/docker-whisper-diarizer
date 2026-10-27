import argparse
import gc
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# Shared pure-Python modules live in ./src (PYTHONPATH=/app/src in the image;
# added here too so the script also works when run from a checkout).
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch
from faster_whisper import WhisperModel
from tqdm import tqdm

from speaker_alignment import AlignmentConfig, TranscriptAligner, build_alignment_pipeline
from speaker_count import SpeakerCountHint
from transcript_models import Segment, SpeakerTurn, WordToken
from transcript_writer import write_segments

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SeniorDiarizerCLI")


@dataclass
class AppConfig:
    input_path: str
    output_path: str
    model_size: str
    speaker_hint: SpeakerCountHint = field(default_factory=SpeakerCountHint)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class IOManager:
    @staticmethod
    def validate_input(path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        if not os.path.isfile(path):
            raise ValueError(f"Input path is not a file: {path}")

    @staticmethod
    def write_output(path: str, segments: List[Segment]) -> None:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                write_segments(f, segments)
            logger.info(f"Transcription saved to: {path}")
        except IOError as e:
            logger.exception(f"Failed to write output file: {e}")
            raise


def _extract_to_wav(input_path: str, wav_path: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-i", input_path, "-ac", "1", "-ar", "16000",
         "-f", "wav", wav_path, "-y", "-loglevel", "error"],
        capture_output=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")


class DiarizationEngine:
    def __init__(self, config: AppConfig, alignment: Optional[TranscriptAligner] = None):
        self.config = config
        self._alignment = alignment or build_alignment_pipeline(config.alignment)
        logger.info(f"Loading Whisper model '{config.model_size}' on {config.device}...")
        self.whisper = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type="float16" if config.device == "cuda" else "int8",
        )
        logger.info("Whisper model loaded.")

    def run(self) -> List[Segment]:
        # Phase 1: Transcribe
        words = self._transcribe()
        if not words:
            logger.warning("No speech detected.")
            return []

        # Free Whisper VRAM before loading pyannote
        del self.whisper
        gc.collect()
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Whisper freed. Starting diarization.")

        # Phase 2: Diarize with pyannote
        turns = self._diarize()
        if not turns:
            logger.warning("No diarization output — using single speaker.")

        # Phases 3-4: assign speakers, smooth short turns, relabel, build segments
        return self._alignment.run(words, turns)

    # ------------------------------------------------------------------
    # Phase 1 – transcription
    # ------------------------------------------------------------------

    def _transcribe(self) -> List[WordToken]:
        logger.info(
            f"Starting transcription: model={self.config.model_size}, "
            f"device={self.config.device}"
        )
        segments_gen, info = self.whisper.transcribe(
            self.config.input_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            word_timestamps=True,
        )
        logger.info(f"Language: '{info.language}' | Duration: {info.duration:.2f}s")

        words: List[WordToken] = []
        with tqdm(total=info.duration, unit="s", desc="Transcribing") as pbar:
            for seg in segments_gen:
                if (seg.end - seg.start) < 0.3:
                    continue
                if seg.words:
                    for w in seg.words:
                        if w.word.strip():
                            words.append(WordToken(w.start, w.end, w.word))
                if seg.end > pbar.n:
                    pbar.update(seg.end - pbar.n)

        logger.info(f"Transcription done: {len(words)} words.")
        return words

    # ------------------------------------------------------------------
    # Phase 2 – diarization via pyannote/speaker-diarization-3.1
    # ------------------------------------------------------------------

    def _diarize(self) -> List[SpeakerTurn]:
        from pyannote.audio import Pipeline

        # HF_TOKEN env var is picked up automatically by huggingface_hub>=0.19.
        # Passing use_auth_token= was removed in huggingface_hub>=0.24.
        logger.info("Loading pyannote/speaker-diarization-3.1...")
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        except Exception as e:
            logger.error(
                f"Failed to load pyannote pipeline: {e}\n"
                "Set HF_TOKEN and accept model terms at:\n"
                "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  https://huggingface.co/pyannote/segmentation-3.0"
            )
            return []

        if self.config.device == "cuda":
            pipeline = pipeline.to(torch.device("cuda"))
            logger.info("Diarization running on GPU.")
        else:
            logger.info("Diarization running on CPU.")

        tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        try:
            _extract_to_wav(self.config.input_path, tmp_wav)

            hint = self.config.speaker_hint
            logger.info(f"Running speaker diarization (speakers: {hint.describe()})...")
            diarization = pipeline(tmp_wav, **hint.to_pipeline_kwargs())

            turns = [
                SpeakerTurn(round(turn.start, 3), round(turn.end, 3), speaker)
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]

            n_speakers = len({t.speaker for t in turns})
            logger.info(
                f"Diarization done: {len(turns)} turns, {n_speakers} speaker(s)."
            )
            return turns

        finally:
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass
            del pipeline
            gc.collect()
            if self.config.device == "cuda":
                torch.cuda.empty_cache()

    # Phases 3-4 (word → speaker, smoothing, relabelling, segments) live in the
    # shared `speaker_alignment` package (src/), injected as `self._alignment`.


def _build_parser() -> argparse.ArgumentParser:
    defaults = AlignmentConfig()
    parser = argparse.ArgumentParser(prog="transcribir.py",
                                     description="Local Neural Diarization Tool")
    parser.add_argument("-i", "--input", required=True, help="Path to input audio/video file")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output text file")
    parser.add_argument("-m", "--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: medium)")

    speakers = parser.add_argument_group("speaker count (forwarded to pyannote)")
    speakers.add_argument("-n", "--speakers", type=int, default=None,
                          help="Exact number of speakers (auto-detect if omitted)")
    speakers.add_argument("--min-speakers", type=int, default=None,
                          help="Minimum number of speakers")
    speakers.add_argument("--max-speakers", type=int, default=None,
                          help="Maximum number of speakers")

    smoothing = parser.add_argument_group("speaker smoothing")
    smoothing.add_argument("--min-turn-words", type=int, default=defaults.min_turn_words,
                           help="Turns with fewer words are merged into their neighbours "
                                f"(default: {defaults.min_turn_words}; 0 disables)")
    smoothing.add_argument("--min-turn-sec", type=float, default=defaults.min_turn_sec,
                           help="Turns shorter than this (seconds) are merged into their "
                                f"neighbours (default: {defaults.min_turn_sec}; 0 disables)")
    smoothing.add_argument("--main-speaker-label", default=None,
                           help="Label for whoever speaks most, e.g. PROFESOR "
                                "(default: SPEAKER_00)")
    smoothing.add_argument("--pause-split-sec", type=float, default=defaults.pause_split_sec,
                           help="Silence (seconds) that starts a new line "
                                f"(default: {defaults.pause_split_sec})")
    return parser


def parse_arguments(argv: Optional[Sequence[str]] = None) -> AppConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        speaker_hint = SpeakerCountHint(
            num_speakers=args.speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        alignment = AlignmentConfig(
            min_turn_words=args.min_turn_words,
            min_turn_sec=args.min_turn_sec,
            main_speaker_label=(args.main_speaker_label or "").strip() or None,
            pause_split_sec=args.pause_split_sec,
        )
        build_alignment_pipeline(alignment)  # validate thresholds before loading models
    except ValueError as e:
        parser.error(str(e))  # exits with status 2 and a usage message

    return AppConfig(
        input_path=args.input,
        output_path=args.output,
        model_size=args.model,
        speaker_hint=speaker_hint,
        alignment=alignment,
    )


if __name__ == "__main__":
    try:
        config = parse_arguments()
        IOManager.validate_input(config.input_path)

        engine = DiarizationEngine(config)
        segments = engine.run()

        if segments:
            IOManager.write_output(config.output_path, segments)
        else:
            logger.warning("No segments to write.")
            sys.exit(1)

    except Exception as e:
        logger.critical(f"Process failed: {e}", exc_info=True)
        sys.exit(1)
