import argparse
import gc
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional, List

import torch
from faster_whisper import WhisperModel
from tqdm import tqdm

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
    num_speakers: Optional[int]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class WordToken:
    start: float
    end: float
    word: str
    speaker: str = "UNKNOWN"


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"


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
                for seg in segments:
                    line = f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.speaker}: {seg.text.strip()}\n"
                    f.write(line)
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
    PAUSE_SPLIT_SEC = 2.0

    def __init__(self, config: AppConfig):
        self.config = config
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
        speaker_segments = self._diarize()

        # Phase 3: Assign speakers to words
        if not speaker_segments:
            logger.warning("No diarization output — using single speaker.")
            for w in words:
                w.speaker = "SPEAKER_00"
        else:
            self._assign_speakers_to_words(words, speaker_segments)

        # Phase 4: Merge into output segments
        return self._merge_words_into_segments(words)

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

    def _diarize(self) -> List[Segment]:
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

            kwargs = {} if self.config.num_speakers is None else {
                "num_speakers": self.config.num_speakers
            }
            logger.info("Running speaker diarization...")
            diarization = pipeline(tmp_wav, **kwargs)

            segments = [
                Segment(
                    start=round(turn.start, 3),
                    end=round(turn.end, 3),
                    text="",
                    speaker=speaker,
                )
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]

            n_speakers = len({s.speaker for s in segments})
            logger.info(
                f"Diarization done: {len(segments)} segments, {n_speakers} speaker(s)."
            )
            return segments

        finally:
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass
            del pipeline
            gc.collect()
            if self.config.device == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 3 – assign speakers to words
    # ------------------------------------------------------------------

    def _assign_speakers_to_words(
        self, words: List[WordToken], speaker_segments: List[Segment]
    ) -> None:
        for word in words:
            midpoint = (word.start + word.end) / 2.0
            word.speaker = self._speaker_at(midpoint, speaker_segments)

    @staticmethod
    def _speaker_at(t: float, segments: List[Segment]) -> str:
        fallback = "SPEAKER_00"
        for seg in segments:
            if seg.start <= t <= seg.end:
                return seg.speaker
            if seg.end < t:
                fallback = seg.speaker
        return fallback

    # ------------------------------------------------------------------
    # Phase 4 – merge words → output segments
    # ------------------------------------------------------------------

    def _merge_words_into_segments(self, words: List[WordToken]) -> List[Segment]:
        if not words:
            return []

        result: List[Segment] = []
        seg_start = words[0].start
        seg_end = words[0].end
        seg_speaker = words[0].speaker
        seg_text = words[0].word

        for w in words[1:]:
            pause = w.start - seg_end
            if w.speaker == seg_speaker and pause <= self.PAUSE_SPLIT_SEC:
                seg_text += w.word
                seg_end = w.end
            else:
                if seg_text.strip():
                    result.append(
                        Segment(seg_start, seg_end, seg_text.strip(), seg_speaker)
                    )
                seg_start = w.start
                seg_end = w.end
                seg_speaker = w.speaker
                seg_text = w.word

        if seg_text.strip():
            result.append(Segment(seg_start, seg_end, seg_text.strip(), seg_speaker))

        return result


def parse_arguments() -> AppConfig:
    parser = argparse.ArgumentParser(description="Local Neural Diarization Tool")
    parser.add_argument("-i", "--input", required=True, help="Path to input audio/video file")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output text file")
    parser.add_argument("-n", "--speakers", type=int, default=None,
                        help="Number of speakers (auto-detect if omitted)")
    parser.add_argument("-m", "--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: medium)")

    args = parser.parse_args()
    return AppConfig(
        input_path=args.input,
        output_path=args.output,
        model_size=args.model,
        num_speakers=args.speakers,
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
