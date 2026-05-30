import ctypes
import gc
import json
import logging
import multiprocessing as mp
import os
import subprocess
import tempfile
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger("TranscriberEngine")


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _malloc_trim() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _rss_mb() -> int:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def _cgroup_mb() -> int:
    for path in ("/sys/fs/cgroup/memory.current",
                 "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(path) as fh:
                return int(fh.read().strip()) // (1024 * 1024)
        except Exception:
            pass
    return -1


# ---------------------------------------------------------------------------
# Whisper subprocess worker
# Runs in its own OS process so the ctranslate2 CUDA context — and all
# WDDM-backed VRAM — is fully released when the process exits.
# ---------------------------------------------------------------------------

def _whisper_worker(
    model_size: str, device: str, input_path: str, output_path: str
) -> None:
    import gc as _gc
    import json as _json
    import logging as _log

    try:
        import onnxruntime as _ort
        _ort.set_default_logger_severity(3)  # must run before faster_whisper imports ORT
    except Exception:
        pass

    import torch as _torch
    from faster_whisper import WhisperModel

    _log.basicConfig(level=_log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _logger = _log.getLogger("WhisperWorker")

    if device == "cuda" and not _torch.cuda.is_available():
        _logger.warning("CUDA not available in subprocess, falling back to CPU.")
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    _logger.info(f"Whisper subprocess: device={device}, compute_type={compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments_gen, info = model.transcribe(
        input_path,
        beam_size=5,
        language="es",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        word_timestamps=True,
    )

    _logger.info(f"Duration: {info.duration:.2f}s. Transcribing...")

    words = []
    last_log = 0.0
    for seg in segments_gen:
        if (seg.end - seg.start) < 0.3:
            continue
        if seg.words:
            for w in seg.words:
                if w.word.strip():
                    words.append({"start": w.start, "end": w.end, "word": w.word})
        if seg.end - last_log > 30:
            pct = (seg.end / info.duration) * 100
            _logger.info(f"Progress: {pct:.1f}% ({seg.end:.0f}s / {info.duration:.0f}s)")
            last_log = seg.end

    try:
        segments_gen.close()
    except Exception:
        pass
    del segments_gen, model
    _gc.collect()

    _logger.info(f"Transcription done: {len(words)} words.")
    with open(output_path, "w", encoding="utf-8") as fh:
        _json.dump({"words": words, "duration": info.duration}, fh)


# ---------------------------------------------------------------------------
# Diarize subprocess worker — pyannote/speaker-diarization-3.1
# Runs in an isolated OS process so all GPU/CPU memory used by the neural
# pipeline (segmentation model + embedding model + clustering) is fully
# released when the process exits. The main process stays flat in memory.
# ---------------------------------------------------------------------------

def _diarize_worker(
    wav_path: str, num_speakers_str: str, output_path: str
) -> None:
    import json as _json
    import logging as _log
    import warnings

    try:
        import onnxruntime as _ort
        _ort.set_default_logger_severity(3)  # must run before pyannote imports ORT
    except Exception:
        pass

    import torch as _torch
    from pyannote.audio import Pipeline

    _log.basicConfig(level=_log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _logger = _log.getLogger("DiarizeWorker")

    # Suppress noisy but harmless third-party logs/warnings
    _log.getLogger("speechbrain").setLevel(_log.WARNING)
    _log.getLogger("speechbrain.utils.quirks").setLevel(_log.WARNING)
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
    warnings.filterwarnings("ignore", message=".*degrees of freedom.*", category=UserWarning)

    # HF_TOKEN env var is automatically picked up by huggingface_hub>=0.19.
    # Passing use_auth_token= was removed in huggingface_hub>=0.24, so we
    # rely on the environment variable set via docker-compose.yml instead.
    _logger.info("Loading pyannote/speaker-diarization-3.1...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    except Exception as e:
        _logger.error(
            f"Failed to load pyannote pipeline: {e}\n"
            "Ensure HF_TOKEN is set in .env and you have accepted the model terms at:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  https://huggingface.co/pyannote/segmentation-3.0"
        )
        raise

    if _torch.cuda.is_available():
        pipeline = pipeline.to(_torch.device("cuda"))
        _logger.info("Diarization running on GPU.")
    else:
        _logger.info("Diarization running on CPU (may be slow for long files).")

    num_speakers = None if num_speakers_str == "None" else int(num_speakers_str)
    kwargs = {} if num_speakers is None else {"num_speakers": num_speakers}

    _logger.info("Running speaker diarization...")
    diarization = pipeline(wav_path, **kwargs)

    segments = [
        {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    n_speakers = len({s["speaker"] for s in segments})
    _logger.info(f"Diarization done: {len(segments)} segments, {n_speakers} speaker(s).")

    with open(output_path, "w", encoding="utf-8") as fh:
        _json.dump(segments, fh)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DiarizationEngine:
    PAUSE_SPLIT_SEC = 2.0

    def __init__(self, model_size: str = "medium", device: str = None):
        self._model_size = model_size
        self._device = device or os.getenv("WHISPER_DEVICE", "cuda")
        logger.info(
            f"Initializing engine. whisper_device={self._device}. "
            "Pyannote diarization runs per-file in subprocess."
        )

    def process_file(self, input_path: str, num_speakers: Optional[int] = None) -> List[Segment]:
        logger.info(f"Processing: {input_path} | RSS: {_rss_mb()} MB | cgroup: {_cgroup_mb()} MB")

        words = self._transcribe_with_cache(input_path)
        if not words:
            logger.warning("No speech detected.")
            return []

        gc.collect()
        _malloc_trim()
        logger.info(f"Post-transcription: RSS: {_rss_mb()} MB | cgroup: {_cgroup_mb()} MB")

        speaker_segments = self._diarize(input_path, num_speakers)
        self._assign_speakers_to_words(words, speaker_segments)
        return self._merge_words_into_segments(words)

    # ------------------------------------------------------------------
    # Phase 1 – transcription (with disk cache to survive restarts)
    # ------------------------------------------------------------------

    def _transcribe_with_cache(self, input_path: str) -> List[WordToken]:
        cache_path = input_path + ".words.json"
        if os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                words = [WordToken(w["start"], w["end"], w["word"]) for w in data["words"]]
                logger.info(f"Loaded {len(words)} words from cache: {cache_path}")
                return words
            except Exception as e:
                logger.warning(f"Cache read failed ({e}), re-transcribing.")

        words = self._transcribe_subprocess(input_path)

        if words:
            try:
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {"words": [{"start": w.start, "end": w.end, "word": w.word} for w in words]},
                        fh,
                    )
                logger.info(f"Transcription cached to {cache_path}")
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")

        return words

    def _transcribe_subprocess(self, input_path: str) -> List[WordToken]:
        tmp_fd, tmp_out = tempfile.mkstemp(suffix=".json")
        os.close(tmp_fd)
        try:
            logger.info("Starting Whisper subprocess...")
            ctx = mp.get_context("spawn")
            proc = ctx.Process(
                target=_whisper_worker,
                args=(self._model_size, self._device, input_path, tmp_out),
            )
            proc.start()
            proc.join(timeout=7200)

            if proc.is_alive():
                proc.kill()
                proc.join()
                logger.error("Whisper subprocess timed out.")
                return []

            if proc.exitcode != 0:
                logger.error(f"Whisper subprocess exited with code {proc.exitcode}.")
                return []

            with open(tmp_out, encoding="utf-8") as fh:
                data = json.load(fh)

            return [WordToken(w["start"], w["end"], w["word"]) for w in data["words"]]

        except Exception as e:
            logger.exception(f"Transcription subprocess error: {e}")
            return []
        finally:
            try:
                os.unlink(tmp_out)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Phase 2 – speaker diarization via pyannote (isolated subprocess)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_to_wav(input_path: str, wav_path: str) -> None:
        result = subprocess.run(
            ["ffmpeg", "-i", input_path, "-ac", "1", "-ar", "16000",
             "-f", "wav", wav_path, "-y", "-loglevel", "error"],
            capture_output=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

    def _diarize(
        self, input_path: str, num_speakers: Optional[int]
    ) -> List[Segment]:
        tmp_wav = input_path + ".diarize.wav"
        tmp_segs = input_path + ".diarize.json"

        try:
            logger.info("Extracting WAV for diarization...")
            self._extract_to_wav(input_path, tmp_wav)

            logger.info(
                f"Spawning diarize subprocess. RSS: {_rss_mb()} MB | cgroup: {_cgroup_mb()} MB"
            )
            ctx = mp.get_context("spawn")
            proc = ctx.Process(
                target=_diarize_worker,
                args=(tmp_wav, str(num_speakers), tmp_segs),
            )
            proc.start()
            proc.join(timeout=7200)

            if proc.is_alive():
                proc.kill()
                proc.join()
                logger.error("Diarize subprocess timed out.")
                return []

            if proc.exitcode != 0:
                logger.error(
                    f"Diarize subprocess failed (exit {proc.exitcode}). "
                    "Check that HF_TOKEN is set and model terms are accepted."
                )
                return []

            logger.info(
                f"Diarize subprocess done. RSS: {_rss_mb()} MB | cgroup: {_cgroup_mb()} MB"
            )

            with open(tmp_segs, encoding="utf-8") as fh:
                data = json.load(fh)

            return [Segment(s["start"], s["end"], "", s["speaker"]) for s in data]

        except RuntimeError as e:
            logger.exception(f"Diarization audio extraction failed: {e}")
            return []
        finally:
            for path in (tmp_wav, tmp_segs):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Phase 3 – assign speakers to words
    # ------------------------------------------------------------------

    def _assign_speakers_to_words(
        self, words: List[WordToken], speaker_segments: List[Segment]
    ) -> None:
        if not speaker_segments:
            for word in words:
                word.speaker = "SPEAKER_00"
            return
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
                    result.append(Segment(seg_start, seg_end, seg_text.strip(), seg_speaker))
                seg_start = w.start
                seg_end = w.end
                seg_speaker = w.speaker
                seg_text = w.word

        if seg_text.strip():
            result.append(Segment(seg_start, seg_end, seg_text.strip(), seg_speaker))

        return result
