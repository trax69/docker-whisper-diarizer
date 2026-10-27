# Auto-Diarizer: Automated Transcription & Speaker Identification

**Auto-Diarizer** is a GPU-accelerated pipeline that transcribes audio/video files and identifies different speakers automatically.

It combines **Faster-Whisper** for speech recognition with **pyannote/speaker-diarization-3.1** for speaker diarization. The system runs as a background service (Docker) that watches folders for new files, or as a standalone CLI tool.

---

## Key Features

* **Watchdog Mode:** Automatically detects files dropped into an input folder, processes them, and moves them to completion.
* **Dual-Engine Architecture:**
    * **ASR:** `faster-whisper` (CTranslate2) for fast, accurate transcription with word-level timestamps.
    * **Diarization:** `pyannote/speaker-diarization-3.1` — end-to-end neural speaker diarization running on GPU.
* **Hallucination Control:** VAD filter and segment post-processing to minimize Whisper hallucinations.
* **GPU Accelerated:** Optimized for NVIDIA GPUs (CUDA 12.4) with PyTorch 2.5.1.
* **Isolated Subprocesses:** Whisper and pyannote each run in a separate OS process so VRAM is fully released between phases.
* **Transcription Cache:** Word-level results are cached to disk so diarization can be re-run without re-transcribing.
* **Dockerized:** Single-command deployment with model cache persistence across container restarts.

---

## Architecture

The pipeline follows these steps:

1. **Input:** Audio/Video file (mp3, wav, mp4, mkv, etc.).
2. **Transcription (subprocess):** Faster-Whisper generates word-level tokens with precise timestamps.
3. **Diarization (subprocess):** pyannote/speaker-diarization-3.1 segments the audio by speaker identity (optionally constrained by `NUM_SPEAKERS` / `MIN_SPEAKERS` / `MAX_SPEAKERS`).
4. **Alignment** (`src/speaker_alignment/`, shared by the daemon and the CLI):
    1. **Assignment:** each word goes to the speaker that overlaps it the most in time; words in a gap go to the *nearest* turn.
    2. **Smoothing:** speaker turns with fewer than `MIN_TURN_WORDS` words or shorter than `MIN_TURN_SEC` seconds are absorbed by their neighbours. This removes the one-word false speakers ("pues", "vale"…) that pyannote produces on breaths or noise, while real interventions survive.
    3. **Relabelling:** speakers are renamed by speaking time, so `SPEAKER_00` is always whoever talks the most (optionally renamed with `MAIN_SPEAKER_LABEL`, e.g. `PROFESOR`).
    4. **Paragraph splitting:** lines longer than `MAX_SEGMENT_SEC` (default 30 s) are split at the next sentence end (`.` `?` `!` `…`), at a comma after twice that time, and forcibly after three times. Timecodes stay exact, so each line can be located in the video.
5. **Output:** A formatted text file with timecodes, speaker labels, and text.

**Output format:**
```
[0.00s - 3.25s] SPEAKER_00: Hello, how are you doing today?
[3.80s - 7.10s] SPEAKER_01: I'm doing well, thanks for asking.
```

---

## Running with Docker (Recommended)

Two Docker services are available: a **watchdog daemon** for batch processing and a **one-shot CLI** for single files.

### Prerequisites

* Docker & Docker Compose
* NVIDIA drivers installed on the host
* **NVIDIA Container Toolkit** (required for Docker GPU access)
* A **HuggingFace token** (`HF_TOKEN`) with access granted to:
    * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Setup

Create a `.env` file in the project root:

```bash
HF_TOKEN=hf_your_token_here
```

---

### Service 1 — Watchdog Daemon (`diarizer`)

Monitors a folder and automatically processes any file dropped into it.

**Start the service:**
```bash
docker compose up -d --build
```

**Usage:**
1. **Drop files** into: `./data_volume/input`
2. The service moves them to `processing` while working.
3. Once finished:
    * **Transcript:** Appears in `./data_volume/output` (same name, `.txt` extension)
    * **Original File:** Moved to `./data_volume/completed`
4. If processing fails, the file is renamed to `ERROR_<filename>` inside `./data_volume/processing`.

> **Note:** Files already present in `input` when the container starts are picked up automatically.
> Only audio/video extensions are queued (mp4, mkv, mov, webm, mp3, m4a, wav, flac, ogg, opus…);
> subtitles, transcripts and partial downloads (`.vtt`, `.srt`, `.txt`, `.part`…) are ignored.

**Configure the model:**
```yaml
# docker-compose.yml
environment:
  - WHISPER_MODEL=large-v3  # Options: tiny, base, small, medium, large-v3
```

**Tune speaker detection** — add any of these to `.env` (blank = default) and run `docker compose up -d`:

| Variable | Default | Effect |
|----------|---------|--------|
| `NUM_SPEAKERS` | *(auto)* | Exact number of speakers. Wins over MIN/MAX. |
| `MIN_SPEAKERS` / `MAX_SPEAKERS` | *(auto)* | Range of speakers pyannote may find. E.g. `MAX_SPEAKERS=3` for a class with few spoken questions. |
| `MIN_TURN_WORDS` | `3` | A speaker turn with fewer words is absorbed by its neighbours. |
| `MIN_TURN_SEC` | `1.5` | A speaker turn shorter than this (seconds) is absorbed by its neighbours. `MIN_TURN_WORDS=0` + `MIN_TURN_SEC=0` disables smoothing. |
| `MAIN_SPEAKER_LABEL` | *(empty)* | Label for whoever speaks most, e.g. `PROFESOR`. Others stay `SPEAKER_01`, `SPEAKER_02`… |
| `PAUSE_SPLIT_SEC` | `2.0` | Silence (seconds) that starts a new line for the same speaker. |
| `MAX_SEGMENT_SEC` | `30` | Lines longer than this are split at the next sentence end (comma at 2×, forced at 3×). `0` keeps whole paragraphs. |

Example `.env` for recorded lectures:
```bash
HF_TOKEN=hf_your_token_here
MAX_SPEAKERS=4
MAIN_SPEAKER_LABEL=PROFESOR
```

> Trade-off: a very short real reply ("Sí", "Gracias") is merged into the surrounding speaker.
> Questions of a few words and a couple of seconds are preserved. Lower the thresholds if you need them.
> Invalid values (e.g. `MIN_SPEAKERS=3` with `MAX_SPEAKERS=2`) stop the daemon at start-up with a clear error.

> The watchdog transcribes in **Spanish** (`language=es`). To change the language, edit `src/engine.py` → `_whisper_worker()`.

---

### Service 2 — One-shot CLI (`transcribir`)

Transcribes a single file on demand. Uses **automatic language detection**.

```bash
# Basic usage
docker compose run --rm transcribir -i /data/input/audio.mp3 -o /data/output/result.txt

# With known number of speakers and a specific model
docker compose run --rm transcribir -i /data/input/video.mp4 -o /data/output/result.txt -n 2 -m large-v3

# Lecture: at most 3 speakers, teacher labelled as PROFESOR
docker compose run --rm transcribir -i /data/input/video.mp4 -o /data/output/result.txt \
    --max-speakers 3 --main-speaker-label PROFESOR
```

Files in `./data_volume/` are accessible inside the container under `/data/`.

---

## CLI Usage (Without Docker)

Run the script directly on a local file.

### 1. Setup Environment
```bash
# Python 3.10+ recommended
pip install -r requirements.txt
export HF_TOKEN=hf_your_token_here
```

### 2. Run Command
```bash
python transcribir.py -i "path/to/audio.mp3" -o "path/to/output.txt" -m large-v3
```

**Arguments:**
| Flag | Description |
|------|-------------|
| `-i` | Path to input audio/video file (required) |
| `-o` | Path to save the output `.txt` file (required) |
| `-m` | Whisper model size (default: `medium`) |
| `-n`, `--speakers` | Exact number of speakers — if known, improves diarization accuracy |
| `--min-speakers` / `--max-speakers` | Range of speakers when the exact number is unknown |
| `--min-turn-words` | Turns with fewer words are absorbed by their neighbours (default: `3`, `0` disables) |
| `--min-turn-sec` | Turns shorter than this are absorbed by their neighbours (default: `1.5`, `0` disables) |
| `--main-speaker-label` | Label for whoever speaks most, e.g. `PROFESOR` |
| `--pause-split-sec` | Silence that starts a new line (default: `2.0`) |
| `--max-segment-sec` | Lines longer than this are split at the next sentence end (default: `30`, `0` disables) |

---

## Project Structure

```text
.
├── Dockerfile              # CUDA 12.4.1 based image definition
├── docker-compose.yml      # GPU reservation, volume mapping, and service definitions
├── requirements.txt        # Python dependencies (PyTorch 2.5.1 + CUDA 12.4)
├── transcribir.py          # One-shot CLI (auto language detection)
├── data_volume/            # Created at runtime — input/output/processing/completed
├── hf_cache/               # Created at runtime — persists HuggingFace + pyannote model downloads
├── tests/                  # unittest suite (no GPU needed for the core logic)
└── src/
    ├── main.py             # Watchdog service & queue manager (Docker entrypoint)
    ├── engine.py           # Faster-Whisper + pyannote in isolated subprocesses (Spanish)
    ├── speaker_alignment/  # Word → speaker assignment, smoothing, relabelling, segments, splitting
    ├── speaker_count.py    # NUM/MIN/MAX speaker hints forwarded to pyannote
    ├── settings.py         # Daemon configuration from environment variables
    ├── media_filter.py     # Which files in input/ are audio/video
    ├── transcript_models.py# WordToken, SpeakerTurn, Segment
    └── transcript_writer.py# Output line format
```

---

## Tests

The alignment logic is pure Python, so the suite runs without a GPU:

```bash
# Locally (Python 3.10+; CLI tests are skipped if torch/faster-whisper are missing)
python -m unittest discover -s tests -t . -v

# Inside the image (runs everything, including the CLI tests)
docker compose run --rm --entrypoint python transcribir -m unittest discover -s /app/tests -t /app -v
```

---

## Notes on Performance

* **First Run:** The Whisper model and pyannote models are downloaded automatically into `./hf_cache/` (pyannote's own cache lives in `./hf_cache/pyannote` via `PYANNOTE_CACHE`). Subsequent runs, even after re-creating the container, use the local cache.
* **pyannote version:** pinned to `>=3.1,<4.0`; the 4.x series changed the pipeline API.
* **VRAM Requirements:**
    * `medium`: ~3 GB VRAM
    * `large-v3`: ~6 GB VRAM (float16) — recommended for best accuracy
    * `small` / `base`: suitable for lower-resource environments
* **Memory Strategy:** Whisper and pyannote run in separate subprocesses. Each subprocess releases all GPU memory when it exits, keeping the main process memory flat across files.

---

## License
[MIT License](LICENSE)
