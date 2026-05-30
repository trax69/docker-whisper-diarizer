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
3. **Diarization (subprocess):** pyannote/speaker-diarization-3.1 segments the audio by speaker identity.
4. **Alignment:** Each word token is assigned to the speaker active at its midpoint.
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

**Configure the model:**
```yaml
# docker-compose.yml
environment:
  - WHISPER_MODEL=large-v3  # Options: tiny, base, small, medium, large-v3
```

> The watchdog transcribes in **Spanish** (`language=es`). To change the language, edit `src/engine.py` → `_whisper_worker()`.

---

### Service 2 — One-shot CLI (`transcribir`)

Transcribes a single file on demand. Uses **automatic language detection**.

```bash
# Basic usage
docker compose run --rm transcribir -i /data/input/audio.mp3 -o /data/output/result.txt

# With known number of speakers and a specific model
docker compose run --rm transcribir -i /data/input/video.mp4 -o /data/output/result.txt -n 2 -m large-v3
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
| `-n` | Number of speakers — if known, improves diarization accuracy |

---

## Project Structure

```text
.
├── Dockerfile              # CUDA 12.4.1 based image definition
├── docker-compose.yml      # GPU reservation, volume mapping, and service definitions
├── requirements.txt        # Python dependencies (PyTorch 2.5.1 + CUDA 12.4)
├── transcribir.py          # Self-contained CLI (auto language detection)
├── data_volume/            # Created at runtime — input/output/processing/completed
├── hf_cache/               # Created at runtime — persists HuggingFace model downloads
└── src/
    ├── main.py             # Watchdog service & queue manager (Docker entrypoint)
    └── engine.py           # Core pipeline: Faster-Whisper + pyannote (Spanish, subprocess isolation)
```

---

## Notes on Performance

* **First Run:** The Whisper model and pyannote models are downloaded automatically from HuggingFace into `./hf_cache/`. Subsequent runs use the local cache.
* **VRAM Requirements:**
    * `medium`: ~3 GB VRAM
    * `large-v3`: ~6 GB VRAM (float16) — recommended for best accuracy
    * `small` / `base`: suitable for lower-resource environments
* **Memory Strategy:** Whisper and pyannote run in separate subprocesses. Each subprocess releases all GPU memory when it exits, keeping the main process memory flat across files.

---

## License
[MIT License](LICENSE)
