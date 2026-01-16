# 🎙️ Auto-Diarizer: Automated Transcription & Speaker Identification

**Auto-Diarizer** is a robust, GPU-accelerated pipeline designed to transcribe audio/video files and identify different speakers (diarization) automatically. 

It combines **Faster-Whisper** for state-of-the-art speech recognition with **SpeechBrain** for speaker embedding and clustering. The system can run as a background service (using Docker) that watches folders for new files, or as a standalone CLI tool.

---

## 🚀 Key Features

* **Watchdog Mode:** Automatically detects files dropped into an input folder, processes them, and moves them to completion.
* **Dual-Engine Architecture:**
    * **ASR:** Uses `faster-whisper` (CTranslate2) for lightning-fast transcription.
    * **Diarization:** Uses `SpeechBrain` (ECAPA-TDNN) to extract voice embeddings and `scikit-learn` to cluster speakers.
* **Hallucination Control:** Includes VAD (Voice Activity Detection) filters and segment post-processing to minimize Whisper hallucinations.
* **GPU Accelerated:** Optimized for NVIDIA GPUs (CUDA 11.8) for high-performance inference.
* **Dockerized:** "Drop-in" deployment with zero environment configuration hell.

---

## 🛠️ Architecture

The pipeline follows these steps:
1.  **Input:** Audio/Video file (mp3, wav, mp4, mkv, etc.).
2.  **Transcription:** Whisper generates text segments with precise timestamps.
3.  **Embedding Extraction:** The system crops the audio for each text segment and passes it through SpeechBrain's Speaker Recognition model to generate a vector (embedding).
4.  **Clustering:** Agglomerative Clustering groups these vectors to assign Speaker IDs (e.g., `SPEAKER_00`, `SPEAKER_01`).
5.  **Output:** A formatted text file with timecodes, speaker labels, and text.

---

## 🐳 Running with Docker (Recommended)

This is the easiest way to run the service. It will act as a "processing factory."

### Prerequisites
* Docker & Docker Compose.
* NVIDIA Drivers installed on the host.
* **NVIDIA Container Toolkit** (required for Docker to access the GPU).

### 1. Start the Service
```bash
docker-compose up -d --build
```

### 2. Usage
Upon starting, the container creates a `data_volume` directory.

1.  **Drop files** into: `./data_volume/input`
2.  The system moves them to `processing`.
3.  Once finished:
    * **Transcript:** Appears in `./data_volume/output`
    * **Original File:** Moved to `./data_volume/completed`

### Configuration
You can change the model size in `docker-compose.yml`:
```yaml
environment:
  - WHISPER_MODEL=medium # Options: tiny, base, small, medium, large-v3
```

---

## 💻 CLI Usage (Manual Mode)

If you prefer to run the script manually on a specific file without Docker.

### 1. Setup Environment
```bash
# It is recommended to use a Virtual Environment (Python 3.10+)
pip install -r requirements.txt
```

### 2. Run Command
```bash
python transcribir.py -i "path/to/audio.mp3" -o "path/to/output.txt" -m medium
```

**Arguments:**
* `-i`: Path to input audio/video file.
* `-o`: Path to save the output text file.
* `-m`: Model size (default: `medium`). Options: `tiny`, `base`, `small`, `medium`, `large-v3`.
* `-n`: (Optional) Number of speakers (if known beforehand, helps the clustering algorithm).

---

## 📂 Project Structure

```text
.
├── Dockerfile              # CUDA 11.8 based image definition
├── docker-compose.yml      # GPU reservation and volume mapping configuration
├── requirements.txt        # Python dependencies
├── transcribir.py          # CLI entry point for manual execution
└── src/
    ├── main.py             # Watchdog service & Queue manager (Docker entrypoint)
    └── engine.py           # Core logic (Whisper + SpeechBrain implementation)
```

## ⚠️ Notes on Performance

* **First Run:** The system will automatically download the Whisper model and SpeechBrain models from HuggingFace. This may take a few minutes depending on your internet connection.
* **VRAM Requirements:**
    * The `medium` model requires approx. 4-6GB VRAM.
    * Use `large-v3` for better accuracy if you have >10GB VRAM.
    * Use `small` or `base` for lower resource environments.

## 📜 License
[MIT License](LICENSE)



