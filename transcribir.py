import argparse
import logging
import os
import sys
from dataclasses import dataclass
from tqdm import tqdm
import librosa
from typing import Optional, List, Union

import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier
from torchaudio.transforms import Resample

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
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"
    embedding: Optional[np.ndarray] = None

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
            logger.info(f"Transcription saved successfully to: {path}")
        except IOError as e:
            logger.error(f"Failed to write output file: {e}")
            raise

class AudioProcessor:
    @staticmethod
    def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
        try:
            signal_np, _ = librosa.load(path, sr=target_sr, mono=True)
            return torch.from_numpy(signal_np)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from video file: {e}")

class DiarizationEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": config.device}
        )
        
        self.whisper = WhisperModel(
            config.model_size, 
            device=config.device, 
            compute_type="float16" if config.device == "cuda" else "int8"
        )

    def run(self) -> List[Segment]:
        logger.info(f"Starting transcription using model: {self.config.model_size}")
        
        # FIX 1: VAD Filter and Anti-Hallucination settings
        segments_generator, info = self.whisper.transcribe(
            self.config.input_path, 
            beam_size=5,
            vad_filter=True, # CRITICAL: Removes silence/noise
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False, # CRITICAL: Prevents loop repetition
            no_speech_threshold=0.6
        )
        
        logger.info(f"Detected language '{info.language}'")
        logger.info(f"Total duration: {info.duration:.2f}s. Transcribing...")

        segments = []
        
        with tqdm(total=info.duration, unit="s", desc="Transcribing") as pbar:
            for s in segments_generator:
                duration = s.end - s.start
                
                # FIX 2: Filter out micro-segments (hallucinations)
                if duration < 0.3: 
                    continue
                    
                segments.append(Segment(s.start, s.end, s.text))
                
                if s.end > pbar.n:
                    pbar.update(s.end - pbar.n)

        if not segments:
            logger.warning("No speech detected.")
            return []

        logger.info("Extracting embeddings for diarization...")
        full_audio = AudioProcessor.load_audio(self.config.input_path)
        
        embeddings = []
        valid_indices = []

        for idx, seg in enumerate(tqdm(segments, desc="Diarizing")):
            start_frame = int(seg.start * 16000)
            end_frame = int(seg.end * 16000)
            audio_crop = full_audio[start_frame:end_frame]
            
            if audio_crop.shape[0] < 3200:
                continue

            with torch.no_grad():
                emb = self.classifier.encode_batch(audio_crop.unsqueeze(0))
                emb = emb.squeeze().cpu().numpy()
                embeddings.append(emb)
                valid_indices.append(idx)

        if not embeddings:
            return segments

        self._cluster_speakers(segments, embeddings, valid_indices)
        return segments

    def _cluster_speakers(self, segments: List[Segment], embeddings: List[np.ndarray], valid_indices: List[int]):
        logger.info("Clustering speakers...")
        X = np.array(embeddings)
        
        n_clusters = self.config.num_speakers
        distance_threshold = 0.7 if n_clusters is None else None

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            metric='cosine',
            linkage='average'
        )
        
        labels = clustering.fit_predict(X)
        
        for i, idx in enumerate(valid_indices):
            segments[idx].speaker = f"SPEAKER_{labels[i]:02d}"
        
        self._fill_gaps(segments)

    def _fill_gaps(self, segments: List[Segment]):
        current_speaker = "SPEAKER_00"
        for seg in segments:
            if seg.speaker == "UNKNOWN":
                seg.speaker = current_speaker
            else:
                current_speaker = seg.speaker

def parse_arguments() -> AppConfig:
    parser = argparse.ArgumentParser(description="Local Neural Diarization Tool")
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to input audio/video file"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Path to save the output text file"
    )
    parser.add_argument(
        "-n", "--speakers", 
        type=int, 
        default=None,
        help="Number of speakers (optional, auto-detect if omitted)"
    )
    parser.add_argument(
        "-m", "--model", 
        default="medium", 
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: medium)"
    )

    args = parser.parse_args()
    return AppConfig(
        input_path=args.input,
        output_path=args.output,
        model_size=args.model,
        num_speakers=args.speakers
    )

if __name__ == "__main__":
    try:
        config = parse_arguments()
        IOManager.validate_input(config.input_path)
        
        engine = DiarizationEngine(config)
        final_segments = engine.run()
        
        IOManager.write_output(config.output_path, final_segments)
        
    except Exception as e:
        logger.critical(f"Process failed: {e}")
        sys.exit(1)