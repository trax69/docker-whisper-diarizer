import logging
import os
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from typing import List, Optional
from dataclasses import dataclass
from faster_whisper import WhisperModel
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

logger = logging.getLogger("TranscriberEngine")

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"

class DiarizationEngine:
    def __init__(self, model_size: str = "medium", device: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing engine on: {self.device}")
        
        try:
            self.sb_device = "cpu" 
            
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.sb_device}
            )
            
            compute_type = "float16" if self.device == "cuda" else "int8"
            self.whisper = WhisperModel(
                model_size, 
                device=self.device, 
                compute_type=compute_type
            )
            logger.info(f"Models loaded. Whisper: {self.device} | SpeechBrain: {self.sb_device}")
            
        except Exception as e:
            logger.critical(f"Failed to load models: {e}")
            raise

    def process_file(self, input_path: str, num_speakers: Optional[int] = None) -> List[Segment]:
        logger.info(f"Transcribing file: {input_path}")
        segments = []
        
        try:
            segments_generator, info = self.whisper.transcribe(
                input_path,
                beam_size=5,
                language="es",
                vad_filter=True, 
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,
                no_speech_threshold=0.6
            )
            
            logger.info(f"Detected duration: {info.duration:.2f}s. Processing...")
            
            last_log = 0
            for s in segments_generator:
                duration = s.end - s.start
                if duration < 0.3: continue
                
                segments.append(Segment(s.start, s.end, s.text))
                
                if s.end - last_log > 30:
                    prog = (s.end / info.duration) * 100
                    logger.info(f"Progress: {prog:.1f}% ({s.end:.0f}s / {info.duration:.0f}s)")
                    last_log = s.end

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise e

        if not segments:
            logger.warning("No speech detected.")
            return []

        logger.info("Extracting embeddings for diarization...")
        try:
            full_audio = self._load_audio(input_path)
            
            embeddings = []
            valid_indices = []

            for idx, seg in enumerate(segments):
                start_frame = int(seg.start * 16000)
                end_frame = int(seg.end * 16000)
                
                if start_frame >= full_audio.shape[1]: continue
                end_frame = min(end_frame, full_audio.shape[1])
                
                audio_crop = full_audio[:, start_frame:end_frame]
                
                if audio_crop.shape[1] < 3200: continue

                with torch.no_grad():
                    emb = self.classifier.encode_batch(audio_crop)
                    emb_np = emb.squeeze().cpu().numpy()
                    embeddings.append(emb_np)
                    valid_indices.append(idx)

            if embeddings:
                self._cluster_speakers(segments, embeddings, valid_indices, num_speakers)
            
            return segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return segments

    def _load_audio(self, path: str) -> torch.Tensor:
        try:
            waveform, sample_rate = torchaudio.load(path)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != 16000:
                resampler = Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                
            return waveform
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")

    def _cluster_speakers(self, segments, embeddings, valid_indices, n_speakers):
        try:
            X = np.array(embeddings)
            distance_threshold = 0.7 if n_speakers is None else None
            
            clustering = AgglomerativeClustering(
                n_clusters=n_speakers,
                distance_threshold=distance_threshold,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(X)
            
            for i, idx in enumerate(valid_indices):
                segments[idx].speaker = f"SPEAKER_{labels[i]:02d}"
                
            current = "SPEAKER_00"
            for seg in segments:
                if seg.speaker == "UNKNOWN": seg.speaker = current
                else: current = seg.speaker
        except Exception as e:
            logger.error(f"Clustering error: {e}")