from dataclasses import dataclass
from pydantic import BaseModel
import numpy as np
from typing import List, Optional

@dataclass(slots=True)
class AudioSegment:
    samples: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float

class AudioQuality(BaseModel):
    passed: bool
    rms: float
    speech_ratio: float
    clipping_ratio: float
    reason: Optional[str] = None

class SpeakerEmbedding(BaseModel):
    embedding: List[float]
    timestamp: float
    duration: float
    session_id: str

class SpeakerVerificationTelemetry(BaseModel):
    model_name: str = "ECAPA-TDNN"
    model_version: str = "voxceleb"
    threshold_version: str = "1.0"
    model_load_time_ms: float
    embedding_time_ms: float
    verification_time_ms: float
    similarity_score: float
    speaker_changed: bool
    verification_enabled: bool
