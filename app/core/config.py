"""
System Configuration Parameters

Purpose:
Defines global settings including chunk sizes, audio sample rates, WebSocket limits, environment configurations, and external API keys.

Module/Component: API Core
"""

"""
System Configuration Parameters

Purpose:
Defines global settings including chunk sizes, audio sample rates, WebSocket limits, environment configurations, and external API keys.

Module/Component: API Core
"""

# =====================================================================
# CONFIGURATION SETTINGS (app/core/config.py)
# =====================================================================
# Purpose: Holds all global settings loaded via pydantic-settings.
#
# Recent Modifications:
# 1. Added `TRANSCRIPT_BUFFER_SIZE` configuration setting (defaulting to 500).
#    This defines the max sliding window size for the transcript event bus
#    in-memory cache buffers.
# =====================================================================

from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import List

class Settings(BaseSettings):
    # App General Settings
    APP_NAME: str = "Clinical Consultation Audio Ingestion"
    API_V1_STR: str = "/v1"
    
    # Audio Parameters
    AUDIO_SAMPLE_RATE: int = 16000  # 16 kHz
    AUDIO_CHANNELS: int = 1         # Mono
    AUDIO_BITS_PER_SAMPLE: int = 16 # 16-bit PCM
    
    # Audio Frame Sizes in Milliseconds (20ms to 50ms)
    MIN_FRAME_DURATION_MS: int = 20
    MAX_FRAME_DURATION_MS: int = 50
    
    # Buffer Configuration
    # We buffer incoming client frames into larger chunks (e.g. 1 second of audio)
    # before they are ready for Scribe V2 transcription layer to optimize latency vs payload size.
    BUFFER_CHUNK_DURATION_SEC: float = 1.0
    
    # Derived Audio Calculation Properties
    @property
    def bytes_per_sample(self) -> int:
        return self.AUDIO_BITS_PER_SAMPLE // 8

    @property
    def min_frame_bytes(self) -> int:
        # samples_per_ms = 16000 / 1000 = 16 samples/ms
        # min_frame_samples = 16 * 20 = 320 samples
        # min_frame_bytes = 320 * 2 = 640 bytes
        samples_per_ms = self.AUDIO_SAMPLE_RATE // 1000
        return samples_per_ms * self.MIN_FRAME_DURATION_MS * self.bytes_per_sample

    @property
    def max_frame_bytes(self) -> int:
        # max_frame_samples = 16 * 50 = 800 samples
        # max_frame_bytes = 800 * 2 = 1600 bytes
        samples_per_ms = self.AUDIO_SAMPLE_RATE // 1000
        return samples_per_ms * self.MAX_FRAME_DURATION_MS * self.bytes_per_sample

    @property
    def buffer_chunk_bytes(self) -> int:
        # 1.0s * 16000 samples/sec * 2 bytes/sample = 32000 bytes
        return int(self.BUFFER_CHUNK_DURATION_SEC * self.AUDIO_SAMPLE_RATE * self.bytes_per_sample)

    # Session Management
    MAX_BUFFER_MEM_LIMIT_BYTES: int = 10 * 1024 * 1024  # 10MB safety threshold per connection
    
    # Security / Auth
    AUTH_SECRET_TOKEN: str = "production-secure-token-change-me"
    ALLOWED_ROLES: List[str] = ["doctor", "patient", "attender"]
    
    # JWT Configurations
    JWT_SECRET_KEY: str = "production-jwt-secret-key-change-me"
    JWT_ALGORITHM: str = "HS256"
    
    # Backpressure Queue Configurations
    QUEUE_MAX_SIZE: int = 100
    QUEUE_OVERFLOW_STRATEGY: str = "drop_oldest"  # options: "drop_oldest", "block", "ignore_and_warn"

    # Transcript Retention Configurations
    # Configures the sliding history buffer window size per session_id (default: 500)
    TRANSCRIPT_BUFFER_SIZE: int = 500

    # ElevenLabs / STT Configurations
    ELEVENLABS_API_KEY: str = ""
    STT_PROVIDER_MODE: str = "production"  # only supported value

    # Pyannote / Diarization Configurations
    HUGGINGFACE_TOKEN: str = ""
    DIARIZATION_MODE: str = "production"  # only supported value
    DIARIZATION_DEBUG: bool = False
    # Minimum speakers expected in diarization (default to 2 for doctor-patient consultation)
    DIARIZATION_MIN_SPEAKERS: int = 2
    # Maximum speakers expected in diarization (default to 3)
    DIARIZATION_MAX_SPEAKERS: int = 3
    # Rolling audio buffer kept per session (seconds)
    DIARIZATION_BUFFER_SECONDS: int = 7
    # Window fed into Pyannote on each inference cycle (seconds, must be <= DIARIZATION_BUFFER_SECONDS)
    DIARIZATION_WINDOW_SECONDS: int = 5
    # How often the inference cycle fires (seconds)
    DIARIZATION_INTERVAL_SECONDS: int = 2

    @model_validator(mode="after")
    def validate_provider_mode(self) -> 'Settings':
        """
        Both providers are production-only.

        The former 'development' modes returned synthetic data through the same code
        path as real output: fabricated transcript text, and speaker labels assigned by
        a `timestamp % 30` cycle. The diarization mock in particular was actively
        harmful, because clinical extraction is gated on the speaker label, so wrong
        labels silently discarded symptoms, history and medications with no error.
        Both are gone. A missing credential now fails at startup instead of degrading
        into fabricated data.
        """
        if self.STT_PROVIDER_MODE != "production":
            raise ValueError(
                f"CLINICAL_COPILOT_STT_PROVIDER_MODE must be 'production' "
                f"(got {self.STT_PROVIDER_MODE!r}). The development mock was removed: it "
                f"emitted placeholder transcript text indistinguishable from real output."
            )
        if not self.ELEVENLABS_API_KEY or not self.ELEVENLABS_API_KEY.strip():
            raise ValueError(
                "CLINICAL_COPILOT_ELEVENLABS_API_KEY must be configured. There is no "
                "offline transcription fallback."
            )
        if self.DIARIZATION_MODE != "production":
            raise ValueError(
                f"CLINICAL_COPILOT_DIARIZATION_MODE must be 'production' "
                f"(got {self.DIARIZATION_MODE!r}). The development mock was removed: it "
                f"assigned speakers on a fixed time cycle, and because clinical "
                f"extraction is gated on the speaker label, those fabricated labels "
                f"silently dropped clinical facts."
            )
        if not self.HUGGINGFACE_TOKEN or not self.HUGGINGFACE_TOKEN.strip():
            raise ValueError(
                "CLINICAL_COPILOT_HUGGINGFACE_TOKEN must be configured (with the "
                "pyannote/speaker-diarization-3.1 licence accepted). There is no "
                "offline diarization fallback."
            )
        return self

    class Config:
        case_sensitive = True
        env_prefix = "CLINICAL_COPILOT_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
