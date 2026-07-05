import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from app.utils.logging import logger
from app.services.speaker_verification.config import (
    SPEAKER_MATCH_THRESHOLD,
    UNKNOWN_THRESHOLD,
    MAX_UNKNOWN_SPEAKERS,
    MINIMUM_CONFIRMATIONS,
    UNKNOWN_EXPIRY_SECONDS
)
from app.services.speaker_verification.embedding_store import cosine_similarity

class RegisteredVoice:
    def __init__(self, speaker_id: str, role: str, initial_embedding: np.ndarray, timestamp: float):
        self.speaker_id = speaker_id
        self.role = role
        self.average_embedding = initial_embedding.copy().astype(np.float32)
        
        # Welford's algorithm parameters for online variance calculation (192-dim vector)
        self._M2_embedding = np.zeros_like(initial_embedding, dtype=np.float32)
        self.running_variance = np.zeros_like(initial_embedding, dtype=np.float32)
        self.running_std_dev = np.zeros_like(initial_embedding, dtype=np.float32)
        
        # Track similarity scores of accepted turns to gauge variance of similarity
        self.similarity_mean = 1.0  # Perfect match initially
        self._M2_similarity = 0.0
        self.similarity_variance = 0.0
        
        self.embedding_count = 1
        self.last_seen = timestamp
        self.locked = False

    def update(self, embedding: np.ndarray, similarity: float, timestamp: float) -> None:
        """
        Updates the running average, running variance, and standard deviation 
        of the embeddings and similarity scores using Welford's online algorithm.
        """
        self.embedding_count += 1
        self.last_seen = timestamp
        
        # 1. Update embedding average and variance (Welford's)
        delta = embedding - self.average_embedding
        self.average_embedding += delta / self.embedding_count
        delta2 = embedding - self.average_embedding
        self._M2_embedding += delta * delta2
        self.running_variance = self._M2_embedding / self.embedding_count
        self.running_std_dev = np.sqrt(self.running_variance)
        
        # 2. Update similarity score average and variance (Welford's)
        sim_delta = similarity - self.similarity_mean
        self.similarity_mean += sim_delta / self.embedding_count
        sim_delta2 = similarity - self.similarity_mean
        self._M2_similarity += sim_delta * sim_delta2
        self.similarity_variance = self._M2_similarity / self.embedding_count

        # Check if stability lock criteria met
        if not self.locked and self.embedding_count >= MINIMUM_CONFIRMATIONS:
            if self.role in ("doctor", "patient", "attender"):
                self.locked = True
                logger.info(
                    f"VoiceRegistry: Voice identity for role '{self.role}' (ID: {self.speaker_id}) is now LOCKED.",
                    extra={"speaker_id": self.speaker_id, "role": self.role, "count": self.embedding_count}
                )


class SessionVoiceRegistry:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.doctor: Optional[RegisteredVoice] = None
        self.patient: Optional[RegisteredVoice] = None
        self.attender: Optional[RegisteredVoice] = None
        self.unknowns: Dict[str, RegisteredVoice] = {}  # Key: Unknown1, Unknown2, etc.

    def get_registered_voices(self) -> List[RegisteredVoice]:
        """Returns all currently active registered voices (clears expired Unknowns first)."""
        self.expire_inactive_unknowns()
        voices = []
        if self.doctor:
            voices.append(self.doctor)
        if self.patient:
            voices.append(self.patient)
        if self.attender:
            voices.append(self.attender)
        voices.extend(self.unknowns.values())
        return voices

    def expire_inactive_unknowns(self) -> None:
        """Removes Unknown speaker records that have been inactive for > UNKNOWN_EXPIRY_SECONDS."""
        now = time.time()
        expired_keys = []
        for key, voice in self.unknowns.items():
            if now - voice.last_seen > UNKNOWN_EXPIRY_SECONDS:
                expired_keys.append(key)
        
        for key in expired_keys:
            logger.info(
                f"VoiceRegistry: Expired inactive speaker {key} (last seen {now - self.unknowns[key].last_seen:.1f}s ago).",
                extra={"session_id": self.session_id, "expired_speaker_id": key}
            )
            del self.unknowns[key]

    def register_voice(self, role: str, embedding: np.ndarray, timestamp: float) -> RegisteredVoice:
        """Registers a new voice mapping for Doctor, Patient, Attender, or a new Unknown speaker."""
        if role == "doctor":
            self.doctor = RegisteredVoice("doctor", "doctor", embedding, timestamp)
            return self.doctor
        elif role == "patient":
            self.patient = RegisteredVoice("patient", "patient", embedding, timestamp)
            return self.patient
        elif role == "attender":
            self.attender = RegisteredVoice("attender", "attender", embedding, timestamp)
            return self.attender
        else:
            # Assign a clean Unknown ID
            self.expire_inactive_unknowns()
            for idx in range(1, MAX_UNKNOWN_SPEAKERS + 1):
                unk_id = f"Unknown{idx}"
                if unk_id not in self.unknowns:
                    voice = RegisteredVoice(unk_id, "UNKNOWN", embedding, timestamp)
                    self.unknowns[unk_id] = voice
                    logger.info(
                        f"VoiceRegistry: Created new speaker profile: {unk_id}",
                        extra={"session_id": self.session_id, "speaker_id": unk_id}
                    )
                    return voice
            
            # Fallback if slots full: Overwrite oldest Unknown
            oldest_key = min(self.unknowns.keys(), key=lambda k: self.unknowns[k].last_seen)
            voice = RegisteredVoice(oldest_key, "UNKNOWN", embedding, timestamp)
            self.unknowns[oldest_key] = voice
            logger.warning(
                f"VoiceRegistry: Reached limit of MAX_UNKNOWN_SPEAKERS. Overwrote oldest slot: {oldest_key}",
                extra={"session_id": self.session_id, "overwritten_speaker_id": oldest_key}
            )
            return voice


class VoiceRegistryManager:
    def __init__(self):
        self._registries: Dict[str, SessionVoiceRegistry] = {}

    def get_registry(self, session_id: str) -> SessionVoiceRegistry:
        if session_id not in self._registries:
            self._registries[session_id] = SessionVoiceRegistry(session_id)
        return self._registries[session_id]

    def clear_session(self, session_id: str) -> None:
        self._registries.pop(session_id, None)
        logger.info(
            "VoiceRegistryManager: Registry cleared for session.", 
            extra={"session_id": session_id}
        )

# Singleton registry manager
voice_registry_manager = VoiceRegistryManager()
