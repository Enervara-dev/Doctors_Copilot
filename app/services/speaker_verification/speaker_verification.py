import asyncio
import time
import numpy as np
from typing import Dict, Optional, Tuple, Set
from pydantic import BaseModel
from app.utils.logging import logger
from app.services.speaker_verification.config import (
    ENABLE_SPEAKER_VERIFICATION,
    SPEAKER_MATCH_THRESHOLD,
    UNKNOWN_THRESHOLD,
    MIN_RMS_THRESHOLD,
    CLIPPING_THRESHOLD,
    MINIMUM_CONFIRMATIONS,
    MIN_SEGMENT_DURATION
)
from app.services.speaker_verification.models import AudioSegment, AudioQuality, SpeakerVerificationTelemetry
from app.services.speaker_verification.backend import SpeechBrainBackend
from app.services.speaker_verification.voice_registry import voice_registry_manager, RegisteredVoice
from app.services.speaker_verification.embedding_store import cosine_similarity

class VerificationResult:
    SKIPPED = "VerificationSkipped"
    VERIFIED = "Verified"
    TENTATIVE = "Tentative"
    UNKNOWN = "Unknown"


class SpeakerVerificationService:
    def __init__(self):
        self._enabled = ENABLE_SPEAKER_VERIFICATION
        self.backend = SpeechBrainBackend()
        self._queue = None
        self._queue_loop = None
        self._worker_task: Optional[asyncio.Task] = None
        self._confirmations: Dict[str, Dict[Tuple[str, str], int]] = {}  # session_id -> {(raw_speaker, role): count}
        self._locked_mappings: Dict[str, Dict[str, str]] = {}  # session_id -> {raw_speaker: role}
        self._role_locks: Dict[str, Set[str]] = {}  # session_id -> {role} (locked roles)
        self._backend_loaded = False
        self._model_load_time_ms = 0.0

    def load_backend(self) -> None:
        """Loads and warms the selected embedding backend."""
        if not self._enabled:
            logger.info("SpeakerVerificationService: Verification is disabled by configuration.")
            return

        if self._backend_loaded:
            return

        try:
            t0 = time.perf_counter()
            self.backend.load_model()
            self._model_load_time_ms = (time.perf_counter() - t0) * 1000.0
            self._backend_loaded = True
            # Start background worker
            self.start_worker()
        except Exception as e:
            logger.error(
                f"SpeakerVerificationService: Failed to load backend. Fallback to Pyannote enabled. Error: {e}",
                exc_info=True
            )
            self._enabled = False

    def start_worker(self) -> None:
        """Starts the background worker queue loop if not already running."""
        if not self._enabled:
            return
            
        try:
            loop = asyncio.get_running_loop()
            
            # Lazily initialize queue for the current running loop
            if getattr(self, "_queue", None) is None or getattr(self, "_queue_loop", None) != loop:
                self._queue = asyncio.Queue()
                self._queue_loop = loop
                
                # If loop changed, cancel the old worker task to start fresh
                if self._worker_task is not None and not self._worker_task.done():
                    self._worker_task.cancel()
                    self._worker_task = None
                
            if self._worker_task is None or self._worker_task.done():
                self._worker_task = loop.create_task(self._bg_worker_loop())
                logger.info("SpeakerVerificationService: Background verification worker task started.")
        except RuntimeError:
            # No running event loop (e.g. during synchronous startup or sync tests)
            logger.warning("SpeakerVerificationService: Cannot start background worker task - no running event loop.")




    def enqueue_verification_task(
        self, session_id: str, start_time: float, end_time: float, proposed_role: str, raw_speaker: str
    ) -> None:
        """Enqueues a diarized segment for background speaker verification."""
        if not self._enabled:
            return
            
        self.start_worker()
        try:
            self._queue.put_nowait((session_id, start_time, end_time, proposed_role, raw_speaker))
        except Exception as e:
            logger.error(
                "SpeakerVerificationService: Failed to enqueue verification task",
                exc_info=True,
                extra={"session_id": session_id, "error": str(e)}
            )

    def passes_audio_quality(self, audio_segment: AudioSegment) -> AudioQuality:
        """Determines if the audio segment is clean and long enough for speaker verification."""
        samples = audio_segment.samples
        duration = audio_segment.end_time - audio_segment.start_time
        
        # 1. Duration check
        if duration < MIN_SEGMENT_DURATION:
            return AudioQuality(
                passed=False, rms=0.0, speech_ratio=0.0, clipping_ratio=0.0,
                reason=f"Duration too short ({duration:.2f}s < {MIN_SEGMENT_DURATION}s)"
            )
            
        # 2. Empty check
        if len(samples) == 0:
            return AudioQuality(
                passed=False, rms=0.0, speech_ratio=0.0, clipping_ratio=0.0,
                reason="Empty audio samples"
            )
            
        # 3. RMS energy check (noise floor)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < MIN_RMS_THRESHOLD:
            return AudioQuality(
                passed=False, rms=rms, speech_ratio=0.0, clipping_ratio=0.0,
                reason=f"Low RMS energy ({rms:.5f} < {MIN_RMS_THRESHOLD})"
            )
            
        # 4. Clipping/distortion check
        clipping_ratio = float(np.sum(np.abs(samples) >= CLIPPING_THRESHOLD) / len(samples))
        if clipping_ratio > 0.05:  # Over 5% samples clipped
            return AudioQuality(
                passed=False, rms=rms, speech_ratio=1.0, clipping_ratio=clipping_ratio,
                reason=f"High clipping ratio ({clipping_ratio:.2%})"
            )
            
        return AudioQuality(passed=True, rms=rms, speech_ratio=1.0, clipping_ratio=clipping_ratio)

    def verify(
        self, session_id: str, audio_segment: AudioSegment, proposed_role: str, raw_speaker: str
    ) -> Tuple[str, float, Optional[str]]:
        """
        Executes similarity thresholding and deterministic audio quality validation.
        Returns:
            Tuple of (VerificationResult, similarity_score, matched_speaker_id)
        """
        # 1. Quality Validation
        quality = self.passes_audio_quality(audio_segment)
        if not quality.passed:
            return VerificationResult.SKIPPED, 0.0, quality.reason

        # Normalise audio before inference (scale peak amplitude to 0.95)
        samples = audio_segment.samples.copy()
        max_amp = np.max(np.abs(samples))
        if max_amp > 0.01:
            samples = samples * (0.95 / max_amp)

        # 2. Extract Embedding
        t0 = time.perf_counter()
        embedding = self.backend.extract_embedding(samples, audio_segment.sample_rate)
        embedding_time = (time.perf_counter() - t0) * 1000.0

        # Get voice registry for session
        registry = voice_registry_manager.get_registry(session_id)
        registered_voices = registry.get_registered_voices()
        now = time.time()

        # 3. Registry Registration & Matching
        # If registry for the proposed role is empty, register the voice embedding
        if proposed_role in ("doctor", "patient", "attender"):
            role_voice = getattr(registry, proposed_role)
            if role_voice is None:
                # Double-check that it does not match any other active registered role
                best_sim = 0.0
                for voice in registered_voices:
                    sim = cosine_similarity(embedding, voice.average_embedding)
                    if sim > best_sim:
                        best_sim = sim
                
                # If it matches an existing role with high confidence, do not create duplicate
                if best_sim >= SPEAKER_MATCH_THRESHOLD:
                    pass
                else:
                    # Register new voice for proposed role
                    registry.register_voice(proposed_role, embedding, now)
                    logger.info(
                        f"SpeakerVerificationService: Registered initial voice for role '{proposed_role}'.",
                        extra={"session_id": session_id, "role": proposed_role}
                    )
                    return VerificationResult.VERIFIED, 1.0, proposed_role

        # Compare against all active registered voices
        best_similarity = 0.0
        matched_voice: Optional[RegisteredVoice] = None

        for voice in registered_voices:
            sim = cosine_similarity(embedding, voice.average_embedding)
            if sim > best_similarity:
                best_similarity = sim
                matched_voice = voice

        # 4. Identity Resolution Logic
        if best_similarity >= SPEAKER_MATCH_THRESHOLD:
            # Verified match
            matched_voice.update(embedding, best_similarity, now)
            return VerificationResult.VERIFIED, best_similarity, matched_voice.speaker_id

        elif best_similarity >= UNKNOWN_THRESHOLD:
            # Tentative match (do not update running average to avoid drift)
            return VerificationResult.TENTATIVE, best_similarity, matched_voice.speaker_id

        else:
            # Unknown speaker
            new_voice = registry.register_voice("UNKNOWN", embedding, now)
            return VerificationResult.UNKNOWN, best_similarity, new_voice.speaker_id

    async def _bg_worker_loop(self) -> None:
        """Asynchronously processes verification tasks queued from the streaming pipeline."""
        from app.services.diarization_worker import diarization_worker_manager
        
        while self._enabled:
            try:
                session_id, start, end, proposed_role, raw_speaker = await self._queue.get()
                
                t_verify_start = time.perf_counter()
                
                # Get the raw AudioSegment slice using exact sample offsets
                segment = diarization_worker_manager.get_session_audio_slice(session_id, start, end)
                if not segment:
                    self._queue.task_done()
                    continue

                # Run verification (computes quality, extracts embedding, evaluates registry)
                t_emb_start = time.perf_counter()
                res_type, similarity, matched_spk = self.verify(session_id, segment, proposed_role, raw_speaker)
                verify_time_ms = (time.perf_counter() - t_verify_start) * 1000.0
                emb_time_ms = (time.perf_counter() - t_emb_start) * 1000.0 if res_type != VerificationResult.SKIPPED else 0.0

                # Log Telemetry
                telemetry = SpeakerVerificationTelemetry(
                    model_name="ECAPA-TDNN",
                    model_version="voxceleb",
                    threshold_version="1.0",
                    model_load_time_ms=self._model_load_time_ms,
                    embedding_time_ms=emb_time_ms,
                    verification_time_ms=verify_time_ms,
                    similarity_score=similarity,
                    speaker_changed=False,
                    verification_enabled=self._enabled
                )

                if res_type == VerificationResult.SKIPPED:
                    self._queue.task_done()
                    continue

                # Handle Identity Stability & Swap Corrections
                if res_type == VerificationResult.VERIFIED:
                    self._process_verified_mapping(session_id, raw_speaker, matched_spk, similarity, telemetry)

                self._queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SpeakerVerificationService: Error in bg worker loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    def _process_verified_mapping(
        self, session_id: str, raw_speaker: str, matched_spk: str, similarity: float, telemetry: SpeakerVerificationTelemetry
    ) -> None:
        """Handles identity stability counters, locking, and mapping swap corrections."""
        if session_id not in self._confirmations:
            self._confirmations[session_id] = {}
        if session_id not in self._locked_mappings:
            self._locked_mappings[session_id] = {}
        if session_id not in self._role_locks:
            self._role_locks[session_id] = set()

        locked_map = self._locked_mappings[session_id]
        role_locks = self._role_locks[session_id]

        # If raw speaker label mapping is already locked, enforce it
        if raw_speaker in locked_map:
            locked_role = locked_map[raw_speaker]
            if locked_role != matched_spk:
                logger.warning(
                    f"SpeakerVerificationService: Prevented mapping raw speaker {raw_speaker} to {matched_spk}. "
                    f"Locked to {locked_role}.",
                    extra={"session_id": session_id, "raw_speaker": raw_speaker, "locked_to": locked_role}
                )
            return

        # stability counter increment
        key = (raw_speaker, matched_spk)
        self._confirmations[session_id][key] = self._confirmations[session_id].get(key, 0) + 1
        count = self._confirmations[session_id][key]

        # Lock mapping if count reaches MINIMUM_CONFIRMATIONS
        if count >= MINIMUM_CONFIRMATIONS:
            locked_map[raw_speaker] = matched_spk
            if matched_spk in ("doctor", "patient", "attender"):
                role_locks.add(matched_spk)

            # Update speaker mapping in speaker alignment service
            from app.services.speaker_alignment import speaker_alignment_service
            current_mapping = speaker_alignment_service._session_mappings.get(session_id)
            
            # Audit log format: medical audit logging standard
            if current_mapping and current_mapping.get(raw_speaker) != matched_spk:
                old_role = current_mapping.get(raw_speaker)
                
                # Update mapping
                speaker_alignment_service._session_mappings[session_id] = {
                    **current_mapping,
                    raw_speaker: matched_spk
                }
                
                telemetry.speaker_changed = True
                
                # Medical Audit Log
                audit_log_message = (
                    f"AUDIT | {time.strftime('%H:%M:%S')} | Pyannote {raw_speaker} "
                    f"-> ECAPA {matched_spk} | Similarity: {similarity:.2f} | Reason: Voice verification correction"
                )
                logger.info(
                    audit_log_message,
                    extra={
                        "session_id": session_id,
                        "raw_speaker": raw_speaker,
                        "old_role": old_role,
                        "new_role": matched_spk,
                        "similarity": similarity,
                        "event": "speaker_correction_audit"
                    }
                )

                # Trigger global session re-alignment to publish corrected events
                try:
                    speaker_alignment_service.realign_session(session_id)
                except Exception as ex:
                    logger.error(
                        "SpeakerVerificationService: Error triggering session realignment",
                        exc_info=True,
                        extra={"session_id": session_id, "error": str(ex)}
                    )

    def clear_session(self, session_id: str) -> None:
        """Clears registry and confirmation cache maps to prevent memory leaks."""
        self._confirmations.pop(session_id, None)
        self._locked_mappings.pop(session_id, None)
        self._role_locks.pop(session_id, None)
        voice_registry_manager.clear_session(session_id)


# Create singleton instance
speaker_verification_service = SpeakerVerificationService()
