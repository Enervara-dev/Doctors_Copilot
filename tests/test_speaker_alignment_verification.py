import unittest
import numpy as np
import time
import asyncio
from unittest.mock import MagicMock, patch

from app.services.providers.models import TranscriptEvent
from app.services.speaker_timeline import speaker_timeline_manager
from app.services.speaker_alignment import speaker_alignment_service
from app.services.diarization_worker import diarization_worker_manager
from app.services.speaker_verification.speaker_verification import (
    speaker_verification_service,
    VerificationResult
)
from app.services.speaker_verification.voice_registry import voice_registry_manager
from app.services.speaker_verification.backend import SpeakerEmbeddingBackend
from app.services.transcript_bus import transcript_bus

class MockEmbeddingBackend(SpeakerEmbeddingBackend):
    def __init__(self):
        pass

    def load_model(self) -> None:
        pass

    def extract_embedding(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        # Generate embedding vector based on first sample value (our mock signal parameter)
        val = samples[0] if len(samples) > 0 else 0.0
        emb = np.zeros(192, dtype=np.float32)
        if val > 0.01:
            emb[0] = 1.0
        elif val < -0.01:
            emb[0] = -1.0
        return emb


class TestSpeakerAlignmentVerification(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session_id = "integration-test-session"
        
        # Reset services
        speaker_timeline_manager.clear_timeline(self.session_id)
        speaker_alignment_service.clear_session(self.session_id)
        speaker_verification_service.clear_session(self.session_id)
        diarization_worker_manager.clear_session(self.session_id)
        
        # Override verification service with a mocked backend
        self.mock_backend = MockEmbeddingBackend()
        speaker_verification_service.backend = self.mock_backend
        speaker_verification_service._enabled = True
        speaker_verification_service._backend_loaded = True
        # Explicitly make sure the background worker is started
        speaker_verification_service.start_worker()

        # Seed local timeline segments
        self.timeline = speaker_timeline_manager.get_timeline(self.session_id)

    async def asyncTearDown(self):
        speaker_timeline_manager.clear_timeline(self.session_id)
        speaker_alignment_service.clear_session(self.session_id)
        speaker_verification_service.clear_session(self.session_id)
        diarization_worker_manager.clear_session(self.session_id)

    async def test_integration_scenario_doctor_only(self):
        """Scenario: Single doctor speaker throughout. Mapped and confirmed doctor."""
        self.timeline.segments = [
            {"start": 0.0, "end": 2.0, "label": "Speaker_0"}
        ]

        # Doctor speaks: fill the whole buffer with 3276 (Doctor mock signal)
        samples = np.full(32000, 3276, dtype=np.int16)
        diarization_worker_manager.feed_audio(self.session_id, samples.tobytes())

        # Align committed turn
        res = {
            "type": "committed",
            "text": "Hello, I am checking your heart rate today.",
            "words": [
                {"text": "Hello,", "start": 0.1, "end": 0.5, "type": "word"},
                {"text": "I", "start": 0.5, "end": 0.8, "type": "word"},
                {"text": "am", "start": 0.8, "end": 1.0, "type": "word"},
                {"text": "checking", "start": 1.0, "end": 1.3, "type": "word"},
                {"text": "your", "start": 1.3, "end": 1.5, "type": "word"},
                {"text": "heart", "start": 1.5, "end": 1.7, "type": "word"},
                {"text": "rate", "start": 1.7, "end": 1.9, "type": "word"},
                {"text": "today.", "start": 1.9, "end": 2.0, "type": "word"}
            ]
        }

        # Align
        events = speaker_alignment_service.align_and_segment(self.session_id, "doctor", res, None)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].speaker_id, "doctor")

        # Let the background worker process verification
        await speaker_verification_service._queue.join()

        # Check mapping matches Doctor
        mapping = speaker_alignment_service._session_mappings.get(self.session_id)
        self.assertEqual(mapping.get("Speaker_0"), "doctor")

    async def test_integration_scenario_doctor_patient_swap_correction(self):
        """
        Scenario: Pyannote swaps Speaker_0 to Speaker_1 mid-consultation for Doctor's voice.
        ECAPA similarity detects that Speaker_1 has same voice as Doctor, restoring Doctor label.
        """
        self.timeline.segments = [
            {"start": 0.0, "end": 2.0, "label": "Speaker_0"}, # Initial Doctor turn
            {"start": 2.0, "end": 4.0, "label": "Speaker_1"}  # Mid consultation swap (should be Doctor)
        ]

        # Feed Doctor audio (fill whole buffer with 3276)
        samples = np.full(64000, 3276, dtype=np.int16)
        diarization_worker_manager.feed_audio(self.session_id, samples.tobytes())

        # Turn 1: Scribe committed
        res1 = {
            "type": "committed",
            "text": "Hello, take a seat.",
            "words": [
                {"text": "Hello,", "start": 0.1, "end": 0.9, "type": "word"},
                {"text": "take", "start": 0.9, "end": 1.2, "type": "word"},
                {"text": "a", "start": 1.2, "end": 1.5, "type": "word"},
                {"text": "seat.", "start": 1.5, "end": 1.8, "type": "word"}
            ]
        }
        events = speaker_alignment_service.align_and_segment(self.session_id, "doctor", res1, None)
        self.assertEqual(events[0].speaker_id, "doctor") # Matches Pyannote Speaker_0 -> doctor

        # Run worker to register Doctor
        await speaker_verification_service._queue.join()

        # Perform 2 more Doctor verification cycles to LOCK the Speaker_0 -> doctor mapping
        for _ in range(2):
            speaker_verification_service.enqueue_verification_task(
                self.session_id, start_time=0.0, end_time=2.0, proposed_role="doctor", raw_speaker="Speaker_0"
            )
        await speaker_verification_service._queue.join()
        
        # Verify Speaker_0 mapping is locked
        self.assertTrue(speaker_verification_service._role_locks[self.session_id])

        # Turn 2: Scribe committed for next segment (Pyannote maps to Speaker_1 but it is actually Doctor's voice)
        res2 = {
            "type": "committed",
            "text": "I will examine your throat.",
            "words": [
                {"text": "I", "start": 2.1, "end": 2.4, "type": "word"},
                {"text": "will", "start": 2.4, "end": 2.6, "type": "word"},
                {"text": "examine", "start": 2.6, "end": 3.0, "type": "word"},
                {"text": "your", "start": 3.0, "end": 3.3, "type": "word"},
                {"text": "throat.", "start": 3.3, "end": 3.8, "type": "word"}
            ]
        }
        events2 = speaker_alignment_service.align_and_segment(self.session_id, "doctor", res2, None)
        # Initially Speaker_1 is mapped to patient as baseline
        self.assertEqual(events2[0].speaker_id, "patient")

        # Now, trigger verification of the Speaker_1 segment
        # It has Doctor voice (5.0), so similarity with Doctor is high (1.0).
        # Run worker to lock mapping corrections
        for _ in range(3):
            speaker_verification_service.enqueue_verification_task(
                self.session_id, start_time=2.0, end_time=4.0, proposed_role="patient", raw_speaker="Speaker_1"
            )
        await speaker_verification_service._queue.join()

        # Check that Speaker_1 has been corrected to doctor
        mapping = speaker_alignment_service._session_mappings.get(self.session_id)
        self.assertEqual(mapping.get("Speaker_1"), "doctor")

    async def test_noise_handling(self):
        """Scenario: Segment with background noise (low RMS) skips verification, retaining original label."""
        self.timeline.segments = [
            {"start": 0.0, "end": 2.0, "label": "Speaker_0"}
        ]
        
        # Silent samples
        samples = np.zeros(32000, dtype=np.int16)
        diarization_worker_manager.feed_audio(self.session_id, samples.tobytes())

        res = {
            "type": "committed",
            "text": "[Music playing or typing noise]",
            "words": [{"text": "[Noise]", "start": 0.1, "end": 1.5, "type": "word"}]
        }
        events = speaker_alignment_service.align_and_segment(self.session_id, "doctor", res, None)
        self.assertEqual(events[0].speaker_id, "doctor") # Retains original

        # Yield to worker
        await speaker_verification_service._queue.join()

        # Since it is low RMS, it should fail passes_audio_quality and skip verification
        # Verify registry doctor remains uninitialized (none)
        registry = speaker_verification_service.backend
        # Registry doctor voice is indeed None
        reg = voice_registry_manager.get_registry(self.session_id)
        self.assertIsNone(reg.doctor)

    async def test_session_cleanup(self):
        """Scenario: Verification service clears all dynamic states upon session clearance."""
        # Setup confirmation maps and role locks
        speaker_verification_service.enqueue_verification_task(
            self.session_id, start_time=0.0, end_time=2.0, proposed_role="doctor", raw_speaker="Speaker_0"
        )
        
        await speaker_verification_service._queue.join()

        # Clear session
        speaker_alignment_service.clear_session(self.session_id)
        
        self.assertNotIn(self.session_id, speaker_verification_service._confirmations)
        self.assertNotIn(self.session_id, speaker_verification_service._locked_mappings)
        self.assertNotIn(self.session_id, speaker_verification_service._role_locks)
