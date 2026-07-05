import unittest
import numpy as np
import time
from unittest.mock import MagicMock, patch

from app.services.speaker_verification.config import (
    SPEAKER_MATCH_THRESHOLD,
    UNKNOWN_THRESHOLD,
    MINIMUM_CONFIRMATIONS
)
from app.services.speaker_verification.models import AudioSegment, AudioQuality
from app.services.speaker_verification.backend import SpeakerEmbeddingBackend
from app.services.speaker_verification.embedding_store import cosine_similarity
from app.services.speaker_verification.voice_registry import (
    voice_registry_manager,
    RegisteredVoice,
    SessionVoiceRegistry
)
from app.services.speaker_verification.speaker_verification import (
    SpeakerVerificationService,
    VerificationResult,
    speaker_verification_service
)

class MockBackend(SpeakerEmbeddingBackend):
    def __init__(self):
        self.load_called = False
        self.extract_called = False

    def load_model(self) -> None:
        self.load_called = True

    def extract_embedding(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        self.extract_called = True
        # Return a simple deterministic embedding vector based on the first sample value
        val = samples[0] if len(samples) > 0 else 0.0
        emb = np.zeros(192, dtype=np.float32)
        emb[0] = val
        return emb


class TestSpeakerVerification(unittest.TestCase):
    def setUp(self):
        self.session_id = "test-session-verification"
        speaker_verification_service.clear_session(self.session_id)
        
        # Instantiate a clean verification service with a mock backend
        self.service = SpeakerVerificationService()
        self.mock_backend = MockBackend()
        self.service.backend = self.mock_backend
        self.service._enabled = True
        self.service._backend_loaded = True

    def tearDown(self):
        self.service.clear_session(self.session_id)
        speaker_verification_service.clear_session(self.session_id)

    def test_audio_quality_checks(self):
        """Test that PassesAudioQuality rejects silent, too short, or clipped audio correctly."""
        # 1. Too short
        short_segment = AudioSegment(
            samples=np.zeros(8000, dtype=np.float32),  # 0.5s at 16kHz
            sample_rate=16000,
            start_time=0.0,
            end_time=0.5
        )
        q = self.service.passes_audio_quality(short_segment)
        self.assertFalse(q.passed)
        self.assertIn("too short", q.reason.lower())

        # 2. Silent (low RMS)
        silent_segment = AudioSegment(
            samples=np.zeros(16000, dtype=np.float32),  # 1.0s at 16kHz
            sample_rate=16000,
            start_time=0.0,
            end_time=1.0
        )
        q = self.service.passes_audio_quality(silent_segment)
        self.assertFalse(q.passed)
        self.assertIn("Low RMS energy", q.reason)

        # 3. Clipped
        clipped_samples = np.zeros(16000, dtype=np.float32)
        # Set 10% samples to 1.0 (clipping)
        clipped_samples[:2000] = 1.0
        clipped_segment = AudioSegment(
            samples=clipped_samples,
            sample_rate=16000,
            start_time=0.0,
            end_time=1.0
        )
        q = self.service.passes_audio_quality(clipped_segment)
        self.assertFalse(q.passed)
        self.assertIn("clipping", q.reason)

        # 4. Valid segment
        valid_samples = np.random.normal(0, 0.1, 16000)
        valid_segment = AudioSegment(
            samples=valid_samples,
            sample_rate=16000,
            start_time=0.0,
            end_time=1.0
        )
        q = self.service.passes_audio_quality(valid_segment)
        self.assertTrue(q.passed)

    def test_cosine_similarity(self):
        """Test direct cosine similarity boundary cases."""
        v1 = np.ones(192)
        v2 = np.ones(192)
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)

        v3 = -np.ones(192)
        self.assertAlmostEqual(cosine_similarity(v1, v3), -1.0)

        v4 = np.zeros(192)
        v4[0] = 1.0
        v5 = np.zeros(192)
        v5[1] = 1.0
        self.assertAlmostEqual(cosine_similarity(v4, v5), 0.0)

    def test_voice_registry_online_welford(self):
        """Test Welford's running average and variance updates in voice registry."""
        initial_emb = np.array([1.0, 2.0, 3.0] + [0.0] * 189)
        voice = RegisteredVoice("test_spk", "doctor", initial_emb, time.time())

        self.assertEqual(voice.embedding_count, 1)
        np.testing.assert_array_almost_equal(voice.average_embedding[:3], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(voice.running_variance[:3], [0.0, 0.0, 0.0])

        # Second update
        second_emb = np.array([2.0, 4.0, 6.0] + [0.0] * 189)
        voice.update(second_emb, 0.90, time.time())

        self.assertEqual(voice.embedding_count, 2)
        # Average should be [1.5, 3.0, 4.5]
        np.testing.assert_array_almost_equal(voice.average_embedding[:3], [1.5, 3.0, 4.5])
        # Variance = ((x - mean_old)*(x - mean_new)) / 2
        # for dim 0: (2 - 1.0)*(2 - 1.5) = 1.0 * 0.5 = 0.5. Divided by 2 = 0.25.
        np.testing.assert_array_almost_equal(voice.running_variance[:3], [0.25, 1.0, 2.25])
        np.testing.assert_array_almost_equal(voice.running_std_dev[:3], [0.5, 1.0, 1.5])

    def test_speaker_verification_and_persistence(self):
        """Test verification mapping updates, locks, and persistent identification."""
        # Setup doctor voice embedding
        doctor_samples = np.random.normal(0, 0.1, 16000)
        doctor_samples[0] = 5.0  # Force mock backend to output first element as 5.0
        
        doctor_segment = AudioSegment(doctor_samples, 16000, 0.0, 1.0)

        # 1. First speaker turn (Doctor is registered)
        res_type, similarity, matched_spk = self.service.verify(
            self.session_id, doctor_segment, proposed_role="doctor", raw_speaker="Speaker_0"
        )
        self.assertEqual(res_type, VerificationResult.VERIFIED)
        self.assertEqual(matched_spk, "doctor")
        self.assertEqual(similarity, 1.0)

        # 2. Subsequent matching turns updates registry
        for idx in range(1, 4):
            # Check stability mappings lock in
            self.service._process_verified_mapping(
                self.session_id, raw_speaker="Speaker_0", matched_spk="doctor", similarity=0.98,
                telemetry=MagicMock()
            )

        # Verify locked map contains Speaker_0 -> doctor
        self.assertIn("Speaker_0", self.service._locked_mappings[self.session_id])
        self.assertEqual(self.service._locked_mappings[self.session_id]["Speaker_0"], "doctor")

        # 3. Patient registration
        patient_samples = np.random.normal(0, 0.1, 16000)
        patient_samples[0] = -5.0  # Force mock backend to output first element as -5.0
        patient_segment = AudioSegment(patient_samples, 16000, 1.0, 2.0)

        # Register patient (which will return VerificationResult.VERIFIED, similarity 1.0 because patient voice wasn't registered yet)
        res_type, similarity, matched_spk = self.service.verify(
            self.session_id, patient_segment, proposed_role="patient", raw_speaker="Speaker_1"
        )
        self.assertEqual(res_type, VerificationResult.VERIFIED)
        self.assertEqual(matched_spk, "patient")
        self.assertEqual(similarity, 1.0)

        # Directly verify the similarity between the registered Doctor and Patient embeddings is negative
        registry = voice_registry_manager.get_registry(self.session_id)
        doc_emb = registry.doctor.average_embedding
        pat_emb = registry.patient.average_embedding
        sim = cosine_similarity(doc_emb, pat_emb)
        self.assertLess(sim, 0.0)


    def test_unknown_speaker_expiration(self):
        """Test that inactive Unknown speakers are correctly purged after timeout."""
        registry = voice_registry_manager.get_registry(self.session_id)
        
        # Register an unknown voice
        dummy_emb = np.zeros(192)
        dummy_emb[0] = 9.9
        
        now = time.time()
        # Add to unknowns directly with timestamp 15 minutes in past
        past_time = now - 900.0
        registry.unknowns["Unknown1"] = RegisteredVoice("Unknown1", "UNKNOWN", dummy_emb, past_time)
        
        # Verify it exists initially
        self.assertIn("Unknown1", registry.unknowns)

        # Expire loop
        registry.expire_inactive_unknowns()
        
        # Verify it is deleted
        self.assertNotIn("Unknown1", registry.unknowns)

    def test_registry_session_isolation(self):
        """Test that Session Voice Registries are isolated."""
        registry_a = voice_registry_manager.get_registry("session-A")
        registry_b = voice_registry_manager.get_registry("session-B")

        dummy_emb = np.zeros(192)
        registry_a.register_voice("doctor", dummy_emb, time.time())

        self.assertIsNotNone(registry_a.doctor)
        self.assertNull(registry_b.doctor)  # Wait, assertNull doesn't exist in unittest, use assertIsNone

    def assertIsNone(self, expr, msg=None):
        super().assertIsNone(expr, msg)

    def assertNull(self, expr):
        self.assertIsNone(expr)
