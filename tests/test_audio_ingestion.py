"""
Ingestion-layer tests: PCM frame validation, buffering/chunking, and the
speaker-alignment behaviours that decide who said what.

This layer had no coverage, which is how a missing `asyncio` import survived in the
WebSocket disconnect path and how duplicate committed transcripts went unnoticed.
"""

import asyncio
import unittest

from app.core.config import settings
from app.services.audio_buffer import AudioFrameValidator, AudioSessionBuffer, InvalidAudioFrameError
from app.services.providers.models import TranscriptEvent
from app.services.speaker_alignment import SpeakerAlignmentService


def pcm(duration_ms: int) -> bytes:
    """Silent 16 kHz mono 16-bit PCM of the given duration."""
    return b"\x00\x00" * int(settings.AUDIO_SAMPLE_RATE * duration_ms / 1000)


class TestAudioFrameValidation(unittest.TestCase):
    def test_accepts_frames_across_the_supported_range(self):
        for ms in (20, 30, 40, 50):
            AudioFrameValidator.validate_pcm_frame(pcm(ms))  # must not raise

    def test_rejects_frame_below_minimum(self):
        with self.assertRaises(InvalidAudioFrameError):
            AudioFrameValidator.validate_pcm_frame(pcm(10))

    def test_rejects_frame_above_maximum(self):
        with self.assertRaises(InvalidAudioFrameError):
            AudioFrameValidator.validate_pcm_frame(pcm(80))

    def test_rejects_odd_length_frame(self):
        """16-bit samples mean an odd byte count is a truncated sample, not audio."""
        with self.assertRaises(InvalidAudioFrameError):
            AudioFrameValidator.validate_pcm_frame(pcm(20) + b"\x00")


class TestAudioSessionBuffer(unittest.IsolatedAsyncioTestCase):
    async def test_frames_accumulate_into_one_second_chunks(self):
        buf = AudioSessionBuffer("s1", "doctor")
        # 50 frames x 20 ms = exactly one second, so exactly one chunk is released.
        for _ in range(50):
            await buf.append_frame(pcm(20))

        self.assertEqual(buf.ready_chunks_queue.qsize(), 1)
        chunk = buf.ready_chunks_queue.get_nowait()
        self.assertEqual(len(chunk), settings.buffer_chunk_bytes)
        self.assertEqual(buf.total_bytes_received, settings.buffer_chunk_bytes)

    async def test_partial_second_is_held_until_flushed(self):
        buf = AudioSessionBuffer("s2", "patient")
        for _ in range(10):  # 200 ms only
            await buf.append_frame(pcm(20))

        self.assertEqual(buf.ready_chunks_queue.qsize(), 0)
        remainder = buf.flush()
        self.assertEqual(len(remainder), 10 * len(pcm(20)))

    async def test_invalid_frame_does_not_corrupt_the_buffer(self):
        buf = AudioSessionBuffer("s3", "doctor")
        await buf.append_frame(pcm(20))
        with self.assertRaises(InvalidAudioFrameError):
            await buf.append_frame(pcm(5))
        self.assertEqual(buf.total_bytes_received, len(pcm(20)))


class TestScribeCommitHandling(unittest.IsolatedAsyncioTestCase):
    """
    ElevenLabs sends every commit twice when timestamps are requested: once plain and
    once with word timings. Acting on both published each turn twice and pushed the
    aligner onto its character-rate estimate, which smears turns together.
    """

    def _provider(self):
        from app.services.providers.scribe_v2 import ScribeV2Provider
        return ScribeV2Provider(api_key="test-key", mode="production")

    async def test_untimestamped_duplicate_is_ignored(self):
        p = self._provider()
        self.assertTrue(p._expects_timestamps)

        class FakeWS:
            def __init__(self, messages):
                self._messages = list(messages)

            async def recv(self):
                return self._messages.pop(0)

        import json
        p._websocket = FakeWS([
            json.dumps({"message_type": "committed_transcript", "text": "Hello, doctor."}),
            json.dumps({"message_type": "committed_transcript_with_timestamps",
                        "text": "Hello, doctor.",
                        "words": [{"text": "Hello,", "start": 0.0, "end": 0.4},
                                  {"text": " doctor.", "start": 0.4, "end": 0.9}]}),
        ])
        p._is_connected = True

        first = await p.receive()
        self.assertEqual(first["type"], "ignored", "plain duplicate must not be published")

        second = await p.receive()
        self.assertEqual(second["type"], "committed")
        self.assertEqual(len(second["words"]), 2, "the timestamped copy is the one to use")


class TestSpeakerAlignmentGrouping(unittest.TestCase):
    def setUp(self):
        self.svc = SpeakerAlignmentService()

    def _event(self, text, speaker, start, end):
        return TranscriptEvent(
            session_id="sess", role="doctor", sequence_number=0, timestamp=0.0,
            transcript=text, text=text, is_final=True, is_partial=False,
            event_id=f"e-{start}", speaker_id=speaker, speaker_label=speaker,
            start_time=start, end_time=end, provider="test", confidence=0.9,
        )

    def test_word_split_across_commits_is_rejoined(self):
        """
        A commit can end mid-word ("...I take aml" / "odipine daily."). Left split, the
        medication never reaches entity extraction.
        """
        events = [
            self._event("I was diagnosed with hypertension and I take aml", "patient", 29.8, 35.3),
            self._event("odipine daily. My father had a heart attack.", "patient", 35.6, 42.5),
        ]
        merged = self.svc._stitch_commit_boundaries(events)

        self.assertEqual(len(merged), 1)
        self.assertIn("amlodipine daily", merged[0].transcript)
        self.assertEqual(merged[0].end_time, 42.5)

    def test_separate_sentences_are_not_glued_together(self):
        """A properly terminated sentence must stay its own turn."""
        events = [
            self._event("Do you have any fever?", "doctor", 15.3, 17.7),
            self._event("No, I do not have a fever.", "patient", 17.7, 24.6),
        ]
        self.assertEqual(len(self.svc._stitch_commit_boundaries(events)), 2)

    def test_same_speaker_complete_sentences_are_not_merged(self):
        events = [
            self._event("Good morning.", "doctor", 0.0, 1.0),
            self._event("What brings you in today?", "doctor", 1.2, 2.8),
        ]
        self.assertEqual(len(self.svc._stitch_commit_boundaries(events)), 2)


class TestClusterRoleAssignment(unittest.TestCase):
    """
    Roles are decided per speaker cluster from evidence pooled over everything that
    cluster said. Scoring sentences individually is what made one voice read as the
    doctor in one turn and the patient in the next.
    """

    def setUp(self):
        self.svc = SpeakerAlignmentService()

    def _seed(self, session_id, clusters):
        self.svc._committed_cache[session_id] = []
        self.svc._cluster_sentences = lambda _sid: clusters

    def test_roles_follow_the_language_each_cluster_uses(self):
        self._seed("sess", {
            "Speaker_0": ["Good morning. What brings you in today?",
                          "Do you have any fever?",
                          "Have you taken any medicine? Let me check your blood pressure."],
            "Speaker_1": ["Doctor, I have been having severe chest pain.",
                          "I was diagnosed with high blood pressure and I take amlodipine.",
                          "My father had a heart attack."],
        })
        mapping = self.svc.assign_cluster_roles("sess")
        self.assertEqual(mapping["Speaker_0"], "doctor")
        self.assertEqual(mapping["Speaker_1"], "patient")

    def test_assignment_is_independent_of_who_speaks_first(self):
        """
        The old mapping hardcoded Speaker_0 to the doctor, so roles inverted whenever
        the patient opened the consultation.
        """
        self._seed("sess", {
            "Speaker_0": ["Doctor, I have been having severe chest pain.",
                          "I was diagnosed with high blood pressure and I take amlodipine."],
            "Speaker_1": ["What brings you in today?",
                          "Let me check your blood pressure. Do you have any fever?"],
        })
        mapping = self.svc.assign_cluster_roles("sess")
        self.assertEqual(mapping["Speaker_0"], "patient")
        self.assertEqual(mapping["Speaker_1"], "doctor")

    def test_one_role_per_cluster(self):
        self._seed("sess", {
            "Speaker_0": ["What brings you in today?"],
            "Speaker_1": ["Doctor, I have a headache."],
            "Speaker_2": ["He has been unwell since yesterday."],
        })
        mapping = self.svc.assign_cluster_roles("sess")
        assigned = [mapping[c] for c in ("Speaker_0", "Speaker_1", "Speaker_2")]
        self.assertEqual(len(set(assigned)), 3, "each cluster gets its own role")


if __name__ == "__main__":
    unittest.main()
