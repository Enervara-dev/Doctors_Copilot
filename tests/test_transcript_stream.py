"""
The transcript egress.

Everything the Doctor application learns about what was said arrives through
this endpoint, so the tests cover the properties a consumer relies on: a
credential bound to one session, a late subscriber caught up rather than left
with a hole, live delivery after that, and the bus released when the stream
ends.

The endpoint is driven as an async generator rather than through `TestClient`.
That is not a workaround — `TestClient` reads a streaming body to completion,
so an endpoint that stays open until the client disconnects deadlocks against
it, and the deadlock says nothing about the endpoint. Iterating the response
body directly exercises the same generator, and can also stop between frames,
which is what makes the live path testable at all.

No audio, no STT provider, no network: events are published onto the bus
directly, which is exactly what the STT worker does.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
import uuid
from typing import Any, List, Optional, Tuple

os.environ.setdefault("CLINICAL_COPILOT_ENABLE_GRAPH", "False")
os.environ.setdefault("CLINICAL_COPILOT_ENABLE_LLM", "False")

from fastapi import HTTPException  # noqa: E402

from app.api.transcript_stream import transcript_stream  # noqa: E402
from app.services.providers.models import TranscriptEvent  # noqa: E402
from app.services.transcript_bus import transcript_bus  # noqa: E402
from app.utils.jwt_utils import generate_jwt_token  # noqa: E402

Frame = Tuple[str, Any]


class StubRequest:
    """The one thing the endpoint asks of a request: whether it is still there."""

    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def utterance(
    session_id: str,
    sequence: int,
    text: str,
    *,
    role: str = "patient",
    final: bool = True,
) -> TranscriptEvent:
    return TranscriptEvent(
        session_id=session_id,
        role=role,
        sequence_number=sequence,
        timestamp=1790000000.0 + sequence,
        transcript=text,
        is_partial=not final,
        is_final=final,
        confidence=0.95,
        event_id=str(uuid.uuid4()),
        speaker_label=role,
        text=text,
    )


def parse(chunk: str) -> Optional[Frame]:
    event_type: Optional[str] = None
    for line in chunk.splitlines():
        if line.startswith("event: "):
            event_type = line[len("event: ") :].strip()
        elif line.startswith("data: ") and event_type:
            return event_type, json.loads(line[len("data: ") :])
    return None


class StreamHarness:
    """Opens the endpoint and pulls frames from it one at a time."""

    def __init__(self, session_id: str, token: Optional[str] = None) -> None:
        self.session_id = session_id
        self.token = token if token is not None else generate_jwt_token(session_id, "doctor")
        self.request = StubRequest()
        self.iterator: Any = None

    async def open(self) -> None:
        response = await transcript_stream(
            self.request, session_id=self.session_id, token=self.token
        )
        self.iterator = response.body_iterator

    async def take(self, count: int, timeout: float = 5.0) -> List[Frame]:
        """Pulls until `count` data frames have arrived, ignoring keep-alives."""
        frames: List[Frame] = []

        async def pull() -> None:
            while len(frames) < count:
                chunk = await self.iterator.__anext__()
                text = chunk.decode() if isinstance(chunk, bytes) else chunk
                parsed = parse(text)
                if parsed is not None:
                    frames.append(parsed)

        await asyncio.wait_for(pull(), timeout=timeout)
        return frames

    async def close(self) -> None:
        self.request.disconnected = True
        if self.iterator is not None:
            await self.iterator.aclose()


class StreamTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.session = f"stream-{uuid.uuid4().hex[:8]}"
        self.addCleanup(transcript_bus.clear_buffer, self.session)


# ══════════════════════════════════════════════════════════════════════════
# Authentication
# ══════════════════════════════════════════════════════════════════════════


class TestAuthentication(StreamTestCase):
    async def test_a_garbage_credential_is_refused(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await transcript_stream(StubRequest(), session_id=self.session, token="nope")
        self.assertEqual(raised.exception.status_code, 403)

    async def test_another_session_s_credential_cannot_read_this_one(self) -> None:
        """
        What makes one clinical service safe to share between consultations:
        speech is not readable across them.
        """
        with self.assertRaises(HTTPException) as raised:
            await transcript_stream(
                StubRequest(),
                session_id=self.session,
                token=generate_jwt_token("a-different-consultation", "doctor"),
            )
        self.assertEqual(raised.exception.status_code, 403)

    async def test_a_valid_credential_opens_the_stream(self) -> None:
        harness = StreamHarness(self.session)
        await harness.open()
        self.addAsyncCleanup(harness.close)

        kind, payload = (await harness.take(1))[0]
        self.assertEqual(kind, "catchup")
        self.assertEqual(payload["session_id"], self.session)


# ══════════════════════════════════════════════════════════════════════════
# Catch-up
# ══════════════════════════════════════════════════════════════════════════


class TestCatchUp(StreamTestCase):
    async def test_a_late_subscriber_receives_what_it_missed(self) -> None:
        transcript_bus.publish(utterance(self.session, 1, "I have chest pain."))
        transcript_bus.publish(utterance(self.session, 2, "It started this morning."))

        harness = StreamHarness(self.session)
        await harness.open()
        self.addAsyncCleanup(harness.close)

        frames = await harness.take(3)
        self.assertEqual(frames[0][0], "catchup", "the gap is declared before it is filled")
        self.assertEqual(frames[0][1]["count"], 2)
        self.assertEqual(
            [payload["transcript"] for _, payload in frames[1:]],
            ["I have chest pain.", "It started this morning."],
        )

    async def test_catch_up_carries_the_whole_event(self) -> None:
        # The consumer rebuilds ordering and speaker attribution from this, so
        # a trimmed payload would not be enough.
        transcript_bus.publish(utterance(self.session, 1, "Hello", role="doctor"))

        harness = StreamHarness(self.session)
        await harness.open()
        self.addAsyncCleanup(harness.close)

        _, payload = (await harness.take(2))[1]
        for field in (
            "session_id",
            "role",
            "sequence_number",
            "timestamp",
            "transcript",
            "is_final",
            "is_partial",
            "event_id",
        ):
            self.assertIn(field, payload)
        self.assertEqual(payload["role"], "doctor")

    async def test_another_session_s_speech_is_not_included(self) -> None:
        other = f"other-{uuid.uuid4().hex[:8]}"
        self.addCleanup(transcript_bus.clear_buffer, other)
        transcript_bus.publish(utterance(self.session, 1, "mine"))
        transcript_bus.publish(utterance(other, 1, "not mine"))

        harness = StreamHarness(self.session)
        await harness.open()
        self.addAsyncCleanup(harness.close)

        frames = await harness.take(2)
        self.assertEqual(frames[1][1]["transcript"], "mine")

    async def test_partials_are_carried_as_well_as_finals(self) -> None:
        # The workspace renders partials; deciding what is clinically stable is
        # the consumer's call, not this endpoint's.
        transcript_bus.publish(utterance(self.session, 1, "I have chest", final=False))
        transcript_bus.publish(utterance(self.session, 1, "I have chest pain.", final=True))

        harness = StreamHarness(self.session)
        await harness.open()
        self.addAsyncCleanup(harness.close)

        frames = await harness.take(3)
        self.assertEqual([payload["is_final"] for _, payload in frames[1:]], [False, True])

    async def test_an_empty_session_still_opens_cleanly(self) -> None:
        harness = StreamHarness(self.session)
        await harness.open()
        self.addAsyncCleanup(harness.close)

        self.assertEqual((await harness.take(1))[0][1]["count"], 0)


# ══════════════════════════════════════════════════════════════════════════
# Live delivery
# ══════════════════════════════════════════════════════════════════════════


class TestLiveDelivery(StreamTestCase):
    async def test_speech_published_after_subscribing_arrives(self) -> None:
        harness = StreamHarness(self.session)
        await harness.open()
        self.addAsyncCleanup(harness.close)
        await harness.take(1)  # the catch-up frame

        transcript_bus.publish(utterance(self.session, 1, "I feel short of breath."))

        _, payload = (await harness.take(1))[0]
        self.assertEqual(payload["transcript"], "I feel short of breath.")

    async def test_live_events_arrive_in_the_order_they_were_published(self) -> None:
        harness = StreamHarness(self.session)
        await harness.open()
        self.addAsyncCleanup(harness.close)
        await harness.take(1)

        for sequence, text in enumerate(["one", "two", "three"], start=1):
            transcript_bus.publish(utterance(self.session, sequence, text))

        frames = await harness.take(3)
        self.assertEqual([payload["transcript"] for _, payload in frames], ["one", "two", "three"])

    async def test_two_speakers_both_reach_the_consumer(self) -> None:
        # Each microphone is its own STT worker, and both publish onto the same
        # bus. A consumer that saw only one of them would transcribe half a
        # consultation.
        harness = StreamHarness(self.session)
        await harness.open()
        self.addAsyncCleanup(harness.close)
        await harness.take(1)

        transcript_bus.publish(utterance(self.session, 1, "Where is the pain?", role="doctor"))
        transcript_bus.publish(utterance(self.session, 1, "In my chest.", role="patient"))

        frames = await harness.take(2)
        self.assertEqual([payload["role"] for _, payload in frames], ["doctor", "patient"])


# ══════════════════════════════════════════════════════════════════════════
# Cleanup
# ══════════════════════════════════════════════════════════════════════════


class TestTheBusIsReleased(StreamTestCase):
    async def test_the_subscription_does_not_outlive_the_stream(self) -> None:
        """
        A leaked subscriber keeps receiving every session's events for the life
        of the process, and holds its queue with them.
        """
        before = len(transcript_bus._subscribers)

        harness = StreamHarness(self.session)
        await harness.open()
        await harness.take(1)
        self.assertEqual(len(transcript_bus._subscribers), before + 1)

        await harness.close()
        self.assertEqual(len(transcript_bus._subscribers), before)

    async def test_a_disconnected_consumer_ends_the_stream(self) -> None:
        before = len(transcript_bus._subscribers)
        harness = StreamHarness(self.session)
        await harness.open()
        await harness.take(1)

        harness.request.disconnected = True
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(harness.iterator.__anext__(), timeout=5.0)

        # Ending on its own, rather than on an explicit close, must release the
        # subscription too — a browser tab closing is the common case.
        self.assertEqual(len(transcript_bus._subscribers), before)


if __name__ == "__main__":
    unittest.main()
