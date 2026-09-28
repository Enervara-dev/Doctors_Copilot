"""
Transcript egress.

The audio ingress takes PCM in and the STT workers publish `TranscriptEvent`s
onto the bus, where the clinical state engine consumes them. Nothing carried
them back out, so a caller could send speech and never learn what was heard.

This is that channel, and deliberately nothing more: a read-only subscription
to the bus, filtered to one session, over Server-Sent Events. It adds no
transcription, no ordering, no clinical logic — those all already exist
upstream, and a second opinion about any of them here would be a second place
to keep them right.

Two properties the consumer depends on:

* **Catch-up before live.** A subscriber that connects after speech has begun,
  or reconnects mid-consultation, is sent the session's buffered events first.
  The bus already keeps a sliding window per session for exactly this, and a
  reconnect that silently skips what it missed is worse than no reconnect.

* **Back-pressure is visible.** A slow consumer's queue is bounded. When it
  overflows the frames are dropped and the consumer is told, rather than the
  server growing a buffer on behalf of a client that cannot keep up.

Authentication is the same call the WebSocket ingress uses, bound to the same
session, so a credential for one consultation cannot read another's speech.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.core.security import verify_session_token
from app.services.providers.models import TranscriptEvent
from app.services.transcript_bus import transcript_bus
from app.utils.logging import logger

router = APIRouter()

#: Frames a subscriber may fall behind by before we start dropping. Roughly a
#: minute of dense two-way speech; past that the consumer is not keeping up and
#: buffering further only delays the moment it finds out.
QUEUE_LIMIT = 512

#: How often the generator wakes to check for a disconnect while idle.
POLL_INTERVAL_SECONDS = 0.05

#: A comment frame at this interval, so a proxy does not close a quiet stream.
KEEPALIVE_SECONDS = 15.0


def _frame(event_type: str, payload: object) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


@router.get("/streaming/transcript")
async def transcript_stream(
    request: Request,
    session_id: str = Query(..., description="Consultation session ID"),
    token: str = Query(..., description="Credential for this session"),
) -> StreamingResponse:
    """
    Streams this session's transcript events as they are published.

    The token arrives as a query parameter rather than a header because this is
    also reachable from an `EventSource`, which cannot set one — the same
    reason the audio WebSocket takes it that way.
    """
    if not verify_session_token(token, expected_session_id=session_id):
        logger.warning(
            "Transcript stream rejected: invalid credential",
            extra={"session_id": session_id, "event": "auth_failure"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "Not valid for this session."},
        )

    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_LIMIT)
    loop = asyncio.get_running_loop()
    dropped = 0

    def on_event(event: TranscriptEvent) -> None:
        # The bus publishes synchronously from the STT worker's thread of
        # control, so the queue is fed thread-safely rather than awaited here.
        nonlocal dropped
        if event.session_id != session_id:
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except asyncio.QueueFull:
            dropped += 1
        except RuntimeError:
            # The loop is closing; the generator's cleanup will unsubscribe.
            pass

    transcript_bus.subscribe(on_event)

    async def frames() -> AsyncIterator[str]:
        nonlocal dropped
        idle = 0.0
        try:
            # What the subscriber missed, before anything live. A reconnecting
            # consumer must not have to guess at the gap.
            replay = transcript_bus.get_recent_events(session_id)
            yield _frame(
                "catchup",
                {"session_id": session_id, "count": len(replay)},
            )
            for event in replay:
                yield _frame("transcript", json.loads(event.model_dump_json()))

            while True:
                if await request.is_disconnected():
                    return

                try:
                    event = queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    idle += POLL_INTERVAL_SECONDS
                    if idle >= KEEPALIVE_SECONDS:
                        idle = 0.0
                        yield ": keep-alive\n\n"
                    continue

                idle = 0.0
                yield _frame("transcript", json.loads(event.model_dump_json()))

                if dropped:
                    # Said plainly rather than hidden: the consumer decides
                    # whether a gap in a clinical transcript matters to it.
                    # Reported once per burst, then reset — a running total
                    # repeated on every frame would be noise, not information.
                    yield _frame("dropped", {"count": dropped})
                    dropped = 0
        finally:
            transcript_bus.unsubscribe(on_event)
            logger.info(
                "Transcript stream closed",
                extra={"session_id": session_id, "dropped": dropped},
            )

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
