"""
Forwards finalized transcript events from the audio pipeline to the Copilot API.

This is the missing link between the two services in this repository:

`app.main` produces `TranscriptEvent`s with a **per-role** sequence number
(`session.py`'s `transcript_seq_counters`, one counter for doctor, one for
patient, one for attender). `copilot_api`'s session contract requires **one**
single, gap-free, strictly increasing sequence per consultation
(`CopilotSession._next_sequence` in `integration/copilot_runtime.py`, starting
at 1). Forwarding `sequence_number` unmodified would have three roles each
claiming sequence 0, 1, 2, ... in parallel: the second role's first event
would be rejected outright (`StaleEventError`, sequence already consumed) or
buffered behind a gap that never closes.

The fix lives here, at the boundary, for the same reason
`integration/doctor_adapter.py` normalises speaker labels at the boundary
rather than teaching the clinical core a second convention: one place owns the
translation, and a misordered or unmappable event fails loudly in a log line
here instead of silently vanishing into a permanently "buffered" state three
layers downstream.

Disabled by default. With `COPILOT_BRIDGE_ENABLED` unset or false, this module
subscribes to nothing, opens no HTTP client, and `app.main` behaves exactly as
it did before this file existed.

Deliberately tolerant of the Copilot API being unreachable: a consultation's
audio must keep recording and transcribing even if the downstream reasoning
service is down, restarting, or not deployed at all. Every failure here is
logged and swallowed, never raised into the ingestion path.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from app.services.providers.models import TranscriptEvent
from app.services.transcript_bus import transcript_bus
from app.utils.logging import logger


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _canonical_speaker_role(value: Optional[str]) -> str:
    """
    Maps whatever `speaker_id` the alignment layer produced onto the three
    roles `CopilotEvent` accepts, plus `unknown`.

    Unlike `doctor_adapter.normalize_speaker`, this never raises: a transient
    `UNKNOWN` during the first few seconds of a session (before diarization has
    resolved a cluster to a role) is expected and must not crash the bridge
    that forwards it. Skipping unmappable events instead is handled by the
    caller — see `_on_transcript_event`.
    """
    candidate = (value or "").strip().lower()
    return candidate if candidate in ("doctor", "patient", "attender") else "unknown"


def _copilot_event_payload(event: TranscriptEvent, sequence: int) -> dict:
    """Builds the JSON body `POST /copilot/v1/sessions/{id}/events` expects."""
    duration = None
    if event.start_time is not None and event.end_time is not None:
        duration = max(0.0, event.end_time - event.start_time)

    return {
        "event_id": event.event_id,
        "consultation_id": event.session_id,
        "sequence": sequence,
        "event_type": "utterance",
        "source": "audio-pipeline",
        "provenance": "SERVICE",
        "payload": {
            "text": event.transcript or event.text or "",
            "speaker_role": _canonical_speaker_role(event.speaker_id),
            "speaker_label": event.speaker_label or "",
            "is_final": True,
            "confidence": event.confidence,
            "offset_seconds": event.start_time,
            "duration_seconds": duration,
        },
    }


@dataclass
class _BridgedSession:
    """Per-session bridge state: the credential to use and the next sequence to assign."""

    token: str
    next_sequence: int = 1  # CopilotSession._next_sequence starts at 1, not 0.
    opened: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class CopilotBridge:
    """
    Subscribes to the transcript bus once, and forwards FINAL events for any
    session that has been explicitly bridged via `ensure_bridged`.

    Only FINAL events are forwarded. Partials are intentionally dropped at
    this boundary: `CopilotSession._apply` already discards non-final events
    after recording live text, and — more importantly — `app.main` mints a
    fresh `event_id` for every partial (`uuid.uuid4()` per call in
    `speaker_alignment.py`), whereas the Copilot contract assumes a partial and
    its eventual final share one id so the final can supersede it. Forwarding
    partials under that mismatched assumption would populate the live-text
    view with several stale, never-superseded entries per utterance rather
    than one that updates. Live partial captions are a real gap this bridge
    does not close; only settled, final transcript facts are forwarded.
    """

    def __init__(self) -> None:
        self._enabled = _flag("COPILOT_BRIDGE_ENABLED", False)
        self._base_url = os.getenv("COPILOT_BRIDGE_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
        self._timeout = float(os.getenv("COPILOT_BRIDGE_TIMEOUT_SECONDS", "10") or 10)
        self._sessions: Dict[str, _BridgedSession] = {}
        self._client = None  # httpx.AsyncClient, created lazily so import stays optional.
        self._subscribed = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _client_or_none(self):
        if not self._enabled:
            return None
        if self._client is None:
            try:
                import httpx
            except ImportError:
                logger.error(
                    "CopilotBridge: COPILOT_BRIDGE_ENABLED is set but httpx is not "
                    "installed; the bridge cannot forward events."
                )
                return None
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self._client

    async def ensure_bridged(self, session_id: str, token: str) -> None:
        """
        Opens a Copilot session for `session_id` the first time any participant
        connects to it, and subscribes the forwarding callback once globally.

        Idempotent: a second call for a session already bridged is a no-op, so
        every WebSocket connection (doctor, patient, attender) can call this
        unconditionally without duplicate-opening the Copilot session.
        """
        if not self._enabled or session_id in self._sessions:
            return

        bridged = _BridgedSession(token=token)
        self._sessions[session_id] = bridged

        if not self._subscribed:
            transcript_bus.subscribe(self._on_transcript_event)
            self._subscribed = True
            logger.info("CopilotBridge: subscribed to the transcript bus.")

        await self._open_copilot_session(session_id, bridged)

    async def _open_copilot_session(self, session_id: str, bridged: _BridgedSession) -> None:
        client = self._client_or_none()
        if client is None:
            return
        try:
            resp = await client.post(
                "/copilot/v1/sessions",
                json={"consultation_id": session_id},
                headers={"Authorization": f"Bearer {bridged.token}"},
            )
            if resp.status_code >= 400:
                logger.error(
                    "CopilotBridge: could not open Copilot session for %s (%s): %s",
                    session_id, resp.status_code, resp.text[:300],
                )
                return
            bridged.opened = True
            logger.info("CopilotBridge: opened Copilot session %s at %s", session_id, self._base_url)
        except Exception as exc:  # noqa: BLE001 - the bridge must never break ingestion
            logger.error(
                "CopilotBridge: Copilot API unreachable while opening session %s: %s",
                session_id, exc,
            )

    async def _on_transcript_event(self, event: TranscriptEvent) -> None:
        if not self._enabled or not event.is_final:
            return

        bridged = self._sessions.get(event.session_id)
        if bridged is None:
            return  # This session was never bridged; nothing to forward to.

        if _canonical_speaker_role(event.speaker_id) == "unknown":
            logger.warning(
                "CopilotBridge: dropping final event %s in %s — speaker %r did not "
                "resolve to doctor/patient/attender, forwarding it would silently "
                "attribute nothing in the Copilot clinical state.",
                event.event_id, event.session_id, event.speaker_id,
            )
            return

        client = self._client_or_none()
        if client is None:
            return

        async with bridged.lock:
            sequence = bridged.next_sequence
            bridged.next_sequence += 1

        body = _copilot_event_payload(event, sequence)
        try:
            resp = await client.post(
                f"/copilot/v1/sessions/{event.session_id}/events",
                json=body,
                headers={"Authorization": f"Bearer {bridged.token}"},
            )
            if resp.status_code >= 400:
                logger.error(
                    "CopilotBridge: event %s (seq %s) rejected by Copilot API (%s): %s",
                    event.event_id, sequence, resp.status_code, resp.text[:300],
                )
            else:
                logger.info(
                    "CopilotBridge: forwarded event %s (seq %s, %s) for %s",
                    event.event_id, sequence, event.speaker_id, event.session_id,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "CopilotBridge: failed to forward event %s for %s: %s",
                event.event_id, event.session_id, exc,
            )

    def clear_session(self, session_id: str) -> None:
        """Drops bridge state for a closed session, so it cannot grow unbounded."""
        self._sessions.pop(session_id, None)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


#: Process-wide singleton, mirroring every other manager in `app/services/`.
copilot_bridge = CopilotBridge()
