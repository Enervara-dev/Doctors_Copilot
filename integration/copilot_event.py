"""
The normalised event contract between the Doctor platform and the Copilot core.

Every field here was derived from one of the two existing implementations —
nothing is aspirational:

* `event_id`, `sequence`, `text`, `confidence`, `provenance` and the
  partial/final distinction come from `TranscriptUtterance`
  (doctor/types/transcript.ts).
* `consultation_id` is the Doctor platform's consultation id, which is also
  what the Python side uses as `session_id`.
* `speaker_role` is the canonical lowercase vocabulary the Python pipeline
  actually branches on (`state_engine.py:126`, plus eight other sites).
* `started_at` / `finalized_at` mirror the Doctor timestamps; `offset_seconds`
  carries the audio-relative time the Python side wants, which the Doctor
  contract does not currently supply.
* `schema_version` and `source` exist because this envelope is meant to survive
  being replayed and, later, moved onto a transport.

What the envelope deliberately does NOT carry: clinical content. It is a
transcript event. Facts, intelligence and state are outputs, not inputs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "copilot-event/1"


class SpeakerRole(str, Enum):
    """
    The canonical roles.

    Lowercase because that is what the clinical pipeline compares against, in
    nine places across seven modules. The Doctor platform emits the same four
    roles uppercase; the adapter lowercases them, and nothing downstream has to
    learn a second convention.
    """

    DOCTOR = "doctor"
    PATIENT = "patient"
    ATTENDER = "attender"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    """
    Only one type is implemented today.

    The Doctor platform's provider contract emits three kinds (`utterance`,
    `clinical-facts`, `clinical-intelligence`), but the latter two flow the
    other way — they are what a Copilot produces, not what it consumes. This
    enum is a single member on purpose, and the field exists so ordering and
    dispatch do not have to change when a second inbound type appears.
    """

    UTTERANCE = "utterance"


class Provenance(str, Enum):
    """Distinguishes synthetic sessions from real platform output."""

    FIXTURE = "FIXTURE"
    SERVICE = "SERVICE"


class UtterancePayload(BaseModel):
    """The spoken content of one utterance."""

    text: str
    speaker_role: SpeakerRole
    speaker_label: str
    is_final: bool
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    started_at: Optional[str] = None
    finalized_at: Optional[str] = None
    #: Seconds from the start of the consultation audio. The Doctor contract
    #: carries only wall-clock ISO timestamps, so this is derived, not received.
    offset_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None


class CopilotEvent(BaseModel):
    """
    One ordered, replayable fact about a consultation.

    Ordering is by `(consultation_id, sequence)` and nothing else — never by
    arrival time. Deduplication is by `event_id`, which is stable across the
    PARTIAL → FINAL update of the same utterance, exactly as the Doctor
    platform intends ("Both share an id so an update replaces rather than
    appends").
    """

    schema_version: str = SCHEMA_VERSION
    event_id: str
    consultation_id: str
    sequence: int
    event_type: EventType = EventType.UTTERANCE
    #: Which system produced this event, for debugging a replay.
    source: str = "doctor-platform"
    provenance: Provenance = Provenance.SERVICE
    payload: UtterancePayload
    #: Anything the source wants to carry through without the core interpreting
    #: it. Never read by the clinical pipeline.
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def dedupe_key(self) -> tuple[str, str, bool]:
        """
        What makes two events the same event.

        The final version of an utterance must replace its partial, so the key
        includes `is_final`: a PARTIAL and a FINAL with the same id are two
        distinct states of one utterance, not duplicates of each other.
        """
        return (self.consultation_id, self.event_id, self.payload.is_final)
