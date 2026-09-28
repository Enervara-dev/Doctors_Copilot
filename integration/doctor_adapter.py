"""
Doctor `TranscriptUtterance` → `CopilotEvent` → `TranscriptEvent`.

The whole reason this module exists is one verified failure: the clinical
pipeline decides whether an utterance carries patient-reported facts by
comparing `speaker_id` against the lowercase literals `"patient"`,
`"attender"` and `"doctor"` — in `state_engine.py:126` and eight other sites
across seven modules. Hand it a diarization label like `SPEAKER_00` and it
extracts nothing, raises nothing, and logs nothing. The clinical state simply
stays empty, which reads like "the consultation contained no findings".

The fix belongs here, at the boundary, not in the reasoning engine. Teaching
seven extraction modules about every upstream labelling convention would spread
one integration concern across the clinical core. Normalising once, on the way
in, keeps the core speaking a single vocabulary — and makes an unmappable
speaker loud instead of silent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, List, Mapping, Optional

from app.services.providers.models import TranscriptEvent
from integration.copilot_event import (
    CopilotEvent,
    EventType,
    Provenance,
    SpeakerRole,
    UtterancePayload,
)


class UnmappableSpeakerError(ValueError):
    """
    Raised when a speaker cannot be resolved to a canonical role.

    Deliberately an error rather than a fallback to UNKNOWN. An unrecognised
    speaker means every patient-reported fact in that utterance is dropped, and
    a silent drop is the exact failure this module exists to prevent. The caller
    can still choose to map it to UNKNOWN explicitly.
    """


#: Diarization labels the Doctor platform is not supposed to emit, but which
#: appear in raw pyannote output. Listed so the error message can say precisely
#: what went wrong rather than "unknown speaker".
_DIARIZATION_LABEL_PREFIXES = ("SPEAKER_", "speaker_", "SPK_", "spk_")


def normalize_speaker(value: Optional[str], *, strict: bool = True) -> SpeakerRole:
    """
    Resolves any upstream speaker representation to a canonical role.

    Accepts the Doctor platform's uppercase `SpeakerRole` ("PATIENT"), the
    already-canonical lowercase form, and common spellings ("caregiver" for
    attender). Rejects raw diarization labels, because mapping `SPEAKER_00` to
    a clinical role is a decision only the diarization layer can make — it is
    not a formatting difference, and guessing would attribute a patient's
    symptoms to whichever voice happened to speak first.
    """
    if value is None:
        if strict:
            raise UnmappableSpeakerError("No speaker was supplied for this utterance.")
        return SpeakerRole.UNKNOWN

    candidate = value.strip()
    if not candidate:
        if strict:
            raise UnmappableSpeakerError("The speaker was empty for this utterance.")
        return SpeakerRole.UNKNOWN

    if candidate.startswith(_DIARIZATION_LABEL_PREFIXES):
        if strict:
            raise UnmappableSpeakerError(
                f"{candidate!r} is a raw diarization label, not a clinical role. "
                "Resolve it to doctor/patient/attender before it reaches the "
                "clinical pipeline: an unresolved label silently produces an "
                "empty clinical state."
            )
        return SpeakerRole.UNKNOWN

    lowered = candidate.lower()
    aliases = {
        "doctor": SpeakerRole.DOCTOR,
        "clinician": SpeakerRole.DOCTOR,
        "physician": SpeakerRole.DOCTOR,
        "patient": SpeakerRole.PATIENT,
        "attender": SpeakerRole.ATTENDER,
        "caregiver": SpeakerRole.ATTENDER,
        "carer": SpeakerRole.ATTENDER,
        "unknown": SpeakerRole.UNKNOWN,
    }
    if lowered in aliases:
        return aliases[lowered]

    if strict:
        raise UnmappableSpeakerError(
            f"{candidate!r} does not map to a canonical role "
            f"({', '.join(r.value for r in SpeakerRole)})."
        )
    return SpeakerRole.UNKNOWN


def _iso_to_epoch(value: Optional[str]) -> Optional[float]:
    """ISO-8601 (the Doctor contract) → epoch seconds (what Python expects)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def utterance_to_copilot_event(
    utterance: Mapping[str, Any],
    *,
    consultation_id: str,
    strict_speaker: bool = True,
    offset_seconds: Optional[float] = None,
    duration_seconds: Optional[float] = None,
) -> CopilotEvent:
    """
    Maps one Doctor `TranscriptUtterance` onto the normalised envelope.

    `utterance` is taken as a mapping rather than a typed object because it
    arrives as JSON over the wire; validating it here is the point.
    """
    status = utterance.get("status")
    if status not in ("PARTIAL", "FINAL"):
        raise ValueError(
            f"status must be PARTIAL or FINAL, got {status!r} — "
            "the Doctor contract defines no third state."
        )

    provenance_raw = utterance.get("provenance", "SERVICE")
    provenance = (
        Provenance(provenance_raw)
        if provenance_raw in (p.value for p in Provenance)
        else Provenance.SERVICE
    )

    return CopilotEvent(
        event_id=str(utterance["id"]),
        consultation_id=consultation_id,
        sequence=int(utterance["sequence"]),
        event_type=EventType.UTTERANCE,
        provenance=provenance,
        payload=UtterancePayload(
            text=utterance.get("text", ""),
            speaker_role=normalize_speaker(
                utterance.get("speaker"), strict=strict_speaker
            ),
            # The Doctor platform resolves a human-readable label; keep it, it is
            # the only place "Attender (son)" survives.
            speaker_label=utterance.get("speakerLabel") or "",
            is_final=status == "FINAL",
            confidence=utterance.get("confidence"),
            started_at=utterance.get("startedAt"),
            finalized_at=utterance.get("finalizedAt"),
            offset_seconds=offset_seconds,
            duration_seconds=duration_seconds,
        ),
    )


def copilot_event_to_transcript_event(event: CopilotEvent) -> TranscriptEvent:
    """
    Maps the envelope onto the model the existing pipeline consumes.

    Two fields carry the same text because the pipeline is inconsistent about
    which it reads: `state_engine` uses `transcript`, other call sites use
    `text`. Populating both is the honest thing to do until that is unified —
    populating one would make behaviour depend on which code path ran.
    """
    payload = event.payload
    started = _iso_to_epoch(payload.started_at)

    return TranscriptEvent(
        session_id=event.consultation_id,
        # `role` is carried for completeness. Note that the clinical pipeline
        # branches on `speaker_id`, NOT on this field.
        role=payload.speaker_role.value,
        sequence_number=event.sequence,
        timestamp=started if started is not None else float(event.sequence),
        transcript=payload.text,
        text=payload.text,
        is_partial=not payload.is_final,
        is_final=payload.is_final,
        confidence=payload.confidence,
        provider=event.source,
        event_id=event.event_id,
        # The field the clinical pipeline actually dispatches on.
        speaker_id=payload.speaker_role.value,
        speaker_label=payload.speaker_label,
        start_time=payload.offset_seconds,
        end_time=(
            payload.offset_seconds + payload.duration_seconds
            if payload.offset_seconds is not None and payload.duration_seconds is not None
            else None
        ),
        version=1,
        updated_at=_iso_to_epoch(payload.finalized_at),
        speaker_confidence=None,
        speaker_method="doctor-platform-resolved",
    )


def utterances_to_transcript_events(
    utterances: Iterable[Mapping[str, Any]],
    *,
    consultation_id: str,
    strict_speaker: bool = True,
) -> List[TranscriptEvent]:
    """
    Maps a whole consultation, in sequence order.

    Sorting by `sequence` is not defensive tidiness: the Doctor contract says
    the client sorts by it and never by arrival time, so a replay that honoured
    arrival order could feed the pipeline a different consultation from the one
    that happened.
    """
    ordered = sorted(utterances, key=lambda u: int(u["sequence"]))
    events: List[TranscriptEvent] = []
    seen: set[tuple[str, str, bool]] = set()

    for utterance in ordered:
        event = utterance_to_copilot_event(
            utterance,
            consultation_id=consultation_id,
            strict_speaker=strict_speaker,
        )
        if event.dedupe_key in seen:
            continue
        seen.add(event.dedupe_key)
        events.append(copilot_event_to_transcript_event(event))

    return events
