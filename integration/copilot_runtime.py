"""
Push-mode runtime for the Copilot core.

The Doctor platform pushes; the existing Python pipeline waits to be polled.
This module is the inversion, and nothing more. It accepts ordered
`CopilotEvent` objects, keeps the small amount of per-consultation state needed
to feed them to the pipeline correctly, and hands structured results to a
callback.

It knows nothing about transport. No HTTP, no sockets, no broker, no database.
A FastAPI layer — when it exists — will be a thin adapter over this, and so
would a WebSocket, an in-process test, or a future event bus.

Three behaviours here are not incidental and are worth stating plainly, because
each was derived from how the existing pipeline actually behaves:

1. **Only FINAL utterances reach the clinical state.** The guard that enforces
   this lives in `ClinicalStateEngine.on_transcript_event` (line 69), not in
   `update_state_from_event`. Calling the latter directly with a PARTIAL does
   mutate state — verified — so the runtime applies the guard itself rather
   than relying on a code path it does not use.

2. **Sequence is the only ordering authority.** The clinical state accumulates,
   so processing utterance 3 before utterance 2 would build a different
   consultation. A gap means "not delivered yet", so the runtime holds the
   later event until the gap closes.

3. **Everything here is derived.** The Doctor platform owns the clinical
   record. What this runtime keeps is a projection of the event stream and can
   be rebuilt by replaying it — which is what makes persistence a later
   decision rather than a prerequisite.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field

from app.services.clinical.models import ClinicalState
from app.services.clinical.state_engine import ClinicalStateEngine
from clinical_intelligence.rule_engine.engine import ClinicalReasoningEngine
from clinical_intelligence.rule_engine.models import ClinicalReasoningResult
from integration.copilot_event import CopilotEvent
from integration.doctor_adapter import copilot_event_to_transcript_event
from integration.semantic_fingerprint import (
    changed_categories,
    fingerprint,
    semantic_view,
    state_fingerprint,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Errors
# ══════════════════════════════════════════════════════════════════════════


class CopilotRuntimeError(Exception):
    """Base for everything this runtime refuses to do."""


class UnknownSessionError(CopilotRuntimeError):
    """An event arrived for a consultation that was never opened."""


class SessionClosedError(CopilotRuntimeError):
    """An event arrived after the consultation was closed."""


class SessionExpiredError(CopilotRuntimeError):
    """
    An event arrived for a consultation that was evicted for being idle.

    Distinct from a close: nobody decided this consultation was over, the
    runtime gave up on it. A caller seeing this knows to re-open and replay
    rather than to treat the consultation as finished.
    """


class StaleEventError(CopilotRuntimeError):
    """
    An event arrived for a sequence already passed.

    Distinct from a duplicate, which is silently ignored: a *different* event
    claiming an already-consumed sequence means the stream is inconsistent, and
    quietly dropping it would hide that.
    """


# ══════════════════════════════════════════════════════════════════════════
# Output contract
# ══════════════════════════════════════════════════════════════════════════


class CopilotResult(BaseModel):
    """
    One structured result, emitted after a FINAL utterance.

    Deliberately thin: it reuses `ClinicalState` and `ClinicalReasoningResult`
    whole rather than re-projecting their contents into a parallel shape that
    would drift. Candidate diagnoses, red flags, investigations, missing
    information, risk factors and contraindications are all already fields on
    `ClinicalReasoningResult` and keep their existing types.
    """

    model_config = {"arbitrary_types_allowed": True}

    consultation_id: str
    #: The sequence of the event that produced this result. Always the event
    #: that caused the *change*, never the one that last ran reasoning.
    sequence: int
    #: The utterance that produced it, so a result can be traced to what was said.
    event_id: str
    clinical_state: ClinicalState
    reasoning: ClinicalReasoningResult
    #: Per-fact origin, as `ClinicalStateEngine` records it.
    provenance: Dict[str, Any] = Field(default_factory=dict)
    #: Digest of the clinical conclusion. Equal digests mean equal opinions.
    fingerprint: str = ""
    #: Which parts of the clinical picture moved since the last emission, so a
    #: consumer can react to "new red flag" without re-diffing the payload.
    changed: List[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when nothing clinical was derived — useful for suppressing noise."""
        return not self.reasoning.candidate_diagnoses and not self.clinical_state.symptoms


class ResultConsumer(Protocol):
    """
    The only thing the runtime knows about its consumer.

    A plain callable satisfies this, which is why the tests pass a list's
    `append` and a future FastAPI layer can pass a broadcast function.
    """

    def __call__(self, result: CopilotResult) -> None: ...


# ══════════════════════════════════════════════════════════════════════════
# Session
# ══════════════════════════════════════════════════════════════════════════


class SessionStatus(str, Enum):
    """
    The minimum lifecycle the push model needs.

    There is no PAUSED. The Doctor platform has one, but pausing there means
    the provider stops emitting — and the absence of events needs no state
    here. A paused consultation is an ACTIVE session that happens to be quiet,
    and inventing a state to represent silence would add a transition nothing
    can observe.
    """

    NEW = "NEW"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    #: Evicted for inactivity. Terminal, like CLOSED, but not a decision anyone
    #: made about the consultation.
    EXPIRED = "EXPIRED"


class CopilotSession:
    """
    One consultation's derived state.

    Isolated by consultation id: the engines, buffers and dedupe sets are all
    per-session, so a failure in one consultation cannot reach another.
    """

    def __init__(
        self,
        consultation_id: str,
        *,
        reasoning_engine: ClinicalReasoningEngine,
        now: float = 0.0,
    ):
        self.consultation_id = consultation_id
        self.status = SessionStatus.NEW
        #: Monotonic reading of the last activity, for idle eviction.
        self.last_seen = now
        #: A private state engine, not the module singleton — two consultations
        #: sharing one would share its per-session dictionaries too.
        self._state_engine = ClinicalStateEngine()
        self._reasoning_engine = reasoning_engine

        #: Next sequence expected. The Doctor platform numbers from 1.
        self._next_sequence = 1
        #: Events ahead of the gap, held until it closes.
        self._pending: Dict[int, CopilotEvent] = {}
        #: Ids of utterances already settled. Partials are not recorded here.
        self._seen: set[str] = set()
        #: Latest text seen for an utterance still in PARTIAL. Tracked so a
        #: consumer can show live text, never fed to the clinical state.
        self._partials: Dict[str, str] = {}
        self._final_count = 0

        #: Throttling state. Suppressing an emission must never discard what
        #: was computed — the next meaningful change is diffed against the
        #: latest reasoning, not against the last one that happened to be sent.
        self._last_view: Optional[Dict[str, Any]] = None
        self._last_fingerprint: Optional[str] = None
        self._latest_result: Optional[CopilotResult] = None
        self._emitted_count = 0
        self._suppressed_count = 0

        #: Reasoning is a pure function of the clinical state, so an unchanged
        #: state digest means the previous conclusion still stands and the pass
        #: can be skipped. See `clinical_state_view` for the verification.
        self._last_state_fingerprint: Optional[str] = None
        self._last_reasoning: Optional[ClinicalReasoningResult] = None
        self._reasoning_runs = 0
        self._reasoning_skipped = 0

    # ── lifecycle ─────────────────────────────────────────────────────────
    def open(self) -> None:
        if self.status is SessionStatus.CLOSED:
            raise SessionClosedError(
                f"Consultation {self.consultation_id} is closed and cannot be reopened."
            )
        if self.status is SessionStatus.EXPIRED:
            raise SessionExpiredError(
                f"Consultation {self.consultation_id} expired and cannot be reopened."
            )
        self.status = SessionStatus.ACTIVE

    def touch(self, now: float) -> None:
        """Records activity, so a busy consultation is never evicted."""
        self.last_seen = now

    def expire(self) -> None:
        """Evicted for inactivity. Terminal, and not reopenable."""
        self._pending.clear()
        self.status = SessionStatus.EXPIRED

    def close(self) -> None:
        """
        Ends the session.

        Buffered events are dropped rather than flushed: they are, by
        definition, waiting on an utterance that will now never arrive, and
        applying them out of order at the end would build a state the
        consultation never passed through.
        """
        if self._pending:
            logger.warning(
                "Closing consultation %s with %d event(s) still waiting on a gap at sequence %d",
                self.consultation_id,
                len(self._pending),
                self._next_sequence,
            )
            self._pending.clear()
        self.status = SessionStatus.CLOSED

    # ── ingestion ─────────────────────────────────────────────────────────
    def ingest(self, event: CopilotEvent) -> List[CopilotResult]:
        """
        Consumes one event and returns whatever results it unblocked.

        A list because a late arrival can close a gap and release several
        buffered events at once: receiving 2 after 1 and 3 processes both 2
        and 3, in order, and returns a result for each FINAL among them.
        """
        if self.status is SessionStatus.CLOSED:
            raise SessionClosedError(
                f"Consultation {self.consultation_id} is closed; event "
                f"{event.event_id} (sequence {event.sequence}) was refused."
            )
        if self.status is SessionStatus.EXPIRED:
            raise SessionExpiredError(
                f"Consultation {self.consultation_id} expired; event "
                f"{event.event_id} (sequence {event.sequence}) was refused."
            )
        if self.status is SessionStatus.NEW:
            self.open()

        if event.consultation_id != self.consultation_id:
            raise CopilotRuntimeError(
                f"Event for {event.consultation_id} was routed to session "
                f"{self.consultation_id}."
            )

        # Only FINAL utterances are deduplicated. A PARTIAL is an overwrite of
        # live text and the platform sends several per utterance as speech
        # settles; treating the second one as a duplicate would freeze the
        # display on the first few words.
        if event.payload.is_final and event.event_id in self._seen:
            logger.debug(
                "Ignoring repeated final event %s in %s",
                event.event_id,
                self.consultation_id,
            )
            return []

        if event.sequence < self._next_sequence:
            raise StaleEventError(
                f"Sequence {event.sequence} was already consumed in "
                f"{self.consultation_id} (expecting {self._next_sequence}). "
                "A new event claiming a past sequence means the stream is "
                "inconsistent."
            )

        if event.sequence > self._next_sequence:
            # A gap. Hold it: the clinical state accumulates, so applying this
            # before its predecessor would build a different consultation.
            self._pending[event.sequence] = event
            return []

        return self._drain(event)

    def _drain(self, event: CopilotEvent) -> List[CopilotResult]:
        """Applies `event`, then anything its arrival unblocked."""
        results: List[CopilotResult] = []
        current: Optional[CopilotEvent] = event

        while current is not None:
            result = self._apply(current)
            if result is not None:
                results.append(result)

            if not current.payload.is_final:
                # A PARTIAL and its FINAL share both id and sequence. Advancing
                # here would make the utterance's own FINAL look stale and it
                # would never be applied — the consultation would lose the
                # settled text of every utterance that was ever partial.
                break

            self._seen.add(current.event_id)
            self._next_sequence = current.sequence + 1
            current = self._pending.pop(self._next_sequence, None)

        return results

    def _apply(self, event: CopilotEvent) -> Optional[CopilotResult]:
        """
        Feeds one in-order event to the pipeline.

        PARTIAL utterances are recorded and go no further. The clinical state
        engine's own subscriber refuses them, and `update_state_from_event`
        does not — so applying that rule here is what keeps a partial's text
        from being extracted and then never retracted when the final differs.
        """
        if not event.payload.is_final:
            self._partials[event.event_id] = event.payload.text
            return None

        # The final supersedes any partial for the same utterance.
        self._partials.pop(event.event_id, None)
        self._final_count += 1

        transcript_event = copilot_event_to_transcript_event(event)
        state = self._state_engine.update_state_from_event(transcript_event)

        state_digest = state_fingerprint(state)
        if state_digest == self._last_state_fingerprint and self._last_reasoning is not None:
            # Nothing the engine reads has moved, so its answer cannot have.
            # The previous conclusion is reused rather than recomputed.
            reasoning = self._last_reasoning
            self._reasoning_skipped += 1
        else:
            # Deterministic only. `run_llm` defaults to False, so no provider is
            # contacted; enrichment is a later, separate decision.
            reasoning = self._reasoning_engine.reason(state)
            self._reasoning_runs += 1
            self._last_reasoning = reasoning
        self._last_state_fingerprint = state_digest

        view = semantic_view(state, reasoning)
        digest = fingerprint(state, reasoning)

        result = CopilotResult(
            consultation_id=self.consultation_id,
            sequence=event.sequence,
            event_id=event.event_id,
            clinical_state=state,
            reasoning=reasoning,
            provenance=dict(self._state_engine.get_provenance(self.consultation_id)),
            fingerprint=digest,
            changed=changed_categories(self._last_view, view),
        )

        # The latest computation is kept whether or not it is emitted: a
        # suppressed result is still the most current picture, and the next
        # comparison must be made against it.
        self._latest_result = result

        is_first = self._last_fingerprint is None
        unchanged = digest == self._last_fingerprint
        self._last_view = view
        self._last_fingerprint = digest

        # The first result always goes out, even a thin one. It establishes the
        # consumer's baseline; without it a consultation that opens quietly
        # would show nothing until something notable happened.
        if unchanged and not is_first:
            self._suppressed_count += 1
            logger.debug(
                "Suppressed an unchanged result for %s at sequence %d",
                self.consultation_id,
                event.sequence,
            )
            return None

        self._emitted_count += 1
        return result

    # ── introspection, for tests and debugging ────────────────────────────
    @property
    def next_expected_sequence(self) -> int:
        return self._next_sequence

    @property
    def pending_sequences(self) -> List[int]:
        return sorted(self._pending)

    @property
    def live_partials(self) -> Dict[str, str]:
        return dict(self._partials)

    @property
    def final_count(self) -> int:
        return self._final_count

    @property
    def reasoning_runs(self) -> int:
        """Reasoning passes actually executed."""
        return self._reasoning_runs

    @property
    def reasoning_skipped(self) -> int:
        """Passes avoided because the clinical state had not moved."""
        return self._reasoning_skipped

    @property
    def emitted_count(self) -> int:
        return self._emitted_count

    @property
    def suppressed_count(self) -> int:
        return self._suppressed_count

    @property
    def last_fingerprint(self) -> Optional[str]:
        return self._last_fingerprint

    @property
    def latest_result(self) -> Optional[CopilotResult]:
        """
        The most recent computation, emitted or not.

        A consumer joining late, or one that wants the current picture rather
        than the last change, reads this.
        """
        return self._latest_result

    def current_state(self) -> ClinicalState:
        return self._state_engine.get_state(self.consultation_id)


# ══════════════════════════════════════════════════════════════════════════
# Runtime
# ══════════════════════════════════════════════════════════════════════════


class CopilotRuntime:
    """
    Routes events to sessions and results to a consumer.

    One reasoning engine is shared across sessions because it is stateless per
    call — it takes a clinical state and returns a result — and constructing
    one per consultation would reload every rule file. Session state is never
    shared.
    """

    #: How long a consultation may sit idle before the runtime gives up on it.
    #: Thirty minutes is longer than any plausible gap inside a live
    #: consultation and short enough that an abandoned one does not outlive the
    #: clinic session that started it.
    DEFAULT_IDLE_TIMEOUT_SECONDS = 30 * 60

    #: How many finished consultations are remembered by id alone. A tombstone
    #: is a string and a reason, so this is a rounding error next to a live
    #: session — but it is what lets a late event be refused rather than
    #: silently starting a second consultation under the same id.
    DEFAULT_RETIRED_CAPACITY = 1024

    def __init__(
        self,
        on_result: Optional[ResultConsumer] = None,
        *,
        reasoning_engine: Optional[ClinicalReasoningEngine] = None,
        idle_timeout_seconds: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        retired_capacity: Optional[int] = None,
    ):
        self._sessions: Dict[str, CopilotSession] = {}
        self._on_result = on_result
        self._reasoning_engine = reasoning_engine or ClinicalReasoningEngine()

        # Injected so eviction is deterministic in tests. `time.monotonic`
        # rather than `time.time`: a clock adjustment must not retire a live
        # consultation, nor keep a dead one alive.
        self._clock = clock
        self._idle_timeout = (
            self.DEFAULT_IDLE_TIMEOUT_SECONDS
            if idle_timeout_seconds is None
            else idle_timeout_seconds
        )
        #: consultation id -> why it is gone. Insertion-ordered so the oldest
        #: tombstone is the one dropped when the cap is reached.
        self._retired: "OrderedDict[str, str]" = OrderedDict()
        self._retired_capacity = (
            self.DEFAULT_RETIRED_CAPACITY
            if retired_capacity is None
            else retired_capacity
        )
        self._expired_count = 0

    # ── eviction ──────────────────────────────────────────────────────────
    def _retire(self, consultation_id: str, reason: str) -> None:
        """Drops the session and remembers only that it is gone, and why."""
        self._sessions.pop(consultation_id, None)
        self._retired.pop(consultation_id, None)
        self._retired[consultation_id] = reason
        while len(self._retired) > self._retired_capacity:
            self._retired.popitem(last=False)

    def _refuse_if_retired(self, consultation_id: str) -> None:
        reason = self._retired.get(consultation_id)
        if reason == "closed":
            raise SessionClosedError(f"Consultation {consultation_id} is closed.")
        if reason == "expired":
            raise SessionExpiredError(
                f"Consultation {consultation_id} expired after "
                f"{self._idle_timeout:g}s of inactivity."
            )

    def evict_idle(self, now: Optional[float] = None) -> List[str]:
        """
        Retires consultations that have gone quiet.

        Swept lazily on each ingest rather than by a background worker: there
        is no thread to supervise, no shutdown to get right, and a runtime that
        is receiving no events has nothing to clean up anyway.
        """
        if self._idle_timeout <= 0:
            return []
        current = self._clock() if now is None else now
        stale = [
            cid
            for cid, session in self._sessions.items()
            if current - session.last_seen > self._idle_timeout
        ]
        for cid in stale:
            self._sessions[cid].expire()
            self._retire(cid, "expired")
            self._expired_count += 1
            logger.info(
                "Expired consultation %s after %gs of inactivity",
                cid,
                self._idle_timeout,
            )
        return stale

    # ── lifecycle ─────────────────────────────────────────────────────────
    def open_session(self, consultation_id: str) -> CopilotSession:
        self._refuse_if_retired(consultation_id)

        existing = self._sessions.get(consultation_id)
        if existing is not None:
            existing.touch(self._clock())
            return existing

        now = self._clock()
        session = CopilotSession(
            consultation_id, reasoning_engine=self._reasoning_engine, now=now
        )
        session.open()
        self._sessions[consultation_id] = session
        return session

    def close_session(self, consultation_id: str) -> None:
        """
        Ends the consultation and releases everything it held.

        The session object is dropped, not parked in a CLOSED state: parking it
        is what made `_sessions` grow without bound. A tombstone remains so a
        late event is refused rather than quietly opening a second consultation
        under the same id.
        """
        session = self._sessions.get(consultation_id)
        if session is None:
            self._refuse_if_retired(consultation_id)
            raise UnknownSessionError(f"No session for consultation {consultation_id}.")
        session.close()
        self._retire(consultation_id, "closed")

    def get_session(self, consultation_id: str) -> CopilotSession:
        session = self._sessions.get(consultation_id)
        if session is None:
            self._refuse_if_retired(consultation_id)
            raise UnknownSessionError(f"No session for consultation {consultation_id}.")
        return session

    @property
    def active_sessions(self) -> List[str]:
        return [
            cid
            for cid, s in self._sessions.items()
            if s.status is SessionStatus.ACTIVE
        ]

    @property
    def session_count(self) -> int:
        """Live sessions held in memory. This is the number being bounded."""
        return len(self._sessions)

    @property
    def expired_count(self) -> int:
        return self._expired_count

    @property
    def retired_ids(self) -> List[str]:
        """Consultations remembered only as tombstones, oldest first."""
        return list(self._retired)

    # ── ingestion ─────────────────────────────────────────────────────────
    def ingest(
        self, event: CopilotEvent, *, auto_open: bool = True
    ) -> List[CopilotResult]:
        """
        Routes one event to its consultation and publishes any results.

        Errors are raised, not swallowed: the caller pushed this event and is
        the only party that can decide what a rejection means. Isolation comes
        from sessions being independent, so a raise here leaves every other
        consultation untouched — which is asserted in the tests.
        """
        # Sweep before routing, so an event arriving after a long quiet period
        # does not resurrect a consultation the runtime has already given up on.
        self.evict_idle()

        session = self._sessions.get(event.consultation_id)
        if session is None:
            self._refuse_if_retired(event.consultation_id)
            if not auto_open:
                raise UnknownSessionError(
                    f"No session for consultation {event.consultation_id}."
                )
            session = self.open_session(event.consultation_id)

        session.touch(self._clock())
        results = session.ingest(event)
        for result in results:
            self._publish(result)
        return results

    def _publish(self, result: CopilotResult) -> None:
        """
        Hands a result to the consumer.

        A consumer that raises must not lose the clinical work that produced
        the result, nor stop the next one being delivered — the same posture
        the Doctor platform takes with its own listeners.
        """
        if self._on_result is None:
            return
        try:
            self._on_result(result)
        except Exception:
            logger.exception(
                "Copilot result consumer failed for consultation %s at sequence %d",
                result.consultation_id,
                result.sequence,
            )
