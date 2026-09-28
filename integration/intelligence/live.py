"""
Intelligence during a live consultation.

The governing constraint is that a clinician must never wait for a model. The
deterministic result is computed, throttled and emitted on the request path in
milliseconds; anything a model has to say arrives afterwards, on its own
channel, clearly attached to the moment it was made about. If the model is
slow, absent or broken, the consultation is unchanged — that is the difference
between an assistant and a dependency.

Three behaviours follow from that, and each is a test:

* **Work is coalesced, not queued.** A consultation that produces five results
  while one graph run is in flight does not accumulate five pending runs. The
  newest wins and the rest are dropped, because nobody wants commentary on a
  clinical picture that has already moved on.

* **Late output is marked, not hidden.** A snapshot carries the fingerprint it
  was made against. If the consultation has moved since, `stale` is true and
  the consumer decides — showing it silently as current would be the worst of
  the options.

* **Degradation is a status, not an absence.** `UNAVAILABLE` and `FAILED` both
  produce a snapshot with no proposals and `degraded` set, so a client can tell
  "the model said nothing useful" from "we have not asked yet".

One worker thread, one pending slot per consultation. No broker, no queue
service, no second process.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from integration.copilot_runtime import CopilotResult
from integration.intelligence.context import ContextFact, CopilotContext, build_context
from integration.intelligence.contract import IntelligenceStatus, Proposal
from integration.intelligence.gateway import IntelligenceGateway, IntelligenceOutcome

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# What a clinician receives
# ══════════════════════════════════════════════════════════════════════════


class ResolvedEvidence(BaseModel):
    """
    A citation, resolved back to the consultation.

    A proposal cites `symptom:symptom_chest_pain`, which means nothing on a
    screen. Resolving it against the context recovers the label, the utterance
    and the patient's own words, so a suggestion can be checked against what
    was actually said rather than taken on trust.
    """

    ref: str
    kind: str
    label: str
    utterance_id: Optional[str] = None
    quote: Optional[str] = None


class ProposalView(BaseModel):
    """One proposal with its citations resolved."""

    proposal_id: str
    kind: str
    content: str
    about_ref: Optional[str] = None
    evidence: List[ResolvedEvidence] = Field(default_factory=list)
    #: Refs that could not be resolved. Empty in practice — the validator
    #: refuses unknown references — but reported rather than assumed.
    unresolved_refs: List[str] = Field(default_factory=list)


class IntelligenceSnapshot(BaseModel):
    """
    Everything the intelligence layer has to say about one moment.

    Deliberately self-describing: a client that reconnects mid-consultation can
    render this without knowing what it missed, and can tell from `fingerprint`
    and `stale` whether it still describes the picture on screen.
    """

    consultation_id: str
    #: The utterance whose result this was computed from.
    sequence: int
    event_id: str
    #: The clinical picture it was made against.
    fingerprint: str

    status: IntelligenceStatus
    provider: str
    provider_version: str
    message: Optional[str] = None

    proposals: List[ProposalView] = Field(default_factory=list)
    #: Refused by the validation boundary. A count rather than the payloads:
    #: rejections are an operator concern, and the gateway logs the detail.
    rejected_count: int = 0

    #: The consultation moved on before this finished.
    stale: bool = False
    #: No proposals because the provider could not answer.
    degraded: bool = False

    elapsed_ms: float = 0.0
    over_budget: bool = False
    generated_at: float = 0.0


SnapshotConsumer = Callable[[IntelligenceSnapshot], None]


def _resolve(proposal: Proposal, context: CopilotContext) -> ProposalView:
    index: Dict[str, ContextFact] = {}
    for fact in context.observed.symptoms:
        index[fact.ref] = fact
    for group in (
        context.deterministic.differential,
        context.deterministic.red_flags,
        context.deterministic.investigations,
        context.deterministic.missing_information,
        context.deterministic.contraindications,
        context.deterministic.risk_factors,
    ):
        for fact in group:
            index[fact.ref] = fact

    evidence: List[ResolvedEvidence] = []
    unresolved: List[str] = []
    for ref in proposal.grounded_in:
        cited = index.get(ref)
        if cited is None:
            unresolved.append(ref)
            continue
        evidence.append(
            ResolvedEvidence(
                ref=cited.ref,
                kind=cited.kind,
                label=cited.label,
                utterance_id=cited.utterance_id,
                quote=cited.quote,
            )
        )

    return ProposalView(
        proposal_id=proposal.proposal_id,
        kind=proposal.kind.value,
        content=proposal.content,
        about_ref=proposal.about_ref,
        evidence=evidence,
        unresolved_refs=unresolved,
    )


def build_snapshot(
    outcome: IntelligenceOutcome, context: CopilotContext, *, stale: bool, now: float
) -> IntelligenceSnapshot:
    """Projects a gateway outcome into the clinician-facing snapshot."""
    return IntelligenceSnapshot(
        consultation_id=outcome.consultation_id,
        sequence=outcome.sequence,
        event_id=context.event_id,
        fingerprint=outcome.fingerprint,
        status=outcome.status,
        provider=outcome.provider,
        provider_version=outcome.provider_version,
        message=outcome.message,
        proposals=[_resolve(p, context) for p in outcome.proposals],
        rejected_count=len(outcome.rejected),
        stale=stale,
        degraded=outcome.status
        in (IntelligenceStatus.UNAVAILABLE, IntelligenceStatus.FAILED)
        or not outcome.proposals,
        elapsed_ms=outcome.elapsed_ms,
        over_budget=outcome.over_budget,
        generated_at=now,
    )


# ══════════════════════════════════════════════════════════════════════════
# The service
# ══════════════════════════════════════════════════════════════════════════


class LiveIntelligenceService:
    """
    Runs the intelligence layer beside a live consultation, never in front of it.

    `submit` returns immediately. One worker thread drains a single pending slot
    per consultation, so a burst of clinical results costs one graph run rather
    than a backlog.

    Not started automatically: a caller that never starts it has a fully
    working clinical system with no intelligence at all, which is the correct
    default and the state every existing test runs in.
    """

    #: Latest snapshot per consultation, bounded like the runtime's sessions.
    DEFAULT_CAPACITY = 256

    def __init__(
        self,
        gateway: IntelligenceGateway,
        on_snapshot: Optional[SnapshotConsumer] = None,
        *,
        capacity: Optional[int] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._gateway = gateway
        self._on_snapshot = on_snapshot
        self._capacity = capacity or self.DEFAULT_CAPACITY
        self._clock = clock

        self._pending: Dict[str, CopilotResult] = {}
        #: The most recent fingerprint submitted, for staleness.
        self._current: Dict[str, str] = {}
        self._snapshots: "OrderedDict[str, IntelligenceSnapshot]" = OrderedDict()

        self._lock = threading.Lock()
        self._work = threading.Condition(self._lock)
        self._idle = threading.Condition(self._lock)
        self._running = False
        self._in_flight = 0
        self._worker: Optional[threading.Thread] = None

        self.submitted = 0
        self.completed = 0
        self.coalesced = 0
        self.stale_dropped = 0

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        self._worker = threading.Thread(
            target=self._run, name="copilot-intelligence", daemon=True
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._work:
            self._running = False
            self._work.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None

    def __enter__(self) -> "LiveIntelligenceService":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._running and self._worker is not None and self._worker.is_alive()

    @property
    def provider_name(self) -> str:
        return self._gateway.provider_name

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending) + self._in_flight

    @property
    def snapshot_count(self) -> int:
        with self._lock:
            return len(self._snapshots)

    # ── submission ────────────────────────────────────────────────────────
    def submit(self, result: CopilotResult) -> None:
        """
        Queues a consultation for intelligence. Returns immediately.

        Replacing a pending result rather than appending is the whole point:
        the clinical picture the model should comment on is the current one.
        """
        with self._work:
            if result.consultation_id in self._pending:
                self.coalesced += 1
            self._pending[result.consultation_id] = result
            self._current[result.consultation_id] = result.fingerprint
            self.submitted += 1
            self._work.notify()

    def latest(self, consultation_id: str) -> Optional[IntelligenceSnapshot]:
        """The most recent snapshot, for a client that connects or reconnects."""
        with self._lock:
            return self._snapshots.get(consultation_id)

    def forget(self, consultation_id: str) -> None:
        """Releases everything held for a consultation that has ended."""
        with self._lock:
            self._pending.pop(consultation_id, None)
            self._current.pop(consultation_id, None)
            self._snapshots.pop(consultation_id, None)

    def wait_until_idle(self, timeout: float = 10.0) -> bool:
        """
        Blocks until nothing is pending or running.

        For tests and for a graceful shutdown. Returns False on timeout rather
        than raising, so a caller can decide whether that matters.
        """
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._pending or self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(timeout=remaining)
        return True

    # ── worker ────────────────────────────────────────────────────────────
    def _run(self) -> None:
        while True:
            with self._work:
                while self._running and not self._pending:
                    self._work.wait(timeout=0.25)
                if not self._running and not self._pending:
                    return
                consultation_id, result = self._pending.popitem()
                self._in_flight += 1

            try:
                self._process(consultation_id, result)
            except Exception:  # noqa: BLE001 - the worker must outlive any one failure
                logger.exception(
                    "Intelligence worker failed for consultation %s", consultation_id
                )
            finally:
                with self._idle:
                    self._in_flight -= 1
                    self.completed += 1
                    self._idle.notify_all()

    def _process(self, consultation_id: str, result: CopilotResult) -> None:
        outcome = self._gateway.consult(result)
        context = build_context(result)

        with self._lock:
            current = self._current.get(consultation_id)
        stale = current is not None and current != outcome.fingerprint
        if stale:
            self.stale_dropped += 1

        snapshot = build_snapshot(
            outcome, context, stale=stale, now=self._clock()
        )

        with self._lock:
            held = self._snapshots.get(consultation_id)
            # A stale snapshot never displaces a fresher one, but it is kept
            # when there is nothing else: something attributed to an earlier
            # moment beats an empty panel.
            if held is None or not stale:
                self._snapshots.pop(consultation_id, None)
                self._snapshots[consultation_id] = snapshot
                while len(self._snapshots) > self._capacity:
                    self._snapshots.popitem(last=False)

        if self._on_snapshot is not None:
            try:
                self._on_snapshot(snapshot)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Snapshot consumer failed for consultation %s", consultation_id
                )
