"""
HTTP surface for the Copilot runtime.

Routes do four things and nothing else: authenticate, deserialize, call the
runtime, and translate a domain error into a status code. Every clinical
decision stays in `CopilotRuntime`, which is why this file contains no
reference to a symptom, a diagnosis or a rule.

Ingestion routes are declared `def`, not `async def`, so FastAPI runs them in
its threadpool. The clinical pipeline is synchronous and can take a moment;
running it on the event loop would stall every other consultation. This is the
concurrency model — no workers, no queue infrastructure, just the threadpool
FastAPI already has, with per-consultation state that is never shared.
"""

from __future__ import annotations

import asyncio
import json
import queue
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, Header, HTTPException, Path, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from copilot_api.auth import Caller, CallerDep, authorize
from integration.copilot_event import CopilotEvent
from integration.copilot_insights import CopilotInsights, to_insights
from integration.intelligence.live import (
    IntelligenceSnapshot,
    LiveIntelligenceService,
)
from integration.copilot_runtime import (
    CopilotResult,
    CopilotRuntime,
    CopilotRuntimeError,
    SessionClosedError,
    SessionExpiredError,
    SessionStatus,
    StaleEventError,
    UnknownSessionError,
)
from integration.doctor_adapter import UnmappableSpeakerError

API_PREFIX = "/copilot/v1"

#: How often the streaming generator checks for a new result. Short enough to
#: be imperceptible, long enough that an idle stream costs nothing.
POLL_INTERVAL_SECONDS = 0.05

#: How long a stream may stay silent before a comment frame is sent.
KEEPALIVE_SECONDS = 15.0


# ══════════════════════════════════════════════════════════════════════════
# Result fan-out for SSE
# ══════════════════════════════════════════════════════════════════════════


class ResultBroker:
    """
    Turns the runtime's single callback into per-consultation subscriptions.

    This is the whole SSE adapter. It satisfies `ResultConsumer` — a plain
    callable taking a `CopilotResult` — so the runtime needs no knowledge that
    anything is streaming.

    Thread-safe `queue.Queue` rather than `asyncio.Queue`: results are produced
    on a threadpool thread during ingestion and consumed on the event loop by
    the streaming generator. A bounded queue drops the oldest rather than
    growing without limit if a client stops reading.
    """

    MAX_BUFFERED = 64

    def __init__(self) -> None:
        self._subscribers: Dict[str, List["queue.Queue[CopilotResult]"]] = {}

    def __call__(self, result: CopilotResult) -> None:
        for q in list(self._subscribers.get(result.consultation_id, ())):
            try:
                q.put_nowait(result)
            except queue.Full:
                try:
                    q.get_nowait()  # drop the oldest; a live view beats a stalled one
                    q.put_nowait(result)
                except queue.Empty:  # pragma: no cover - racing drain
                    pass

    def subscribe(self, consultation_id: str) -> "queue.Queue[CopilotResult]":
        q: "queue.Queue[CopilotResult]" = queue.Queue(maxsize=self.MAX_BUFFERED)
        self._subscribers.setdefault(consultation_id, []).append(q)
        return q

    def unsubscribe(self, consultation_id: str, q: "queue.Queue[CopilotResult]") -> None:
        subscribers = self._subscribers.get(consultation_id)
        if not subscribers:
            return
        if q in subscribers:
            subscribers.remove(q)
        if not subscribers:
            self._subscribers.pop(consultation_id, None)

    def subscriber_count(self, consultation_id: str) -> int:
        return len(self._subscribers.get(consultation_id, ()))


class SnapshotBroker:
    """
    The same fan-out, for intelligence snapshots.

    A separate broker rather than a generalised one: clinical results and model
    commentary travel on different schedules and must never share a backlog. A
    slow or absent intelligence layer cannot delay a clinical frame, and a
    stalled intelligence subscriber drops its own oldest, not the clinician's.
    """

    MAX_BUFFERED = 16

    def __init__(self) -> None:
        self._subscribers: Dict[str, List["queue.Queue[IntelligenceSnapshot]"]] = {}

    def __call__(self, snapshot: IntelligenceSnapshot) -> None:
        for q in list(self._subscribers.get(snapshot.consultation_id, ())):
            try:
                q.put_nowait(snapshot)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(snapshot)
                except queue.Empty:  # pragma: no cover - racing drain
                    pass

    def subscribe(self, consultation_id: str) -> "queue.Queue[IntelligenceSnapshot]":
        q: "queue.Queue[IntelligenceSnapshot]" = queue.Queue(maxsize=self.MAX_BUFFERED)
        self._subscribers.setdefault(consultation_id, []).append(q)
        return q

    def unsubscribe(
        self, consultation_id: str, q: "queue.Queue[IntelligenceSnapshot]"
    ) -> None:
        subscribers = self._subscribers.get(consultation_id)
        if not subscribers:
            return
        if q in subscribers:
            subscribers.remove(q)
        if not subscribers:
            self._subscribers.pop(consultation_id, None)

    def subscriber_count(self, consultation_id: str) -> int:
        return len(self._subscribers.get(consultation_id, ()))


# ══════════════════════════════════════════════════════════════════════════
# Request / response shapes
# ══════════════════════════════════════════════════════════════════════════


class CreateSessionRequest(BaseModel):
    consultation_id: str = Field(min_length=1)


class SessionView(BaseModel):
    """
    What the transport says about a consultation.

    Counters rather than clinical content: a caller polling this wants to know
    the session is alive and keeping up, not to re-read the record.
    """

    consultation_id: str
    status: SessionStatus
    next_expected_sequence: int
    pending_sequences: List[int]
    final_count: int
    reasoning_runs: int
    reasoning_skipped: int
    emitted_count: int
    suppressed_count: int
    live_partials: Dict[str, str]
    latest_result: Optional[CopilotResult] = None


class IngestResponse(BaseModel):
    """
    The outcome of one pushed event.

    `results` is a list because a late arrival can close a sequence gap and
    release several buffered events at once. It is empty for a PARTIAL, for a
    duplicate, for an event still waiting on a gap, and for a settled utterance
    whose conclusion did not change — `accepted` and `buffered` say which.
    """

    accepted: bool
    buffered: bool
    results: List[CopilotResult]
    #: The clinician-facing projection of the last result in `results`. Present
    #: only when something was emitted, so a live panel can render straight
    #: from the ingest response without a follow-up read.
    insights: Optional[CopilotInsights] = None


def _error(status_code: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, **extra}},
    )


# ══════════════════════════════════════════════════════════════════════════
# Application
# ══════════════════════════════════════════════════════════════════════════


def create_app(
    runtime: Optional[CopilotRuntime] = None,
    *,
    broker: Optional[ResultBroker] = None,
    intelligence: Optional[LiveIntelligenceService] = None,
    snapshot_broker: Optional[SnapshotBroker] = None,
    lifespan: Optional[Any] = None,
) -> FastAPI:
    """
    Builds the transport.

    The runtime is injectable so a test can supply one with a fake clock — the
    same seam session eviction already uses — without the API owning any
    lifecycle policy of its own.
    """
    result_broker = broker or ResultBroker()
    copilot = runtime or CopilotRuntime(result_broker)
    # Optional by design. With no service the API is exactly what it was, and
    # every existing test runs in that configuration.
    snapshots = snapshot_broker or SnapshotBroker()

    app = FastAPI(
        title="ENERVARA Copilot",
        description="Transport boundary around the Copilot clinical runtime.",
        version="1.0.0",
        # Supplied by the composition root, which owns the worker lifecycle.
        # None keeps the app usable in a test with no background thread.
        lifespan=lifespan,
    )
    app.state.runtime = copilot
    app.state.broker = result_broker
    app.state.intelligence = intelligence
    app.state.snapshots = snapshots

    # ── domain errors become HTTP, explicitly ─────────────────────────────
    # Each mapping is a statement about what the failure means, not a default.
    # Nothing here leaks a traceback.

    @app.exception_handler(UnknownSessionError)
    def _unknown(_r: Request, exc: UnknownSessionError) -> JSONResponse:
        return _error(status.HTTP_404_NOT_FOUND, "SESSION_NOT_FOUND", str(exc))

    @app.exception_handler(SessionClosedError)
    def _closed(_r: Request, exc: SessionClosedError) -> JSONResponse:
        # 409: the consultation exists in the caller's world, but its state
        # forbids the action. Retrying unchanged will not help.
        return _error(status.HTTP_409_CONFLICT, "SESSION_CLOSED", str(exc))

    @app.exception_handler(SessionExpiredError)
    def _expired(_r: Request, exc: SessionExpiredError) -> JSONResponse:
        # 410: it was here and is deliberately gone. Distinct from 409 because
        # the caller's correct response differs — reopen and replay, rather
        # than accept that the consultation is over.
        return _error(status.HTTP_410_GONE, "SESSION_EXPIRED", str(exc))

    @app.exception_handler(StaleEventError)
    def _stale(_r: Request, exc: StaleEventError) -> JSONResponse:
        # 409: the event contradicts a sequence already consumed. The stream is
        # inconsistent; a different event, not a retry, is what is needed.
        return _error(status.HTTP_409_CONFLICT, "STALE_EVENT", str(exc))

    @app.exception_handler(UnmappableSpeakerError)
    def _speaker(_r: Request, exc: UnmappableSpeakerError) -> JSONResponse:
        # 422: the payload is well-formed but cannot be acted on. A raw
        # diarization label silently produces an empty clinical state, so it is
        # refused here exactly as the domain refuses it.
        return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, "INVALID_SPEAKER", str(exc))

    @app.exception_handler(CopilotRuntimeError)
    def _runtime(_r: Request, exc: CopilotRuntimeError) -> JSONResponse:
        # The base class, for anything the runtime refuses that has no more
        # specific mapping. Still a client error, never a 500.
        return _error(status.HTTP_409_CONFLICT, "RUNTIME_REFUSED", str(exc))

    @app.exception_handler(ValidationError)
    def _validation(_r: Request, exc: ValidationError) -> JSONResponse:
        return _error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "VALIDATION_ERROR",
            "The event did not match the CopilotEvent contract.",
            details=json.loads(exc.json()),
        )

    # ── routes ────────────────────────────────────────────────────────────

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """
        Liveness and readiness in one.

        `status` reflects the clinical core only. A degraded or absent
        intelligence layer is reported but never makes the service unhealthy —
        an orchestrator must not restart a container that is serving clinical
        output correctly because a model is down.
        """
        body: Dict[str, Any] = {
            "status": "healthy",
            "sessions": copilot.session_count,
            "expired_sessions": copilot.expired_count,
        }

        if intelligence is None:
            body["intelligence"] = {"enabled": False}
            return body

        body["intelligence"] = {
            "enabled": True,
            "worker_running": intelligence.is_running,
            "provider": intelligence.provider_name,
            "submitted": intelligence.submitted,
            "completed": intelligence.completed,
            "coalesced": intelligence.coalesced,
            "stale_dropped": intelligence.stale_dropped,
            "pending": intelligence.pending_count,
            "tracked_consultations": intelligence.snapshot_count,
        }
        return body

    @app.post(f"{API_PREFIX}/sessions", status_code=status.HTTP_201_CREATED)
    def create_session(
        body: CreateSessionRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        # The consultation id is in the body here, so the same authorization
        # rule is applied by hand rather than through the path dependency.
        authorize(body.consultation_id, authorization)
        session = copilot.open_session(body.consultation_id)
        return {
            "consultation_id": session.consultation_id,
            "status": session.status,
        }

    @app.post(f"{API_PREFIX}/sessions/{{consultation_id}}/events")
    def ingest_event(
        consultation_id: str = Path(...),
        event: CopilotEvent = Body(...),
        caller: Caller = CallerDep,
    ) -> IngestResponse:
        if event.consultation_id != consultation_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "CONSULTATION_MISMATCH",
                    "message": (
                        f"The event names consultation {event.consultation_id}, "
                        f"but was posted to {consultation_id}."
                    ),
                },
            )

        # A first event opens the consultation implicitly, so there may be no
        # session to read yet. Asking for one here would 404 the very case the
        # implicit open exists to serve.
        try:
            before: Optional[int] = copilot.get_session(consultation_id).next_expected_sequence
        except UnknownSessionError:
            before = None

        results = copilot.ingest(event)
        after = copilot.get_session(consultation_id).next_expected_sequence

        # Nothing emitted and the sequence did not advance: the event is held
        # behind a gap. Saying so beats an empty 200 the caller cannot read.
        buffered = (
            not results
            and before is not None
            and after == before
            and event.payload.is_final
        )
        # Handed off after the clinical answer is built, never before. The
        # call returns immediately; the model's reply arrives later on its own
        # channel. Only meaningful changes are submitted, because the runtime
        # has already suppressed everything else.
        if intelligence is not None and results:
            intelligence.submit(results[-1])

        return IngestResponse(
            accepted=True,
            buffered=buffered,
            results=results,
            insights=to_insights(results[-1]) if results else None,
        )

    @app.get(f"{API_PREFIX}/sessions/{{consultation_id}}")
    def get_session(
        consultation_id: str = Path(...),
        caller: Caller = CallerDep,
    ) -> SessionView:
        session = copilot.get_session(consultation_id)
        return SessionView(
            consultation_id=session.consultation_id,
            status=session.status,
            next_expected_sequence=session.next_expected_sequence,
            pending_sequences=session.pending_sequences,
            final_count=session.final_count,
            reasoning_runs=session.reasoning_runs,
            reasoning_skipped=session.reasoning_skipped,
            emitted_count=session.emitted_count,
            suppressed_count=session.suppressed_count,
            live_partials=session.live_partials,
            latest_result=session.latest_result,
        )

    @app.get(f"{API_PREFIX}/sessions/{{consultation_id}}/insights")
    def get_insights(
        consultation_id: str = Path(...),
        caller: Caller = CallerDep,
    ) -> CopilotInsights:
        """
        The six clinician-facing surfaces for the current state of the
        consultation.

        Served from the latest result the runtime holds — emitted or
        suppressed — because a panel opened mid-consultation wants the current
        picture, not the last thing that changed. Nothing is recomputed.
        """
        session = copilot.get_session(consultation_id)
        latest = session.latest_result
        if latest is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NO_RESULT_YET",
                    "message": (
                        "This consultation has produced no settled utterance yet, "
                        "so there is nothing to show."
                    ),
                },
            )
        return to_insights(latest)

    @app.get(f"{API_PREFIX}/sessions/{{consultation_id}}/intelligence")
    def get_intelligence(
        consultation_id: str = Path(...),
        caller: Caller = CallerDep,
    ) -> IntelligenceSnapshot:
        """
        The latest model commentary, for a client that has just connected.

        Separate from `/insights` on purpose: that endpoint is the
        deterministic clinical picture and must never vary with whether a model
        answered. A client rendering only the clinical panel never calls this.

        404 when nothing has been produced — which is also the answer when no
        intelligence layer is configured at all, so a client needs no special
        case for a deployment without one.
        """
        copilot.get_session(consultation_id)  # 404/410 before anything else

        snapshot = (
            intelligence.latest(consultation_id) if intelligence is not None else None
        )
        if snapshot is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NO_INTELLIGENCE_YET",
                    "message": (
                        "No intelligence has been produced for this consultation."
                    ),
                },
            )
        return snapshot

    @app.post(f"{API_PREFIX}/sessions/{{consultation_id}}/close")
    def close_session(
        consultation_id: str = Path(...),
        caller: Caller = CallerDep,
    ) -> Dict[str, Any]:
        copilot.close_session(consultation_id)
        if intelligence is not None:
            intelligence.forget(consultation_id)
        return {"consultation_id": consultation_id, "status": SessionStatus.CLOSED}

    @app.get(f"{API_PREFIX}/sessions/{{consultation_id}}/stream")
    def stream_results(
        consultation_id: str = Path(...),
        caller: Caller = CallerDep,
    ) -> StreamingResponse:
        """
        Server-sent events carrying each emitted result.

        Only a transport: the same results are available from the session
        endpoint, so a client that cannot hold a stream open loses nothing.
        Throttling has already happened upstream — what arrives here is only
        the changes.
        """
        copilot.get_session(consultation_id)  # 404/410 before the stream opens
        subscription = result_broker.subscribe(consultation_id)
        insight_subscription = snapshots.subscribe(consultation_id)
        # What a reconnecting client missed. Sent once, before live frames, so
        # a reconnect does not leave the panel blank until the next utterance.
        caught_up = (
            intelligence.latest(consultation_id) if intelligence is not None else None
        )

        async def events() -> Any:
            # Polled with `get_nowait` rather than a blocking get on a
            # threadpool thread. A blocking get is not cancellable: when the
            # client disconnects, the thread stays parked on the queue and the
            # generator cannot finish, so the subscription is never released.
            # Yielding to the loop instead keeps one coroutine per subscriber
            # and lets a disconnect unwind immediately.
            idle = 0.0
            try:
                yield ": connected\n\n"
                if caught_up is not None:
                    yield (
                        "event: insight\n"
                        f"data: {caught_up.model_dump_json()}\n\n"
                    )
                while True:
                    # Clinical frames first, always. If both channels have
                    # something ready the deterministic one goes out ahead of
                    # the commentary about it.
                    try:
                        result = subscription.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        idle = 0.0
                        yield f"event: result\ndata: {result.model_dump_json()}\n\n"
                        continue

                    try:
                        snapshot = insight_subscription.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        idle = 0.0
                        yield f"event: insight\ndata: {snapshot.model_dump_json()}\n\n"
                        continue

                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    idle += POLL_INTERVAL_SECONDS
                    if idle >= KEEPALIVE_SECONDS:
                        # A comment frame, so a proxy does not close an idle
                        # stream during a quiet stretch.
                        idle = 0.0
                        yield ": keep-alive\n\n"
            finally:
                result_broker.unsubscribe(consultation_id, subscription)
                snapshots.unsubscribe(consultation_id, insight_subscription)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
