from __future__ import annotations
import inspect
import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Type

from clinical_intelligence.workflow.protocols import IEventBus

logger = logging.getLogger(__name__)


class EventBus(IEventBus):
    """Central pub/sub dispatcher routing strongly-typed events to synchronous and asynchronous handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[Type[Any], List[Callable[[Any], Any]]] = {}
        self._history: Dict[str, List[Any]] = {}

    def subscribe(self, event_type: Type[Any], handler: Callable[[Any], Any]) -> None:
        """Register a callback handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
        logger.debug(f"EventBus: Subscribed handler {handler.__name__} to {event_type.__name__}")

    def unsubscribe(self, event_type: Type[Any], handler: Callable[[Any], Any]) -> None:
        """Remove a registered callback handler."""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.debug(f"EventBus: Unsubscribed handler {handler.__name__} from {event_type.__name__}")

    def publish(self, event: Any) -> None:
        """Publish event to all registered handlers and append to session history."""
        session_id = getattr(event, "session_id", "global")
        if session_id not in self._history:
            self._history[session_id] = []
        self._history[session_id].append(event)
        
        # Automatically record event on the TimelineEngine
        try:
            from clinical_intelligence.core.container import container
            from clinical_intelligence.workflow.protocols import ITimelineEngine, ISessionRepository
            te = container.resolve(ITimelineEngine)
            repo = container.resolve(ISessionRepository)
            session = repo.load(session_id)
            stage = session.consultation_stage if session else "Unknown"
            
            # Construct and append entry snapshot to timeline list
            actor = getattr(event, "actor", "clinician" if "Question" in type(event).__name__ else "system")
            entry = {
                "timestamp": event.timestamp if hasattr(event, "timestamp") else time.time(),
                "session_id": session_id,
                "stage": stage,
                "actor": actor,
                "event_type": type(event).__name__,
                "request_id": getattr(event, "request_id", None),
                "version_snapshot": {
                    "rule_engine": session.rule_engine_version if session else "1.0.0",
                    "workflow": session.workflow_version if session else "1.0.0",
                },
                "payload_summary": str(event),
            }
            if session:
                session.timeline.append(entry)
                repo.save(session)
        except Exception:
            pass

        event_type = type(event)
        handlers = self._handlers.get(event_type, [])
        
        logger.debug(f"EventBus: Publishing event {event_type.__name__} for session {session_id} to {len(handlers)} handlers")

        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"EventBus: Error executing handler {handler.__name__} on event {event_type.__name__}: {e}")

    def replay(self, session_id: str) -> List[Any]:
        """Retrieve and replay all chronological events associated with a session."""
        events = self._history.get(session_id, [])
        logger.info(f"EventBus: Replaying {len(events)} events for session {session_id}")
        return list(events)
