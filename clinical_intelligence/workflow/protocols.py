from __future__ import annotations

"""
protocols.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IEventBus(ABC):
    """Abstract interface for event publication, subscription, and replay."""

    @abstractmethod
    def publish(self, event: Any) -> None:
        """Publish an event to all registered handlers (sync and async)."""
        pass

    @abstractmethod
    def subscribe(self, event_type: type, handler: Any) -> None:
        """Register a handler callback for a specific event type."""
        pass

    @abstractmethod
    def unsubscribe(self, event_type: type, handler: Any) -> None:
        """Unregister a handler callback."""
        pass

    @abstractmethod
    def replay(self, session_id: str) -> List[Any]:
        """Retrieve and replay all chronological events associated with a session."""
        pass


class ISessionManager(ABC):
    """Abstract coordinator interface managing session life cycle checkpoints."""

    @abstractmethod
    def create_session(self, session_id: str, patient_id: str, clinician_id: str) -> Any:
        """Initialize and persist a new consultation session context."""
        pass

    @abstractmethod
    def load_session(self, session_id: str) -> Optional[Any]:
        """Fetch and lock a session state from repository storage."""
        pass

    @abstractmethod
    def save_session(self, session: Any) -> None:
        """Persist session updates to storage and release locks."""
        pass

    @abstractmethod
    def archive_session(self, session_id: str) -> None:
        """Transition session state to completed/archived storage."""
        pass

    @abstractmethod
    def destroy_session(self, session_id: str) -> None:
        """Completely purge session from active and persistent database."""
        pass

    @abstractmethod
    def get_lock(self, session_id: str) -> Any:
        """Return the asyncio.Lock corresponding to a unique session context."""
        pass


class IWorkflowEngine(ABC):
    """Abstract logic tracker for progression, stages, and action items."""

    @abstractmethod
    def determine_stage(self, session: Any) -> str:
        """Determine and validate the stage of clinical consultation."""
        pass

    @abstractmethod
    def determine_next_action(self, session: Any) -> Dict[str, Any]:
        """Estimate completion percentage and next clinical workflows steps."""
        pass


class IQuestionController(ABC):
    """Abstract filter scheduling adaptive questions."""

    @abstractmethod
    def get_next_questions(self, session: Any, limit: int) -> List[Any]:
        """Prioritize, deduplicate, and schedule next questions to present."""
        pass


class IReasoningCoordinator(ABC):
    """Abstract orchestrator routing data across reasoning segments."""

    @abstractmethod
    def coordinate_reasoning(self, session: Any, run_llm: bool, debug: bool) -> Any:
        """Coordinate execution sequentially across engines using increment trackers."""
        pass


class ITimelineEngine(ABC):
    """Abstract recorder for session progression histories."""

    @abstractmethod
    def record_event(self, session_id: str, stage: str, event: Any) -> None:
        """Emit chronological event data to session timeline log."""
        pass

    @abstractmethod
    def get_timeline(self, session_id: str) -> List[Any]:
        """Load list of chronological event details of a session."""
        pass

    @abstractmethod
    def take_snapshot(self, session: Any, stage_name: str) -> None:
        """Persist state snapshot of session at a major stage boundary."""
        pass


class ISessionRepository(ABC):
    """Abstract storage interface mapping sessions persistence."""

    @abstractmethod
    def save(self, session: Any) -> None:
        """Save a session's state changes to storage."""
        pass

    @abstractmethod
    def load(self, session_id: str) -> Optional[Any]:
        """Retrieve a session by its unique ID."""
        pass

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Delete a session's active state."""
        pass

    @abstractmethod
    def archive(self, session_id: str) -> None:
        """Archive an completed session."""
        pass

    @abstractmethod
    def list_active(self) -> List[Any]:
        """List active consultation sessions."""
        pass
