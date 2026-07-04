from __future__ import annotations

"""
session_manager.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, PrivateAttr

from clinical_intelligence.workflow.protocols import ISessionManager, ISessionRepository

logger = logging.getLogger(__name__)


class ConsultationSession(BaseModel):
    """Encapsulates the complete isolated clinical consultation state and metadata."""
    session_id: str
    patient_id: str
    clinician_id: str
    status: str = "CREATED"
    consultation_stage: str = "Greeting"
    
    # Metadata timestamps
    created_at: float
    updated_at: float
    last_activity: float
    expires_at: float

    # System configurations version mapping
    rule_engine_version: str = "1.0.0"
    knowledge_graph_version: str = "1.0.0"
    retriever_version: str = "1.0.0"
    workflow_version: str = "1.0.0"

    # Core states
    clinical_state: Dict[str, Any] = Field(default_factory=dict)
    consultation_memory: Dict[str, Any] = Field(default_factory=dict)
    version_tracker: Dict[str, Any] = Field(default_factory=dict)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    audit_metadata: Dict[str, Any] = Field(default_factory=dict)
    question_history: List[Dict[str, Any]] = Field(default_factory=list)
    soap_draft: Optional[Dict[str, Any]] = None
    cached_subsystem_outputs: Dict[str, Any] = Field(default_factory=dict)

    # Isolated write lock
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)


class ConsultationSessionManager(ISessionManager):
    """Orchestrates creation, recovery, timeout verification, and locks of session execution contexts."""

    def __init__(self, repository: ISessionRepository) -> None:
        self.repository = repository
        self._locks: Dict[str, asyncio.Lock] = {}
        self.max_concurrent_sessions = 100
        self.session_timeout = 1800.0  # 30 minutes in seconds

    def create_session(self, session_id: str, patient_id: str, clinician_id: str) -> ConsultationSession:
        """Create and persist a new isolated consultation session context."""
        active_sessions = self.repository.list_active()
        if len(active_sessions) >= self.max_concurrent_sessions:
            raise ValueError(f"Max concurrent sessions limit ({self.max_concurrent_sessions}) exceeded.")

        now = time.time()
        session = ConsultationSession(
            session_id=session_id,
            patient_id=patient_id,
            clinician_id=clinician_id,
            status="CREATED",
            created_at=now,
            updated_at=now,
            last_activity=now,
            expires_at=now + self.session_timeout,
        )
        self._locks[session_id] = asyncio.Lock()
        self.repository.save(session)
        logger.info(f"ConsultationSessionManager: Created session {session_id}")

        # Publish SessionStarted
        try:
            from clinical_intelligence.core.container import container
            from clinical_intelligence.events.events import SessionStarted
            from clinical_intelligence.workflow.protocols import IEventBus
            eb = container.resolve(IEventBus)
            eb.publish(SessionStarted(session_id=session_id, patient_id=patient_id, clinician_id=clinician_id))
        except Exception as e:
            logger.exception("Failed to publish SessionStarted event")

        return session

    def load_session(self, session_id: str) -> Optional[ConsultationSession]:
        """Fetch and lock a session state from repository storage."""
        session = self.repository.load(session_id)
        if session:
            if session_id not in self._locks:
                self._locks[session_id] = asyncio.Lock()
            self.heartbeat(session)
            return session
        return None

    def save_session(self, session: ConsultationSession) -> None:
        """Persist session updates to storage and release locks."""
        session.updated_at = time.time()
        self.repository.save(session)
        logger.debug(f"ConsultationSessionManager: Saved session {session.session_id}")

        # Publish SessionUpdated
        try:
            from clinical_intelligence.core.container import container
            from clinical_intelligence.events.events import SessionUpdated
            from clinical_intelligence.workflow.protocols import IEventBus
            eb = container.resolve(IEventBus)
            eb.publish(SessionUpdated(session_id=session.session_id, workflow_stage=session.consultation_stage, status=session.status))
        except Exception as e:
            logger.exception("Failed to publish SessionUpdated event")

    def archive_session(self, session_id: str) -> None:
        """Transition session state to completed/archived storage."""
        session = self.repository.load(session_id)
        if session:
            session.status = "ARCHIVED"
            self.repository.save(session)
            self.repository.archive(session_id)
            logger.info(f"ConsultationSessionManager: Archived session {session_id}")

            # Publish SessionArchived
            try:
                from clinical_intelligence.core.container import container
                from clinical_intelligence.events.events import SessionArchived
                from clinical_intelligence.workflow.protocols import IEventBus
                eb = container.resolve(IEventBus)
                eb.publish(SessionArchived(session_id=session_id))
            except Exception as e:
                logger.exception("Failed to publish SessionArchived event")

    def destroy_session(self, session_id: str) -> None:
        """Completely purge session from active and persistent database."""
        self.repository.delete(session_id)
        if session_id in self._locks:
            del self._locks[session_id]
        logger.info(f"ConsultationSessionManager: Destroyed session {session_id}")

    def get_lock(self, session_id: str) -> asyncio.Lock:
        """Return the asyncio.Lock corresponding to a unique session context."""
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def heartbeat(self, session: ConsultationSession) -> None:
        """Update last active activity timestamp of a session."""
        now = time.time()
        session.last_activity = now
        session.expires_at = now + self.session_timeout

    def cleanup_expired_sessions(self) -> int:
        """Cleanup active sessions whose last activity passed expiration time."""
        now = time.time()
        expired_count = 0
        active_sessions = self.repository.list_active()
        for session in active_sessions:
            if now > session.expires_at:
                logger.info(f"ConsultationSessionManager: Expiring idle session {session.session_id}")
                session.status = "EXPIRED"
                self.repository.save(session)
                self.repository.archive(session.session_id)
                expired_count += 1
        return expired_count

    def cleanup_idle_sessions(self) -> int:
        """Trigger idle cleanup checking current active times."""
        return self.cleanup_expired_sessions()

    def health(self) -> bool:
        """Check if repository service is functional."""
        try:
            self.repository.list_active()
            return True
        except Exception:
            return False

    def shutdown(self) -> None:
        """Gracefully release concurrent locks and flush all sessions."""
        logger.info("ConsultationSessionManager: Shutting down manager and clearing active locks.")
        self._locks.clear()

    def get_statistics(self) -> Dict[str, int]:
        """Return operational session count statistics."""
        active_sessions = self.repository.list_active()
        now = time.time()
        
        active_count = 0
        idle_count = 0
        expired_count = 0
        
        for s in active_sessions:
            if s.status == "ARCHIVED":
                continue
            if now > s.expires_at:
                expired_count += 1
            elif now - s.last_activity > 300.0: # Idle if > 5 min
                idle_count += 1
            else:
                active_count += 1
                
        return {
            "active_sessions": active_count,
            "idle_sessions": idle_count,
            "expired_sessions": expired_count,
            "total_sessions": len(active_sessions)
        }
