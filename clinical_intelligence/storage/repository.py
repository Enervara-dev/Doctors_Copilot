from __future__ import annotations

"""
repository.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import copy
import logging
from typing import Any, Dict, List, Optional

from clinical_intelligence.workflow.protocols import ISessionRepository

logger = logging.getLogger(__name__)


class MemorySessionRepository(ISessionRepository):
    """Memory-backed concrete implementation of ISessionRepository."""

    def __init__(self) -> None:
        self._active_store: Dict[str, Any] = {}
        self._archived_store: Dict[str, Any] = {}

    def save(self, session: Any) -> None:
        """Save session state using deep copy to ensure isolation."""
        session_id = session.session_id
        self._active_store[session_id] = copy.deepcopy(session)
        logger.debug(f"MemorySessionRepository: Saved session {session_id}")

    def load(self, session_id: str) -> Optional[Any]:
        """Load session state, returning a copy to ensure session isolation."""
        if session_id in self._active_store:
            logger.debug(f"MemorySessionRepository: Loaded session {session_id}")
            return copy.deepcopy(self._active_store[session_id])
        if session_id in self._archived_store:
            logger.debug(f"MemorySessionRepository: Loaded archived session {session_id}")
            return copy.deepcopy(self._archived_store[session_id])
        return None

    def delete(self, session_id: str) -> None:
        """Purge session completely from storage."""
        if session_id in self._active_store:
            del self._active_store[session_id]
            logger.info(f"MemorySessionRepository: Deleted active session {session_id}")
        if session_id in self._archived_store:
            del self._archived_store[session_id]
            logger.info(f"MemorySessionRepository: Deleted archived session {session_id}")

    def archive(self, session_id: str) -> None:
        """Move active session to archived store."""
        if session_id in self._active_store:
            session = self._active_store[session_id]
            session.status = "ARCHIVED"
            self._archived_store[session_id] = session
            del self._active_store[session_id]
            logger.info(f"MemorySessionRepository: Archived session {session_id}")
        elif session_id in self._archived_store:
            logger.warning(f"MemorySessionRepository: Session {session_id} is already archived.")
        else:
            raise ValueError(f"MemorySessionRepository: Session '{session_id}' not found to archive.")

    def list_active(self) -> List[Any]:
        """Return list of active sessions."""
        return copy.deepcopy(list(self._active_store.values()))
