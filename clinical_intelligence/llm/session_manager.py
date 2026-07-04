from __future__ import annotations

"""
session_manager.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging
from typing import Any, Dict, Optional, Set

from clinical_intelligence.llm.orchestrator import LLMResultCache

logger = logging.getLogger(__name__)


class LLMSessionManager:
    """
    Manages session-level LLM contexts, deduplicates requests,
    and performs graceful cleanup after consultation completion.
    """

    def __init__(self) -> None:
        self.active_sessions: Set[str] = set()
        self.session_caches: Dict[str, LLMResultCache] = {}
        self.active_requests: Dict[str, Set[str]] = {} # session_id -> set of active task names

    def register_session(self, session_id: str) -> None:
        """Register a new consultation session context."""
        self.active_sessions.add(session_id)
        if session_id not in self.session_caches:
            self.session_caches[session_id] = LLMResultCache()
        self.active_requests[session_id] = set()
        logger.info(f"LLMSessionManager: Registered session context '{session_id}'")

    def get_cache(self, session_id: str) -> LLMResultCache:
        """Get or create session-specific result cache."""
        if session_id not in self.session_caches:
            self.session_caches[session_id] = LLMResultCache()
        return self.session_caches[session_id]

    def track_request_start(self, session_id: str, task_name: str) -> None:
        """Record the start of an LLM task execution."""
        if session_id in self.active_requests:
            self.active_requests[session_id].add(task_name)

    def track_request_end(self, session_id: str, task_name: str) -> None:
        """Record completion of an LLM task execution."""
        if session_id in self.active_requests and task_name in self.active_requests[session_id]:
            self.active_requests[session_id].remove(task_name)

    def cancel_active_requests(self, session_id: str) -> None:
        """Graceful cancellation of all outstanding requests for a session."""
        if session_id in self.active_requests:
            active = list(self.active_requests[session_id])
            if active:
                logger.info(f"LLMSessionManager: Cancelling active tasks {active} for session '{session_id}'")
            self.active_requests[session_id].clear()

    def cleanup_session(self, session_id: str) -> None:
        """Flush temporary caches, release buffers, and unregister session context."""
        self.cancel_active_requests(session_id)
        
        # Release cache memory
        if session_id in self.session_caches:
            self.session_caches[session_id].clear()
            del self.session_caches[session_id]

        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)

        if session_id in self.active_requests:
            del self.active_requests[session_id]
            
        logger.info(f"LLMSessionManager: Cleaned up and released resources for session '{session_id}'")
