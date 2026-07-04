from __future__ import annotations

"""
timeline.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import copy
import logging
import time
from typing import Any, Dict, List, Optional

from clinical_intelligence.workflow.protocols import ITimelineEngine

logger = logging.getLogger(__name__)


class TimelineEngine(ITimelineEngine):
    """Chronologically logs events, manages audit records, and captures stage-level session snapshots."""

    def __init__(self) -> None:
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    def record_event(self, session_id: str, stage: str, event: Any) -> None:
        """Constructs and appends a timeline entry context directly to the session."""
        from clinical_intelligence.core.container import container
        from clinical_intelligence.workflow.protocols import ISessionRepository
        
        try:
            repo = container.resolve(ISessionRepository)
            session = repo.load(session_id)
            if not session:
                return
        except Exception:
            return

        actor = getattr(event, "actor", "clinician" if "Question" in type(event).__name__ else "system")
        entry = {
            "timestamp": time.time(),
            "session_id": session_id,
            "stage": stage,
            "actor": actor,
            "event_type": type(event).__name__,
            "version_snapshot": {
                "rule_engine": session.rule_engine_version,
                "workflow": session.workflow_version,
            },
            "payload_summary": str(event),
        }
        
        session.timeline.append(entry)
        repo.save(session)
        logger.debug(f"TimelineEngine: Recorded event {type(event).__name__} for session {session_id}")

    def get_timeline(self, session_id: str) -> List[Any]:
        """Returns the list of all timeline entries for a session."""
        from clinical_intelligence.core.container import container
        from clinical_intelligence.workflow.protocols import ISessionRepository
        try:
            repo = container.resolve(ISessionRepository)
            session = repo.load(session_id)
            if session:
                return list(session.timeline)
        except Exception:
            pass
        return []

    def take_snapshot(self, session: Any, stage_name: str) -> None:
        """Create and capture a lightweight snapshot of the session state at a major stage boundary."""
        snap_key = f"{session.session_id}:{stage_name}"
        snapshot = {
            "session_id": session.session_id,
            "stage": stage_name,
            "timestamp": time.time(),
            "status": session.status,
            "clinical_state": copy.deepcopy(session.clinical_state),
            "consultation_memory": copy.deepcopy(session.consultation_memory),
            "version_tracker": copy.deepcopy(session.version_tracker),
        }
        self._snapshots[snap_key] = snapshot
        logger.info(f"TimelineEngine: Captured stage snapshot '{stage_name}' for session {session.session_id}")

    def get_snapshot(self, session_id: str, stage_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific stage boundary snapshot."""
        return self._snapshots.get(f"{session_id}:{stage_name}")

    def export_timeline(self, session_id: str) -> List[Any]:
        """Export timeline logs."""
        return self.get_timeline(session_id)
