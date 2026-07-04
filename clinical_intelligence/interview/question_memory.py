from __future__ import annotations

"""
Asked Inquiries Memory Manager

Purpose:
Tracks explored, skipped, refused, or deferred items in session logs to avoid duplicate prompts.

Module/Component: Decision Support (CDSS)
"""

"""
Asked Inquiries Memory Manager

Purpose:
Tracks explored, skipped, refused, or deferred items in session logs to avoid duplicate prompts.

Module/Component: Decision Support (CDSS)
"""

"""
Asked Inquiries Memory Manager

Purpose:
Tracks explored, skip, refused, or deferred items in session logs to avoid duplicate prompts.

Module/Component: Decision Support (CDSS)
"""

import logging
from typing import Any, List, Set

logger = logging.getLogger(__name__)

"""
Question Memory Manager.

Created for Module 7 (Adaptive Clinical Decision Support & Consultation Intelligence).
This module tracks previously asked, answered, refused, skipped, and deferred findings
using the session timeline and memory register. It prevents repeating questions and
correctly penalizes skipped or refused inquiries.
"""


class QuestionMemoryManager:
    """Tracks asked, answered, skipped, refused, unknown, and deferred questions/actions to avoid repeat asking."""

    def is_blocked(self, finding: str, session: Any) -> bool:
        """Determines if a finding is blocked from being asked again based on memory state."""
        finding_clean = finding.lower().strip()
        
        # 1. Check direct question history
        q_history = getattr(session, "question_history", []) or []
        for q in q_history:
            if finding_clean in q.lower():
                return True
                
        # 2. Check consultation memory
        memory = getattr(session, "consultation_memory", {}) or {}
        
        blocked_statuses = ["refused", "answered", "unknown", "skipped"]
        for status in blocked_statuses:
            status_list = memory.get(status, []) or []
            if any(finding_clean == item.lower().strip() for item in status_list):
                return True
                
        return False

    def mark_asked(self, finding: str, session: Any) -> None:
        """Record that a finding was queried in the history list."""
        if not hasattr(session, "question_history") or session.question_history is None:
            session.question_history = []
        if finding not in session.question_history:
            session.question_history.append(finding)

    def mark_status(self, finding: str, status: str, session: Any) -> None:
        """Update consultation memory status (e.g. refused, answered, unknown, deferred)."""
        if not hasattr(session, "consultation_memory") or session.consultation_memory is None:
            session.consultation_memory = {}
            
        if status not in session.consultation_memory:
            session.consultation_memory[status] = []
            
        if finding not in session.consultation_memory[status]:
            session.consultation_memory[status].append(finding)
