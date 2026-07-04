from __future__ import annotations

"""
state_machine.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging
from typing import Any

from clinical_intelligence.config.workflow import (
    CLINICAL_STAGES,
    STAGED_TRANSITIONS,
    ROLLBACK_ALLOWED_STAGES,
)

logger = logging.getLogger(__name__)


class InvalidTransitionError(ValueError):
    """Exception raised when an illegal stage transition is requested."""
    pass


class ConsultationStateMachine:
    """Validates and applies workflow state transitions based on config rules."""

    def transition_to(self, session: Any, next_stage: str) -> None:
        """Move the session to a new stage if it is a legal transition."""
        current_stage = session.consultation_stage
        
        if next_stage not in CLINICAL_STAGES:
            raise InvalidTransitionError(f"Stage '{next_stage}' is not a valid consultation stage.")

        # completed/emergency escape checks
        if next_stage != "Completed":
            allowed_next = STAGED_TRANSITIONS.get(current_stage, set())
            if next_stage not in allowed_next:
                raise InvalidTransitionError(
                    f"Illegal stage transition from '{current_stage}' to '{next_stage}'."
                )

        logger.info(f"StateMachine: Transitioning session {session.session_id} from '{current_stage}' to '{next_stage}'")
        session.consultation_stage = next_stage
        session.status = "REASONING"

    def pause(self, session: Any) -> None:
        """Pause consultation execution, waiting for doctor inputs."""
        session.status = "WAITING_FOR_INPUT"
        logger.info(f"StateMachine: Paused session {session.session_id}")

    def resume(self, session: Any) -> None:
        """Resume active consultation workflows."""
        session.status = "ACTIVE"
        logger.info(f"StateMachine: Resumed session {session.session_id}")

    def rollback(self, session: Any, target_stage: str) -> None:
        """Rollback session context back to a previous stage."""
        if target_stage not in ROLLBACK_ALLOWED_STAGES:
            raise InvalidTransitionError(f"Rollback to stage '{target_stage}' is not allowed.")
        logger.warning(
            f"StateMachine: Rolling back session {session.session_id} from '{session.consultation_stage}' to '{target_stage}'"
        )
        session.consultation_stage = target_stage
        session.status = "REASONING"

    def cancel(self, session: Any) -> None:
        """Cancel consultation, archiving its active context."""
        session.status = "ARCHIVED"
        logger.info(f"StateMachine: Cancelled session {session.session_id}")

    def timeout(self, session: Any) -> None:
        """Timeout session execution on inactivity."""
        session.status = "EXPIRED"
        logger.info(f"StateMachine: Session {session.session_id} expired due to timeout.")

    def restart(self, session: Any) -> None:
        """Restart consultation from Greeting."""
        logger.info(f"StateMachine: Restarting session {session.session_id} from Greeting.")
        session.consultation_stage = "Greeting"
        session.status = "ACTIVE"
