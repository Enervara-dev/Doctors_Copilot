from __future__ import annotations

"""
workflow_engine.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging
from typing import Any, Dict, List

from clinical_intelligence.workflow.protocols import IWorkflowEngine
from clinical_intelligence.config.workflow import (
    STAGE_PROGRESS_ESTIMATES,
    STAGE_FACT_REQUIREMENTS,
)

logger = logging.getLogger(__name__)


class ConsultationWorkflowEngine(IWorkflowEngine):
    """Deterministic, configuration-driven logic resolving stages and next action recommendation paths."""

    def determine_stage(self, session: Any) -> str:
        """Determine and validate the stage of clinical consultation based on facts."""
        state = session.clinical_state
        symptoms = state.get("symptoms", []) or []
        history = state.get("medical_history", []) or []
        medications = state.get("current_medications", []) or []
        
        # Check active status of symptoms
        active_symptoms = [s for s in symptoms if s.get("status") == "Active" or str(s.get("present", "True")).lower() == "true"]

        current = session.consultation_stage

        # Deterministic progression progression checks
        if not active_symptoms:
            return "Greeting"
        if len(active_symptoms) == 1 and not history:
            return "Chief Complaint"
        if len(active_symptoms) > 1 and not history:
            return "HPI"
        if history and not medications:
            return "PMH"
        if medications and current in ("Greeting", "Chief Complaint", "HPI", "PMH"):
            return "Medication Review"
            
        return current

    def determine_next_action(self, session: Any) -> Dict[str, Any]:
        """Estimate completion percentage and next recommended actions."""
        stage = session.consultation_stage
        state = session.clinical_state
        
        progress = STAGE_PROGRESS_ESTIMATES.get(stage, 0)
        
        # Audit missing required facts for the current stage
        required = STAGE_FACT_REQUIREMENTS.get(stage, [])
        missing = []
        for fact in required:
            if not state.get(fact):
                missing.append(fact)

        next_action = "Collect details"
        if stage == "Greeting":
            next_action = "Inquire about the patient's chief complaint."
        elif stage == "Chief Complaint":
            next_action = "Expand details on the duration and onset of symptoms."
        elif stage == "HPI":
            next_action = "Inquire about past medical history and underlying conditions."
        elif stage == "PMH":
            next_action = "Conduct a complete review of active patient medications."
        elif stage == "Medication Review":
            next_action = "Inquire about known drug allergies."
        elif stage == "Allergy Review":
            next_action = "Perform physical examination checks."
        elif stage == "Examination":
            next_action = "Review diagnostic candidate rankings."
        elif stage == "Assessment":
            next_action = "Formulate a treatment and management plan."
        elif stage == "Plan":
            next_action = "Provide patient education and follow-up guidance."
        elif stage == "Follow-up":
            next_action = "Finalize the SOAP note and complete consultation."

        return {
            "stage": stage,
            "completion_percentage": progress,
            "missing_information": missing,
            "next_action": next_action,
            "needs_reasoning": len(missing) == 0,
        }
