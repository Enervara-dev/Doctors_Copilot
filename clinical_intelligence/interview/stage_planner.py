from __future__ import annotations

"""
Workflow Stage Transition Planner

Purpose:
Tracks consultation boundaries (Greeting, Chief Complaint, HPI, PMH, Meds, Allergies, Exam, Assessment) and schedules transitions.

Module/Component: Decision Support (CDSS)
"""

"""
Workflow Stage Transition Planner

Purpose:
Tracks consultation boundaries (Greeting, Chief Complaint, HPI, PMH, Meds, Allergies, Exam, Assessment) and schedules transitions.

Module/Component: Decision Support (CDSS)
"""

"""
Workflow Stage Transition Planner

Purpose:
Tracks consultation boundaries (Greeting, Chief Complaint, HPI, PMH, Meds, Allergies, Exam, Assessment) and schedules transitions.

Module/Component: Decision Support (CDSS)
"""

import logging
from typing import Any, Dict, List, Optional
from clinical_intelligence.rule_engine.models import ClinicalDecisionRecommendation, DecisionAction, ActionPriority, ConsultationStage

logger = logging.getLogger(__name__)

"""
Stage Planner.

Created for Module 7 (Adaptive Clinical Decision Support & Consultation Intelligence).
This module manages sequential stage transition logic (from Greeting to Chief Complaint,
HPI, PMH, Medications, Allergies, Exam, Investigations, Assessment, Plan, and Completed).
It recommends ADVANCE_STAGE actions once section completeness rules/thresholds are met.
"""

# Ordered list of stages for transition reference
ORDERED_STAGES = [
    ConsultationStage.GREETING,
    ConsultationStage.CHIEF_COMPLAINT,
    ConsultationStage.HPI,
    ConsultationStage.PMH,
    ConsultationStage.MEDICATION_REVIEW,
    ConsultationStage.ALLERGY_REVIEW,
    ConsultationStage.EXAMINATION,
    ConsultationStage.INVESTIGATIONS,
    ConsultationStage.ASSESSMENT,
    ConsultationStage.PLAN,
    ConsultationStage.FOLLOW_UP,
    ConsultationStage.COMPLETED,
]


class StagePlanner:
    """Plans stage transition recommendations based on stage completeness thresholds."""

    def plan_recommendation(
        self,
        current_stage: ConsultationStage,
        completeness_details: Dict[str, float],
        session: Any
    ) -> Optional[ClinicalDecisionRecommendation]:
        
        # Find index of current stage
        try:
            curr_idx = ORDERED_STAGES.index(current_stage)
        except ValueError:
            return None

        if curr_idx >= len(ORDERED_STAGES) - 1:
            # We are already completed
            return None

        next_stage = ORDERED_STAGES[curr_idx + 1]
        
        # Deterministic checks for transition readiness based on completeness and state contents
        should_advance = False
        reason = ""
        rule_src = ""
        
        state = getattr(session, "clinical_state", {}) or {}

        if current_stage == ConsultationStage.GREETING:
            # Advance as soon as we have a symptom or chief complaint
            if state.get("chief_complaint") or state.get("symptoms"):
                should_advance = True
                reason = "Chief complaint detected. Recommend moving to symptom detail taking."
                rule_src = "workflow.greeting.symptom_detected"
                
        elif current_stage == ConsultationStage.CHIEF_COMPLAINT:
            if state.get("symptoms"):
                should_advance = True
                reason = "Symptom identified. Recommend moving to History of Present Illness (HPI) expansion."
                rule_src = "workflow.complaint.symptom_exists"
                
        elif current_stage == ConsultationStage.HPI:
            # If subjective completeness >= 0.70 or we have >= 2 symptoms/details
            hpi_comp = completeness_details.get("subjective", 0.0)
            if hpi_comp >= 0.70 or len(state.get("symptoms", [])) >= 2:
                should_advance = True
                reason = f"HPI subjective history completeness is high ({int(hpi_comp*100)}%). Recommend moving to Past Medical History (PMH)."
                rule_src = "workflow.hpi.completeness_threshold"
                
        elif current_stage == ConsultationStage.PMH:
            pmh_comp = completeness_details.get("pmh", 0.0)
            if pmh_comp >= 0.70 or state.get("medical_history"):
                should_advance = True
                reason = "Past Medical History recorded. Recommend moving to Medication Review."
                rule_src = "workflow.pmh.history_exists"
                
        elif current_stage == ConsultationStage.MEDICATION_REVIEW:
            # Medications review complete
            if state.get("current_medications") or state.get("allergies"):
                should_advance = True
                reason = "Medication review complete. Recommend moving to Allergy Review."
                rule_src = "workflow.medication.complete"
                
        elif current_stage == ConsultationStage.ALLERGY_REVIEW:
            should_advance = True
            reason = "Allergies reviewed. Recommend advancing to physical Examination stage."
            rule_src = "workflow.allergy.complete"
            
        elif current_stage == ConsultationStage.EXAMINATION:
            # Exam complete when vitals/examination findings recorded
            exam_comp = completeness_details.get("examination", 0.0)
            if exam_comp >= 0.80 or state.get("examination"):
                should_advance = True
                reason = f"Physical examination details recorded. Recommend moving to Investigations."
                rule_src = "workflow.exam.completeness_threshold"
                
        elif current_stage == ConsultationStage.INVESTIGATIONS:
            # Investigations ordered or reviewed
            should_advance = True
            reason = "Recommended investigations evaluated. Recommend advancing to diagnostic Assessment stage."
            rule_src = "workflow.investigations.complete"
            
        elif current_stage == ConsultationStage.ASSESSMENT:
            should_advance = True
            reason = "Differential assessment complete. Recommend advancing to treatment Plan formulation."
            rule_src = "workflow.assessment.complete"
            
        elif current_stage == ConsultationStage.PLAN:
            should_advance = True
            reason = "Treatment plan formulated. Recommend moving to Follow-up stage."
            rule_src = "workflow.plan.complete"
            
        elif current_stage == ConsultationStage.FOLLOW_UP:
            should_advance = True
            reason = "Follow-up recommendations established. Recommend completing the consultation."
            rule_src = "workflow.followup.complete"

        if should_advance:
            # Action action priority is HIGH if we've successfully satisfied the section
            priority = ActionPriority.HIGH
            if current_stage == ConsultationStage.FOLLOW_UP:
                action_type = DecisionAction.COMPLETE_CONSULTATION
            else:
                action_type = DecisionAction.ADVANCE_STAGE

            return ClinicalDecisionRecommendation(
                action=action_type,
                priority=priority,
                confidence=0.90,
                rationale=reason,
                affected_diagnoses=[],
                expected_information_gain=1.0,
                target_finding=next_stage.value,
                rule_sources=[rule_src]
            )

        return None
