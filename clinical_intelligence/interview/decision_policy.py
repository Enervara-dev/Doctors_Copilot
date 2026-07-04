from __future__ import annotations

"""
Central Decision Policy Gater

Purpose:
Applies safety red flag bypass overrides and workflow stage constraints to select the next recommendation.

Module/Component: Decision Support (CDSS)
"""

"""
Central Decision Policy Gater

Purpose:
Applies safety red flag bypass overrides and workflow stage constraints to select the next recommendation.

Module/Component: Decision Support (CDSS)
"""

"""
Central Decision Policy Gater

Purpose:
Applies safety red flag bypass overrides and workflow stage constraints to select the next recommendation.

Module/Component: Decision Support (CDSS)
"""

import logging
import time
from typing import Any, Dict, List, Optional

from clinical_intelligence.interview.evidence_gap_analyzer import DiseaseGap
from clinical_intelligence.interview.question_planner import QuestionPlanner
from clinical_intelligence.interview.examination_planner import ExaminationPlanner
from clinical_intelligence.interview.investigation_planner import InvestigationPlanner
from clinical_intelligence.interview.stage_planner import StagePlanner
from clinical_intelligence.rule_engine.models import (
    ClinicalDecisionRecommendation,
    DecisionAction,
    ActionPriority,
    ConsultationStage,
    DecisionTelemetry
)

logger = logging.getLogger(__name__)

"""
Decision Policy Coordinator.

Created for Module 7 (Adaptive Clinical Decision Support & Consultation Intelligence).
This module serves as the main CDSS orchestrator. It aggregates raw recommendations from all sub-planners,
and enforces two major sets of clinical rules:
1. Red Flag Override: Elevates critical investigations/exams immediately if safety flags fire.
2. Workflow Constraints: Restricts exams, diagnostics, or follow-ups during history-taking.
It outputs the final recommended action.
"""


class DecisionPolicy:
    """Central Clinical Decision Support orchestrator applying Red Flag Overrides and Workflow Constraints."""

    def __init__(self) -> None:
        self.question_planner = QuestionPlanner()
        self.examination_planner = ExaminationPlanner()
        self.investigation_planner = InvestigationPlanner()
        self.stage_planner = StagePlanner()

    def determine_next_action(
        self,
        gaps: List[DiseaseGap],
        session: Any,
        candidate_diagnoses: List[Any],
        red_flags: List[Any],
        recommended_investigations: List[Any],
        current_stage: ConsultationStage
    ) -> tuple[List[ClinicalDecisionRecommendation], Optional[ClinicalDecisionRecommendation], DecisionTelemetry]:
        
        start_time = time.perf_counter()
        
        telemetry = DecisionTelemetry(
            candidate_count=len(candidate_diagnoses)
        )
        
        has_red_flags = len(red_flags) > 0
        
        # 1. Plan raw recommendations from all planners
        q_recs = self.question_planner.plan_recommendations(gaps, session, candidate_diagnoses, current_stage.value)
        exam_recs = self.examination_planner.plan_recommendations(gaps, session, candidate_diagnoses, current_stage.value, has_red_flags)
        inv_recs = self.investigation_planner.plan_recommendations(gaps, session, candidate_diagnoses, recommended_investigations, current_stage.value, has_red_flags)
        
        # Determine section completeness for stage advancement
        completeness_details: Dict[str, float] = {}
        if gaps:
            completeness_details["subjective"] = sum(g.completeness for g in gaps) / len(gaps)
            completeness_details["pmh"] = completeness_details["subjective"]
            completeness_details["examination"] = completeness_details["subjective"]
        else:
            completeness_details["subjective"] = 0.0
            completeness_details["pmh"] = 0.0
            completeness_details["examination"] = 0.0
            
        stage_rec = self.stage_planner.plan_recommendation(current_stage, completeness_details, session)

        all_recs: List[ClinicalDecisionRecommendation] = []
        all_recs.extend(q_recs)
        all_recs.extend(exam_recs)
        all_recs.extend(inv_recs)
        if stage_rec:
            all_recs.append(stage_rec)

        # 2. Apply Red Flag Override Check
        next_rec: Optional[ClinicalDecisionRecommendation] = None
        
        if has_red_flags:
            telemetry.red_flag_override = True
            # Find any critical investigation recommendation (e.g. ECG for cardiac symptoms)
            critical_invs = [r for r in inv_recs if r.priority == ActionPriority.CRITICAL]
            critical_exams = [r for r in exam_recs if r.priority == ActionPriority.CRITICAL]
            
            if critical_invs:
                next_rec = critical_invs[0]
            elif critical_exams:
                next_rec = critical_exams[0]
            elif inv_recs:
                # Upgrade highest priority investigation
                inv_recs[0].priority = ActionPriority.CRITICAL
                inv_recs[0].rationale = f"Emergency Red Flag alert detected. {inv_recs[0].rationale}"
                next_rec = inv_recs[0]

        # 3. Apply Workflow Constraints (if no red flag override is active)
        if not next_rec:
            # We are in history stage (Greeting, Chief Complaint, HPI, PMH, Medication, Allergy)
            history_stages = [
                ConsultationStage.GREETING,
                ConsultationStage.CHIEF_COMPLAINT,
                ConsultationStage.HPI,
                ConsultationStage.PMH,
                ConsultationStage.MEDICATION_REVIEW,
                ConsultationStage.ALLERGY_REVIEW
            ]
            if current_stage in history_stages:
                telemetry.workflow_constraint_triggered = True
                
                # Under constraints, we only allow ASK_QUESTION or ADVANCE_STAGE recommendations as next_recommendation
                allowed_recs = [r for r in all_recs if r.action in [DecisionAction.ASK_QUESTION, DecisionAction.ADVANCE_STAGE]]
                if allowed_recs:
                    # Sort by utility
                    allowed_recs.sort(key=lambda r: r.confidence, reverse=True)
                    next_rec = allowed_recs[0]
                else:
                    # Fallback to stage transition if no question available
                    if stage_rec:
                        next_rec = stage_rec

        # 4. Default utility-based selection if not yet chosen
        if not next_rec and all_recs:
            # Sort all recommendations by utility score
            # Highest priority first, then highest confidence
            def sort_key(r: ClinicalDecisionRecommendation) -> tuple[int, float]:
                p_val = {"critical": 4, "high": 3, "normal": 2, "low": 1}[r.priority.value]
                return (p_val, r.confidence)
                
            sorted_recs = sorted(all_recs, key=sort_key, reverse=True)
            next_rec = sorted_recs[0]

        # 5. Handle NO_ACTION fallback
        if not next_rec:
            next_rec = ClinicalDecisionRecommendation(
                action=DecisionAction.NO_ACTION,
                priority=ActionPriority.LOW,
                confidence=1.0,
                rationale="No pending evidence gaps, investigations, or workflow progression actions required.",
                affected_diagnoses=[],
                expected_information_gain=0.0,
                target_finding="none",
                rule_sources=["workflow.policy.no_action"]
            )
            all_recs.append(next_rec)

        telemetry.decision_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        telemetry.planner_selected = next_rec.action.value
        telemetry.utility_score = next_rec.confidence * 10.0

        return all_recs, next_rec, telemetry
