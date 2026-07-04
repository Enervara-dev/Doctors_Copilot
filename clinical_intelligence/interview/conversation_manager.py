from __future__ import annotations

"""
CDSS Conversation Manager Orchestrator

Purpose:
Orchestrates gap analysis, completeness metrics, memory trackers, and decision policies to produce consultation recommendations.

Module/Component: Decision Support (CDSS)
"""

"""
CDSS Conversation Manager Orchestrator

Purpose:
Orchestrates gap analysis, completeness metrics, memory trackers, and decision policies to produce consultation recommendations.

Module/Component: Decision Support (CDSS)
"""

"""
CDSS Conversation Manager Orchestrator

Purpose:
Orchestrates gap analysis, completeness metrics, memory trackers, and decision policies to produce consultation recommendations.

Module/Component: Decision Support (CDSS)
"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from clinical_intelligence.interview.evidence_gap_analyzer import EvidenceGapAnalyzer
from clinical_intelligence.interview.completeness import CompletenessPlanner
from clinical_intelligence.interview.decision_policy import DecisionPolicy
from clinical_intelligence.evidence.disease_profile_registry import DiseaseProfileRegistry
from clinical_intelligence.rule_engine.models import (
    ConsultationDecisionResult,
    ConsultationStage,
    ClinicalReasoningResult
)

logger = logging.getLogger(__name__)

"""
Conversation Manager.

Created for Module 7 (Adaptive Clinical Decision Support & Consultation Intelligence).
This class integrates the evidence gap analyzer, completeness planners, and central decision
policy. It evaluates incoming reasoning results against active session history and returns
the final ConsultationDecisionResult.
"""


class ConversationManager:
    """Orchestrates gap analysis, completeness planner, memory managers, and decision policies."""

    def __init__(self) -> None:
        self.gap_analyzer = EvidenceGapAnalyzer()
        self.completeness_planner = CompletenessPlanner()
        self.decision_policy = DecisionPolicy()

    def generate_decision(
        self,
        session: Any,
        reasoning_result: ClinicalReasoningResult,
        registry: DiseaseProfileRegistry
    ) -> ConsultationDecisionResult:
        
        # 1. Resolve and parse type-safe ConsultationStage
        stage_str = getattr(session, "consultation_stage", "Greeting")
        
        # Normalize stage string to match enum values
        try:
            current_stage = ConsultationStage(stage_str)
        except ValueError:
            current_stage = ConsultationStage.GREETING
            
        clinical_state = session.clinical_state
        if isinstance(clinical_state, BaseModel):
            clinical_state_dict = clinical_state.model_dump()
        else:
            clinical_state_dict = clinical_state

        # 2. Execute Evidence Gap Analysis
        gaps = self.gap_analyzer.analyze(
            clinical_state=clinical_state_dict,
            candidate_diagnoses=reasoning_result.candidate_diagnoses,
            registry=registry
        )

        # 3. Calculate Section and Overall Completeness
        completeness_details = self.completeness_planner.compute_completeness(
            gaps=gaps,
            clinical_state=clinical_state_dict
        )
        progress = completeness_details.get("overall", 5.0)

        # 4. Determine Next Best Actions via Decision Policy
        recs, next_rec, telemetry = self.decision_policy.determine_next_action(
            gaps=gaps,
            session=session,
            candidate_diagnoses=reasoning_result.candidate_diagnoses,
            red_flags=reasoning_result.red_flags,
            recommended_investigations=reasoning_result.recommended_investigations,
            current_stage=current_stage
        )

        # 5. Compile final type-safe ConsultationDecisionResult
        decision_result = ConsultationDecisionResult(
            recommendations=recs,
            next_recommendation=next_rec,
            workflow_stage=current_stage,
            consultation_progress=round(progress, 2),
            telemetry=telemetry
        )

        # Register the chosen action in memory to avoid duplicate loops
        if next_rec and next_rec.action == "ask_question":
            self.decision_policy.question_planner.memory_manager.mark_asked(next_rec.target_finding, session)
            
        return decision_result
