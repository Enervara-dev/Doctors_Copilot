from __future__ import annotations

"""
Diagnostic Lab & Imaging Planner

Purpose:
Recommends guidelines-backed imaging/labs, upgrading priorities to CRITICAL if emergency red flags fire.

Module/Component: Decision Support (CDSS)
"""

"""
Diagnostic Lab & Imaging Planner

Purpose:
Recommends guidelines-backed imaging/labs, upgrading priorities to CRITICAL if emergency red flags fire.

Module/Component: Decision Support (CDSS)
"""

"""
Diagnostic Lab & Imaging Planner

Purpose:
Recommends guidelines-backed imaging/labs, upgrading priorities to CRITICAL if emergency red flags fire.

Module/Component: Decision Support (CDSS)
"""

import logging
from typing import Any, Dict, List
from clinical_intelligence.interview.evidence_gap_analyzer import DiseaseGap
from clinical_intelligence.interview.question_memory import QuestionMemoryManager
from clinical_intelligence.rule_engine.models import ClinicalDecisionRecommendation, DecisionAction, ActionPriority

logger = logging.getLogger(__name__)

"""
Investigation Planner.

Created for Module 7 (Adaptive Clinical Decision Support & Consultation Intelligence).
This module extracts recommended clinical investigations (labs, radiology, or ECG tests)
from candidate disease profile guidelines and recommends them. It schedules them during
the Investigations stage of the workflow, and upgrades their priority to CRITICAL when active
emergency red flags require immediate testing.
"""


class InvestigationPlanner:
    """Selects and plans guideline-recommended investigations based on candidate diagnoses and active alerts."""

    def __init__(self, memory_manager: QuestionMemoryManager = None) -> None:
        self.memory_manager = memory_manager or QuestionMemoryManager()

    def plan_recommendations(
        self,
        gaps: List[DiseaseGap],
        session: Any,
        candidate_diagnoses: List[Any],
        recommended_investigations: List[Any],
        current_stage: str,
        has_red_flags: bool = False
    ) -> List[ClinicalDecisionRecommendation]:
        
        recommendations: List[ClinicalDecisionRecommendation] = []
        ordered_investigations = set()
        
        # Pull ordered investigations from session state
        state = getattr(session, "clinical_state", {}) or {}
        treatment_plan = state.get("treatment_plan", {}) or {}
        ordered_list = treatment_plan.get("investigations", []) or []
        for ordered in ordered_list:
            ordered_investigations.add(ordered.lower().strip())

        for inv in recommended_investigations:
            inv_name = getattr(inv, "name", "") if not isinstance(inv, dict) else inv.get("name", "")
            inv_id = getattr(inv, "investigation_id", "") if not isinstance(inv, dict) else inv.get("investigation_id", "")
            inv_priority = getattr(inv, "priority", "ROUTINE") if not isinstance(inv, dict) else inv.get("priority", "ROUTINE")
            inv_reason = getattr(inv, "reason", "") if not isinstance(inv, dict) else inv.get("reason", "")
            
            if not inv_name:
                continue

            inv_clean = inv_name.lower().strip()
            if inv_clean in ordered_investigations or self.memory_manager.is_blocked(inv_name, session):
                continue

            # Clinical Utility calculation
            base_score = 3.0
            
            priority_score = 0.0
            if inv_priority == "URGENT":
                priority_score = 5.0
            elif inv_priority == "HIGH":
                priority_score = 3.0
                
            # Stage match check (Investigations stage matches perfectly)
            workflow_score = 4.0 if current_stage == "Investigations" else 0.0
            
            # Workflow constraints: discourage investigations in early history phase unless overridden by red flags
            constraint_penalty = 0.0
            if current_stage in ["Greeting", "Chief Complaint", "HPI", "PMH", "Medication Review", "Allergy Review", "Examination"] and not has_red_flags:
                constraint_penalty = 4.0

            # Red flag boost (e.g. ECG / Troponin for chest pain)
            red_flag_boost = 0.0
            if has_red_flags and any(x in inv_clean for x in ["ecg", "electrocardiogram", "troponin", "ct brain", "head ct", "chest x-ray", "cxr"]):
                red_flag_boost = 6.0

            utility_score = base_score + priority_score + workflow_score + red_flag_boost - constraint_penalty
            
            # Priority
            if red_flag_boost > 0 and has_red_flags:
                priority = ActionPriority.CRITICAL
            elif utility_score >= 6.0 or inv_priority == "URGENT":
                priority = ActionPriority.HIGH
            elif utility_score >= 3.0:
                priority = ActionPriority.NORMAL
            else:
                priority = ActionPriority.LOW

            rec_confidence = min(0.99, max(0.1, utility_score / 12.0))
            
            # Identify affected diagnoses
            affected = []
            for candidate in candidate_diagnoses:
                dis_name = getattr(candidate, "name", None) or candidate.get("name")
                if dis_name:
                    affected.append(dis_name)

            recommendations.append(ClinicalDecisionRecommendation(
                action=DecisionAction.RECOMMEND_INVESTIGATION,
                priority=priority,
                confidence=round(rec_confidence, 2),
                rationale=inv_reason or f"Order investigation '{inv_name}' based on active clinical guidelines.",
                affected_diagnoses=affected,
                expected_information_gain=2.0,
                target_finding=inv_name,
                rule_sources=[f"investigation.{inv_id}", f"workflow.stage.{current_stage}"]
            ))

        # ----------------------------------------------------
        # EXTENSION POINT: External Data Sources & Telemetry
        # ----------------------------------------------------
        # Planners can query wearable stats, ECG data streams, or imaging findings here.
        # Example wearable diagnostic logic check:
        # if query_wearable_data_stream(session.session_id):
        #      recommendations.append(plan_wearable_rec())
        
        # Sort descending by utility
        def sort_key(r: ClinicalDecisionRecommendation) -> float:
            p_weight = {"critical": 10.0, "high": 5.0, "normal": 2.0, "low": 0.0}[r.priority.value]
            return r.expected_information_gain + p_weight

        recommendations.sort(key=sort_key, reverse=True)
        return recommendations


def query_wearable_data_stream(session_id: str) -> bool:
    """Hook for future integrations with continuous glucose monitoring or heart rate telemetry."""
    return False
