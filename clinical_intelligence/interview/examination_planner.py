from __future__ import annotations

"""
Vital & Exam Check Recommendation Planner

Purpose:
Suggests physical exams or vital sign targets from profiles, scheduling checks appropriately based on workflow stage.

Module/Component: Decision Support (CDSS)
"""

"""
Vital & Exam Check Recommendation Planner

Purpose:
Suggests physical exams or vital sign targets from profiles, scheduling checks appropriately based on workflow stage.

Module/Component: Decision Support (CDSS)
"""

"""
Vital & Exam Check Recommendation Planner

Purpose:
Suggests physical exams or vital sign targets from profiles, scheduling checks appropriately based on workflow stage.

Module/Component: Decision Support (CDSS)
"""

import logging
from typing import Any, Dict, List
from clinical_intelligence.interview.evidence_gap_analyzer import DiseaseGap
from clinical_intelligence.interview.question_memory import QuestionMemoryManager
from clinical_intelligence.rule_engine.models import ClinicalDecisionRecommendation, DecisionAction, ActionPriority

logger = logging.getLogger(__name__)

"""
Examination Planner.

Created for Module 7 (Adaptive Clinical Decision Support & Consultation Intelligence).
This module scans candidate disease profiles to identify missing vital sign criteria or
physical examination maneuvers. It recommends these checks, scheduling them heavily during
the Examination stage of the workflow (or earlier if critical red flags warrant immediate assessment).
"""


class ExaminationPlanner:
    """Identifies missing vitals or exam maneuvers from disease profiles and plans recommendations."""

    def __init__(self, memory_manager: QuestionMemoryManager = None) -> None:
        self.memory_manager = memory_manager or QuestionMemoryManager()

    def plan_recommendations(
        self,
        gaps: List[DiseaseGap],
        session: Any,
        candidate_diagnoses: List[Any],
        current_stage: str,
        has_red_flags: bool = False
    ) -> List[ClinicalDecisionRecommendation]:
        
        recommendations: List[ClinicalDecisionRecommendation] = []
        missing_exams: Dict[str, List[str]] = {}  # exam/vital -> affected disease names
        
        top_disease_id = None
        if candidate_diagnoses:
            top_disease_id = getattr(candidate_diagnoses[0], "disease_id", None) or candidate_diagnoses[0].get("disease_id")

        for gap in gaps:
            for finding in gap.missing_findings:
                finding_clean = finding.lower().strip()
                if is_vital_or_exam_finding(finding_clean):
                    missing_exams.setdefault(finding, []).append(gap.disease_name)

        for exam, affected_diseases in missing_exams.items():
            if self.memory_manager.is_blocked(exam, session):
                continue
                
            exam_clean = exam.lower().strip()
            
            # Clinical Utility calculation
            base_score = 2.0
            
            # High confidence candidate diagnostic match
            critical_score = 0.0
            for gap in gaps:
                if gap.disease_id == top_disease_id and exam in gap.missing_findings:
                    critical_score = 3.0
                    break
                    
            # Workflow stage check (Examination stage matches perfectly)
            workflow_score = 4.0 if current_stage == "Examination" else 0.0
            
            # Apply severe workflow constraints penalty if we recommend exams during subjective history phase
            # unless a red flag is present (which overrides the constraint)
            constraint_penalty = 0.0
            if current_stage in ["Greeting", "Chief Complaint", "HPI", "PMH", "Medication Review", "Allergy Review"] and not has_red_flags:
                constraint_penalty = 4.0
                
            # Vitals / Emergency priority checking
            red_flag_score = 0.0
            if exam_clean in ["spo2", "oxygen saturation", "bp", "blood pressure", "pulse", "temperature", "heart rate", "auscultation", "auscultate lungs"]:
                if has_red_flags or any(any(x in dis for x in ["Influenza", "COVID", "Pneumonia", "Coronary", "Stroke"]) for dis in affected_diseases):
                    red_flag_score = 5.0

            utility_score = base_score + critical_score + workflow_score + red_flag_score - constraint_penalty
            
            # Priority
            if red_flag_score > 0 and has_red_flags:
                priority = ActionPriority.CRITICAL
            elif utility_score >= 5.0:
                priority = ActionPriority.HIGH
            elif utility_score >= 2.0:
                priority = ActionPriority.NORMAL
            else:
                priority = ActionPriority.LOW

            rec_confidence = min(0.99, max(0.1, utility_score / 12.0))

            recommendations.append(ClinicalDecisionRecommendation(
                action=DecisionAction.RECOMMEND_EXAM,
                priority=priority,
                confidence=round(rec_confidence, 2),
                rationale=f"Recommend checking physical finding/vital '{exam}' for: {', '.join(affected_diseases)}.",
                affected_diagnoses=affected_diseases,
                expected_information_gain=round(len(affected_diseases) * 1.0, 2),
                target_finding=exam,
                rule_sources=[f"examination.{exam_clean}.missing", f"workflow.stage.{current_stage}"]
            ))

        # Sort descending by utility
        def sort_key(r: ClinicalDecisionRecommendation) -> float:
            p_weight = {"critical": 10.0, "high": 5.0, "normal": 2.0, "low": 0.0}[r.priority.value]
            return r.expected_information_gain + p_weight

        recommendations.sort(key=sort_key, reverse=True)
        return recommendations


def is_vital_or_exam_finding(finding: str) -> bool:
    vitals = ["bp", "pulse", "temperature", "temp", "spo2", "weight", "height", "blood pressure", "heart rate", "oxygen saturation", "respiratory rate"]
    exams = ["throat", "auscultation", "lungs", "chest", "abdomen", "abdomen palpation", "neurological exam", "gait", "pupils", "heart sounds"]
    finding_clean = finding.lower().strip()
    return finding_clean in vitals or any(ex in finding_clean for ex in exams)
