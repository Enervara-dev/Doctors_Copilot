from __future__ import annotations

"""
Inquiry Utility Planner

Purpose:
Calculates Clinical Utility Scores for missing cardinal and supportive symptoms, prioritizing high-value target questions.

Module/Component: Decision Support (CDSS)
"""

"""
Inquiry Utility Planner

Purpose:
Calculates Clinical Utility Scores for missing cardinal and supportive symptoms, prioritizing high-value target questions.

Module/Component: Decision Support (CDSS)
"""

"""
Inquiry Utility Planner

Purpose:
Calculates Clinical Utility Scores for missing cardinal and supportive symptoms, prioritizing high-value target questions.

Module/Component: Decision Support (CDSS)
"""

import logging
from typing import Any, Dict, List
from clinical_intelligence.interview.evidence_gap_analyzer import DiseaseGap
from clinical_intelligence.interview.question_memory import QuestionMemoryManager
from clinical_intelligence.rule_engine.models import ClinicalDecisionRecommendation, DecisionAction, ActionPriority

logger = logging.getLogger(__name__)

"""
Question Planner.

Created for Module 7 (Adaptive Clinical Decision Support & Consultation Intelligence).
This module dynamically scores and schedules missing symptom inquiries using clinical utility
calculations, prioritizing highly discriminative, red-flag relevant, and workflow-appropriate
cardinal and supportive symptoms.
"""

# Predefined clean natural questions mapping for standard clinical symptoms
CLINICAL_QUESTIONS_MAP = {
    "fever": "Have you had a fever or elevated temperature?",
    "cough": "Are you experiencing a cough?",
    "headache": "Do you have a headache?",
    "chest pain": "Are you experiencing any chest pain or pressure?",
    "shortness of breath": "Are you feeling short of breath?",
    "chills": "Have you had any chills?",
    "body aches": "Are you experiencing body aches or muscle pain?",
    "fatigue": "Have you been feeling fatigued or unusually tired?",
    "sputum": "Are you coughing up any sputum or phlegm?",
    "sore throat": "Do you have a sore throat?",
    "congestion": "Are you experiencing nasal congestion or a runny nose?",
    "rhinorrhea": "Do you have a runny nose?",
    "dyspnea": "Are you experiencing shortness of breath?",
    "nausea": "Have you been feeling nauseous?",
    "vomiting": "Have you had any vomiting?",
    "dizziness": "Are you experiencing any dizziness or lightheadedness?",
    "sweating": "Have you been sweating heavily?",
    "diaphoresis": "Have you been experiencing heavy sweating?",
}


class QuestionPlanner:
    """Identifies missing subjective/HPI symptoms and plans natural clinical questions prioritized by utility."""

    def __init__(self, memory_manager: QuestionMemoryManager = None) -> None:
        self.memory_manager = memory_manager or QuestionMemoryManager()

    def plan_recommendations(
        self,
        gaps: List[DiseaseGap],
        session: Any,
        candidate_diagnoses: List[Any],
        current_stage: str
    ) -> List[ClinicalDecisionRecommendation]:
        
        recommendations: List[ClinicalDecisionRecommendation] = []
        
        # 1. Gather all unique missing symptoms across gaps
        # (Filter out vitals, examination findings, and modifier keys if they aren't subjective symptoms)
        missing_symptoms: Dict[str, List[str]] = {}  # symptom -> list of disease names affected
        
        # Let's inspect active candidates to prioritize cardinal symptoms of high-confidence candidates
        top_disease_id = None
        if candidate_diagnoses:
            top_disease_id = getattr(candidate_diagnoses[0], "disease_id", None) or candidate_diagnoses[0].get("disease_id")

        # In gaps: missing_findings is a list of findings from profiles
        for gap in gaps:
            for finding in gap.missing_findings:
                # We prioritize symptoms (lowercase string matching standard symptoms or not vital/exam)
                finding_clean = finding.lower().strip()
                if is_vital_or_exam(finding_clean):
                    continue
                missing_symptoms.setdefault(finding_clean, []).append(gap.disease_name)

        # 2. Score each unique missing symptom
        for sym, affected_diseases in missing_symptoms.items():
            if self.memory_manager.is_blocked(sym, session):
                continue

            # Clinical Utility calculation
            # a) Discrimination: how many candidate diagnoses are affected
            discrimination = len(affected_diseases) * 1.5
            
            # b) Red flag importance
            red_flag_score = 0.0
            if sym in ["chest pain", "shortness of breath", "dyspnea", "sudden numbness", "facial droop", "slurred speech", "vision loss", "confusion"]:
                red_flag_score = 5.0
                
            # c) Workflow stage alignment (subjective history matches HPI/Chief Complaint)
            workflow_score = 3.0 if current_stage in ["Chief Complaint", "HPI"] else 0.5
            
            # d) Critical Evidence: cardinal symptom for top differential
            critical_score = 0.0
            for gap in gaps:
                if gap.disease_id == top_disease_id and sym in gap.missing_findings:
                    critical_score = 4.0
                    break
                    
            # e) Memory penalty (previously deferred)
            memory_penalty = 0.0
            memory = getattr(session, "consultation_memory", {}) or {}
            deferred_list = memory.get("deferred", []) or []
            if any(sym == d.lower().strip() for d in deferred_list):
                memory_penalty = 1.5

            utility_score = discrimination + red_flag_score + workflow_score + critical_score - memory_penalty
            
            # Action priority mapping
            if utility_score >= 8.0:
                priority = ActionPriority.CRITICAL if red_flag_score > 0 else ActionPriority.HIGH
            elif utility_score >= 4.0:
                priority = ActionPriority.NORMAL
            else:
                priority = ActionPriority.LOW
                
            # Formulate the natural question
            question_text = CLINICAL_QUESTIONS_MAP.get(sym, f"Have you experienced any {sym}?")
            
            # Confidence in recommendation is derived from utility/discrimination
            rec_confidence = min(0.99, max(0.1, utility_score / 12.0))

            recommendations.append(ClinicalDecisionRecommendation(
                action=DecisionAction.ASK_QUESTION,
                priority=priority,
                confidence=round(rec_confidence, 2),
                rationale=f"Queries missing symptom '{sym}' to evaluate or distinguish: {', '.join(affected_diseases)}.",
                affected_diagnoses=affected_diseases,
                expected_information_gain=round(discrimination, 2),
                target_finding=sym,
                rule_sources=[f"symptom.{sym}.missing", f"workflow.stage.{current_stage}"]
            ))

        # Sort descending by utility (expected_information_gain + priority weight)
        def sort_key(r: ClinicalDecisionRecommendation) -> float:
            p_weight = {"critical": 10.0, "high": 5.0, "normal": 2.0, "low": 0.0}[r.priority.value]
            return r.expected_information_gain + p_weight

        recommendations.sort(key=sort_key, reverse=True)
        return recommendations


def is_vital_or_exam(finding: str) -> bool:
    vitals = ["bp", "pulse", "temperature", "temp", "spo2", "weight", "height", "blood pressure", "heart rate", "oxygen saturation", "respiratory rate"]
    exams = ["throat", "auscultation", "lungs", "chest", "abdomen", "abdomen palpation", "neurological exam", "gait", "pupils", "heart sounds"]
    finding_clean = finding.lower().strip()
    return finding_clean in vitals or any(ex in finding_clean for ex in exams)
