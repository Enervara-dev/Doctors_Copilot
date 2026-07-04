from __future__ import annotations

"""
question_planner.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging
from typing import List, Dict, Any
from clinical_intelligence.evidence.evidence_models import DiseaseProfile, EvidenceResult
from clinical_intelligence.rule_engine.models import AdaptiveQuestionItem

logger = logging.getLogger(__name__)

class AdaptiveQuestionPlanner:
    """
    Deterministic Question Planner.
    Uses missing evidence from Candidate Diagnoses to generate targeted clinical questions
    for the clinician.
    """

    def plan_questions(
        self,
        ranked_candidates: List[Any],
        disease_profiles: Dict[str, DiseaseProfile]
    ) -> List[AdaptiveQuestionItem]:
        questions: List[AdaptiveQuestionItem] = []

        # Only process top 3 highest confidence candidate diagnoses
        for candidate in ranked_candidates[:3]:
            disease_id = candidate.disease_id
            profile = disease_profiles.get(disease_id)
            if not profile:
                continue

            # Gather missing cardinal symptoms
            missing_syms = candidate.missing_symptoms

            for sym in missing_syms[:2]:  # Limit questions per disease to prevent overload
                sym_clean = sym.lower().strip()
                
                # Question templates
                if "pain" in sym_clean:
                    q_text = f"Are you experiencing any {sym_clean}?"
                elif sym_clean in ["fever", "cough", "shortness of breath", "sweating", "dizziness", "nausea"]:
                    q_text = f"Do you have a history of or current {sym_clean}?"
                else:
                    q_text = f"Have you experienced {sym_clean} recently?"

                questions.append(AdaptiveQuestionItem(
                    question=q_text,
                    purpose=f"Evaluate missing cardinal symptom for suspected '{candidate.name}'",
                    related_diagnosis=candidate.name,
                    priority="High" if candidate.confidence >= 0.75 else "Medium",
                    clinical_importance=f"Cardinal indicator for {candidate.name}"
                ))

        return questions
