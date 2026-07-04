from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
from clinical_intelligence.rule_engine.scores import ClinicalScoreEvaluator
from clinical_intelligence.rule_engine.models import ClinicalScoreResult

class qSOFAEvaluator(ClinicalScoreEvaluator):
    name = "qSOFA"

    def calculate(
        self,
        *,
        state_data: Dict[str, Any],
        vitals: Dict[str, Optional[float]],
        present_symptoms: Set[str],
        negated_symptoms: Set[str],
        medical_history: List[str],
        current_medications: List[str],
        allergies: List[str],
        age: Optional[int],
        gender: Optional[str]
    ) -> Optional[ClinicalScoreResult]:
        systolic_bp = vitals.get("bp_systolic")
        rr = vitals.get("respiration_rate") or vitals.get("respiratory_rate")

        if systolic_bp is None or rr is None:
            return None

        evidence: List[str] = []
        score = 0

        # 1. Respiration Rate >= 22
        rr_pt = 1 if rr >= 22 else 0
        score += rr_pt
        evidence.append(f"Respiration Rate: {rr} bpm ({'+1 pt' if rr_pt else '0 pts'})")

        # 2. Systolic BP <= 100
        sbp_pt = 1 if systolic_bp <= 100 else 0
        score += sbp_pt
        evidence.append(f"Systolic BP: {systolic_bp} mmHg ({'+1 pt' if sbp_pt else '0 pts'})")

        # 3. Altered Mentation
        altered_consciousness_terms = [
            "confusion", "unconscious", "unresponsive", "somnolence", 
            "altered mental status", "altered mentation", "coma", "lethargy", "stupor"
        ]
        is_altered = any(any(term in sym.lower() for term in altered_consciousness_terms) for sym in present_symptoms)
        ment_pt = 1 if is_altered else 0
        score += ment_pt
        evidence.append(f"Altered Mentation: {'Yes' if is_altered else 'No'} ({'+1 pt' if ment_pt else '0 pts'})")

        if score >= 2:
            interpretation = f"High Sepsis Risk (Score {score})"
            recs = [
                "Evaluate for sepsis and organ dysfunction.",
                "Measure lactate and search for source of infection.",
                "Consider intensive care monitoring or escalation."
            ]
        else:
            interpretation = f"Low Sepsis Risk (Score {score})"
            recs = ["Low risk of death or prolonged ICU stay. Continue clinical monitoring as indicated."]

        return ClinicalScoreResult(
            name=self.name,
            score=float(score),
            interpretation=interpretation,
            evidence=evidence,
            recommendations=recs
        )
