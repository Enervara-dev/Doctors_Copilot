from __future__ import annotations

"""
curb65.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import re
from typing import Any, Dict, List, Optional, Set
from clinical_intelligence.rule_engine.scores import ClinicalScoreEvaluator
from clinical_intelligence.rule_engine.models import ClinicalScoreResult

class CURB65Evaluator(ClinicalScoreEvaluator):
    name = "CURB-65"

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
        # Required parameters
        rr = vitals.get("respiration_rate") or vitals.get("respiratory_rate")
        sbp = vitals.get("bp_systolic")
        dbp = vitals.get("bp_diastolic")

        if rr is None or sbp is None or dbp is None or age is None:
            return None

        # BUN / Urea lookup
        bun_raw = (
            state_data.get("vital_signs", {}).get("bun") or 
            state_data.get("labs", {}).get("bun") or 
            state_data.get("lab_results", {}).get("bun") or
            state_data.get("bun") or
            state_data.get("vital_signs", {}).get("urea") or
            state_data.get("labs", {}).get("urea") or
            state_data.get("lab_results", {}).get("urea") or
            state_data.get("urea")
        )
        if bun_raw is None:
            return None

        # Convert BUN/Urea to float
        try:
            match = re.search(r'(\d+(?:\.\d+)?)', str(bun_raw))
            val = float(match.group(1)) if match else float(bun_raw)
        except (ValueError, TypeError):
            return None

        evidence: List[str] = []
        score = 0

        # 1. Confusion (new onset)
        altered_consciousness_terms = [
            "confusion", "unconscious", "unresponsive", "somnolence", 
            "altered mental status", "altered mentation"
        ]
        is_confused = any(any(term in sym.lower() for term in altered_consciousness_terms) for sym in present_symptoms)
        c_pt = 1 if is_confused else 0
        score += c_pt
        evidence.append(f"Confusion: {'Yes' if is_confused else 'No'} ({'+1 pt' if c_pt else '0 pts'})")

        # 2. Urea / BUN > 19 mg/dL (or > 7 mmol/L)
        is_high_urea = False
        if val > 19.0:
            is_high_urea = True
        elif val > 7.0:
            is_high_urea = True

        u_pt = 1 if is_high_urea else 0
        score += u_pt
        evidence.append(f"Urea/BUN: {val} ({'+1 pt' if u_pt else '0 pts'})")

        # 3. Respiratory Rate >= 30 bpm
        r_pt = 1 if rr >= 30 else 0
        score += r_pt
        evidence.append(f"Respiration Rate: {rr} bpm ({'+1 pt' if r_pt else '0 pts'})")

        # 4. Blood Pressure: SBP < 90 or DBP <= 60 mmHg
        b_pt = 1 if (sbp < 90 or dbp <= 60) else 0
        score += b_pt
        evidence.append(f"Blood Pressure: {sbp}/{dbp} mmHg ({'+1 pt' if b_pt else '0 pts'})")

        # 5. Age >= 65 years
        age_pt = 1 if age >= 65 else 0
        score += age_pt
        evidence.append(f"Age: {age} years ({'+1 pt' if age_pt else '0 pts'})")

        # Interpretation & Recommendations
        if score <= 1:
            interpretation = f"Low Mortality Risk (Score {score})"
            recs = ["Outpatient management is usually suitable. 30-day mortality risk is low (<3.0%)."]
        elif score == 2:
            interpretation = f"Moderate Mortality Risk (Score {score})"
            recs = ["Consider short-stay inpatient admission or close outpatient monitoring. Mortality risk ~9.2%."]
        else: # score >= 3
            interpretation = f"High Mortality Risk (Score {score})"
            recs = [
                "Urgent inpatient admission required. Mortality risk is high (15-22%).",
                f"If score is {score} (4-5), assess immediately for intensive care unit (ICU) admission."
            ]

        return ClinicalScoreResult(
            name=self.name,
            score=float(score),
            interpretation=interpretation,
            evidence=evidence,
            recommendations=recs
        )
