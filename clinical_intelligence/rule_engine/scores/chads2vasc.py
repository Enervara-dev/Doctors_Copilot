from __future__ import annotations

"""
chads2vasc.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

from typing import Any, Dict, List, Optional, Set
from clinical_intelligence.rule_engine.scores import ClinicalScoreEvaluator
from clinical_intelligence.rule_engine.models import ClinicalScoreResult

class CHA2DS2VAScEvaluator(ClinicalScoreEvaluator):
    name = "CHA2DS2-VASc"

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
        # Required: Diagnosis of Atrial Fibrillation must be present in history or symptoms
        history_all = [h.lower() for h in medical_history] + [s.lower() for s in present_symptoms]
        has_af = any("atrial fibrillation" in item or "a-fib" in item or item == "af" for item in history_all)
        if not has_af or age is None or gender is None:
            return None

        evidence: List[str] = []
        score = 0

        # 1. Congestive Heart Failure / LV dysfunction
        has_chf = any(any(term in item for term in ["congestive heart failure", "heart failure", "chf", "left ventricular dysfunction", "lvd"]) for item in history_all)
        chf_pts = 1 if has_chf else 0
        score += chf_pts
        evidence.append(f"Congestive Heart Failure/LVD: {'Yes' if has_chf else 'No'} (+{chf_pts} pt)")

        # 2. Hypertension history
        has_htn = any(any(term in item for term in ["hypertension", "high blood pressure", "htn"]) for item in history_all)
        htn_pts = 1 if has_htn else 0
        score += htn_pts
        evidence.append(f"Hypertension History: {'Yes' if has_htn else 'No'} (+{htn_pts} pt)")

        # 3. Age criteria
        age_pts = 0
        if age >= 75:
            age_pts = 2
            evidence.append(f"Age {age} (>=75) (+2 pts)")
        elif 65 <= age <= 74:
            age_pts = 1
            evidence.append(f"Age {age} (65-74) (+1 pt)")
        else:
            evidence.append(f"Age {age} (<65) (+0 pts)")
        score += age_pts

        # 4. Diabetes History
        has_dm = any(any(term in item for term in ["diabetes", "diabetic", "type 2 diabetes", "dm2", "t2dm"]) for item in history_all)
        dm_pts = 1 if has_dm else 0
        score += dm_pts
        evidence.append(f"Diabetes History: {'Yes' if has_dm else 'No'} (+{dm_pts} pt)")

        # 5. Stroke / TIA / Thromboembolism history
        has_stroke = any(any(term in item for term in ["stroke", "tia", "transient ischemic attack", "thromboembolism", "ischemic stroke"]) for item in history_all)
        stroke_pts = 2 if has_stroke else 0
        score += stroke_pts
        evidence.append(f"Stroke/TIA/Thromboembolism History: {'Yes' if has_stroke else 'No'} (+{stroke_pts} pts)")

        # 6. Vascular disease (prior MI, PAD, or aortic plaque)
        has_vascular = any(any(term in item for term in ["myocardial infarction", "heart attack", "pad", "peripheral artery disease", "aortic plaque"]) for item in history_all)
        vasc_pts = 1 if has_vascular else 0
        score += vasc_pts
        evidence.append(f"Vascular Disease History: {'Yes' if has_vascular else 'No'} (+{vasc_pts} pt)")

        # 7. Sex Category (Female)
        is_female = gender.lower() in ["female", "woman", "f"]
        sex_pts = 1 if is_female else 0
        score += sex_pts
        evidence.append(f"Sex Category: Female ({'Yes' if is_female else 'No'}) (+{sex_pts} pt)")

        # Interpretation & Recommendations
        # Correct adjustment for gender-only score
        adjusted_score_for_tx = score
        if is_female and score == 1:
            adjusted_score_for_tx = 0  # Female with no other risk factors has low risk

        if adjusted_score_for_tx == 0:
            interpretation = f"Low Stroke Risk (Score {score})"
            recs = ["No oral anticoagulant therapy recommended. Low risk of thromboembolic events."]
        elif adjusted_score_for_tx == 1:
            interpretation = f"Moderate Stroke Risk (Score {score})"
            recs = ["Consider oral anticoagulation (DOAC preferred over Warfarin). Individualized decision."]
        else: # >= 2
            interpretation = f"High Stroke Risk (Score {score})"
            recs = ["Oral anticoagulation therapy is recommended (DOAC preferred over Warfarin) unless contraindicated."]

        return ClinicalScoreResult(
            name=self.name,
            score=float(score),
            interpretation=interpretation,
            evidence=evidence,
            recommendations=recs
        )
