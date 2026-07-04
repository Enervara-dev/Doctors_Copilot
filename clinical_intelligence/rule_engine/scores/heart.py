from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set
from clinical_intelligence.rule_engine.scores import ClinicalScoreEvaluator
from clinical_intelligence.rule_engine.models import ClinicalScoreResult

class HEARTEvaluator(ClinicalScoreEvaluator):
    name = "HEART Score"

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
        # Required: Chest pain must be present
        has_chest_pain = any("chest pain" in s.lower() or "chest discomfort" in s.lower() for s in present_symptoms) or \
                         any("chest pain" in cc.lower() for cc in state_data.get("chief_complaint", []))
        if not has_chest_pain or age is None:
            return None

        # Look for ECG and Troponin in state data / labs / vitals to ensure we don't guess
        ecg_raw = state_data.get("ecg") or state_data.get("ecg_findings") or state_data.get("vital_signs", {}).get("ecg")
        troponin_raw = (
            state_data.get("troponin") or 
            state_data.get("labs", {}).get("troponin") or 
            state_data.get("lab_results", {}).get("troponin") or 
            state_data.get("vital_signs", {}).get("troponin")
        )

        if ecg_raw is None or troponin_raw is None:
            return None

        evidence: List[str] = []
        score = 0

        # 1. History (H)
        # Highly suspicious: typical characteristics (crushing, radiating, heavy) + sweating/nausea
        # Moderately suspicious: chest pain present without specific atypical features
        # Slightly suspicious: chest pain atypical
        chest_pain_chars = ["radiating", "crushing", "heaviness", "squeezing", "retrosternal", "pressure"]
        has_typical_chars = any(any(c in sym.lower() for c in chest_pain_chars) for sym in present_symptoms)
        has_associated = any(any(s in sym.lower() for s in ["sweating", "diaphoresis", "shortness of breath", "dyspnea"]) for sym in present_symptoms)

        h_pt = 0
        if has_typical_chars and has_associated:
            h_pt = 2
            evidence.append("History: Highly suspicious typical cardiac pain (+2 pts)")
        elif has_typical_chars or has_associated:
            h_pt = 1
            evidence.append("History: Moderately suspicious cardiac pain (+1 pt)")
        else:
            h_pt = 0
            evidence.append("History: Slightly/non-suspicious cardiac pain (+0 pts)")
        score += h_pt

        # 2. ECG (E)
        # Significant ST-deviation: 2 points
        # Nonspecific repolarisation disturbance / LBBB / LVH: 1 point
        # Normal: 0 points
        ecg_str = str(ecg_raw).lower()
        e_pt = 0
        if any(term in ecg_str for term in ["st elevation", "st depression", "st-elevation", "st-depression", "st deviation"]):
            e_pt = 2
            evidence.append(f"ECG: Significant ST deviation: '{ecg_raw}' (+2 pts)")
        elif any(term in ecg_str for term in ["lbbb", "lvh", "repolarization disturbance", "t wave inversion", "nonspecific"]):
            e_pt = 1
            evidence.append(f"ECG: Nonspecific disturbance / LBBB: '{ecg_raw}' (+1 pt)")
        else:
            e_pt = 0
            evidence.append("ECG: Normal (+0 pts)")
        score += e_pt

        # 3. Age (A)
        a_pt = 0
        if age >= 65:
            a_pt = 2
        elif 45 <= age <= 64:
            a_pt = 1
        else:
            a_pt = 0
        score += a_pt
        evidence.append(f"Age: {age} years (+{a_pt} pts)")

        # 4. Risk Factors (R)
        # Risk factors count: diabetes, hypertension, hyperlipidemia, smoking, family history of CAD, obesity
        # Prior vascular disease (CAD, MI, stroke, PCI, CABG) -> automatic 2 points
        risk_count = 0
        rf_terms = ["hypertension", "diabetes", "hyperlipidemia", "high cholesterol", "smoking", "smoker", "obesity", "family history of coronary", "family history of cad"]
        for term in rf_terms:
            if any(term in item.lower() for item in medical_history + current_medications + list(present_symptoms)):
                risk_count += 1

        has_prior_cad = any(any(term in item.lower() for term in ["cad", "coronary artery", "myocardial infarction", "heart attack", "pci", "cabg", "stroke"]) for item in medical_history)

        r_pt = 0
        if has_prior_cad or risk_count >= 3:
            r_pt = 2
            evidence.append(f"Risk Factors: Prior CAD/Vascular disease or >=3 risk factors (RF count: {risk_count}) (+2 pts)")
        elif 1 <= risk_count <= 2:
            r_pt = 1
            evidence.append(f"Risk Factors: 1-2 risk factors (RF count: {risk_count}) (+1 pt)")
        else:
            r_pt = 0
            evidence.append("Risk Factors: No risk factors detected (+0 pts)")
        score += r_pt

        # 5. Troponin (T)
        # >= 3x normal: 2 points
        # 1-3x normal: 1 point
        # <= normal / normal: 0 points
        trop_str = str(troponin_raw).lower()
        t_pt = 0
        if "elevated" in trop_str or "positive" in trop_str or "high" in trop_str:
            if "highly" in trop_str or "3x" in trop_str or "critically" in trop_str:
                t_pt = 2
                evidence.append(f"Troponin: Highly elevated (>=3x limit): '{troponin_raw}' (+2 pts)")
            else:
                t_pt = 1
                evidence.append(f"Troponin: Elevated (1-3x limit): '{troponin_raw}' (+1 pt)")
        else:
            t_pt = 0
            evidence.append("Troponin: Normal (+0 pts)")
        score += t_pt

        # Interpretation & Recommendations
        if score <= 3:
            interpretation = f"Low Risk (Score {score})"
            recs = ["Major Adverse Cardiac Event (MACE) risk is low (<1.7%). Discharged home may be considered."]
        elif 4 <= score <= 6:
            interpretation = f"Moderate Risk (Score {score})"
            recs = ["MACE risk is moderate (12-17.6%). Inpatient admission and clinical observation/testing recommended."]
        else: # score >= 7
            interpretation = f"High Risk (Score {score})"
            recs = ["MACE risk is high (>50%). Immediate invasive treatment/evaluation (coronary angiography) recommended."]

        return ClinicalScoreResult(
            name=self.name,
            score=float(score),
            interpretation=interpretation,
            evidence=evidence,
            recommendations=recs
        )
