from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
from clinical_intelligence.rule_engine.scores import ClinicalScoreEvaluator
from clinical_intelligence.rule_engine.models import ClinicalScoreResult

class TIMIEvaluator(ClinicalScoreEvaluator):
    name = "TIMI"

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
        # Required: Chest pain / suspected cardiac event
        has_chest_pain = any("chest pain" in s.lower() or "chest discomfort" in s.lower() for s in present_symptoms) or \
                         any("chest pain" in cc.lower() for cc in state_data.get("chief_complaint", []))
        if not has_chest_pain or age is None:
            return None

        # Look for ECG and Troponin to ensure we don't guess
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

        # 1. Age >= 65
        age_pt = 1 if age >= 65 else 0
        score += age_pt
        evidence.append(f"Age: {age} years ({'+1 pt' if age_pt else '0 pts'})")

        # 2. >= 3 CAD Risk Factors
        risk_count = 0
        rf_terms = ["hypertension", "diabetes", "hyperlipidemia", "high cholesterol", "smoking", "smoker", "obesity", "family history of coronary", "family history of cad"]
        for term in rf_terms:
            if any(term in item.lower() for item in medical_history + current_medications + list(present_symptoms)):
                risk_count += 1
        rf_pt = 1 if risk_count >= 3 else 0
        score += rf_pt
        evidence.append(f"Vascular Risk Factors: Count is {risk_count} ({'+1 pt' if rf_pt else '0 pts'})")

        # 3. Prior coronary stenosis >= 50% (prior CAD/MI/revascularization history)
        has_prior_cad = any(any(term in item.lower() for term in ["cad", "coronary artery", "myocardial infarction", "heart attack", "pci", "cabg"]) for item in medical_history)
        cad_pt = 1 if has_prior_cad else 0
        score += cad_pt
        evidence.append(f"Prior CAD history (stenosis >=50%): {'Yes' if has_prior_cad else 'No'} ({'+1 pt' if cad_pt else '0 pts'})")

        # 4. Aspirin use in last 7 days
        has_aspirin = any("aspirin" in item.lower() for item in current_medications)
        asa_pt = 1 if has_aspirin else 0
        score += asa_pt
        evidence.append(f"Aspirin use in last 7 days: {'Yes' if has_aspirin else 'No'} ({'+1 pt' if asa_pt else '0 pts'})")

        # 5. Severe angina (>= 2 episodes in 24 hours)
        has_recurrent = any("recurrent" in s.lower() or "multiple" in s.lower() or "angina" in s.lower() for s in present_symptoms)
        ang_pt = 1 if has_recurrent else 0
        score += ang_pt
        evidence.append(f"Severe Angina (recurrent episodes): {'Yes' if has_recurrent else 'No'} ({'+1 pt' if ang_pt else '0 pts'})")

        # 6. ST-segment deviation >= 0.5 mm
        ecg_str = str(ecg_raw).lower()
        ecg_pt = 1 if any(term in ecg_str for term in ["st elevation", "st depression", "st-elevation", "st-depression", "st deviation"]) else 0
        score += ecg_pt
        evidence.append(f"ECG ST-segment deviation >= 0.5mm: {'Yes' if ecg_pt else 'No'} ({'+1 pt' if ecg_pt else '0 pts'})")

        # 7. Elevated cardiac markers (Troponin positive)
        trop_str = str(troponin_raw).lower()
        trop_pt = 1 if ("elevated" in trop_str or "positive" in trop_str or "high" in trop_str) else 0
        score += trop_pt
        evidence.append(f"Elevated Cardiac Markers (Troponin): {'Yes' if trop_pt else 'No'} ({'+1 pt' if trop_pt else '0 pts'})")

        # Interpretation & Recommendations
        # TIMI Score 0-2 (Low risk), 3-4 (Intermediate), 5-7 (High)
        if score <= 2:
            interpretation = f"Low Risk (Score {score})"
            recs = ["Low risk of 14-day MACE (<8.3%). Consider outpatient workup or medical management."]
        elif 3 <= score <= 4:
            interpretation = f"Intermediate Risk (Score {score})"
            recs = ["Intermediate risk of 14-day MACE (13-20%). Admit for observation and serial biomarkers."]
        else: # score >= 5
            interpretation = f"High Risk (Score {score})"
            recs = ["High risk of 14-day MACE (>26%). Prompt invasive strategy (angiography) and intensive medical management recommended."]

        return ClinicalScoreResult(
            name=self.name,
            score=float(score),
            interpretation=interpretation,
            evidence=evidence,
            recommendations=recs
        )
