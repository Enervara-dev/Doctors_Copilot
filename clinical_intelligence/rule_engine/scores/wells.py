from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
from clinical_intelligence.rule_engine.scores import ClinicalScoreEvaluator
from clinical_intelligence.rule_engine.models import ClinicalScoreResult

class WellsEvaluator(ClinicalScoreEvaluator):
    name = "Wells Score (PE)"

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
        pulse = vitals.get("pulse") or vitals.get("heart_rate")
        if pulse is None:
            return None

        # Check for dyspnea or pleuritic chest pain to warrant running the Wells score
        has_pe_trigger = any(term in s.lower() for s in present_symptoms for term in ["dyspnea", "shortness of breath", "chest pain", "pleuritic chest pain"])
        if not has_pe_trigger:
            return None

        evidence: List[str] = []
        score = 0.0

        # 1. Clinical signs/symptoms of DVT (leg swelling, pain with palpation)
        dvt_terms = ["leg swelling", "leg pain", "dvt symptoms", "calf swelling", "calf tenderness", "deep vein thrombosis"]
        has_dvt = any(any(term in item.lower() for term in dvt_terms) for item in list(present_symptoms) + medical_history)
        dvt_pts = 3.0 if has_dvt else 0.0
        score += dvt_pts
        evidence.append(f"Clinical signs/symptoms of DVT: {'Yes' if has_dvt else 'No'} (+{dvt_pts} pts)")

        # 2. PE is #1 diagnosis or equally likely
        # True if both chest pain and dyspnea are present without typical cardiac signs, or PE is explicitly mentioned
        has_dyspnea = any("dyspnea" in s.lower() or "shortness of breath" in s.lower() for s in present_symptoms)
        has_chest_pain = any("chest pain" in s.lower() for s in present_symptoms)
        pe_likely = (has_dyspnea and has_chest_pain) or any("pulmonary embolism" in item.lower() or "pe" == item.lower() for item in state_data.get("diagnosis_tentative", []))
        pe_pts = 3.0 if pe_likely else 0.0
        score += pe_pts
        evidence.append(f"PE is #1 diagnosis or equally likely: {'Yes' if pe_likely else 'No'} (+{pe_pts} pts)")

        # 3. Heart rate > 100 bpm
        hr_pts = 1.5 if pulse > 100 else 0.0
        score += hr_pts
        evidence.append(f"Heart Rate: {pulse} bpm ({'+1.5 pts' if hr_pts else '0 pts'})")

        # 4. Immobilization >= 3 days or surgery in previous 4 weeks
        immobilization_terms = ["immobilization", "bed rest", "paralysis", "surgery", "postoperative", "post-operative", "hip fracture", "knee replacement"]
        has_immobilization = any(any(term in item.lower() for term in immobilization_terms) for item in medical_history + list(present_symptoms))
        immob_pts = 1.5 if has_immobilization else 0.0
        score += immob_pts
        evidence.append(f"Immobilization or recent surgery: {'Yes' if has_immobilization else 'No'} (+{immob_pts} pts)")

        # 5. Previous PE or DVT
        prior_pe_dvt = any(any(term in item.lower() for term in ["prior pe", "prior dvt", "previous pe", "previous dvt", "history of pe", "history of dvt"]) for item in medical_history)
        prior_pts = 1.5 if prior_pe_dvt else 0.0
        score += prior_pts
        evidence.append(f"Previous PE/DVT: {'Yes' if prior_pe_dvt else 'No'} (+{prior_pts} pts)")

        # 6. Hemoptysis
        has_hemoptysis = any("hemoptysis" in s.lower() or "coughing blood" in s.lower() or "coughing up blood" in s.lower() for s in present_symptoms)
        hemo_pts = 1.0 if has_hemoptysis else 0.0
        score += hemo_pts
        evidence.append(f"Hemoptysis: {'Yes' if has_hemoptysis else 'No'} (+{hemo_pts} pts)")

        # 7. Malignancy
        has_malignancy = any(any(term in item.lower() for term in ["cancer", "malignancy", "chemotherapy", "metastasis", "lymphoma", "sarcoma", "tumor"]) for item in medical_history)
        mal_pts = 1.0 if has_malignancy else 0.0
        score += mal_pts
        evidence.append(f"Malignancy: {'Yes' if has_malignancy else 'No'} (+{mal_pts} pts)")

        # Interpretation & Recommendations
        # 3-tier model
        if score < 2.0:
            interpretation = f"Low Clinical Probability (Score {score})"
            recs = ["Consider D-dimer test to rule out Pulmonary Embolism. If negative, PE can be safely excluded."]
        elif 2.0 <= score <= 6.0:
            interpretation = f"Moderate Clinical Probability (Score {score})"
            recs = ["Perform high-sensitivity D-dimer test or proceed directly to CT Pulmonary Angiography (CTPA)."]
        else: # score > 6.0
            interpretation = f"High Clinical Probability (Score {score})"
            recs = ["Proceed directly to CT Pulmonary Angiography (CTPA) or V/Q Scan. Consider empiric anticoagulation if no contraindications."]

        return ClinicalScoreResult(
            name=self.name,
            score=score,
            interpretation=interpretation,
            evidence=evidence,
            recommendations=recs
        )
