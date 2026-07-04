from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
from clinical_intelligence.rule_engine.scores import ClinicalScoreEvaluator
from clinical_intelligence.rule_engine.models import ClinicalScoreResult

class NEWS2Evaluator(ClinicalScoreEvaluator):
    name = "NEWS2"

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
        # Required vitals. NEWS2 cannot be calculated if core parameters are missing.
        rr = vitals.get("respiration_rate") or vitals.get("respiratory_rate")
        spo2 = vitals.get("spo2")
        systolic_bp = vitals.get("bp_systolic")
        pulse = vitals.get("pulse") or vitals.get("heart_rate")
        temp = vitals.get("temperature")

        if rr is None or spo2 is None or systolic_bp is None or pulse is None or temp is None:
            return None

        evidence: List[str] = []
        total_score = 0

        # 1. Respiration Rate
        rr_score = 0
        if rr <= 8:
            rr_score = 3
        elif 9 <= rr <= 11:
            rr_score = 1
        elif 12 <= rr <= 20:
            rr_score = 0
        elif 21 <= rr <= 24:
            rr_score = 2
        else: # rr >= 25
            rr_score = 3
        total_score += rr_score
        evidence.append(f"Respiration Rate: {rr} bpm (+{rr_score} pts)")

        # 2. SpO2 Scale (Scale 2 for COPD patients, Scale 1 for normal)
        is_copd = any("copd" in h.lower() or "chronic obstructive pulmonary disease" in h.lower() for h in medical_history)
        
        # Check supplemental oxygen
        on_oxygen = any(
            any(term in x.lower() for term in ["supplemental oxygen", "nasal cannula", "o2 therapy", "oxygen therapy"])
            for x in current_medications + list(present_symptoms) + medical_history
        ) or state_data.get("vital_signs", {}).get("oxygen") in [True, "true", "True"]

        spo2_score = 0
        if is_copd:
            evidence.append("Using SpO2 Scale 2 (COPD patient history detected)")
            if spo2 <= 83:
                spo2_score = 3
            elif 84 <= spo2 <= 85:
                spo2_score = 2
            elif 86 <= spo2 <= 87:
                spo2_score = 1
            elif 88 <= spo2 <= 92:
                spo2_score = 0
            elif 93 <= spo2 <= 94:
                spo2_score = 1 if on_oxygen else 0
            elif 95 <= spo2 <= 96:
                spo2_score = 2 if on_oxygen else 0
            else: # spo2 >= 97
                spo2_score = 3 if on_oxygen else 0
        else:
            evidence.append("Using SpO2 Scale 1")
            if spo2 <= 91:
                spo2_score = 3
            elif 92 <= spo2 <= 93:
                spo2_score = 2
            elif 94 <= spo2 <= 95:
                spo2_score = 1
            else: # spo2 >= 96
                spo2_score = 0
        total_score += spo2_score
        evidence.append(f"SpO2: {spo2}% (+{spo2_score} pts)")

        # 3. Supplemental Oxygen
        o2_score = 2 if on_oxygen else 0
        total_score += o2_score
        evidence.append(f"Supplemental Oxygen: {'Yes' if on_oxygen else 'No'} (+{o2_score} pts)")

        # 4. Systolic Blood Pressure
        sbp_score = 0
        if systolic_bp <= 90:
            sbp_score = 3
        elif 91 <= systolic_bp <= 100:
            sbp_score = 2
        elif 101 <= systolic_bp <= 110:
            sbp_score = 1
        elif 111 <= systolic_bp <= 219:
            sbp_score = 0
        else: # systolic_bp >= 220
            sbp_score = 3
        total_score += sbp_score
        evidence.append(f"Systolic BP: {systolic_bp} mmHg (+{sbp_score} pts)")

        # 5. Pulse (Heart Rate)
        pulse_score = 0
        if pulse <= 40:
            pulse_score = 3
        elif 41 <= pulse <= 50:
            pulse_score = 1
        elif 51 <= pulse <= 90:
            pulse_score = 0
        elif 91 <= pulse <= 110:
            pulse_score = 1
        elif 111 <= pulse <= 130:
            pulse_score = 2
        else: # pulse >= 131
            pulse_score = 3
        total_score += pulse_score
        evidence.append(f"Pulse: {pulse} bpm (+{pulse_score} pts)")

        # 6. Consciousness (ACVPU)
        altered_consciousness_terms = [
            "confusion", "unconscious", "unresponsive", "somnolence", 
            "altered mental status", "altered mentation", "coma", "lethargy", "stupor"
        ]
        is_altered = any(any(term in sym.lower() for term in altered_consciousness_terms) for sym in present_symptoms)
        acvpu_score = 3 if is_altered else 0
        total_score += acvpu_score
        evidence.append(f"Consciousness Level: {'Altered/CVPU' if is_altered else 'Alert/A'} (+{acvpu_score} pts)")

        # 7. Temperature
        temp_score = 0
        if temp <= 35.0:
            temp_score = 3
        elif 35.1 <= temp <= 36.0:
            temp_score = 1
        elif 36.1 <= temp <= 38.0:
            temp_score = 0
        elif 38.1 <= temp <= 39.0:
            temp_score = 1
        else: # temp >= 39.1
            temp_score = 2
        total_score += temp_score
        evidence.append(f"Temperature: {temp}°C (+{temp_score} pts)")

        # Interpretation & Recommendations
        if total_score == 0:
            interpretation = "Low Risk (Score 0)"
            recs = ["Clinical monitoring frequency: 12-hourly. Continue routine ward care."]
        elif 1 <= total_score <= 4:
            interpretation = f"Low Risk (Score {total_score})"
            recs = ["Ward-based response. Clinical monitoring frequency: 4 to 6-hourly. Alert registered nurse."]
        elif 5 <= total_score <= 6 or any(s == 3 for s in [rr_score, spo2_score, sbp_score, pulse_score, acvpu_score, temp_score]):
            interpretation = f"Medium Risk (Score {total_score} or single parameter scoring 3)"
            recs = ["Urgent ward-based response / Acute team assessment. Clinical monitoring frequency: hourly. Escalation to medical team."]
        else:
            interpretation = f"High Risk (Score {total_score})"
            recs = ["Emergency/Critical Care response. Continuous monitoring of vitals. Active call to registrar/intensive care outreach team."]

        return ClinicalScoreResult(
            name=self.name,
            score=float(total_score),
            interpretation=interpretation,
            evidence=evidence,
            recommendations=recs
        )
