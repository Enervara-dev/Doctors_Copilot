"""
Clinical State Schema Validator

Purpose:
Validates the integrity, types, and schema boundaries of ClinicalState objects to prevent downstream reasoning errors.

Module/Component: Clinical Ingestion
"""

"""
Clinical State Schema Validator

Purpose:
Validates the integrity, types, and schema boundaries of ClinicalState objects to prevent downstream reasoning errors.

Module/Component: Clinical Ingestion
"""

import time
from typing import Dict, Any, List
from app.services.clinical.models import ClinicalState, SymptomEntity, VitalObservation

class ClinicalStateValidator:
    @staticmethod
    def validate(state: ClinicalState) -> ClinicalState:
        """
        Validates, cleans, and deduplicates the clinical state data in place.
        Records all state modifications and conflict resolutions in the state's audit_trail.
        """
        # 1. Deduplicate Symptoms with Conflict Resolution Order
        symptoms_map: Dict[str, Dict[str, Any]] = {}
        for sym in state.symptoms:
            name = sym.get("name", "").strip().lower()
            if not name:
                continue
            
            if name not in symptoms_map:
                symptoms_map[name] = sym
            else:
                existing = symptoms_map[name]
                old_status = existing.get("status")
                new_status = sym.get("status")
                
                # Deterministic Conflict Resolution:
                # 1. Negation (highest priority)
                if old_status == "Negated" or new_status == "Negated":
                    resolved_status = "Negated"
                    reason = "Negation conflict resolution priority"
                # 2. Explicit patient confirmation / active status
                elif new_status == "Active":
                    resolved_status = "Active"
                    reason = "Patient confirmation priority"
                else:
                    resolved_status = old_status
                    reason = "Retained existing status"
                
                if old_status != resolved_status:
                    existing["status"] = resolved_status
                    log_entry = {
                        "timestamp": time.time(),
                        "action": "resolve_symptom_conflict",
                        "field": f"symptoms.{name}.status",
                        "old_value": old_status,
                        "new_value": resolved_status,
                        "reason": reason
                    }
                    state.audit_trail.append(log_entry)
                
                # Merge duration/severity if not present in existing
                if sym.get("duration") and not existing.get("duration"):
                    existing["duration"] = sym["duration"]
                if sym.get("severity") and not existing.get("severity"):
                    existing["severity"] = sym["severity"]
                if sym.get("confidence", 0.0) > existing.get("confidence", 0.0):
                    existing["confidence"] = sym["confidence"]
                
                # Merge custom temporal features
                for key in ["onset", "progression", "frequency", "character", "source_text", "utterance_id", "speaker"]:
                    if sym.get(key) and not existing.get(key):
                        existing[key] = sym[key]

        state.symptoms = list(symptoms_map.values())

        # 2. Deduplicate Current Medications (reported by patient/attender)
        seen_meds = set()
        cleaned_meds = []
        for med in state.current_medications:
            m_clean = med.strip()
            if not m_clean:
                continue
            m_lower = m_clean.lower()
            # Handle potential formats e.g. "reported: Tylenol, normalized: Paracetamol"
            if m_lower not in seen_meds:
                seen_meds.add(m_lower)
                cleaned_meds.append(m_clean)
            else:
                state.audit_trail.append({
                    "timestamp": time.time(),
                    "action": "remove_duplicate_medication",
                    "field": "current_medications",
                    "value": m_clean,
                    "reason": "Deduplication"
                })
        state.current_medications = cleaned_meds

        # Deduplicate medicines in TreatmentPlan
        seen_tx_meds = set()
        cleaned_tx_meds = []
        for med_dict in state.treatment_plan.medicines:
            name = med_dict.get("name", "").strip().lower()
            if not name:
                continue
            if name not in seen_tx_meds:
                seen_tx_meds.add(name)
                cleaned_tx_meds.append(med_dict)
        state.treatment_plan.medicines = cleaned_tx_meds

        # 3. Clean list fields (demographics / empty fields)
        state.chief_complaint = [c.strip() for c in state.chief_complaint if c and c.strip()]
        state.medical_history = [m.strip() for m in state.medical_history if m and m.strip()]
        state.allergies = [a.strip() for a in state.allergies if a and a.strip()]

        if state.patient_info.age is not None:
            if state.patient_info.age < 0 or state.patient_info.age > 150:
                old_age = state.patient_info.age
                state.patient_info.age = None
                state.audit_trail.append({
                    "timestamp": time.time(),
                    "action": "invalidate_age",
                    "field": "patient_info.age",
                    "old_value": old_age,
                    "new_value": None,
                    "reason": "Out of bounds age (0-150)"
                })

        # 4. Validate Vitals Observations
        vitals_fields = ["bp", "pulse", "temperature", "spo2", "weight", "height"]
        for field in vitals_fields:
            obs = getattr(state.vital_signs, field)
            if obs:
                # If empty value and empty qualitative, discard
                if not obs.value and not obs.qualitative:
                    setattr(state.vital_signs, field, None)
                    state.audit_trail.append({
                        "timestamp": time.time(),
                        "action": "discard_empty_vital",
                        "field": f"vital_signs.{field}",
                        "reason": "Both value and qualitative were empty"
                    })
                # Check for temperature out-of-bounds checks
                elif field == "temperature" and obs.value:
                    try:
                        t_val = float(obs.value)
                        # Normal human body temp is roughly 25-45C or 75-115F
                        if not (25.0 <= t_val <= 45.0) and not (75.0 <= t_val <= 115.0):
                            setattr(state.vital_signs, field, None)
                            state.audit_trail.append({
                                "timestamp": time.time(),
                                "action": "invalidate_temperature",
                                "field": "vital_signs.temperature",
                                "old_value": obs.value,
                                "new_value": None,
                                "reason": "Temperature value out of physiological bounds"
                            })
                    except ValueError:
                        pass

        return state
