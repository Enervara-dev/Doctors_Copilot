from __future__ import annotations

"""
Evidence Gaps Analyzer

Purpose:
Matches active clinical state against compiled disease profiles to classify findings into positive, negative, and missing lists.

Module/Component: Decision Support (CDSS)
"""

"""
Evidence Gaps Analyzer

Purpose:
Matches active clinical state against compiled disease profiles to classify findings into positive, negative, and missing lists.

Module/Component: Decision Support (CDSS)
"""

"""
Evidence Gaps Analyzer

Purpose:
Matches active clinical state against compiled disease profiles to classify findings into positive, negative, and missing lists.

Module/Component: Decision Support (CDSS)
"""

import logging
from typing import Any, Dict, List, Set, Optional
from pydantic import BaseModel, Field

from clinical_intelligence.evidence.evidence_models import DiseaseProfile
from clinical_intelligence.evidence.disease_profile_registry import DiseaseProfileRegistry

logger = logging.getLogger(__name__)

"""
Evidence Gap Analyzer.

Created for Module 7 (Adaptive Clinical Decision Support & Consultation Intelligence).
This module analyzes the active patient clinical state against disease profiles in the registry,
extracting positive findings (symptoms/exams already confirmed), negative findings (negated symptoms),
and missing findings (unexplored symptoms/exams listed in disease rules).
It computes a completeness percentage per disease candidate to guide targeted questioning.
"""


class DiseaseGap(BaseModel):
    disease_id: str
    disease_name: str
    positive_findings: List[str] = Field(default_factory=list)
    negative_findings: List[str] = Field(default_factory=list)
    missing_findings: List[str] = Field(default_factory=list)
    completeness: float = 0.0


def normalize_symptom_name(name: str, registry: DiseaseProfileRegistry) -> str:
    key = name.lower().strip()
    if key in registry.alias_map:
        return registry.alias_map[key]
    
    # Prefix-agnostic fallback
    if key.startswith("symptom_"):
        stripped = key[8:]
        if stripped in registry.alias_map:
            return registry.alias_map[stripped]
        key = stripped
        
    best = None
    best_len = 0
    for alias, canonical in registry.alias_map.items():
        pattern = registry.compiled_regexes.get(alias)
        if pattern and pattern.search(key) and len(alias) > best_len:
            best = canonical
            best_len = len(alias)
    return best or key


class EvidenceGapAnalyzer:
    """Computes positive, negative, and missing findings, along with completeness for active candidate diagnoses."""

    def analyze(
        self,
        clinical_state: Dict[str, Any],
        candidate_diagnoses: List[Any],
        registry: DiseaseProfileRegistry
    ) -> List[DiseaseGap]:
        
        # 1. Parse present and negated symptoms from ClinicalState
        present_symptoms: Set[str] = set()
        negated_symptoms: Set[str] = set()
        
        symptoms_list = clinical_state.get("symptoms", []) or []
        for sym in symptoms_list:
            name = sym.get("name", "")
            if not name:
                continue
            is_present = sym.get("present", True)
            canonical = normalize_symptom_name(name, registry)
            if is_present:
                present_symptoms.add(canonical)
            else:
                negated_symptoms.add(canonical)

        # 2. Parse vitals present
        vitals_state = clinical_state.get("vital_signs", {}) or {}
        present_vitals: Set[str] = set()
        for v_name, v_val in vitals_state.items():
            if vval_is_present(v_val):
                present_vitals.add(v_name.lower().strip())

        # 3. Parse physical exam findings present
        exam_state = clinical_state.get("examination", []) or []
        present_exams: Set[str] = set()
        for exam in exam_state:
            finding = exam.get("finding", "") if isinstance(exam, dict) else getattr(exam, "finding", "")
            if finding:
                present_exams.add(finding.lower().strip())

        # 4. Parse history/risk factors present
        history_state = clinical_state.get("medical_history", []) or []
        risk_state = clinical_state.get("risk_factors", []) or []
        present_history: Set[str] = set()
        for h in history_state:
            present_history.add(h.lower().strip())
        for r in risk_state:
            present_history.add(r.lower().strip())

        gaps: List[DiseaseGap] = []

        # 5. Process each candidate diagnosis
        for candidate in candidate_diagnoses:
            disease_id = getattr(candidate, "disease_id", None) or candidate.get("disease_id")
            profile = registry.get_profile(disease_id)
            if not profile:
                continue

            positive: List[str] = []
            negative: List[str] = []
            missing: List[str] = []
            
            # Check cardinal symptoms
            for item in profile.cardinal_symptoms:
                sym_name = item.get("name", "")
                canonical = normalize_symptom_name(sym_name, registry)
                if canonical in present_symptoms:
                    positive.append(sym_name)
                elif canonical in negated_symptoms:
                    negative.append(sym_name)
                else:
                    missing.append(sym_name)

            # Check supportive symptoms
            for item in profile.supportive_symptoms:
                sym_name = item.get("name", "")
                canonical = normalize_symptom_name(sym_name, registry)
                if canonical in present_symptoms:
                    positive.append(sym_name)
                elif canonical in negated_symptoms:
                    negative.append(sym_name)
                else:
                    missing.append(sym_name)

            # Check vital signs
            for item in profile.vitals:
                v_name = item.get("name", "")
                v_name_clean = v_name.lower().strip()
                if v_name_clean in present_vitals:
                    positive.append(v_name)
                else:
                    missing.append(v_name)

            # Check physical exam
            for exam_finding in profile.examination:
                exam_clean = exam_finding.lower().strip()
                # partial or direct match
                is_positive = any(exam_clean in pe for pe in present_exams)
                if is_positive:
                    positive.append(exam_finding)
                else:
                    missing.append(exam_finding)

            # Check risk factor/history modifiers
            for rf_name in profile.risk_factor_modifiers.keys():
                rf_clean = rf_name.lower().strip()
                is_positive = any(rf_clean in ph for ph in present_history)
                if is_positive:
                    positive.append(rf_name)
                else:
                    missing.append(rf_name)

            total_findings = len(positive) + len(negative) + len(missing)
            completeness = (len(positive) + len(negative)) / max(1, total_findings)

            gaps.append(DiseaseGap(
                disease_id=profile.id,
                disease_name=profile.name,
                positive_findings=positive,
                negative_findings=negative,
                missing_findings=missing,
                completeness=completeness
            ))

        return gaps


def vval_is_present(val: Any) -> bool:
    if not val:
        return False
    if isinstance(val, dict):
        return bool(val.get("value") or val.get("qualitative"))
    return bool(getattr(val, "value", None) or getattr(val, "qualitative", None))
