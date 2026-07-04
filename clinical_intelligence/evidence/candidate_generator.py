from __future__ import annotations

"""
candidate_generator.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

from typing import Any, Dict, List, Set

from clinical_intelligence.evidence.disease_profile_registry import DiseaseProfileRegistry

class CandidateRetriever:
    """
    Retrieves primary candidate disease IDs using direct cues (symptoms, chief complaints)
    to maximize recall from direct evidence.
    """
    def __init__(self, registry: DiseaseProfileRegistry) -> None:
        self.registry = registry

    def retrieve(self, symptoms: List[str], chief_complaints: List[str]) -> Set[str]:
        candidates: Set[str] = set()
        from clinical_intelligence.rule_engine.matcher import _partial_match, _normalise
        
        # 1. Lookup symptoms in inverse index
        for sym in symptoms:
            sym_clean = sym.lower().strip()
            canonical = _partial_match(sym_clean, self.registry.alias_map, self.registry.compiled_regexes) or _normalise(sym_clean, self.registry.alias_map)
            
            if canonical not in self.registry.alias_map and sym_clean.startswith("symptom_"):
                stripped = sym_clean[8:]
                canonical = _partial_match(stripped, self.registry.alias_map, self.registry.compiled_regexes) or _normalise(stripped, self.registry.alias_map)

            for dis_id in self.registry.get_by_symptom(canonical):
                candidates.add(dis_id)
            for idx_sym, dis_ids in self.registry.symptom_index.items():
                if canonical in idx_sym or idx_sym in canonical:
                    for dis_id in dis_ids:
                        candidates.add(dis_id)

        # 2. Lookup chief complaints
        for cc in chief_complaints:
            cc_clean = cc.lower().strip()
            canonical = _partial_match(cc_clean, self.registry.alias_map, self.registry.compiled_regexes) or _normalise(cc_clean, self.registry.alias_map)
            
            if canonical not in self.registry.alias_map and cc_clean.startswith("symptom_"):
                stripped = cc_clean[8:]
                canonical = _partial_match(stripped, self.registry.alias_map, self.registry.compiled_regexes) or _normalise(stripped, self.registry.alias_map)

            for dis_id in self.registry.get_by_symptom(canonical):
                candidates.add(dis_id)
            for idx_sym, dis_ids in self.registry.symptom_index.items():
                if canonical in idx_sym or idx_sym in canonical:
                    for dis_id in dis_ids:
                        candidates.add(dis_id)

        return candidates


class CandidateExpander:
    """
    Expands the candidate list to include secondary diseases that are:
    1. Clinically connected to the already retrieved primary candidates (e.g. same category/specialty).
    2. Supported by other patient contexts like history risk factors, vitals, or examinations.
    """
    def __init__(self, registry: DiseaseProfileRegistry) -> None:
        self.registry = registry

    def expand(
        self,
        retrieved_candidates: Set[str],
        history: List[str],
        risk_factors: List[str],
        vitals: Dict[str, Any],
        examination: List[str]
    ) -> Set[str]:
        expanded_candidates: Set[str] = set(retrieved_candidates)
        if not retrieved_candidates:
            return expanded_candidates

        # Determine related categories from retrieved candidates to prevent set explosion
        related_categories: Set[str] = set()
        for cand_id in retrieved_candidates:
            profile = self.registry.get_profile(cand_id)
            if profile:
                related_categories.add(profile.category.lower().strip())

        # Scan all profiles to check expansion eligibility
        for profile in self.registry.get_all_profiles():
            if profile.id in expanded_candidates:
                continue

            # Candidate must be connected (e.g., share the same clinical category)
            if profile.category.lower().strip() not in related_categories:
                continue

            # Check if any expansion triggers match the patient context
            has_trigger = False

            # Risk Factors & History triggers
            for rf in risk_factors + history:
                rf_clean = rf.lower().strip()
                if rf_clean in [k.lower().strip() for k in profile.risk_factor_modifiers.keys()]:
                    has_trigger = True
                    break

            # Examination triggers
            if not has_trigger and profile.examination:
                active_exams = [e.lower().strip() for e in examination]
                for exam in profile.examination:
                    if exam.lower().strip() in active_exams:
                        has_trigger = True
                        break

            # Vitals triggers
            if not has_trigger and profile.vitals and vitals:
                # If vitals are defined, matching any vital criteria can act as a trigger
                has_trigger = True

            if has_trigger:
                expanded_candidates.add(profile.id)

        return expanded_candidates
