from __future__ import annotations

"""
providers.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import time
import uuid
from typing import Any, Dict, List, Type
from clinical_intelligence.evidence.evidence_models import (
    EvidenceProvider,
    DiseaseProfile,
    EvidenceResult,
    EvidenceContribution
)
from clinical_intelligence.evidence.evidence_context import EvidenceContext
from clinical_intelligence.rule_engine.scorer import _parse_vitals, _evaluate_vital_condition
from clinical_intelligence.rule_engine.matcher import _build_alias_map, _partial_match, _normalise
from clinical_intelligence.config.evidence import EXCLUSION_PENALTY

class EvidenceProviderRegistry:
    """Registry managing evidence providers dynamically."""

    def __init__(self) -> None:
        self.providers: Dict[str, Type[EvidenceProvider]] = {}

    def register(self, name: str, provider_cls: Type[EvidenceProvider]) -> None:
        self.providers[name] = provider_cls

    def get_providers(self) -> Dict[str, Type[EvidenceProvider]]:
        return self.providers


# --- Providers ---

class SymptomProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        start_time = time.perf_counter()
        contributions: List[EvidenceContribution] = []

        # Get active patient symptoms
        state_symptoms = context.state.get("symptoms", []) or []
        present_symptoms = [s for s in state_symptoms if s.get("status") == "Active" or str(s.get("present", "True")).lower() == "true"]

        for s in present_symptoms:
            s_name = s.get("name", "").lower().strip()
            canonical = _partial_match(s_name, context.alias_map, context.compiled_regexes) or _normalise(s_name, context.alias_map)

            if canonical not in context.alias_map and s_name.startswith("symptom_"):
                stripped = s_name[8:]
                canonical = _partial_match(stripped, context.alias_map, context.compiled_regexes) or _normalise(stripped, context.alias_map)

            # Check if cardinal
            cardinal_match = None
            for card in disease.cardinal_symptoms:
                card_name = card["name"].lower().strip()
                card_canonical = _partial_match(card_name, context.alias_map, context.compiled_regexes) or _normalise(card_name, context.alias_map)
                if card_canonical not in context.alias_map and card_name.startswith("symptom_"):
                    stripped = card_name[8:]
                    card_canonical = _partial_match(stripped, context.alias_map, context.compiled_regexes) or _normalise(stripped, context.alias_map)
                if card_canonical == canonical:
                    cardinal_match = card
                    break

            if cardinal_match:
                weight = float(cardinal_match.get("weight", 0.25))
                conf_str = s.get("confidence", "High")
                conf = 1.0 if conf_str == "High" else (0.75 if conf_str == "Medium" else 0.50)
                contributions.append(EvidenceContribution(
                    trace_id=str(uuid.uuid4()),
                    source_id=s.get("entity_id"),
                    timestamp=float(s.get("updated_at", 0.0) or 0.0),
                    source="symptom",
                    finding=s.get("display_name", s_name),
                    polarity="positive",
                    confidence=conf,
                    weight=weight,
                    explanation=f"Cardinal symptom '{s.get('display_name', s_name)}' present (confidence: {conf_str})."
                ))

            # Check if supportive
            supportive_match = None
            for supp in disease.supportive_symptoms:
                supp_name = supp["name"].lower().strip()
                supp_canonical = _partial_match(supp_name, context.alias_map, context.compiled_regexes) or _normalise(supp_name, context.alias_map)
                if supp_canonical not in context.alias_map and supp_name.startswith("symptom_"):
                    stripped = supp_name[8:]
                    supp_canonical = _partial_match(stripped, context.alias_map, context.compiled_regexes) or _normalise(stripped, context.alias_map)
                if supp_canonical == canonical:
                    supportive_match = supp
                    break

            if supportive_match:
                weight = float(supportive_match.get("weight", 0.20))
                conf_str = s.get("confidence", "High")
                conf = 1.0 if conf_str == "High" else (0.75 if conf_str == "Medium" else 0.50)
                contributions.append(EvidenceContribution(
                    trace_id=str(uuid.uuid4()),
                    source_id=s.get("entity_id"),
                    timestamp=float(s.get("updated_at", 0.0) or 0.0),
                    source="symptom",
                    finding=s.get("display_name", s_name),
                    polarity="positive",
                    confidence=conf,
                    weight=weight,
                    explanation=f"Supportive symptom '{s.get('display_name', s_name)}' present (confidence: {conf_str})."
                ))

        score = sum(c.weight * c.confidence for c in contributions)
        execution_time = (time.perf_counter() - start_time) * 1000.0

        return EvidenceResult(
            provider="SymptomProvider",
            score=score,
            contributions=contributions,
            execution_time_ms=execution_time,
            confidence=1.0 if contributions else 0.0
        )


class VitalProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        start_time = time.perf_counter()
        contributions: List[EvidenceContribution] = []

        vital_signs = context.state.get("vital_signs", {}) or {}
        parsed_vitals = _parse_vitals(vital_signs)

        for criterion in disease.vitals:
            condition = criterion.get("condition", "")
            threshold = float(criterion.get("threshold", 0))
            modifier = float(criterion.get("modifier", 0.0))

            if _evaluate_vital_condition(condition, threshold, parsed_vitals):
                contributions.append(EvidenceContribution(
                    trace_id=str(uuid.uuid4()),
                    source="vital",
                    finding=condition,
                    polarity="positive",
                    confidence=1.0,
                    weight=modifier,
                    explanation=f"Vital sign criterion met: {condition} (threshold: {threshold}, modifier: {modifier})"
                ))

        score = sum(c.weight for c in contributions)
        execution_time = (time.perf_counter() - start_time) * 1000.0

        return EvidenceResult(
            provider="VitalProvider",
            score=score,
            contributions=contributions,
            execution_time_ms=execution_time
        )


class ExaminationProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        start_time = time.perf_counter()
        contributions: List[EvidenceContribution] = []

        examination_findings = context.state.get("examination", []) or []

        for f in examination_findings:
            f_name = f.get("finding", "").lower().strip() if isinstance(f, dict) else str(getattr(f, "finding", "")).lower().strip()
            
            for doc_exam in disease.examination:
                if doc_exam.lower().strip() == f_name:
                    contributions.append(EvidenceContribution(
                        trace_id=str(uuid.uuid4()),
                        source="examination",
                        finding=doc_exam,
                        polarity="positive",
                        confidence=1.0,
                        weight=0.10,
                        explanation=f"Examination finding matched: '{doc_exam}'"
                    ))

        score = sum(c.weight for c in contributions)
        execution_time = (time.perf_counter() - start_time) * 1000.0

        return EvidenceResult(
            provider="ExaminationProvider",
            score=score,
            contributions=contributions,
            execution_time_ms=execution_time
        )


class HistoryProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        start_time = time.perf_counter()
        contributions: List[EvidenceContribution] = []

        medical_history = context.state.get("medical_history", []) or []

        for h in medical_history:
            h_clean = h.lower().strip()
            for rf_id, mod in disease.risk_factor_modifiers.items():
                if rf_id.lower().strip() == h_clean:
                    contributions.append(EvidenceContribution(
                        trace_id=str(uuid.uuid4()),
                        source="history",
                        finding=h,
                        polarity="positive",
                        confidence=1.0,
                        weight=float(mod),
                        explanation=f"Patient history has condition: '{h}' (modifier: {mod})"
                    ))

        score = sum(c.weight for c in contributions)
        execution_time = (time.perf_counter() - start_time) * 1000.0

        return EvidenceResult(
            provider="HistoryProvider",
            score=score,
            contributions=contributions,
            execution_time_ms=execution_time
        )


class DemographicProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        start_time = time.perf_counter()
        contributions: List[EvidenceContribution] = []

        patient_info = context.state.get("patient_info", {}) or {}
        age = patient_info.get("age") if isinstance(patient_info, dict) else getattr(patient_info, "age", None)
        gender = patient_info.get("gender") if isinstance(patient_info, dict) else getattr(patient_info, "gender", None)

        if disease.age_range and age is not None:
            min_age = disease.age_range.get("min", 0)
            max_age = disease.age_range.get("max", 200)
            modifier = float(disease.age_range.get("modifier", 0.0))
            if min_age <= age <= max_age:
                contributions.append(EvidenceContribution(
                    trace_id=str(uuid.uuid4()),
                    source="demographics",
                    finding=f"Age: {age}",
                    polarity="positive",
                    confidence=1.0,
                    weight=modifier,
                    explanation=f"Age ({age}) falls in the risk range [{min_age}-{max_age}]"
                ))

        if disease.gender_modifier and gender:
            req_gender = disease.gender_modifier.get("gender", "")
            modifier = float(disease.gender_modifier.get("modifier", 0.0))
            if req_gender.lower() == gender.lower():
                contributions.append(EvidenceContribution(
                    trace_id=str(uuid.uuid4()),
                    source="demographics",
                    finding=f"Gender: {gender}",
                    polarity="positive",
                    confidence=1.0,
                    weight=modifier,
                    explanation=f"Gender ({gender}) matches disease modifier"
                ))

        score = sum(c.weight for c in contributions)
        execution_time = (time.perf_counter() - start_time) * 1000.0

        return EvidenceResult(
            provider="DemographicProvider",
            score=score,
            contributions=contributions,
            execution_time_ms=execution_time
        )


class RiskFactorProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        start_time = time.perf_counter()
        contributions: List[EvidenceContribution] = []

        # Check identified risk factors from risk evaluator
        # Note: ClinicalState has risk_factors list
        active_rf_ids = [rf.get("factor_id", "").lower().strip() if isinstance(rf, dict) else getattr(rf, "factor_id", "").lower().strip() for rf in context.state.get("risk_factors", []) or []]

        for rf_id, mod in disease.risk_factor_modifiers.items():
            if rf_id.lower().strip() in active_rf_ids:
                contributions.append(EvidenceContribution(
                    trace_id=str(uuid.uuid4()),
                    source="risk_factor",
                    finding=rf_id,
                    polarity="positive",
                    confidence=1.0,
                    weight=float(mod),
                    explanation=f"Risk factor '{rf_id}' active (modifier: {mod})"
                ))

        score = sum(c.weight for c in contributions)
        execution_time = (time.perf_counter() - start_time) * 1000.0

        return EvidenceResult(
            provider="RiskFactorProvider",
            score=score,
            contributions=contributions,
            execution_time_ms=execution_time
        )


class KnowledgeGraphProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        start_time = time.perf_counter()
        contributions: List[EvidenceContribution] = []

        # Read from graph_prefetch
        # We pre-fetched batch evidence and stored in graph_prefetch
        data = context.graph_prefetch.get(disease.id) or {}
        paths = data.get("relationship_paths", [])

        if paths:
            # Add trace contribution for Knowledge Graph verification
            matched_nodes = []
            for path in paths:
                for node in path.nodes:
                    if node.id not in matched_nodes:
                        matched_nodes.append(node.id)

            contributions.append(EvidenceContribution(
                trace_id=str(uuid.uuid4()),
                source="knowledge_graph",
                finding=f"KG nodes: {', '.join(matched_nodes[:5])}",
                polarity="positive",
                confidence=1.0,
                weight=0.03,
                explanation=f"Knowledge Graph supports disease with {len(paths)} relationship paths."
            ))

        score = sum(c.weight for c in contributions)
        execution_time = (time.perf_counter() - start_time) * 1000.0

        return EvidenceResult(
            provider="KnowledgeGraphProvider",
            score=score,
            contributions=contributions,
            execution_time_ms=execution_time,
            metadata=data
        )


class GuidelineProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        start_time = time.perf_counter()
        contributions: List[EvidenceContribution] = []

        # Read from guideline_prefetch
        guidelines = context.guideline_prefetch.get(disease.id.lower().strip()) or []

        for gl in guidelines:
            title = gl.get("title", "Clinical Guideline")
            evidence_lvl = gl.get("evidence_level", "Level C")
            contributions.append(EvidenceContribution(
                trace_id=str(uuid.uuid4()),
                source="guidelines",
                finding=title,
                polarity="positive",
                confidence=1.0,
                weight=0.02,
                explanation=f"Clinical guideline '{title}' ({evidence_lvl}) provides retrieval matches."
            ))

        score = sum(c.weight for c in contributions)
        execution_time = (time.perf_counter() - start_time) * 1000.0

        return EvidenceResult(
            provider="GuidelineProvider",
            score=score,
            contributions=contributions,
            execution_time_ms=execution_time,
            metadata={"guidelines": guidelines}
        )


class ExclusionProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        start_time = time.perf_counter()
        contributions: List[EvidenceContribution] = []

        # Active patient symptoms
        state_symptoms = context.state.get("symptoms", []) or []
        present_symptoms = [s for s in state_symptoms if s.get("status") == "Active" or str(s.get("present", "True")).lower() == "true"]

        for excl in disease.exclusionary_symptoms:
            excl_clean = excl.lower().strip()
            excl_canonical = _partial_match(excl_clean, context.alias_map, context.compiled_regexes) or _normalise(excl_clean, context.alias_map)
            if excl_canonical not in context.alias_map and excl_clean.startswith("symptom_"):
                stripped = excl_clean[8:]
                excl_canonical = _partial_match(stripped, context.alias_map, context.compiled_regexes) or _normalise(stripped, context.alias_map)
            
            for s in present_symptoms:
                s_name = s.get("name", "").lower().strip()
                canonical = _partial_match(s_name, context.alias_map, context.compiled_regexes) or _normalise(s_name, context.alias_map)
                
                if canonical not in context.alias_map and s_name.startswith("symptom_"):
                    stripped = s_name[8:]
                    canonical = _partial_match(stripped, context.alias_map, context.compiled_regexes) or _normalise(stripped, context.alias_map)
                
                if canonical == excl_canonical:
                    contributions.append(EvidenceContribution(
                        trace_id=str(uuid.uuid4()),
                        source="exclusion",
                        finding=excl,
                        polarity="negative",
                        confidence=1.0,
                        weight=EXCLUSION_PENALTY,
                        explanation=f"Exclusionary symptom present: '{excl}' (penalty: -{EXCLUSION_PENALTY})"
                    ))
                    break

        score = sum(-c.weight for c in contributions)
        execution_time = (time.perf_counter() - start_time) * 1000.0

        return EvidenceResult(
            provider="ExclusionProvider",
            score=score,
            contributions=contributions,
            execution_time_ms=execution_time
        )


# Future placeholders
class LaboratoryProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        return EvidenceResult(provider="LaboratoryProvider", score=0.0)

class ImagingProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        return EvidenceResult(provider="ImagingProvider", score=0.0)

class ECGProvider(EvidenceProvider):
    def evaluate(self, context: EvidenceContext, disease: DiseaseProfile) -> EvidenceResult:
        return EvidenceResult(provider="ECGProvider", score=0.0)
