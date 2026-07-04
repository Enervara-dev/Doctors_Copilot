from __future__ import annotations

"""
pipeline.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging
import time
from typing import Any, Dict, List, Set, Type, Optional
from pydantic import BaseModel, Field

from clinical_intelligence.evidence.evidence_models import (
    DiseaseProfile,
    EvidenceResult,
    EvidenceProvider
)
from clinical_intelligence.evidence.evidence_context import EvidenceContext
from clinical_intelligence.evidence.providers import (
    EvidenceProviderRegistry,
    SymptomProvider,
    VitalProvider,
    ExaminationProvider,
    HistoryProvider,
    DemographicProvider,
    RiskFactorProvider,
    KnowledgeGraphProvider,
    GuidelineProvider,
    LaboratoryProvider,
    ImagingProvider,
    ECGProvider,
    ExclusionProvider
)
from clinical_intelligence.evidence.candidate_generator import CandidateRetriever, CandidateExpander
from clinical_intelligence.evidence.normalizer import EvidenceNormalizer
from clinical_intelligence.evidence.hybrid_scorer import HybridEvidenceScorer, DiseaseScore
from clinical_intelligence.evidence.confidence_normalizer import ConfidenceNormalizer
from clinical_intelligence.evidence.differential_ranker import DifferentialRanker
from clinical_intelligence.evidence.question_planner import AdaptiveQuestionPlanner
from clinical_intelligence.evidence.explanation_builder import ExplanationBuilder
from clinical_intelligence.rule_engine.models import CandidateDiagnosis, AdaptiveQuestionItem, DiseaseExplanation

logger = logging.getLogger(__name__)

class ReasoningTelemetry(BaseModel):
    candidate_retrieval_ms: float = 0.0
    candidate_expansion_ms: float = 0.0
    provider_latency_ms: float = 0.0
    normalization_ms: float = 0.0
    scoring_ms: float = 0.0
    calibration_ms: float = 0.0
    ranking_ms: float = 0.0
    total_latency_ms: float = 0.0

class ReasoningTrace(BaseModel):
    request_id: str
    session_id: str
    retrieved_candidate_ids: List[str] = Field(default_factory=list)
    expanded_candidate_ids: List[str] = Field(default_factory=list)
    provider_outputs: Dict[str, List[EvidenceResult]] = Field(default_factory=dict)
    score_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    ranking: List[str] = Field(default_factory=list)
    telemetry: Optional[ReasoningTelemetry] = None


class ReasoningPipeline:
    """
    Orchestration Pipeline for Evidence-Based Clinical Reasoning.
    Decoupled and independently testable logic flow from raw ClinicalState to Candidate Differential rankings.
    """

    def __init__(self, registry: Any) -> None:
        self.registry = registry

        # Initialize Candidate selectors
        self.retriever = CandidateRetriever(self.registry)
        self.expander = CandidateExpander(self.registry)

        # Initialize Providers Registry
        self.provider_registry = EvidenceProviderRegistry()
        self._register_default_providers()

        # Initialize Scorer and Rankers
        self.normalizer = EvidenceNormalizer()
        self.scorer = HybridEvidenceScorer()
        self.confidence_normalizer = ConfidenceNormalizer()
        self.ranker = DifferentialRanker()
        self.question_planner = AdaptiveQuestionPlanner()
        self.explanation_builder = ExplanationBuilder()

    def _register_default_providers(self) -> None:
        self.provider_registry.register("SymptomProvider", SymptomProvider)
        self.provider_registry.register("VitalProvider", VitalProvider)
        self.provider_registry.register("ExaminationProvider", ExaminationProvider)
        self.provider_registry.register("HistoryProvider", HistoryProvider)
        self.provider_registry.register("DemographicProvider", DemographicProvider)
        self.provider_registry.register("RiskFactorProvider", RiskFactorProvider)
        self.provider_registry.register("KnowledgeGraphProvider", KnowledgeGraphProvider)
        self.provider_registry.register("GuidelineProvider", GuidelineProvider)
        self.provider_registry.register("ExclusionProvider", ExclusionProvider)
        self.provider_registry.register("LaboratoryProvider", LaboratoryProvider)
        self.provider_registry.register("ImagingProvider", ImagingProvider)
        self.provider_registry.register("ECGProvider", ECGProvider)

    def run(
        self,
        state: Dict[str, Any],
        graph_prefetch: Dict[str, Any] = None,
        guideline_prefetch: Dict[str, Any] = None,
        request_id: str = "default-req-id",
        cache: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        start_pipeline = time.perf_counter()
        telemetry = ReasoningTelemetry()

        session_id = state.get("session_id", "unknown-session")
        symptoms_raw = [s.get("name", "") for s in state.get("symptoms", []) or [] if s.get("status") == "Active"]
        chief_complaints = state.get("chief_complaint", []) or []

        # 1. Candidate Retrieval
        start_ret = time.perf_counter()
        retrieved = self.retriever.retrieve(symptoms_raw, chief_complaints)
        telemetry.candidate_retrieval_ms = (time.perf_counter() - start_ret) * 1000.0

        # 2. Candidate Expansion
        start_exp = time.perf_counter()
        history = state.get("medical_history", []) or []
        risk_factors = [rf.get("factor_id", "") if isinstance(rf, dict) else getattr(rf, "factor_id", "") for rf in state.get("risk_factors", []) or []]
        vitals = state.get("vital_signs", {}) or {}
        examination = [e.get("finding", "") if isinstance(e, dict) else getattr(e, "finding", "") for e in state.get("examination", []) or []]

        active_candidates = self.expander.expand(retrieved, history, risk_factors, vitals, examination)
        telemetry.candidate_expansion_ms = (time.perf_counter() - start_exp) * 1000.0

        # 3. Build Context
        profiles = {cid: self.registry.get_profile(cid) for cid in active_candidates if self.registry.get_profile(cid)}
        context = EvidenceContext(
            state=state,
            disease_profiles=profiles,
            graph_prefetch=graph_prefetch or {},
            guideline_prefetch=guideline_prefetch or {},
            cache=cache if cache is not None else {},
            alias_map=self.registry.alias_map,
            compiled_regexes=self.registry.compiled_regexes
        )

        # 4. Evaluate Providers via Provider Registry (enforcing caching)
        start_prov = time.perf_counter()
        evidence_by_disease: Dict[str, List[EvidenceResult]] = {}
        
        provider_instances: Dict[str, EvidenceProvider] = {}
        for name, p_cls in self.provider_registry.get_providers().items():
            provider_instances[name] = p_cls()

        for cid in active_candidates:
            profile = profiles.get(cid)
            if not profile:
                continue

            evidence_by_disease[cid] = []

            for name, provider in provider_instances.items():
                # Check cache hit
                cached = context.get_cached_result(name, cid)
                if cached is not None:
                    evidence_by_disease[cid].append(cached)
                else:
                    try:
                        res = provider.evaluate(context, profile)
                        context.set_cached_result(name, cid, res)
                        evidence_by_disease[cid].append(res)
                    except Exception as e:
                        logger.warning(f"ReasoningPipeline: Provider '{name}' failed for disease '{cid}': {e}")

        telemetry.provider_latency_ms = (time.perf_counter() - start_prov) * 1000.0

        # 5. Normalization, Scoring, and Calibration
        start_norm = time.perf_counter()
        scores: List[DiseaseScore] = []
        
        for cid in active_candidates:
            profile = profiles[cid]
            results = evidence_by_disease[cid]

            # Flatten all contributions
            contributions = [c for r in results for c in r.contributions]
            
            # Normalize
            normalized_contribs = self.normalizer.normalize(contributions)

            # Score
            score_res = self.scorer.score(cid, normalized_contribs)
            
            # Calibrate / Map confidence text
            conf_str = self.confidence_normalizer.normalize_confidence_level(score_res.confidence)
            score_res.score_breakdown["calibrated_bracket"] = conf_str

            scores.append(score_res)

        telemetry.normalization_ms = (time.perf_counter() - start_norm) * 1000.0
        telemetry.scoring_ms = telemetry.normalization_ms  # Scoring is performed inline

        # 6. Rank Differential Diagnoses
        start_rank = time.perf_counter()
        ranked_candidates = self.ranker.rank(scores, profiles, evidence_by_disease)
        telemetry.ranking_ms = (time.perf_counter() - start_rank) * 1000.0

        # 7. Adaptive Question Planning
        questions = self.question_planner.plan_questions(ranked_candidates, profiles)

        # 8. Structured Explanation Building
        questions_map = {}
        for q in questions:
            for c in ranked_candidates:
                if c.name == q.related_diagnosis:
                    questions_map.setdefault(c.disease_id, []).append(q.question)

        explanations = self.explanation_builder.build_explanations(scores, evidence_by_disease, profiles, questions_map)

        # 9. Build ReasoningTrace
        telemetry.total_latency_ms = (time.perf_counter() - start_pipeline) * 1000.0
        
        trace = ReasoningTrace(
            request_id=request_id,
            session_id=session_id,
            retrieved_candidate_ids=list(retrieved),
            expanded_candidate_ids=list(active_candidates),
            provider_outputs=evidence_by_disease,
            score_breakdown={s.disease_id: s.score_breakdown for s in scores},
            ranking=[c.disease_id for c in ranked_candidates],
            telemetry=telemetry
        )

        return {
            "candidates": ranked_candidates,
            "questions": questions,
            "explanations": explanations,
            "trace": trace,
            "cache": context.cache
        }
