from __future__ import annotations

"""
Deterministic Reasoning Engine Orchestrator

Purpose:
Executes zero-LLM reasoning passes over patient state, running symptoms, risk factors, and vitals against rulesets.

Module/Component: Rule Engine Core
"""

"""
Deterministic Reasoning Engine Orchestrator

Purpose:
Executes zero-LLM reasoning passes over patient state, running symptoms, risk factors, and vitals against rulesets.

Module/Component: Rule Engine Core
"""

"""
Clinical Reasoning Engine — Orchestrator.

This is the sole public entry point of Module 1 (Rule-Based Clinical
Reasoning Engine). It coordinates all sub-components and returns a fully
explainable ClinicalReasoningResult.

Usage::

    from clinical_intelligence.rule_engine.engine import ClinicalReasoningEngine

    engine = ClinicalReasoningEngine()
    result = engine.reason(clinical_state)   # ClinicalState from Milestone 1
    print(result.model_dump_json(indent=2))
"""

import logging
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Set

from clinical_intelligence.rule_engine.contraindications import ContraindicationChecker
from clinical_intelligence.rule_engine.exceptions import InvalidClinicalStateError
from clinical_intelligence.rule_engine.explainability import ExplainabilityTracer
from clinical_intelligence.rule_engine.interactions import DrugInteractionDetector
from clinical_intelligence.rule_engine.investigations import InvestigationRecommender
from clinical_intelligence.rule_engine.loader import RuleLoader, get_loader
from clinical_intelligence.rule_engine.matcher import SymptomMatcher
from clinical_intelligence.rule_engine.missing_info import MissingInfoDetector
from clinical_intelligence.rule_engine.models import (
    ClinicalAlert,
    ClinicalReasoningResult,
    DebugDiagnostics,
)
from clinical_intelligence.rule_engine.red_flags import RedFlagDetector
from clinical_intelligence.rule_engine.risk_evaluator import RiskFactorEvaluator
from clinical_intelligence.rule_engine.scorer import DiseaseScorer, _parse_vitals
from clinical_intelligence.rule_engine.scores import evaluate_all

# Knowledge Graph imports for Milestone 2 Module 2
from clinical_intelligence.knowledge_graph.graph_manager import get_graph_manager
from clinical_intelligence.knowledge_graph.graph_queries import GraphQueryEngine
from clinical_intelligence.knowledge_graph.cache import GraphQueryCache
from clinical_intelligence.knowledge_graph.service import KnowledgeGraphService
from clinical_intelligence.rule_engine.aggregator import EvidenceAggregator, GraphEvidenceProvider, VectorEvidenceProvider
from clinical_intelligence.knowledge_graph.graph_models import GraphMetadata

# RAG & LLM Orchestration imports (added for Milestone 2 Module 3)
from clinical_intelligence.rag.retriever import ClinicalGuidelinesRetriever
from clinical_intelligence.llm.task_manager import LLMTaskManager
from clinical_intelligence.llm.runner import execute_llm_tasks_sync
from clinical_intelligence.config.runtime import ENABLE_RAG, ENABLE_GRAPH, ENABLE_LLM
from clinical_intelligence.rule_engine.models import (
    AdaptiveQuestionItem,
    LLMExplainability,
    LLMTelemetrySummary,
    TaskTelemetry,
)

logger = logging.getLogger(__name__)

# ── Alias normalisation helpers (shared with matcher) ────────────────────
from clinical_intelligence.rule_engine.matcher import _build_alias_map, _partial_match, _normalise


class ClinicalReasoningEngine:
    """
    Deterministic, LLM-free Clinical Reasoning Engine.

    Accepts a ClinicalState (or its dict representation) and returns a
    fully explainable ClinicalReasoningResult.

    All reasoning is driven by configurable YAML rules — no LLM is used.
    """

    ENGINE_VERSION = "1.0.0"

    def __init__(self, loader: Optional[RuleLoader] = None) -> None:
        self._loader = loader or get_loader()
        logger.info(
            "ClinicalReasoningEngine: Initialised. Rules loaded: %s",
            self._loader.get().version_summary(),
        )
        # Initialize graph components for Module 2
        self._graph_manager = get_graph_manager()
        self._graph_cache = GraphQueryCache()
        self._graph_service = KnowledgeGraphService(self._graph_manager, self._graph_cache)
        
        # Initialize RAG and LLM components for Module 3
        self._retriever = ClinicalGuidelinesRetriever()
        self._llm_task_manager = LLMTaskManager()

    def _get_graph_metadata(self) -> Optional[GraphMetadata]:
        """Fetch GraphMetadata record from the database if active and populated."""
        if not self._graph_manager.is_active:
            return None
        try:
            records = self._graph_manager.run_query("MATCH (m:GraphMetadata {id: 'graph_metadata'}) RETURN m")
            if records and "m" in records[0]:
                m_node = records[0]["m"]
                import ast
                db_ver = {}
                db_ver_raw = m_node.get("rule_database_version", "{}")
                try:
                    db_ver = ast.literal_eval(db_ver_raw)
                except Exception:
                    pass
                return GraphMetadata(
                    graph_version=m_node.get("graph_version", "1.0.0"),
                    knowledge_version=m_node.get("knowledge_version", "unknown"),
                    import_timestamp=m_node.get("import_timestamp", ""),
                    rule_database_version=db_ver
                )
        except Exception as e:
            logger.warning(f"Failed to query graph metadata: {e}")
        return None

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def reason(self, clinical_state: Any, debug: bool = False, run_llm: bool = False, request_id: Optional[str] = None) -> ClinicalReasoningResult:
        """
        Perform a complete deterministic clinical reasoning pass.

        Parameters
        ----------
        clinical_state:
            Either a ClinicalState Pydantic model (from Milestone 1) or
            a plain dict with the same schema.
        debug:
            If True, enables performance diagnostics profiling.

        Returns
        -------
        ClinicalReasoningResult
            Fully typed, fully explainable structured clinical intelligence.
        """
        # Reset graph metrics before reasoning pass
        self._graph_service.query_engine.reset_metrics()
        self._graph_service.cache.reset_metrics()

        # ── 0. Optional Debug Instrumentation Start ──────────────────────
        start_time = 0.0
        if debug:
            tracemalloc.start()
            start_time = time.perf_counter()

        # ── 0. Normalise input ──────────────────────────────────────────
        state_data = self._extract_state_data(clinical_state)
        session_id = state_data.get("session_id", "unknown")

        # ── 1. Hot-reload rules ─────────────────────────────────────────
        rules = self._loader.get()
        tracer = ExplainabilityTracer()
        tracer.start()

        logger.info(
            "ClinicalReasoningEngine: Starting reasoning pass for session=%s",
            session_id,
        )

        # ── 2. Extract relevant state fields ────────────────────────────
        present_symptoms = self._get_present_symptoms(state_data)
        negated_symptoms = self._get_negated_symptoms(state_data)
        all_symptom_names_present: Set[str] = set()

        vital_signs: Dict[str, Optional[str]] = state_data.get("vital_signs", {}) or {}
        age: Optional[int] = state_data.get("patient_info", {}).get("age") if state_data.get("patient_info") else None
        gender: Optional[str] = state_data.get("patient_info", {}).get("gender") if state_data.get("patient_info") else None
        medical_history: List[str] = state_data.get("medical_history", []) or []
        current_medications: List[str] = state_data.get("current_medications", []) or []
        allergies: List[str] = state_data.get("allergies", []) or []
        family_history = state_data.get("family_history", []) or []

        # Build alias map for red-flag canonical matching
        alias_map = _build_alias_map(rules.symptoms)
        canonical_present: Set[str] = set()
        canonical_negated: Set[str] = set()
        for sym in present_symptoms:
            raw = sym.get("name", "")
            canonical = _partial_match(raw, alias_map) or _normalise(raw, alias_map)
            if canonical not in alias_map and raw.lower().startswith("symptom_"):
                stripped = raw[8:]
                canonical = _partial_match(stripped, alias_map) or _normalise(stripped, alias_map)
            canonical_present.add(canonical)
        for neg_name in negated_symptoms:
            canonical = _partial_match(neg_name, alias_map) or _normalise(neg_name, alias_map)
            if canonical not in alias_map and neg_name.lower().startswith("symptom_"):
                stripped = neg_name[8:]
                canonical = _partial_match(stripped, alias_map) or _normalise(stripped, alias_map)
            canonical_negated.add(canonical)

        tracer.add_step(
            component="Engine",
            description="Clinical state parsed and normalised",
            input_facts=[
                f"session_id={session_id}",
                f"present_symptoms={len(present_symptoms)}",
                f"negated_symptoms={len(negated_symptoms)}",
                f"age={age}, gender={gender}",
                f"history_items={len(medical_history)}",
                f"medications={len(current_medications)}",
                f"allergies={len(allergies)}",
            ],
            output="State normalised successfully",
        )

        # ── 3. Evaluate Clinical Scores (Deterministic Scoring Modules) ──
        parsed_vitals = _parse_vitals(vital_signs)
        clinical_scores = evaluate_all(
            state_data=state_data,
            vitals=parsed_vitals,
            present_symptoms=canonical_present,
            negated_symptoms=canonical_negated,
            medical_history=medical_history,
            current_medications=current_medications,
            allergies=allergies,
            age=age,
            gender=gender
        )

        for score in clinical_scores:
            tracer.add_step(
                component="Scores",
                description=f"Calculated clinical score: {score.name}",
                input_facts=score.evidence,
                output=f"{score.name} score = {score.score} ({score.interpretation})",
            )

        # ── 4. Risk Factor Evaluation ────────────────────────────────────
        rf_evaluator = RiskFactorEvaluator(rules.risk_factors)
        identified_risk_factors, identified_rf_ids = rf_evaluator.evaluate(
            medical_history=medical_history,
            current_medications=current_medications,
            allergies=allergies,
            family_history=family_history,
            tracer=tracer,
        )

        # ── 5. Symptom Matching ──────────────────────────────────────────
        matcher = SymptomMatcher(rules.diseases, rules.symptoms, rule_set=rules)
        match_results = matcher.match(
            present_symptoms=present_symptoms,
            negated_symptoms=negated_symptoms,
            tracer=tracer,
        )

        # ── 6. Disease Confidence Scoring ────────────────────────────────
        scorer = DiseaseScorer()
        candidate_diagnoses = scorer.score(
            match_results=match_results,
            vital_signs=vital_signs,
            age=age,
            gender=gender,
            identified_risk_factor_ids=identified_rf_ids,
            tracer=tracer,
            clinical_scores=clinical_scores,
        )

        # ── 7. Red Flag Detection ────────────────────────────────────────
        red_flag_detector = RedFlagDetector(rules.red_flags)
        red_flags = red_flag_detector.detect(
            present_symptoms=canonical_present,
            negated_symptoms=canonical_negated,
            vital_signs_raw=vital_signs,
            medical_history=medical_history,
            risk_factor_ids=identified_rf_ids,
            age=age,
            tracer=tracer,
        )

        # ── 8. Investigation Recommendations ─────────────────────────────
        investigator = InvestigationRecommender(rules.investigations)
        recommended_investigations = investigator.recommend(
            candidates=candidate_diagnoses,
            red_flags=red_flags,
            tracer=tracer,
        )

        # ── 9. Contraindication Checking ─────────────────────────────────
        ci_checker = ContraindicationChecker(rules.contraindications)
        contraindications = ci_checker.check(
            allergies=allergies,
            medical_history=medical_history,
            current_medications=current_medications,
            recommended_investigations=recommended_investigations,
            tracer=tracer,
        )

        # ── 10. Drug Interaction Detection ────────────────────────────────
        di_detector = DrugInteractionDetector(rules.drug_interactions)
        drug_interactions = di_detector.detect(
            current_medications=current_medications,
            tracer=tracer,
        )

        # ── 11. Missing Information Detection ────────────────────────────
        missing_detector = MissingInfoDetector()
        missing_information = missing_detector.detect(
            clinical_state_data=state_data,
            candidates=candidate_diagnoses,
            tracer=tracer,
        )

        # ── 12. Aggregate Clinical Alerts ─────────────────────────────────
        clinical_alerts: List[ClinicalAlert] = []
        clinical_alerts.extend(red_flag_detector.to_clinical_alerts(red_flags))
        clinical_alerts.extend(ci_checker.to_clinical_alerts(contraindications))
        clinical_alerts.extend(di_detector.to_clinical_alerts(drug_interactions))
        # Risk-factor info alerts
        for rf in identified_risk_factors:
            clinical_alerts.append(
                ClinicalAlert(
                    alert_id=f"alert_rf_{rf.factor_id}",
                    alert_type="risk_factor",
                    severity="INFO",
                    title=f"Risk Factor: {rf.name}",
                    message=f"Patient has {rf.name} — increases likelihood of associated conditions.",
                    supporting_evidence=[f"Found in: {rf.source}"],
                )
            )
        # Missing-info alerts
        for mi in missing_information[:5]:  # Top 5 only to avoid noise
            clinical_alerts.append(
                ClinicalAlert(
                    alert_id=f"alert_mi_{mi.field}",
                    alert_type="missing_info",
                    severity="LOW",
                    title=f"Missing Information: {mi.field.replace('_', ' ').title()}",
                    message=mi.question,
                    supporting_evidence=[f"Relevant for: {', '.join(mi.relevant_conditions[:2])}"],
                    recommended_action=mi.question,
                )
            )

        # ── 13. Build Explainability Metadata ────────────────────────────
        explainability_metadata = tracer.finish(
            loaded_files=self._loader.loaded_files(),
        )

        # ── 14. Optional Debug Diagnostics Profile End ─────────────────
        debug_diagnostics = None
        if debug:
            time_taken_ms = (time.perf_counter() - start_time) * 1000.0
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Compile debug telemetry details
            triggered_rules_ids = []
            rule_scores_dict = {}
            guidelines_list = []

            for c in candidate_diagnoses:
                triggered_rules_ids.extend(c.rules_fired)
                rule_scores_dict[c.disease_id] = c.confidence
                if c.guideline:
                    guidelines_list.append(c.guideline.guideline_name)

            for rf in red_flags:
                triggered_rules_ids.append(rf.flag_id)
            for ci in contraindications:
                triggered_rules_ids.append(ci.contraindication_id)
            for di in drug_interactions:
                triggered_rules_ids.append(di.interaction_id)

            clinical_scores_dict = {cs.name: cs.score for cs in clinical_scores}
            all_evidence = []
            for c in candidate_diagnoses:
                all_evidence.extend(c.supporting_evidence)

            debug_diagnostics = DebugDiagnostics(
                triggered_rules=list(set(triggered_rules_ids)),
                rule_scores=rule_scores_dict,
                clinical_scores=clinical_scores_dict,
                evidence=list(set(all_evidence)),
                contraindications=[ci.reason for ci in contraindications],
                guidelines_used=list(set(guidelines_list)),
                time_taken_ms=round(time_taken_ms, 2),
                memory_used_bytes=peak_memory,
                request_id=request_id,
                # Graph query telemetry metrics (added for Module 2)
                graph_query_count=self._graph_service.query_engine.query_count,
                graph_traversal_time_ms=round(self._graph_service.query_engine.total_traversal_time_ms, 2),
                graph_cache_hits=self._graph_service.cache.hits,
                graph_cache_misses=self._graph_service.cache.misses,
                nodes_traversed=self._graph_service.query_engine.nodes_traversed,
                relationships_traversed=self._graph_service.query_engine.relationships_traversed,
                neo4j_execution_time_ms=round(self._graph_service.query_engine.neo4j_execution_time_ms, 2),
            )

        # ── 14.3 Retrieve Explainable Guidelines from Vector Store (Module 3) ──
        guidelines_retrieved = []
        if ENABLE_RAG:
            try:
                guidelines_retrieved = self._retriever.retrieve_explainable_guidelines(
                    symptoms=list(canonical_present),
                    candidates=[{"disease_id": c.disease_id, "category": c.category, "confidence": c.confidence} for c in candidate_diagnoses],
                    risk_factors=state_data.get("medical_history", []) or [],
                    medications=state_data.get("current_medications", []) or [],
                    age=float(state_data.get("patient_info", {}).get("age", 0.0) or 0.0),
                    gender=str(state_data.get("patient_info", {}).get("gender", "") or ""),
                    red_flags=[rf.condition for rf in red_flags],
                    scores={cs.name: cs.score for cs in clinical_scores},
                    stage="Assessment"
                )
            except Exception as e:
                logger.warning(f"Guidelines retriever execution failed: {e}")

        # ── 14.5 Enrich Candidate Diagnoses via Pluggable Evidence Aggregator ──────
        providers = []
        if ENABLE_GRAPH:
            providers.append(GraphEvidenceProvider(self._graph_service))
        if ENABLE_RAG:
            providers.append(VectorEvidenceProvider(self._retriever, state_data))
            
        aggregator = EvidenceAggregator(providers)
        enriched_candidates = aggregator.aggregate_evidence(candidate_diagnoses, list(canonical_present))
        
        graph_meta = self._get_graph_metadata()

        # ── 14.7 Coordinate Hybrid Clinical Intelligence LLM Tasks (Module 3) ──
        llm_res = {
            "llm_adaptive_questions": [],
            "llm_differential_explanations": {},
            "llm_recommended_investigations": [],
            "llm_clinical_summary": None,
            "llm_soap_notes": None,
            "llm_patient_explanation": None,
            "llm_emergency_escalation": None,
            "llm_explainability": None,
            "needs_human_review": False,
        }
        if ENABLE_LLM and (run_llm or debug):
            self._llm_task_manager.task_telemetry_registry.clear()
            self._llm_task_manager.orchestrator.llm_calls_telemetry.clear()
            
            try:
                # Extract graph paths for LLM
                graph_paths = []
                if ENABLE_GRAPH and candidate_diagnoses:
                    paths_map = self._graph_service.query_engine.batch_find_relationship_paths(
                        list(canonical_present), [c.disease_id for c in candidate_diagnoses]
                    )
                    for plist in paths_map.values():
                        graph_paths.extend(plist)

                # Isolated event loop execution inside execute_llm_tasks_sync
                llm_res = execute_llm_tasks_sync(
                    self._llm_task_manager,
                    session_id,
                    state_data,
                    candidate_diagnoses,
                    red_flags,
                    identified_risk_factors,
                    state_data.get("current_medications", []) or [],
                    missing_information,
                    guidelines_retrieved,
                    graph_paths,
                    False
                )
            except Exception as lex:
                logger.warning(f"ClinicalReasoningEngine: LLMTaskManager failed: {lex}")

        # Update DebugDiagnostics with LLM Telemetry
        llm_telemetry_summary = None
        if debug and ENABLE_LLM and (run_llm or debug):
            from clinical_intelligence.rule_engine.models import (
                LLMTelemetry,
                RetrieverTelemetry,
                ProviderTelemetry,
                LLMTelemetrySummary,
            )
            
            calls = [
                LLMTelemetry(
                    prompt_execution_id=c.get("prompt_execution_id", ""),
                    model=c["model"],
                    provider=c["provider"],
                    prompt_id=c["prompt_id"],
                    prompt_version=c["prompt_version"],
                    provider_name=c.get("provider_name"),
                    provider_version=c.get("provider_version"),
                    retriever_version=c.get("retriever_version"),
                    graph_version=c.get("graph_version"),
                    context_version=c.get("context_version"),
                    retrieval_time_ms=c.get("retrieval_time_ms"),
                    validation_time_ms=c.get("validation_time_ms"),
                    formatting_time_ms=c.get("formatting_time_ms"),
                    total_llm_pipeline_time_ms=c.get("total_llm_pipeline_time_ms"),
                    input_tokens=c.get("input_tokens", 0),
                    output_tokens=c.get("output_tokens", 0),
                    latency_ms=c.get("latency_ms", 0.0),
                    retry_count=c.get("retry_count", 0),
                    cache_hit=c.get("cache_hit", False),
                    validation_failed=c.get("validation_failed", False),
                    execution_timestamp=c.get("execution_timestamp", "")
                )
                for c in self._llm_task_manager.orchestrator.llm_calls_telemetry
            ]
            
            provs = [
                ProviderTelemetry(
                    provider_name="Gemini",
                    is_available=self._llm_task_manager.orchestrator.gemini.health(),
                    error_rate=self._llm_task_manager.orchestrator.gemini.error_count / max(1, len(self._llm_task_manager.orchestrator.gemini.latency_tracker)),
                    timeout_count=self._llm_task_manager.orchestrator.gemini.timeout_count,
                    average_latency_ms=sum(self._llm_task_manager.orchestrator.gemini.latency_tracker) / max(1, len(self._llm_task_manager.orchestrator.gemini.latency_tracker))
                )
            ]
            
            ret_telemetry = RetrieverTelemetry(
                retrieved_documents=len(guidelines_retrieved),
                filtered_documents=len(self._retriever.store.documents) - len(guidelines_retrieved),
                average_similarity=sum(g.get("retrieval_score", 0.0) for g in guidelines_retrieved) / max(1, len(guidelines_retrieved))
            )
            
            llm_telemetry_summary = LLMTelemetrySummary(
                total_requests=self._llm_task_manager.orchestrator.total_requests,
                successful_requests=self._llm_task_manager.orchestrator.successful_requests,
                retries=self._llm_task_manager.orchestrator.retries,
                failures=self._llm_task_manager.orchestrator.failures,
                provider_switches=self._llm_task_manager.orchestrator.provider_switches,
                tasks=self._llm_task_manager.task_telemetry_registry,
                providers=provs,
                retriever=ret_telemetry,
                llm_calls=calls
            )

            if debug_diagnostics:
                debug_diagnostics.llm_telemetry = llm_telemetry_summary

            from clinical_intelligence.core.container import container
            from clinical_intelligence.workflow.protocols import ISessionRepository, ISessionManager
            try:
                repo = container.resolve(ISessionRepository)
                session = repo.load(session_id)
                if session and "telemetry" in session.audit_metadata:
                    manager = container.resolve(ISessionManager)
                    stats = manager.get_statistics()
                    
                    wf_telemetry = session.audit_metadata["telemetry"]
                    # Update count fields from manager stats
                    wf_telemetry.active_session_count = stats.get("active_sessions", 0)
                    wf_telemetry.idle_session_count = stats.get("idle_sessions", 0)
                    wf_telemetry.archived_session_count = stats.get("total_sessions", 0) - stats.get("active_sessions", 0)
                    
                    if debug_diagnostics:
                        debug_diagnostics.workflow_telemetry = wf_telemetry
            except Exception:
                pass

        # ── 15. Assemble Final Result ─────────────────────────────────────
        result = ClinicalReasoningResult(
            session_id=session_id,
            candidate_diagnoses=enriched_candidates,
            red_flags=red_flags,
            risk_factors=identified_risk_factors,
            recommended_investigations=recommended_investigations,
            missing_information=missing_information,
            contraindications=contraindications,
            drug_interactions=drug_interactions,
            clinical_alerts=clinical_alerts,
            explainability=explainability_metadata,
            engine_version=self.ENGINE_VERSION,
            rule_set_version=rules.version_summary(),
            clinical_scores=clinical_scores,
            debug_diagnostics=debug_diagnostics,
            graph_metadata=graph_meta,
            graph_status=self._graph_service.health(),
            llm_adaptive_questions=llm_res.get("llm_adaptive_questions", []) or [],
            llm_differential_explanations=llm_res.get("llm_differential_explanations", {}) or {},
            llm_recommended_investigations=llm_res.get("llm_recommended_investigations", []) or [],
            llm_clinical_summary=llm_res.get("llm_clinical_summary"),
            llm_soap_notes=llm_res.get("llm_soap_notes"),
            llm_patient_explanation=llm_res.get("llm_patient_explanation"),
            llm_emergency_escalation=llm_res.get("llm_emergency_escalation"),
            llm_explainability=llm_res.get("llm_explainability"),
            needs_human_review=llm_res.get("needs_human_review", False),
            pipeline_version="1.0.0",
            retriever_version=self._retriever.store.version,
            prompt_version="1.0.0",
            provider_version="1.0.0"
        )

        logger.info(
            "ClinicalReasoningEngine: Reasoning complete. "
            "session=%s | candidates=%d | red_flags=%d | investigations=%d | "
            "contraindications=%d | interactions=%d | missing=%d | "
            "duration_ms=%.1f",
            session_id,
            len(candidate_diagnoses),
            len(red_flags),
            len(recommended_investigations),
            len(contraindications),
            len(drug_interactions),
            len(missing_information),
            explainability_metadata.reasoning_duration_ms,
        )

        return result

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _extract_state_data(clinical_state: Any) -> Dict[str, Any]:
        """
        Accept either a Pydantic ClinicalState model or a plain dict.
        Raises InvalidClinicalStateError if neither.
        """
        if isinstance(clinical_state, dict):
            state_dict = dict(clinical_state)
        elif hasattr(clinical_state, "model_dump"):
            state_dict = clinical_state.model_dump()
        elif hasattr(clinical_state, "dict"):
            state_dict = clinical_state.dict()
        else:
            raise InvalidClinicalStateError(
                field="clinical_state",
                reason=f"Expected ClinicalState or dict, got {type(clinical_state).__name__}",
            )

        # Normalize symptom names and IDs to be rule-friendly
        symptoms = state_dict.get("symptoms", []) or []
        normalized_symptoms = []
        for s in symptoms:
            s_copy = dict(s)
            raw_name = s_copy.get("display_name") or s_copy.get("canonical_name") or s_copy.get("name", "")
            if raw_name.lower().startswith("symptom_"):
                raw_name = raw_name[8:]
            s_copy["name"] = raw_name.lower().strip()
            normalized_symptoms.append(s_copy)
        state_dict["symptoms"] = normalized_symptoms

        # Normalize chief complaints
        cc = state_dict.get("chief_complaint", []) or []
        normalized_cc = []
        for item in cc:
            if item.lower().startswith("symptom_"):
                item = item[8:]
            normalized_cc.append(item.lower().strip())
        state_dict["chief_complaint"] = normalized_cc

        return state_dict

    @staticmethod
    def _get_present_symptoms(state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return symptoms with status == 'Active' (or present == 'True')."""
        symptoms = state_data.get("symptoms", []) or []
        return [
            s for s in symptoms
            if s.get("status") == "Active" or str(s.get("present", "True")).lower() == "true"
        ]

    @staticmethod
    def _get_negated_symptoms(state_data: Dict[str, Any]) -> List[str]:
        """Return symptom names with status Negated or Resolved."""
        symptoms = state_data.get("symptoms", []) or []
        return [
            s["name"]
            for s in symptoms
            if s.get("status") in ("Negated", "Resolved")
            or str(s.get("present", "True")).lower() == "false"
        ]
