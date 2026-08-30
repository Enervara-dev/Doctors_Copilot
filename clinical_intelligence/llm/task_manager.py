from __future__ import annotations

"""
task_manager.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import hashlib
import time
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Set

from clinical_intelligence.llm.orchestrator import LLMOrchestrator, HallucinationGuard
from clinical_intelligence.llm.session_manager import LLMSessionManager
from clinical_intelligence.llm.context_builder import ContextBuilder
from clinical_intelligence.llm.response_formatter import ResponseFormatter
from clinical_intelligence.config.runtime import TASK_DEFINITIONS, TaskDefinition
from clinical_intelligence.rule_engine.models import (
    AdaptiveQuestionItem,
    EvidenceCitation,
    LLMExplainability,
    TaskTelemetry,
)

logger = logging.getLogger(__name__)



def _cache_scope(session_id: str, context: str) -> str:
    """
    Cache scope for one session's LLM results.

    Keying on session_id alone freezes SOAP notes, summaries and adaptive questions
    for the whole TTL, so a consultation that later gains vitals or symptoms keeps
    serving the note written before they were known. Folding a digest of the prompt
    context into the key still collapses duplicate calls while the clinical picture
    is unchanged, but regenerates as soon as it moves.
    """
    digest = hashlib.sha1((context or "").encode("utf-8")).hexdigest()[:12]
    return f"{session_id}:{digest}"


class LLMTaskManager:
    """
    Coordinates prioritized LLM tasks, asynchronous parallel DAG jobs execution,
    emergency bypass branches, and registers task-specific telemetries.
    """

    def __init__(
        self,
        orchestrator: Optional[LLMOrchestrator] = None,
        session_manager: Optional[LLMSessionManager] = None
    ) -> None:
        self.orchestrator = orchestrator or LLMOrchestrator()
        self.session_manager = session_manager or LLMSessionManager()
        self.task_telemetry_registry: List[TaskTelemetry] = []

    async def run(
        self,
        session_id: str,
        state: Dict[str, Any],
        candidates: List[Any],
        red_flags: List[Any],
        risk_factors: List[Any],
        medications: List[str],
        missing_info: List[Any],
        guidelines: List[Any],
        graph_paths: List[Any],
        is_complete: bool = False
    ) -> Dict[str, Any]:
        """
        Public entry point for executing LLM reasoning tasks synchronously/asynchronously.
        """
        self.session_manager.register_session(session_id)
        session_cache = self.session_manager.get_cache(session_id)
        self.orchestrator.cache = session_cache

        # 1. Delegate context formatting to ContextBuilder
        context_str, stage, progress = ContextBuilder.build_minimal_context(
            state, candidates, red_flags, risk_factors, medications, missing_info, guidelines, graph_paths, is_complete
        )

        cache_scope = _cache_scope(session_id, context_str)

        template_params = {
            "context": context_str,
            "stage": stage,
            "progress_percent": progress,
            "question_limit": 3,
        }

        # 2. Extract boundaries for HallucinationGuard
        rule_diseases = {c.disease_id for c in candidates}
        graph_investigations = {p.nodes[-1].id for p in graph_paths if p.nodes}
        guidelines_tests = {t for gl in guidelines for t in gl.get("matching_symptoms", [])}
        hallucination_guard = HallucinationGuard(rule_diseases, graph_investigations, guidelines_tests)

        results = {
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

        # Priority 1: Emergency Escalation Bypass
        from clinical_intelligence.config.runtime import ENABLE_EMERGENCY_WORKFLOW
        if red_flags and ENABLE_EMERGENCY_WORKFLOW:
            logger.info("Emergency Red Flags detected! Running emergency escalation bypass branch.")
            telemetry_order = 1
            escalation_start = time.perf_counter()
            self.session_manager.track_request_start(session_id, "EmergencyEscalation")
            
            res_json, _ = await self.orchestrator.execute_prompt(
                prompt_id="emergency_escalation_v1",
                template_params=template_params,
                cache_key=f"emergency:{cache_scope}",
                hallucination_guard=hallucination_guard
            )
            
            self.session_manager.track_request_end(session_id, "EmergencyEscalation")
            latency = (time.perf_counter() - escalation_start) * 1000.0
            
            self.task_telemetry_registry.append(
                TaskTelemetry(
                    task_name="EmergencyEscalation",
                    priority=1,
                    execution_order=telemetry_order,
                    latency_ms=latency,
                    status="success" if res_json else "failed"
                )
            )

            # Delegate formatting to ResponseFormatter
            esc_model, _ = ResponseFormatter.format_emergency_escalation(res_json)
            results["llm_emergency_escalation"] = esc_model
            
            # Construct summary fallback
            from clinical_intelligence.rule_engine.models import ClinicalSummaryModel
            results["llm_clinical_summary"] = ClinicalSummaryModel(
                assessment=res_json.get("alert_header", "EMERGENCY CLINICAL ESCALATION REQUIRED") if res_json else "EMERGENCY ESCALATION",
                key_findings=res_json.get("first_responder_actions", []) if res_json else [],
                unresolved_issues=["Stabilize vitals"],
                next_steps=[res_json.get("stabilization_notes", "Cardiology alert")] if res_json else ["Secure access"]
            )
            results["needs_human_review"] = True

            # Load cached background tasks if available
            aq_cached = session_cache.get(f"adap:{cache_scope}")
            if aq_cached and "questions" in aq_cached:
                results["llm_adaptive_questions"] = [
                    AdaptiveQuestionItem(
                        question=q.get("question", ""),
                        purpose=q.get("purpose", ""),
                        related_diagnosis=q.get("related_diagnosis", ""),
                        priority=q.get("priority", "ROUTINE"),
                        clinical_importance=q.get("clinical_importance", "Low")
                    )
                    for q in aq_cached["questions"]
                ]

            de_cached = session_cache.get(f"diff:{cache_scope}")
            if de_cached and "explanations" in de_cached:
                results["llm_differential_explanations"] = de_cached.get("explanations", {})

            ir_cached = session_cache.get(f"inve:{cache_scope}")
            if ir_cached:
                models, _ = ResponseFormatter.format_investigations(ir_cached)
                results["llm_recommended_investigations"] = models

            cs_cached = session_cache.get(f"clin:{cache_scope}")
            if cs_cached:
                model, _ = ResponseFormatter.format_clinical_summary(cs_cached)
                results["llm_clinical_summary"] = model

            soap_cached = session_cache.get(f"soap:{cache_scope}")
            if soap_cached:
                model, _ = ResponseFormatter.format_soap(soap_cached)
                results["llm_soap_notes"] = model

            pe_cached = session_cache.get(f"pati:{cache_scope}")
            if pe_cached:
                model, _ = ResponseFormatter.format_patient_explanation(pe_cached)
                results["llm_patient_explanation"] = model
            
            # Asynchronous background execution of other tasks
            asyncio.create_task(
                self._run_remaining_tasks_in_background(
                    session_id, template_params, hallucination_guard, state
                )
            )
            return results

        # 3. Dynamic DAG Scheduler parsing TASK_DEFINITIONS configuration
        # Filter enabled definitions
        enabled_defs = {td.task_id: td for td in TASK_DEFINITIONS if td.enabled and td.task_id != "EmergencyEscalation"}
        completed_tasks: Set[str] = set()
        execution_counter = 1

        while len(completed_tasks) < len(enabled_defs):
            ready_tasks = []
            for task_id, td in enabled_defs.items():
                if task_id in completed_tasks:
                    continue
                # All listed dependencies must be finished or not enabled
                dep_met = all(dep in completed_tasks or dep not in enabled_defs for dep in td.dependencies)
                if dep_met:
                    ready_tasks.append(td)

            if not ready_tasks:
                logger.error("LLMTaskManager: Deadlock or cyclic dependency detected in task configuration.")
                break

            # Execute ready tasks concurrently
            coros = []
            for td in ready_tasks:
                coros.append(
                    self._execute_single_task(
                        td, session_id, template_params, hallucination_guard, state, results, execution_counter
                    )
                )
                execution_counter += 1

            await asyncio.gather(*coros)

            # Mark completed
            for td in ready_tasks:
                completed_tasks.add(td.task_id)

        # 4. Generate structured explainability audit metadata
        citations = []
        for c in candidates:
            for rule in c.rules_fired:
                citations.append(EvidenceCitation(rule_id=rule, evidence_type="Rule"))
        for path in graph_paths:
            if path.nodes:
                citations.append(EvidenceCitation(graph_path_id=path.nodes[-1].id, evidence_type="Graph"))
        for gl in guidelines:
            citations.append(
                EvidenceCitation(
                    guideline_id=gl.get("id"),
                    guideline_version=gl.get("version"),
                    evidence_type="Guideline"
                )
            )

        results["llm_explainability"] = LLMExplainability(
            generated_from_prompt_id="adaptive_questions_v1",
            prompt_version="1.0.0",
            citations=citations,
            reasoning_summary="Merged Rule Engine findings with local Guideline Vector Store indices.",
            confidence_rating="High" if candidates and candidates[0].confidence > 0.75 else "Medium"
        )

        return results

    async def _execute_single_task(
        self,
        td: TaskDefinition,
        session_id: str,
        template_params: Dict[str, Any],
        hallucination_guard: HallucinationGuard,
        state: Dict[str, Any],
        results: Dict[str, Any],
        execution_order: int
    ) -> None:
        """Executes a single workflow task, records telemetries, and formats the outputs."""
        task_id = td.task_id
        self.session_manager.track_request_start(session_id, task_id)
        start_t = time.perf_counter()

        prompt_id_map = {
            "AdaptiveQuestions": "adaptive_questions_v1",
            "InvestigationRecommendations": "investigation_v1",
            "DifferentialExplanation": "differential_explanation_v1",
            "ClinicalSummary": "clinical_summary_v1",
            "SOAPGeneration": "soap_v1",
            "PatientExplanation": "patient_explanation_v1",
        }
        
        prompt_id = prompt_id_map.get(task_id)
        if not prompt_id:
            logger.error(f"Unknown prompt target mapping for task '{task_id}'")
            self.session_manager.track_request_end(session_id, task_id)
            return

        answered = []
        if task_id == "AdaptiveQuestions":
            answered = [s.get("name", "") for s in state.get("symptoms", []) if s.get("status") in ("Active", "Negated")]

        # Execute prompt asynchronously without executor thread pools
        res_json, _ = await self.orchestrator.execute_prompt(
            prompt_id=prompt_id,
            template_params=template_params,
            cache_key=f"{task_id[:4].lower()}:{_cache_scope(session_id, template_params.get('context', ''))}",
            hallucination_guard=hallucination_guard,
            answered_questions=answered
        )

        self.session_manager.track_request_end(session_id, task_id)
        latency = (time.perf_counter() - start_t) * 1000.0

        self.task_telemetry_registry.append(
            TaskTelemetry(
                task_name=task_id,
                priority=td.priority,
                execution_order=execution_order,
                latency_ms=latency,
                status="success" if res_json else "failed"
            )
        )

        # Delegate JSON mapping formatting to ResponseFormatter
        if res_json:
            if task_id == "AdaptiveQuestions" and "questions" in res_json:
                results["llm_adaptive_questions"] = [
                    AdaptiveQuestionItem(
                        question=q.get("question", ""),
                        purpose=q.get("purpose", ""),
                        related_diagnosis=q.get("related_diagnosis", ""),
                        priority=q.get("priority", "ROUTINE"),
                        clinical_importance=q.get("clinical_importance", "Low")
                    )
                    for q in res_json["questions"]
                ]
            elif task_id == "InvestigationRecommendations":
                models, _ = ResponseFormatter.format_investigations(res_json)
                results["llm_recommended_investigations"] = models
            elif task_id == "DifferentialExplanation":
                results["llm_differential_explanations"] = res_json.get("explanations", {})
            elif task_id == "ClinicalSummary":
                model, _ = ResponseFormatter.format_clinical_summary(res_json)
                results["llm_clinical_summary"] = model
            elif task_id == "SOAPGeneration":
                model, _ = ResponseFormatter.format_soap(res_json)
                results["llm_soap_notes"] = model
            elif task_id == "PatientExplanation":
                model, _ = ResponseFormatter.format_patient_explanation(res_json)
                results["llm_patient_explanation"] = model

    async def _run_remaining_tasks_in_background(
        self,
        session_id: str,
        template_params: Dict[str, Any],
        hallucination_guard: HallucinationGuard,
        state: Dict[str, Any]
    ) -> None:
        """Runs background backfill tasks for emergency cases asynchronously."""
        try:
            self.session_manager.register_session(session_id)
            session_cache = self.session_manager.get_cache(session_id)
            self.orchestrator.cache = session_cache

            enabled_defs = {td.task_id: td for td in TASK_DEFINITIONS if td.enabled and td.task_id != "EmergencyEscalation"}
            completed_tasks: Set[str] = set()
            execution_counter = 1
            bg_results: Dict[str, Any] = {}
            while len(completed_tasks) < len(enabled_defs):
                ready_tasks = []
                for task_id, td in enabled_defs.items():
                    if task_id in completed_tasks:
                        continue
                    dep_met = all(dep in completed_tasks or dep not in enabled_defs for dep in td.dependencies)
                    if dep_met:
                        ready_tasks.append(td)
                if not ready_tasks:
                    break
                coros = []
                for td in ready_tasks:
                    coros.append(
                        self._execute_single_task(
                            td, session_id, template_params, hallucination_guard, state, bg_results, execution_counter
                        )
                    )
                    execution_counter += 1
                await asyncio.gather(*coros)
                for td in ready_tasks:
                    completed_tasks.add(td.task_id)
        except Exception as e:
            logger.warning(f"Background task execution failed: {e}")
