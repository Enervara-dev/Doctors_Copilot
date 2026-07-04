from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional

from clinical_intelligence.workflow.protocols import IReasoningCoordinator
from clinical_intelligence.session.version_tracker import StateVersionTracker
from clinical_intelligence.rule_engine.engine import ClinicalReasoningEngine
from clinical_intelligence.llm.runner import execute_llm_tasks_sync

logger = logging.getLogger(__name__)


class ReasoningCoordinator(IReasoningCoordinator):
    """Orchestrates pipeline execution using StateVersionTracker to skip unchanged subsystems."""

    def __init__(self, engine: Optional[ClinicalReasoningEngine] = None) -> None:
        self.engine = engine or ClinicalReasoningEngine()
        
        # Operational metrics tracking
        self.reasoning_invoked = 0
        self.reasoning_skipped = 0
        self.graph_invoked = 0
        self.graph_skipped = 0
        self.retriever_invoked = 0
        self.retriever_skipped = 0
        self.llm_invoked = 0
        self.llm_skipped = 0

    def coordinate_reasoning(self, session: Any, run_llm: bool = True, debug: bool = True) -> Any:
        """
        Coordinates execution sequentially across engines.
        Skips modules if versions match.
        """
        import uuid
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        start_t = time.perf_counter()
        
        tracker = StateVersionTracker(session.version_tracker, session)
        
        current_clinical_state = session.clinical_state
        current_guidelines = session.cached_subsystem_outputs.get("guidelines", [])
        current_graph = session.cached_subsystem_outputs.get("graph_paths", [])
        
        changes = tracker.update_hashes(current_clinical_state, current_guidelines, current_graph)
        
        needs_reason = tracker.needs_reasoning(changes)
        needs_g = tracker.needs_graph(changes)
        needs_ret = tracker.needs_retrieval(changes)
        needs_l = tracker.needs_llm(changes) and run_llm
        
        from clinical_intelligence.config import runtime
        
        orig_graph = runtime.ENABLE_GRAPH
        orig_rag = runtime.ENABLE_RAG
        orig_llm = runtime.ENABLE_LLM
        
        runtime.ENABLE_GRAPH = orig_graph and (needs_g or not session.cached_subsystem_outputs.get("graph_paths"))
        runtime.ENABLE_RAG = orig_rag and (needs_ret or not session.cached_subsystem_outputs.get("guidelines"))
        runtime.ENABLE_LLM = orig_llm and needs_l
        
        # Resolve EventBus and publish ReasoningStarted
        try:
            from clinical_intelligence.core.container import container
            from clinical_intelligence.workflow.protocols import IEventBus
            from clinical_intelligence.events.events import (
                ReasoningStarted, ReasoningCompleted, GraphCompleted, RetrievalCompleted, LLMCompleted
            )
            eb = container.resolve(IEventBus)
            eb.publish(ReasoningStarted(session_id=session.session_id, request_id=request_id))
        except Exception as e:
            logger.exception("Failed to publish ReasoningStarted event")
            eb = None

        # Run or reuse reasoning
        if needs_reason or "reasoning_result" not in session.cached_subsystem_outputs:
            logger.info(f"ReasoningCoordinator: Executing reasoning for session {session.session_id} [Request ID: {request_id}]")
            self.reasoning_invoked += 1
            
            result = self.engine.reason(current_clinical_state, debug=debug, run_llm=runtime.ENABLE_LLM, request_id=request_id)
            
            session.cached_subsystem_outputs["reasoning_result"] = result
            session.cached_subsystem_outputs["llm_adaptive_questions"] = result.llm_adaptive_questions
            session.cached_subsystem_outputs["llm_differential_explanations"] = result.llm_differential_explanations
            session.cached_subsystem_outputs["llm_recommended_investigations"] = result.llm_recommended_investigations
            session.cached_subsystem_outputs["llm_clinical_summary"] = result.llm_clinical_summary
            session.cached_subsystem_outputs["llm_soap_notes"] = result.llm_soap_notes
            session.cached_subsystem_outputs["llm_patient_explanation"] = result.llm_patient_explanation
            session.cached_subsystem_outputs["llm_emergency_escalation"] = result.llm_emergency_escalation
            session.cached_subsystem_outputs["llm_explainability"] = result.llm_explainability
            
            if eb:
                eb.publish(ReasoningCompleted(session_id=session.session_id, candidate_diagnoses_count=len(result.candidate_diagnoses), request_id=request_id))
                if runtime.ENABLE_GRAPH:
                    eb.publish(GraphCompleted(session_id=session.session_id, query_count=1, request_id=request_id))
                if runtime.ENABLE_RAG:
                    eb.publish(RetrievalCompleted(session_id=session.session_id, retrieved_count=len(result.llm_explainability.citations if result.llm_explainability else []), request_id=request_id))
                if runtime.ENABLE_LLM:
                    eb.publish(LLMCompleted(session_id=session.session_id, latency_ms=result.debug_diagnostics.time_taken_ms if result.debug_diagnostics else 0.0, request_id=request_id))

            if runtime.ENABLE_GRAPH:
                self.graph_invoked += 1
            else:
                self.graph_skipped += 1
                
            if runtime.ENABLE_RAG:
                self.retriever_invoked += 1
            else:
                self.retriever_skipped += 1

            if runtime.ENABLE_LLM:
                self.llm_invoked += 1
            else:
                self.llm_skipped += 1
        else:
            logger.info(f"ReasoningCoordinator: Skipping reasoning (cache hit) for session {session.session_id} [Request ID: {request_id}]")
            self.reasoning_skipped += 1
            self.graph_skipped += 1
            self.retriever_skipped += 1
            self.llm_skipped += 1
            
            result = session.cached_subsystem_outputs["reasoning_result"]
            
        runtime.ENABLE_GRAPH = orig_graph
        runtime.ENABLE_RAG = orig_rag
        runtime.ENABLE_LLM = orig_llm
        
        duration_ms = (time.perf_counter() - start_t) * 1000.0
        
        from clinical_intelligence.rule_engine.models import WorkflowTelemetry
        session.audit_metadata["telemetry"] = WorkflowTelemetry(
            workflow_stage=session.consultation_stage,
            execution_duration=duration_ms,
            session_version=session.rule_engine_version,
            workflow_transitions=len(session.timeline),
            pause_count=session.audit_metadata.get("pause_count", 0),
            resume_count=session.audit_metadata.get("resume_count", 0),
            rollback_count=session.audit_metadata.get("rollback_count", 0),
            skipped_reasoning=self.reasoning_skipped,
            skipped_graph=self.graph_skipped,
            skipped_retrieval=self.retriever_skipped,
            skipped_llm=self.llm_skipped,
            emitted_events=session.audit_metadata.get("emitted_events_count", 0),
            timeline_entries=len(session.timeline),
            active_session_count=0,
            idle_session_count=0,
            archived_session_count=0,
            request_id=request_id,
        )

        return result
