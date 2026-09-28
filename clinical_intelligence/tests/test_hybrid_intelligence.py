from __future__ import annotations

"""
Test suite for hybrid_intelligence

Purpose:
Defines automated unit and integration tests to verify code coverage and correctness.

Module/Component: Test Suites
"""

import unittest
import asyncio
from unittest import mock
from typing import Any, Dict, List

from clinical_intelligence.vector_store.store import ClinicalVectorStore
from clinical_intelligence.rag.retriever import ClinicalGuidelinesRetriever
from clinical_intelligence.llm.context_builder import ContextBuilder
from clinical_intelligence.llm.providers import GeminiProvider
from clinical_intelligence.llm.orchestrator import LLMOrchestrator, LLMResultCache
from clinical_intelligence.llm.task_manager import LLMTaskManager
from clinical_intelligence.llm.session_manager import LLMSessionManager
from clinical_intelligence.validators.schema_validator import JSONSchemaValidator
from clinical_intelligence.validators.response_validator import ClinicalResponseValidator
from clinical_intelligence.validators.hallucination_guard import HallucinationGuard
from clinical_intelligence.rule_engine.engine import ClinicalReasoningEngine


class TestHybridIntelligence(unittest.IsolatedAsyncioTestCase):
    """Automated test suite validating the hybrid clinical intelligence RAG & Orchestration pipeline."""

    def setUp(self) -> None:
        self.store = ClinicalVectorStore()
        self.retriever = ClinicalGuidelinesRetriever(self.store)
        self.session_manager = LLMSessionManager()
        self.orchestrator = LLMOrchestrator()
        self.task_manager = LLMTaskManager(self.orchestrator, self.session_manager)

    def test_guidelines_retrieval(self) -> None:
        """Verify guidelines are retrieved and ranked correctly."""
        results = self.retriever.retrieve_explainable_guidelines(
            symptoms=["chest pain", "sweating"],
            candidates=[{"disease_id": "unstable_angina", "category": "cardiology", "confidence": 0.85}],
            risk_factors=["hypertension"],
            medications=["aspirin"],
            age=65.0,
            gender="male",
            red_flags=[],
            scores={},
            stage="Assessment"
        )
        self.assertIsNotNone(results)
        if results:
            self.assertTrue(any("Chest Pain" in r["title"] for r in results))
            self.assertIn("retrieval_score", results[0])
            self.assertIn("retrieval_reason", results[0])

    def test_context_builder_stages(self) -> None:
        """Verify consultation stages and progress are inferred accurately."""
        # 1. Greeting stage
        stage, progress = ContextBuilder.infer_stage_and_progress([], [], [], [], False)
        self.assertEqual(stage, "Greeting")
        self.assertEqual(progress, 5)

        # 2. History taking stage
        stage, progress = ContextBuilder.infer_stage_and_progress(
            [{"name": "chest pain", "status": "Active"}, {"name": "sweating", "status": "Active"}], [], [], [], False
        )
        self.assertEqual(stage, "History of Present Illness")
        self.assertEqual(progress, 40)

        # 3. Assessment stage
        stage, progress = ContextBuilder.infer_stage_and_progress(
            [{"name": "chest pain", "status": "Active"}], ["hypertension"], ["aspirin"], [1], True
        )
        self.assertEqual(stage, "Assessment")
        self.assertEqual(progress, 85)

    async def test_llm_orchestrator_reports_unavailable(self) -> None:
        """
        With no healthy provider the orchestrator returns nothing.

        It used to substitute MockLLMProvider text here, which reached the caller in
        the same fields as a real answer and could not be told apart from one.

        The absence of a provider is established here rather than assumed. It used
        to depend on the ambient environment having no key, so on a machine that
        has one this test made a live API call and then failed for the wrong
        reason — the selection logic reads these two names at call time.
        """
        with mock.patch.multiple(
            "clinical_intelligence.llm.orchestrator",
            GEMINI_API_KEY="",
            OPENAI_API_KEY="",
        ):
            res_json, telemetry = await self.orchestrator.execute_prompt(
                prompt_id="adaptive_questions_v1",
                template_params={
                    "context": "Patient has acute chest pain.",
                    "stage": "Assessment",
                    "progress_percent": 85,
                    "question_limit": 2
                }
            )
        self.assertIsNone(res_json, "no provider answered, so there must be no result")
        self.assertEqual(telemetry.get("status"), "unavailable")
        self.assertIsNotNone(telemetry)
        self.assertIn("prompt_id", telemetry)

    def test_orchestrator_caching(self) -> None:
        """Verify compound caching hit overrides work correctly."""
        cache = LLMResultCache()
        cache.set("test_key", {"mock_key": "mock_val"})
        
        val = cache.get("test_key")
        self.assertEqual(val, {"mock_key": "mock_val"})
        self.assertEqual(cache.hits, 1)

    def test_validators_hallucinations(self) -> None:
        """Verify validators check JSON formats and filter hallucinated advice."""
        guard = HallucinationGuard(
            rule_diseases={"unstable_angina", "pneumonia"},
            graph_investigations={"ECG", "Troponin"},
            guidelines_tests=set()
        )
        
        # Test diagnosis verification
        ok, verified = guard.verify_diagnoses(["unstable_angina", "hallucinated_disease"])
        self.assertFalse(ok)
        self.assertEqual(verified, ["unstable_angina"])

        # Test investigation verification
        ok, verified = guard.verify_investigations([
            {"name": "ECG Interpretation", "priority": "URGENT"},
            {"name": "Dangling Investigation", "priority": "ROUTINE"}
        ])
        self.assertFalse(ok)
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["name"], "ECG Interpretation")

    def test_response_formatter(self) -> None:
        """Verify response formatter maps raw json to typed models."""
        from clinical_intelligence.llm.response_formatter import ResponseFormatter
        raw_soap = {
            "subjective": "Subjective part",
            "objective": "Objective part",
            "assessment": "Assessment part",
            "plan": "Plan part"
        }
        soap, duration = ResponseFormatter.format_soap(raw_soap)
        self.assertIsNotNone(soap)
        self.assertEqual(soap.subjective, "Subjective part")
        self.assertGreater(duration, 0.0)

    async def test_provider_circuit_breaker(self) -> None:
        """Verify that provider circuit breaker opens on successive failures."""
        provider = GeminiProvider("gemini-2.5-flash")
        self.assertTrue(provider.health())
        
        # Simulate consecutive failures
        provider.record_failure()
        provider.record_failure()
        provider.record_failure()
        
        self.assertFalse(provider.health())
        self.assertEqual(provider.circuit_state, "OPEN")

    def test_pipeline_integration(self) -> None:
        """Verify end-to-end reasoning engine integrates LLM results cleanly."""
        engine = ClinicalReasoningEngine()
        
        state = {
            "session_id": "test_session_123",
            "symptoms": [
                {"name": "Chest Pain", "present": True, "status": "Active"},
                {"name": "Sweating", "present": True, "status": "Active"}
            ],
            "patient_info": {"age": 62, "gender": "male"},
            "medical_history": ["Hypertension"],
            "current_medications": []
        }

        # Run with run_llm=True
        result = engine.reason(state, debug=True, run_llm=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.session_id, "test_session_123")
        self.assertTrue(result.pipeline_version == "1.0.0")
        
        # The enrichment fields are populated only when a provider actually answered.
        # Without one they stay empty and llm_status says so explicitly, rather than
        # carrying templated text indistinguishable from a real answer.
        self.assertIn(result.llm_status, ("ok", "unavailable", "disabled"))
        if result.llm_status != "ok":
            self.assertEqual(result.llm_adaptive_questions, [])
            self.assertIsNone(result.llm_clinical_summary)
            self.assertIsNone(result.llm_soap_notes)
        
        # Verify Telemetry inclusion in DebugDiagnostics
        self.assertIsNotNone(result.debug_diagnostics)
        self.assertIsNotNone(result.debug_diagnostics.llm_telemetry)
        self.assertGreater(result.debug_diagnostics.llm_telemetry.total_requests, 0)
