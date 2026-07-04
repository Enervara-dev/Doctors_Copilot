from __future__ import annotations
import time
import unittest
import tracemalloc
from typing import Any, Dict

from clinical_intelligence.rule_engine.engine import ClinicalReasoningEngine


class BenchmarkPipeline(unittest.TestCase):
    """Benchmarks processing latency and memory footprint bounds for components."""

    def setUp(self) -> None:
        self.engine = ClinicalReasoningEngine()
        self.state = {
            "session_id": "benchmark_session_999",
            "symptoms": [
                {"name": "Chest Pain", "present": True, "status": "Active"},
                {"name": "Sweating", "present": True, "status": "Active"}
            ],
            "patient_info": {"age": 60, "gender": "female"},
            "medical_history": ["Hypertension"],
            "current_medications": []
        }

    def test_pipeline_latencies(self) -> None:
        """Measure average latency and memory usage for engine components."""
        tracemalloc.start()
        
        # 1. Benchmark Rule Engine & Graph Reasoning Pass (LLM disabled)
        start_t = time.perf_counter()
        res_baseline = self.engine.reason(self.state, debug=True, run_llm=False)
        baseline_time = (time.perf_counter() - start_t) * 1000.0
        
        # Extract graph latency from reasoning metadata
        graph_latency = 0.0
        if res_baseline.debug_diagnostics and res_baseline.debug_diagnostics.graph_query_count > 0:
            graph_latency = res_baseline.debug_diagnostics.graph_traversal_time_ms

        rule_engine_latency = max(0.0, baseline_time - graph_latency)

        # 2. Benchmark Full Hybrid Clinical Reasoning Pass (LLM enabled)
        start_llm = time.perf_counter()
        res_llm = self.engine.reason(self.state, debug=True, run_llm=True)
        llm_pipeline_time = (time.perf_counter() - start_llm) * 1000.0

        # Retrieve specific component latency details from telemetries
        retrieval_latency = 0.0
        llm_latency = 0.0
        validation_latency = 0.0
        formatting_latency = 0.0

        if res_llm.debug_diagnostics and res_llm.debug_diagnostics.llm_telemetry:
            telemetry = res_llm.debug_diagnostics.llm_telemetry
            # Sum up latency across LLM task executions
            llm_latency = sum(task.latency_ms for task in telemetry.tasks)
            # Retrieve validation, retrieval, and formatting latency from calls
            for call in telemetry.llm_calls:
                validation_latency += call.validation_time_ms or 0.0
                retrieval_latency += call.retrieval_time_ms or 0.0
                formatting_latency += call.formatting_time_ms or 0.0

        # 3. Benchmark Caching Performance Pass (Identical State)
        start_cached = time.perf_counter()
        res_cached = self.engine.reason(self.state, debug=True, run_llm=True)
        cached_time = (time.perf_counter() - start_cached) * 1000.0

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print("\n=== PIPELINE PERFORMANCE BENCHMARK ===")
        print(f"Rule Engine Latency:         {rule_engine_latency:.2f} ms")
        print(f"Knowledge Graph Latency:     {graph_latency:.2f} ms")
        print(f"RAG Retrieval Latency:       {retrieval_latency:.2f} ms")
        print(f"LLM Task Execution Latency:  {llm_latency:.2f} ms")
        print(f"Validation Guard Latency:    {validation_latency:.2f} ms")
        print(f"Pydantic Formatting Latency: {formatting_latency:.2f} ms")
        print(f"---------------------------------------")
        print(f"Total Pipeline Latency:      {llm_pipeline_time:.2f} ms")
        print(f"Cached Pipeline Latency:     {cached_time:.2f} ms")
        print(f"Peak Memory Allocated:       {peak / 1024.0:.2f} KB")
        print("=======================================")

        # Assert latencies are within reasonable bounds
        self.assertLess(baseline_time, 2000.0) # Baseline under 2s
        self.assertLess(cached_time, llm_pipeline_time) # Caching is faster than full LLM calls
