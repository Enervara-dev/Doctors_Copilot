from __future__ import annotations

"""
orchestrator.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

from clinical_intelligence.prompts.registry import PromptRegistry, PromptDefinition
from clinical_intelligence.llm.providers import BaseLLMProvider, GeminiProvider, OpenAIProvider, MockLLMProvider
from clinical_intelligence.validators.schema_validator import JSONSchemaValidator
from clinical_intelligence.validators.response_validator import ClinicalResponseValidator
from clinical_intelligence.validators.hallucination_guard import HallucinationGuard
from clinical_intelligence.config.runtime import PROVIDER_TIMEOUT_SECONDS, RETRY_LIMIT
from clinical_intelligence.config.llm import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TEMPERATURE,
    GEMINI_API_KEY,
    MAX_RESPONSE_TOKENS,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


class LLMResultCache:
    """Caching layer mapping compound keys to parsed LLM outputs."""

    def __init__(self, ttl: float = 300.0) -> None:
        self.ttl = ttl
        self._store: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._store:
            val, expiry = self._store[key]
            if time.time() < expiry:
                self.hits += 1
                return val
            else:
                del self._store[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self._store[key] = (value, time.time() + self.ttl)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0


class LLMOrchestrator:
    """
    Coordinates LLM providers, health routing, transaction retry guardrails,
    and cache lookup overrides. Exposes execution performance telemetries.
    """

    def __init__(self, cache: Optional[LLMResultCache] = None) -> None:
        self.registry = PromptRegistry()
        self.cache = cache or LLMResultCache()
        
        # Instantiate available provider channels. Model, temperature and the token
        # ceiling come from config: hardcoding them here left MAX_RESPONSE_TOKENS and
        # DEFAULT_TEMPERATURE unused, so tuning them had no effect.
        self.gemini = GeminiProvider(
            DEFAULT_GEMINI_MODEL, temperature=DEFAULT_TEMPERATURE, max_tokens=MAX_RESPONSE_TOKENS
        )
        self.openai = OpenAIProvider(
            DEFAULT_OPENAI_MODEL, temperature=DEFAULT_TEMPERATURE, max_tokens=MAX_RESPONSE_TOKENS
        )
        self.mock_llm = MockLLMProvider("mock-model")

        # Telemetry metrics counters
        self.total_requests = 0
        self.successful_requests = 0
        self.retries = 0
        self.failures = 0
        self.provider_switches = 0
        self.llm_calls_telemetry: List[Dict[str, Any]] = []

    async def execute_prompt(
        self,
        prompt_id: str,
        template_params: Dict[str, Any],
        cache_key: Optional[str] = None,
        hallucination_guard: Optional[HallucinationGuard] = None,
        answered_questions: Optional[List[str]] = None,
        retrieval_time_ms: Optional[float] = None,
        validation_time_ms: Optional[float] = None,
        formatting_time_ms: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes prompt generation asynchronously with health routing and validations.
        """
        self.total_requests += 1
        start_time = time.perf_counter()
        execution_id = str(uuid.uuid4())

        prompt_def = self.registry.get(prompt_id)
        if not prompt_def:
            logger.error(f"Prompt definition '{prompt_id}' not found in registry.")
            return None, {"error": "Prompt registry miss"}

        # 1. Caching override check
        if cache_key:
            cached_res = self.cache.get(cache_key)
            if cached_res:
                logger.info(f"LLMOrchestrator: Caching hit for prompt '{prompt_id}'")
                telemetry = {
                    "prompt_execution_id": execution_id,
                    "model": "cached",
                    "provider": "cache",
                    "provider_name": "cache",
                    "provider_version": "1.0.0",
                    "prompt_id": prompt_id,
                    "prompt_version": prompt_def.version,
                    "retriever_version": "1.0.0",
                    "graph_version": "1.0.0",
                    "context_version": "1.0.0",
                    "cache_hit": True,
                    "retry_count": 0,
                    "latency_ms": (time.perf_counter() - start_time) * 1000.0,
                    "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                self.successful_requests += 1
                return cached_res, telemetry

        # Format context
        prompt_text = prompt_def.template_string.format(**template_params)

        # 2. Select Healthy Provider (sort based on success rate and average latency)
        active_candidates = []
        if GEMINI_API_KEY and self.gemini.health():
            active_candidates.append(self.gemini)
        if OPENAI_API_KEY and self.openai.health():
            active_candidates.append(self.openai)
            
        if active_candidates:
            # Route to provider with highest success rate, breaking ties with lowest average latency
            active_candidates.sort(key=lambda p: (-p.rolling_success_rate, p.average_latency))
            providers_chain = active_candidates + [self.mock_llm]
        else:
            providers_chain = [self.mock_llm]

        active_provider = providers_chain[0]
        retry_count = 0
        parsed_json: Optional[Dict[str, Any]] = None
        validation_failed = False
        val_start = 0.0

        # 3. Execution & Guardrails Validation Retry Loop
        while retry_count < RETRY_LIMIT:
            try:
                # Eagerly call model asynchronously
                raw_text = await active_provider.generate(prompt_text, schema=prompt_def.expected_json_schema)
                
                # Check Schema Syntax Validation
                val_start = time.perf_counter()
                schema_ok, parsed = JSONSchemaValidator.validate(raw_text, prompt_def.expected_json_schema.get("required"))
                if not schema_ok or not parsed:
                    validation_failed = True
                    retry_count += 1
                    self.retries += 1
                    continue

                # Check Content Validation
                content_ok = True
                if "questions" in parsed:
                    questions_ok, cleaned_qs = ClinicalResponseValidator.validate_adaptive_questions(
                        parsed.get("questions", []), answered_questions or []
                    )
                    if questions_ok:
                        parsed["questions"] = cleaned_qs
                    else:
                        content_ok = False

                if "subjective" in parsed:
                    if not ClinicalResponseValidator.validate_soap(parsed):
                        content_ok = False

                # Check Hallucination Guard Cross-References
                if hallucination_guard and content_ok:
                    if "questions" in parsed:
                        diagnoses = [q.get("related_diagnosis", "") for q in parsed["questions"]]
                        diag_ok, verified_diags = hallucination_guard.verify_diagnoses(diagnoses)
                        if not diag_ok:
                            parsed["questions"] = [
                                q for q in parsed["questions"] 
                                if q.get("related_diagnosis") in verified_diags
                            ]
                    
                    if "investigations" in parsed:
                        tests_ok, verified_tests = hallucination_guard.verify_investigations(parsed["investigations"])
                        if not tests_ok:
                            parsed["investigations"] = verified_tests

                if not content_ok:
                    validation_failed = True
                    retry_count += 1
                    self.retries += 1
                    continue

                parsed_json = parsed
                validation_failed = False
                break

            except Exception as e:
                logger.warning(f"Provider {active_provider.__class__.__name__} failed on try {retry_count}: {e}")
                retry_count += 1
                self.retries += 1
                
                if len(providers_chain) > 1 and retry_count < RETRY_LIMIT:
                    active_provider = providers_chain[1]
                    self.provider_switches += 1

        # Fallback to Mock LLM if all API retries failed
        if parsed_json is None:
            logger.error(f"All LLM API retries failed for prompt '{prompt_id}'. Falling back to MockLLMProvider.")
            try:
                raw_text = await self.mock_llm.generate(prompt_text, schema=prompt_def.expected_json_schema)
                _, parsed_json = JSONSchemaValidator.validate(raw_text, prompt_def.expected_json_schema.get("required"))
                active_provider = self.mock_llm
            except Exception as ex:
                logger.critical(f"MockLLMProvider fallback failed: {ex}")
                parsed_json = None

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        val_time = (time.perf_counter() - val_start) * 1000.0 if val_start > 0 else 0.0
        
        # Telemetry updates
        telemetry = {
            "prompt_execution_id": execution_id,
            "model": active_provider.model_name,
            "provider": active_provider.__class__.__name__,
            "provider_name": active_provider.__class__.__name__,
            "provider_version": "1.0.0",
            "prompt_id": prompt_id,
            "prompt_version": prompt_def.version,
            "retriever_version": "1.0.0",
            "graph_version": "1.0.0",
            "context_version": "1.0.0",
            "retrieval_time_ms": retrieval_time_ms,
            "validation_time_ms": validation_time_ms or val_time,
            "formatting_time_ms": formatting_time_ms,
            "total_llm_pipeline_time_ms": latency_ms,
            "latency_ms": latency_ms,
            "retry_count": retry_count,
            "cache_hit": False,
            "validation_failed": validation_failed,
            "input_tokens": len(prompt_text) // 4,
            "output_tokens": len(str(parsed_json)) // 4 if parsed_json else 0,
            "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self.llm_calls_telemetry.append(telemetry)
        if parsed_json:
            self.successful_requests += 1
            if cache_key:
                self.cache.set(cache_key, parsed_json)
        else:
            self.failures += 1

        return parsed_json, telemetry
