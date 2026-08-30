from __future__ import annotations

"""
providers.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import httpx
import json
import logging
import ssl
import time
from typing import Any, Dict, List, Optional

from clinical_intelligence.config.llm import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    THINKING_BUDGET_TOKENS,
)
from clinical_intelligence.config.runtime import CIRCUIT_BREAKER, PROVIDER_TIMEOUT_SECONDS


def _build_ssl_context() -> ssl.SSLContext:
    """
    TLS context that trusts the operating system's certificate store.

    httpx verifies against certifi's bundle by default, which does not contain the
    root CA of a TLS-inspecting corporate proxy or endpoint-security agent. Those
    CAs are installed in the OS store, so loading it keeps certificate verification
    fully enabled while still working behind such a proxy.
    """
    context = ssl.create_default_context()
    try:
        context.load_default_certs(ssl.Purpose.SERVER_AUTH)
    except Exception:  # pragma: no cover - platform dependent
        pass
    return context


_SSL_CONTEXT = _build_ssl_context()


# Keys Gemini's responseSchema accepts. Anything else (notably `additionalProperties`
# and `$schema`) makes the request 400, so unknown keywords are dropped rather than
# forwarded.
_SCHEMA_KEYS = {"type", "format", "description", "nullable", "enum",
                "items", "properties", "required", "propertyOrdering"}


def _to_response_schema(schema):
    """Reduces a JSON Schema fragment to the subset Gemini will accept."""
    if not isinstance(schema, dict):
        return None
    out = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {k: _to_response_schema(v) for k, v in value.items()}
        elif key == "items":
            out[key] = _to_response_schema(value)
        else:
            out[key] = value
    return out or None


def _read_gemini_text(resp_data: Dict[str, Any]) -> str:
    """
    Pulls the reply text out of a Gemini response.

    A candidate can come back with no `parts` at all - `MAX_TOKENS` reached while
    thinking, or a safety block - and indexing straight into parts[0] turns that into
    an opaque KeyError. Report why instead, so the failure is diagnosable rather than
    just another circuit-breaker tick.
    """
    candidates = resp_data.get("candidates") or []
    if not candidates:
        raise ValueError(f"Gemini returned no candidates: {resp_data.get('promptFeedback', {})}")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()

    if not text:
        reason = candidate.get("finishReason", "UNKNOWN")
        usage = resp_data.get("usageMetadata", {})
        raise ValueError(
            f"Gemini returned no text (finishReason={reason}, "
            f"thinkingTokens={usage.get('thoughtsTokenCount', 0)}, "
            f"maxOutputTokens exhausted before the reply began)"
        )

    if candidate.get("finishReason") == "MAX_TOKENS":
        logger.warning(
            "Gemini hit maxOutputTokens; the reply is truncated and will fail schema "
            "validation. Raise CLINICAL_COPILOT_MAX_RESPONSE_TOKENS or cap thinking "
            "with CLINICAL_COPILOT_THINKING_BUDGET."
        )
    return text

logger = logging.getLogger(__name__)


class BaseLLMProvider:
    """Base interface for all LLM client channels with circuit breaker checks."""

    def __init__(self, model_name: str, temperature: float = 0.2, max_tokens: int = 1024) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.latency_tracker: List[float] = []
        
        # Telemetry metrics counters
        self.error_count = 0
        self.timeout_count = 0
        self.rate_limit_count = 0
        self.last_used_timestamp = 0.0

        # Health state monitoring metrics
        self.circuit_state = "CLOSED" # CLOSED, OPEN, HALF_OPEN
        self.consecutive_failures = 0
        self.last_failure_timestamp = 0.0
        self.successful_calls_count = 0
        self.failed_calls_count = 0
        self.rolling_success_rate = 1.0
        self.average_latency = 0.0

        # Load configurable thresholds
        self.failure_threshold = CIRCUIT_BREAKER.get("failure_threshold", 3)
        self.cooldown_seconds = CIRCUIT_BREAKER.get("cooldown_seconds", 30.0)
        self.half_open_requests = CIRCUIT_BREAKER.get("half_open_requests", 1)

    async def generate(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

    async def generate_stream(self, prompt: str, schema: Optional[Dict[str, Any]] = None):
        """Streaming generator placeholder (future compatibility constraint)."""
        yield await self.generate(prompt, schema)

    def health(self) -> bool:
        """Check if provider is available, routing through Circuit Breaker logic."""
        if self.circuit_state == "CLOSED":
            return True
        if self.circuit_state == "OPEN":
            # Check cooldown threshold
            if time.time() - self.last_failure_timestamp > self.cooldown_seconds:
                self.circuit_state = "HALF_OPEN"
                logger.info(f"Circuit Breaker for provider {self.__class__.__name__} transitioned to HALF_OPEN (cooldown passed)")
                return True
            return False
        if self.circuit_state == "HALF_OPEN":
            return True
        return True

    def record_success(self, latency_ms: float) -> None:
        """Update metrics on successful network execution."""
        self.consecutive_failures = 0
        self.successful_calls_count += 1
        self.last_used_timestamp = time.time()
        self.latency_tracker.append(latency_ms)
        self.average_latency = sum(self.latency_tracker) / max(1, len(self.latency_tracker))
        self.rolling_success_rate = self.successful_calls_count / (self.successful_calls_count + self.failed_calls_count)

        if self.circuit_state == "HALF_OPEN":
            self.circuit_state = "CLOSED"
            logger.info(f"Circuit Breaker for provider {self.__class__.__name__} closed successfully.")

    def record_failure(self, is_timeout: bool = False, is_rate_limit: bool = False) -> None:
        """Update metrics and transition breaker on execution errors."""
        self.consecutive_failures += 1
        self.failed_calls_count += 1
        self.error_count += 1
        self.last_failure_timestamp = time.time()
        self.rolling_success_rate = self.successful_calls_count / (self.successful_calls_count + self.failed_calls_count)

        if is_timeout:
            self.timeout_count += 1
        if is_rate_limit:
            self.rate_limit_count += 1

        if self.circuit_state in ("CLOSED", "HALF_OPEN") and self.consecutive_failures >= self.failure_threshold:
            self.circuit_state = "OPEN"
            logger.warning(
                f"Circuit Breaker for provider {self.__class__.__name__} opened after {self.consecutive_failures} failures. "
                f"Provider will be disabled for {self.cooldown_seconds} seconds."
            )

    def warmup(self) -> None:
        """Warm up network connections on startup."""
        pass

    def shutdown(self) -> None:
        """Close connection pools on cleanup."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Integrates with Google Gemini API via HTTP REST queries using async HTTP."""

    async def generate(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> str:
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API key is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        
        gen_config: Dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
        }
        if THINKING_BUDGET_TOKENS is not None:
            gen_config["thinkingConfig"] = {"thinkingBudget": THINKING_BUDGET_TOKENS}
        if schema:
            gen_config["responseMimeType"] = "application/json"
            # Constrain the reply to the shape the validator requires. Asking only for
            # "application/json" left the model free to invent its own envelope - it
            # returned {"consultation_notes": {...}} for the SOAP prompt, which the
            # validator rejected for missing top-level keys, so every well-formed
            # response was thrown away and replaced with canned fallback text.
            response_schema = _to_response_schema(schema)
            if response_schema:
                gen_config["responseSchema"] = response_schema

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_config
        }

        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_SECONDS, verify=_SSL_CONTEXT) as client:
                response = await client.post(url, json=body, headers=headers)
                
            if response.status_code != 200:
                is_rate_limit = (response.status_code == 429)
                self.record_failure(is_rate_limit=is_rate_limit)
                response.raise_for_status()

            resp_data = response.json()
            text = _read_gemini_text(resp_data)

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.record_success(latency_ms)
            return text

        except httpx.TimeoutException as te:
            self.record_failure(is_timeout=True)
            logger.warning(f"Gemini HTTP request timed out: {te}")
            raise te
        except Exception as e:
            self.record_failure()
            logger.warning(f"Gemini HTTP request failed: {e}")
            raise e


class OpenAIProvider(BaseLLMProvider):
    """Integrates with OpenAI Chat Completions API via HTTP REST queries using async HTTP."""

    async def generate(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> str:
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not configured.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if schema:
            body["response_format"] = {"type": "json_object"}

        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_SECONDS, verify=_SSL_CONTEXT) as client:
                response = await client.post(url, json=body, headers=headers)
                
            if response.status_code != 200:
                is_rate_limit = (response.status_code == 429)
                self.record_failure(is_rate_limit=is_rate_limit)
                response.raise_for_status()

            resp_data = response.json()
            text = resp_data["choices"][0]["message"]["content"]
            
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.record_success(latency_ms)
            return text

        except httpx.TimeoutException as te:
            self.record_failure(is_timeout=True)
            logger.warning(f"OpenAI HTTP request timed out: {te}")
            raise te
        except Exception as e:
            self.record_failure()
            logger.warning(f"OpenAI HTTP request failed: {e}")
            raise e


class MockLLMProvider(BaseLLMProvider):
    """Offline mock generator fallback returning schema-compliant clinical JSONs."""

    async def generate(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> str:
        prompt_lower = prompt.lower()
        start_time = time.perf_counter()
        
        # Simulate local database parsing latency
        time.sleep(0.001)

        # Extract suspected candidates from prompt to satisfy hallucination guard
        import re
        candidates = []
        if "top candidates:" in prompt_lower:
            match = re.search(r"top candidates:\s*(.*?)(?:\n\n|\Z)", prompt_lower, re.DOTALL)
            if match:
                candidates = re.findall(r"-\s*([^(]+)", match.group(1))
                candidates = [c.strip() for c in candidates]
        
        # Also check for individual known disease names in prompt
        diseases = [
            "acute bronchitis", "influenza-like illness", "ischemic stroke", "hypertensive crisis",
            "hypothyroidism", "acute coronary syndrome", "unstable angina", "congestive heart failure",
            "pulmonary embolism", "sepsis", "pneumonia", "copd exacerbation", "asthma attack", "migraine",
            "tension-type headache", "meningitis", "type 2 diabetes mellitus", "gastro-oesophageal reflux disease",
            "peptic ulcer disease", "appendicitis", "acute cholecystitis", "acute pancreatitis", "urinary tract infection",
            "pyelonephritis", "deep vein thrombosis", "generalised anxiety disorder", "panic attack",
            "viral upper respiratory infection", "viral pharyngitis", "acute pharyngitis"
        ]
        for d in diseases:
            if d in prompt_lower:
                candidates.append(d)

        # Deduplicate candidates
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            candidates = ["general"]

        target_diag = candidates[0]
        target_diag_und = target_diag.replace(" ", "_")
        
        if "follow-up questions" in prompt_lower or "questions" in prompt_lower:
            mock_json = json.dumps({
                "questions": [
                    {
                        "question": "Has the chest pain radiated to your left arm or jaw?" if "chest" in prompt_lower else f"How long have you been experiencing symptoms of {target_diag}?",
                        "purpose": "Evaluate for acute coronary syndrome or myocardial ischemia" if "chest" in prompt_lower else f"Evaluate for suspected {target_diag}",
                        "related_diagnosis": "unstable_angina" if "chest" in prompt_lower else target_diag_und,
                        "priority": "HIGH",
                        "clinical_importance": "HIGH"
                    },
                    {
                        "question": "Does taking deep breaths make the chest pain feel worse?" if "chest" in prompt_lower else "Have the symptoms been worsening or improving?",
                        "purpose": "Evaluate for pleuritic chest pain causes like pneumonia or pulmonary embolism" if "chest" in prompt_lower else "Check disease progression",
                        "related_diagnosis": "pneumonia" if "chest" in prompt_lower else target_diag_und,
                        "priority": "ROUTINE",
                        "clinical_importance": "MEDIUM"
                    }
                ]
            })
            
        elif "soap" in prompt_lower:
            # Never invent observations: echo the vitals carried on the prompt, and say so
            # plainly when none were recorded. A fallback note claiming normal vitals for a
            # hypertensive, tachycardic patient is a patient-safety hazard.
            vitals_match = re.search(r"^vitals:\s*(.+)$", prompt, re.IGNORECASE | re.MULTILINE)
            objective = (
                f"Recorded vitals: {vitals_match.group(1).strip()}."
                if vitals_match
                else "No vital signs were recorded during this consultation."
            )
            mock_json = json.dumps({
                "subjective": f"Patient is presenting with symptoms suggestive of {target_diag}.",
                "objective": objective,
                "assessment": f"Suspected {target_diag}.",
                "plan": f"Conservative management, monitoring, and follow-up for symptoms of {target_diag}."
            })
            
        elif "clinical summary" in prompt_lower:
            mock_json = json.dumps({
                "assessment": f"Symptom profile match for {target_diag}.",
                "key_findings": ["Fever" if "fever" in prompt_lower else "General symptoms"],
                "unresolved_issues": ["Confirm specific laboratory tests"],
                "next_steps": [f"Initiate care plan for {target_diag}"]
            })
            
        elif "investigation" in prompt_lower:
            mock_json = json.dumps({
                "investigations": [
                    {
                        "name": f"CBC (Complete Blood Count) for {target_diag}",
                        "reason": f"Indicated to evaluate systemic response for suspected {target_diag}.",
                        "priority": "ROUTINE",
                        "guideline_reference": "NICE Guidelines"
                    }
                ]
            })
            
        elif "ranked in the current order" in prompt_lower or "differential_explanation" in prompt_lower:
            explanations = {}
            for cand in candidates:
                cand_und = cand.replace(" ", "_")
                explanations[cand_und] = f"Ranked based on patient symptom profile and match score for {cand}."
            mock_json = json.dumps({
                "explanations": explanations
            })
            
        elif "patient" in prompt_lower:
            mock_json = json.dumps({
                "explanation_text": f"We are checking your symptoms to rule out conditions like {target_diag}.",
                "reassurance": "Your doctor is reviewing your symptoms and test results.",
                "warnings": ["Please seek immediate medical attention if you experience difficulty breathing or chest pain."]
            })
            
        else:
            mock_json = json.dumps({
                "alert_header": "Emergency Red Flag Alert",
                "first_responder_actions": ["Call emergency cardiology team", "Secure intravenous access"],
                "stabilization_notes": "Monitor vitals continuously and prepare defibrillation pad."
            })

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.record_success(latency_ms)
        return mock_json
