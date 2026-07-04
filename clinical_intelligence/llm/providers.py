from __future__ import annotations
import httpx
import json
import logging
import time
from typing import Any, Dict, List, Optional

from clinical_intelligence.config.llm import GEMINI_API_KEY, OPENAI_API_KEY
from clinical_intelligence.config.runtime import CIRCUIT_BREAKER, PROVIDER_TIMEOUT_SECONDS

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
        if schema:
            gen_config["responseMimeType"] = "application/json"

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_config
        }

        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_SECONDS) as client:
                response = await client.post(url, json=body, headers=headers)
                
            if response.status_code != 200:
                is_rate_limit = (response.status_code == 429)
                self.record_failure(is_rate_limit=is_rate_limit)
                response.raise_for_status()

            resp_data = response.json()
            text = resp_data["candidates"][0]["content"]["parts"][0]["text"]
            
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
            async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_SECONDS) as client:
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
        
        if "follow-up questions" in prompt_lower or "questions" in prompt_lower:
            mock_json = json.dumps({
                "questions": [
                    {
                        "question": "Has the chest pain radiated to your left arm or jaw?",
                        "purpose": "Evaluate for acute coronary syndrome or myocardial ischemia",
                        "related_diagnosis": "Acute Coronary Syndrome",
                        "priority": "HIGH",
                        "clinical_importance": "HIGH"
                    },
                    {
                        "question": "Does taking deep breaths make the chest pain feel worse?",
                        "purpose": "Evaluate for pleuritic chest pain causes like pneumonia or pulmonary embolism",
                        "related_diagnosis": "Pneumonia",
                        "priority": "ROUTINE",
                        "clinical_importance": "MEDIUM"
                    }
                ]
            })
            
        elif "soap" in prompt_lower:
            mock_json = json.dumps({
                "subjective": "Patient is a 68-year-old male presenting with acute chest pain, dyspnea, and sweating.",
                "objective": "BP 90/60, pulse 110, respiratory rate 24, SpO2 95%. ECG shows ST elevation.",
                "assessment": "ACS / Unstable Angina. Patient at high cardiac event risk (HEART Score 9.0).",
                "plan": "Administer oxygen, interpret urgent 12-lead ECG, check serial troponins. Absolute contraindicated: steroids due to diabetes."
            })
            
        elif "clinical summary" in prompt_lower:
            mock_json = json.dumps({
                "assessment": "Cardiovascular distress showing significant tachycardia and chest pressure.",
                "key_findings": ["Chest pain", "Dyspnea", "ST Elevation"],
                "unresolved_issues": ["Confirm exact troponin values"],
                "next_steps": ["Transfer to acute cardiology bay"]
            })
            
        elif "investigation" in prompt_lower:
            mock_json = json.dumps({
                "investigations": [
                    {
                        "name": "12-Lead ECG Interpretation",
                        "reason": "Indicated immediately for patients presenting with chest pain to rule out STEMI.",
                        "priority": "URGENT",
                        "guideline_reference": "AHA/ACC Chest Pain Guideline 2021"
                    },
                    {
                        "name": "High-Sensitivity Troponin Assays",
                        "reason": "Indicated to evaluate myocardial infarction within 1 hour.",
                        "priority": "URGENT",
                        "guideline_reference": "AHA/ACC Chest Pain Guideline 2021"
                    }
                ]
            })
            
        elif "ranked in the current order" in prompt_lower or "differential_explanation" in prompt_lower:
            mock_json = json.dumps({
                "explanations": {
                    "unstable_angina": "Ranked highest due to active exertional chest pain and high-risk cardiovascular profile.",
                    "acs": "Ranked second because of matching cardiac symptom triad (chest pain, sweating, dyspnea) and high score index."
                }
            })
            
        elif "patient" in prompt_lower:
            mock_json = json.dumps({
                "explanation_text": "We are closely checking your heart function because of the pressure in your chest.",
                "reassurance": "You are currently in a monitored clinic environment.",
                "warnings": ["Please notify the team immediately if the chest pain worsens or spreads to your arm."]
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
