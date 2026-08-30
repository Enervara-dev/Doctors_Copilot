"""
Structured-output contract for the Gemini channel.

Asking only for `responseMimeType: application/json` left the model free to choose
its own envelope: the SOAP prompt came back as {"consultation_notes": {...}}, the
validator rejected it for missing top-level keys, and the pipeline silently served
canned fallback text. The reply must be constrained to the schema the validator
checks, and a truncated or empty reply must say so rather than raising an opaque
KeyError.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

from clinical_intelligence.llm.providers import (
    GeminiProvider,
    _read_gemini_text,
    _to_response_schema,
)
from clinical_intelligence.prompts.registry import PromptRegistry


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise AssertionError("should not be called on a 200")


class FakeClient:
    """Captures the request body so the generationConfig can be asserted on."""

    last_body = None

    def __init__(self, *_, **__):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def post(self, _url, json=None, headers=None):
        FakeClient.last_body = json
        return FakeResponse({
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": '{"subjective":"s","objective":"o",'
                                              '"assessment":"a","plan":"p"}'}]},
            }],
            "usageMetadata": {"thoughtsTokenCount": 120, "candidatesTokenCount": 40},
        })


class TestResponseSchema(unittest.TestCase):
    def test_registry_schema_is_forwarded_as_response_schema(self):
        soap = PromptRegistry().get("soap_v1").expected_json_schema
        provider = GeminiProvider("gemini-2.5-flash")

        with patch("clinical_intelligence.llm.providers.GEMINI_API_KEY", "test-key"), \
             patch("clinical_intelligence.llm.providers.httpx.AsyncClient", FakeClient):
            text = asyncio.run(provider.generate("prompt", schema=soap))

        cfg = FakeClient.last_body["generationConfig"]
        self.assertEqual(cfg.get("responseMimeType"), "application/json")
        self.assertIn("responseSchema", cfg, "the schema must constrain the reply, not just the mime type")
        self.assertEqual(sorted(cfg["responseSchema"]["properties"]),
                         ["assessment", "objective", "plan", "subjective"])
        self.assertEqual(sorted(json.loads(text)), ["assessment", "objective", "plan", "subjective"])

    def test_no_schema_means_no_response_schema(self):
        provider = GeminiProvider("gemini-2.5-flash")
        with patch("clinical_intelligence.llm.providers.GEMINI_API_KEY", "test-key"), \
             patch("clinical_intelligence.llm.providers.httpx.AsyncClient", FakeClient):
            asyncio.run(provider.generate("prompt", schema=None))
        self.assertNotIn("responseSchema", FakeClient.last_body["generationConfig"])

    def test_unsupported_keywords_are_dropped(self):
        """Forwarding `additionalProperties` or `$schema` makes the request 400."""
        cleaned = _to_response_schema({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "properties": {"questions": {"type": "array",
                                         "items": {"type": "object", "additionalProperties": True,
                                                   "properties": {"question": {"type": "string"}}}}},
            "required": ["questions"],
        })
        self.assertNotIn("$schema", cleaned)
        self.assertNotIn("additionalProperties", cleaned)
        self.assertNotIn("additionalProperties", cleaned["properties"]["questions"]["items"])
        self.assertEqual(cleaned["required"], ["questions"])

    def test_every_registered_prompt_schema_survives_conversion(self):
        registry = PromptRegistry()
        for prompt_id in ("soap_v1", "adaptive_questions_v1", "clinical_summary_v1",
                          "investigation_v1", "patient_explanation_v1"):
            schema = registry.get(prompt_id).expected_json_schema
            converted = _to_response_schema(schema)
            self.assertIsNotNone(converted, prompt_id)
            for key in schema.get("required", []):
                self.assertIn(key, converted["properties"], f"{prompt_id}: {key} lost in conversion")


class TestTruncatedReplies(unittest.TestCase):
    def test_no_parts_reports_the_finish_reason(self):
        """Thinking can consume the whole budget, leaving a candidate with no parts."""
        with self.assertRaises(ValueError) as ctx:
            _read_gemini_text({
                "candidates": [{"finishReason": "MAX_TOKENS", "content": {}}],
                "usageMetadata": {"thoughtsTokenCount": 1024},
            })
        self.assertIn("MAX_TOKENS", str(ctx.exception))
        self.assertIn("1024", str(ctx.exception))

    def test_no_candidates_reports_prompt_feedback(self):
        with self.assertRaises(ValueError) as ctx:
            _read_gemini_text({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})
        self.assertIn("SAFETY", str(ctx.exception))

    def test_multipart_replies_are_joined(self):
        text = _read_gemini_text({
            "candidates": [{"finishReason": "STOP",
                            "content": {"parts": [{"text": '{"a":'}, {"text": '1}'}]}}]
        })
        self.assertEqual(json.loads(text), {"a": 1})


class TestTokenBudget(unittest.TestCase):
    def test_orchestrator_uses_the_configured_ceiling(self):
        """A ceiling sized for the JSON alone truncates it once thinking is counted."""
        from clinical_intelligence.config.llm import MAX_RESPONSE_TOKENS
        from clinical_intelligence.llm.orchestrator import LLMOrchestrator

        self.assertGreaterEqual(MAX_RESPONSE_TOKENS, 2048,
                                "needs headroom for thinking tokens plus the reply")
        self.assertEqual(LLMOrchestrator().gemini.max_tokens, MAX_RESPONSE_TOKENS)


class TestRunsInsideARunningLoop(unittest.IsolatedAsyncioTestCase):
    """
    The reasoning engine is called from a FastAPI request, so the LLM runner has to
    re-enter an already-running event loop. That path imports nest_asyncio lazily and
    the package was never declared, so under the server every task raised
    ModuleNotFoundError — while the same code passed from a plain script, where the
    loop is not running.

    Since the mock fallback was removed the llm_* fields are empty whenever no
    provider answers, so the assertion is on llm_status: the run must report a
    definite outcome rather than dying on the import.
    """

    async def test_reasoning_populates_llm_fields_under_a_running_loop(self):
        from clinical_intelligence.integration import reason_from_state

        state = {
            "session_id": "loop-check",
            "patient_info": {"age": 58, "gender": "male"},
            "medical_history": ["Hypertension"],
            "current_medications": ["Amlodipine"],
            "vital_signs": {"bp": {"value": "160/95", "unit": "mmHg"}},
            "symptoms": [{"name": "chest pain", "severity": "severe", "present": True, "status": "Active"}],
        }
        self.assertTrue(asyncio.get_running_loop().is_running())
        result = await asyncio.to_thread(reason_from_state, state)

        self.assertIn(result.llm_status, ("ok", "unavailable", "disabled"),
                      "the running-loop path must complete and report an outcome")
        if result.llm_status == "ok":
            self.assertIsNotNone(result.llm_soap_notes)
        else:
            # No stand-in text: empty means empty.
            self.assertIsNone(result.llm_soap_notes)
            self.assertEqual(result.llm_adaptive_questions, [])

    async def test_runner_dependency_is_importable(self):
        """runner.py imports this only when the loop is running, so a missing
        install shows up in production and never in a unit test."""
        import importlib
        importlib.import_module("nest_asyncio")


if __name__ == "__main__":
    unittest.main()
