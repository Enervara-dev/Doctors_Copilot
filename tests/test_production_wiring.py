"""
The deployable assembly.

Configuration becomes objects in exactly one place, so this file tests that
place: what each setting produces, that the worker starts and stops with the
application, that health reports honestly, and — the one that matters most —
that every intelligence failure mode leaves the clinical API untouched.

Offline. The only model client that appears here is scripted or unavailable.
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict
from unittest import mock

os.environ.setdefault("CLINICAL_COPILOT_ENABLE_GRAPH", "False")
os.environ.setdefault("CLINICAL_COPILOT_ENABLE_LLM", "False")

from fastapi.testclient import TestClient  # noqa: E402

from app.utils.jwt_utils import generate_jwt_token  # noqa: E402
from copilot_api.bootstrap import (  # noqa: E402
    PLACEHOLDER_JWT_SECRET,
    build_application,
    build_intelligence,
    build_provider,
    check_auth_secret,
    create_production_app,
)
from copilot_api.main import API_PREFIX, SnapshotBroker  # noqa: E402
from copilot_api.settings import CopilotSettings  # noqa: E402
from integration.copilot_event import (  # noqa: E402
    CopilotEvent,
    Provenance,
    SpeakerRole,
    UtterancePayload,
)
from integration.intelligence.langgraph_provider import (  # noqa: E402
    LangGraphIntelligenceProvider,
)
from integration.intelligence.model_client import ModelUnavailable  # noqa: E402
from integration.intelligence.model_clients import (  # noqa: E402
    PlatformModelClient,
    _parse_json_object,
    build_model_client,
)
from integration.intelligence.providers import (  # noqa: E402
    DeterministicIntelligenceProvider,
    NullIntelligenceProvider,
)

CARDIAC = [
    "I am a 62 year old male. I have severe chest pain and I am sweating.",
    "I also feel short of breath.",
]


def utterance(consultation_id: str, sequence: int, text: str) -> Dict[str, Any]:
    return CopilotEvent(
        event_id=f"utt-{sequence:03d}",
        consultation_id=consultation_id,
        sequence=sequence,
        provenance=Provenance.FIXTURE,
        payload=UtterancePayload(
            text=text,
            speaker_role=SpeakerRole.PATIENT,
            speaker_label="Patient",
            is_final=True,
            confidence=0.95,
            started_at="2026-09-28T09:00:00.000Z",
            finalized_at="2026-09-28T09:00:04.000Z",
        ),
    ).model_dump(mode="json")


def headers(consultation_id: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {generate_jwt_token(consultation_id, 'doctor')}"}


def no_model_configured():
    """
    Hides every model key for the duration of the block.

    Without this, a developer machine that has a real key turns a wiring test
    into a live API call — which this suite must never make. `build_model_client`
    reads the keys when it is called, so patching them around `build_application`
    is enough to force the declared `UnavailableModelClient` path.
    """
    return mock.patch.multiple(
        "clinical_intelligence.config.llm",
        GEMINI_API_KEY="",
        OPENAI_API_KEY="",
    )


# ══════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════


class TestSettings(unittest.TestCase):
    def test_the_default_is_a_clinical_only_deployment(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in list(os.environ):
                if key.startswith("COPILOT_"):
                    del os.environ[key]
            settings = CopilotSettings.from_env()

        self.assertFalse(settings.intelligence_enabled)

    def test_flags_and_numbers_are_read_from_the_environment(self) -> None:
        env = {
            "COPILOT_INTELLIGENCE_ENABLED": "true",
            "COPILOT_INTELLIGENCE_PROVIDER": "deterministic",
            "COPILOT_INTELLIGENCE_BUDGET_MS": "1234",
            "COPILOT_SESSION_IDLE_TIMEOUT_SECONDS": "60",
        }
        with mock.patch.dict(os.environ, env):
            settings = CopilotSettings.from_env()

        self.assertTrue(settings.intelligence_enabled)
        self.assertEqual(settings.intelligence_provider, "deterministic")
        self.assertEqual(settings.intelligence_budget_ms, 1234.0)
        self.assertEqual(settings.session_idle_timeout_seconds, 60.0)

    def test_an_unparseable_number_falls_back_rather_than_crashing(self) -> None:
        with mock.patch.dict(os.environ, {"COPILOT_INTELLIGENCE_BUDGET_MS": "soon"}):
            self.assertEqual(CopilotSettings.from_env().intelligence_budget_ms, 8000.0)

    def test_describe_carries_no_secret(self) -> None:
        described = json.dumps(CopilotSettings().describe()).lower()
        for forbidden in ("key", "secret", "token", "password"):
            self.assertNotIn(forbidden, described)


# ══════════════════════════════════════════════════════════════════════════
# Provider selection
# ══════════════════════════════════════════════════════════════════════════


class TestProviderSelection(unittest.TestCase):
    def test_null_by_name(self) -> None:
        provider = build_provider(CopilotSettings(intelligence_provider="null"))
        self.assertIsInstance(provider, NullIntelligenceProvider)

    def test_deterministic_by_name(self) -> None:
        provider = build_provider(CopilotSettings(intelligence_provider="deterministic"))
        self.assertIsInstance(provider, DeterministicIntelligenceProvider)

    def test_langgraph_by_name(self) -> None:
        with no_model_configured():
            provider = build_provider(
                CopilotSettings(intelligence_provider="langgraph")
            )
        self.assertIsInstance(provider, LangGraphIntelligenceProvider)

    def test_a_typo_degrades_the_optional_layer_rather_than_failing_to_boot(self) -> None:
        provider = build_provider(CopilotSettings(intelligence_provider="langraph"))
        self.assertIsInstance(provider, NullIntelligenceProvider)

    def test_no_service_is_built_when_the_layer_is_off(self) -> None:
        self.assertIsNone(
            build_intelligence(CopilotSettings(intelligence_enabled=False), SnapshotBroker())
        )


# ══════════════════════════════════════════════════════════════════════════
# The model adapter
# ══════════════════════════════════════════════════════════════════════════


class TestModelAdapter(unittest.TestCase):
    def test_a_missing_key_yields_an_unavailable_client_not_an_exception(self) -> None:
        with mock.patch("clinical_intelligence.config.llm.GEMINI_API_KEY", ""):
            client = build_model_client(provider_name="gemini")
        self.assertEqual(client.name, "unavailable")

    def test_an_unknown_provider_yields_an_unavailable_client(self) -> None:
        self.assertEqual(build_model_client(provider_name="mistral").name, "unavailable")

    def test_plain_json_is_parsed(self) -> None:
        self.assertEqual(_parse_json_object('{"a": 1}'), {"a": 1})

    def test_a_fenced_block_is_recovered(self) -> None:
        self.assertEqual(
            _parse_json_object('Here you go:\n```json\n{"a": 1}\n```'), {"a": 1}
        )

    def test_a_preamble_before_braces_is_recovered(self) -> None:
        self.assertEqual(_parse_json_object('Sure! {"a": 1} Hope that helps.'), {"a": 1})

    def test_unrecoverable_text_raises_model_unavailable(self) -> None:
        with self.assertRaises(ModelUnavailable):
            _parse_json_object("I cannot help with that.")

    def test_a_json_array_is_not_accepted_as_an_object(self) -> None:
        with self.assertRaises(ModelUnavailable):
            _parse_json_object("[1, 2, 3]")

    def test_a_provider_exception_becomes_model_unavailable(self) -> None:
        class Failing:
            model_name = "x"

            async def generate(self, prompt, schema=None):
                raise RuntimeError("upstream 503")

        client = PlatformModelClient(Failing())
        with self.assertRaises(ModelUnavailable):
            client.complete(prompt="p", schema={}, purpose="synthesis")

    def test_an_empty_reply_becomes_model_unavailable(self) -> None:
        class Empty:
            model_name = "x"

            async def generate(self, prompt, schema=None):
                return "   "

        with self.assertRaises(ModelUnavailable):
            PlatformModelClient(Empty()).complete(prompt="p", schema={}, purpose="s")

    def test_a_successful_call_returns_parsed_json(self) -> None:
        class Good:
            model_name = "x"

            async def generate(self, prompt, schema=None):
                return '```json\n{"summary": "ok", "refs": []}\n```'

        parsed = PlatformModelClient(Good()).complete(
            prompt="p", schema={}, purpose="synthesis"
        )
        self.assertEqual(parsed["summary"], "ok")


# ══════════════════════════════════════════════════════════════════════════
# The assembled application
# ══════════════════════════════════════════════════════════════════════════


class TestAssembledApplication(unittest.TestCase):
    def test_the_default_deployment_serves_clinical_output_with_no_config(self) -> None:
        app, service = build_application(CopilotSettings())
        self.assertIsNone(service)

        cid = "prod-1"
        with TestClient(app) as client:
            response = client.post(
                f"{API_PREFIX}/sessions/{cid}/events",
                json=utterance(cid, 1, CARDIAC[0]),
                headers=headers(cid),
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["results"])

            insights = client.get(
                f"{API_PREFIX}/sessions/{cid}/insights", headers=headers(cid)
            )
            self.assertEqual(insights.status_code, 200)
            self.assertTrue(insights.json()["red_flags"])

            intelligence = client.get(
                f"{API_PREFIX}/sessions/{cid}/intelligence", headers=headers(cid)
            )
            self.assertEqual(intelligence.status_code, 404)

    def test_the_worker_starts_and_stops_with_the_application(self) -> None:
        app, service = build_application(
            CopilotSettings(intelligence_enabled=True, intelligence_provider="deterministic")
        )
        self.assertIsNotNone(service)
        self.assertFalse(service.is_running, "not started before the app runs")

        with TestClient(app):
            self.assertTrue(service.is_running)

        self.assertFalse(service.is_running, "stopped when the app shut down")

    def test_the_full_path_runs_under_the_assembled_app(self) -> None:
        app, service = build_application(
            CopilotSettings(intelligence_enabled=True, intelligence_provider="deterministic")
        )
        cid = "prod-2"
        with TestClient(app) as client:
            for sequence, text in enumerate(CARDIAC, start=1):
                self.assertEqual(
                    client.post(
                        f"{API_PREFIX}/sessions/{cid}/events",
                        json=utterance(cid, sequence, text),
                        headers=headers(cid),
                    ).status_code,
                    200,
                )
            self.assertTrue(service.wait_until_idle(30))

            snapshot = client.get(
                f"{API_PREFIX}/sessions/{cid}/intelligence", headers=headers(cid)
            )
            self.assertEqual(snapshot.status_code, 200)
            body = snapshot.json()
            self.assertEqual(body["consultation_id"], cid)
            self.assertTrue(body["proposals"])
            for proposal in body["proposals"]:
                self.assertTrue(proposal["evidence"])

    def test_a_model_that_never_answers_leaves_the_clinical_api_intact(self) -> None:
        """
        The configuration a real deployment hits first: the layer is on, no key
        is set. Clinical output must be indistinguishable from a deployment
        with the layer off.
        """
        with no_model_configured():
            app, service = build_application(
                CopilotSettings(
                    intelligence_enabled=True, intelligence_provider="langgraph"
                )
            )
        cid = "prod-3"
        with TestClient(app) as client:
            for sequence, text in enumerate(CARDIAC, start=1):
                client.post(
                    f"{API_PREFIX}/sessions/{cid}/events",
                    json=utterance(cid, sequence, text),
                    headers=headers(cid),
                )
            self.assertTrue(service.wait_until_idle(30))

            insights = client.get(
                f"{API_PREFIX}/sessions/{cid}/insights", headers=headers(cid)
            )
            self.assertEqual(insights.status_code, 200)
            self.assertTrue(insights.json()["red_flags"])

            snapshot = client.get(
                f"{API_PREFIX}/sessions/{cid}/intelligence", headers=headers(cid)
            ).json()
            self.assertTrue(snapshot["degraded"])
            self.assertEqual(snapshot["proposals"], [])

    def test_closing_a_consultation_releases_both_layers(self) -> None:
        app, service = build_application(
            CopilotSettings(intelligence_enabled=True, intelligence_provider="deterministic")
        )
        cid = "prod-4"
        with TestClient(app) as client:
            client.post(
                f"{API_PREFIX}/sessions/{cid}/events",
                json=utterance(cid, 1, CARDIAC[0]),
                headers=headers(cid),
            )
            service.wait_until_idle(20)
            self.assertIsNotNone(service.latest(cid))

            client.post(f"{API_PREFIX}/sessions/{cid}/close", headers=headers(cid))
            self.assertIsNone(service.latest(cid))
            self.assertEqual(client.get("/health").json()["sessions"], 0)


# ══════════════════════════════════════════════════════════════════════════
# The signing key
# ══════════════════════════════════════════════════════════════════════════


class TestAuthSecretGuard(unittest.TestCase):
    """
    The placeholder in `app/core/config.py` is a *working* secret, so nothing
    fails without this guard: the service comes up and authenticates against a
    key that is published in the repository.
    """

    def test_the_placeholder_refuses_to_start(self) -> None:
        with mock.patch("app.core.config.settings.JWT_SECRET_KEY", PLACEHOLDER_JWT_SECRET):
            with self.assertRaises(RuntimeError) as raised:
                check_auth_secret()

        message = str(raised.exception)
        # The prefixed name, not the bare one: the settings class ignores
        # `JWT_SECRET_KEY`, so naming it here would send someone in circles.
        self.assertIn("CLINICAL_COPILOT_JWT_SECRET_KEY", message)
        self.assertIn("COPILOT_ALLOW_INSECURE_AUTH", message, "says how to proceed")

    def test_a_configured_secret_passes(self) -> None:
        with mock.patch(
            "app.core.config.settings.JWT_SECRET_KEY", "a-real-deployment-secret"
        ):
            check_auth_secret()  # no raise

    def test_the_opt_out_is_honoured(self) -> None:
        with mock.patch("app.core.config.settings.JWT_SECRET_KEY", PLACEHOLDER_JWT_SECRET):
            check_auth_secret(allow_insecure=True)  # no raise

    def test_the_guard_runs_on_the_real_entry_point(self) -> None:
        """
        `create_production_app` is what the container command names, so that is
        where the check has to bite — not merely in a helper nobody calls.
        """
        env = {"COPILOT_ALLOW_INSECURE_AUTH": "false"}
        with mock.patch("app.core.config.settings.JWT_SECRET_KEY", PLACEHOLDER_JWT_SECRET):
            with mock.patch.dict(os.environ, env):
                with self.assertRaises(RuntimeError):
                    create_production_app()

    def test_the_entry_point_builds_with_a_configured_secret(self) -> None:
        with mock.patch(
            "app.core.config.settings.JWT_SECRET_KEY", "a-real-deployment-secret"
        ):
            with mock.patch.dict(os.environ, {}, clear=False):
                for key in list(os.environ):
                    if key.startswith("COPILOT_"):
                        del os.environ[key]
                app = create_production_app()

        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)

    def test_a_test_assembly_is_not_subject_to_the_guard(self) -> None:
        """
        `build_application` must stay usable with whatever the machine has, or
        every test in this repository would depend on a deployment variable.
        """
        app, _ = build_application(CopilotSettings())
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)


# ══════════════════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════════════════


class TestHealth(unittest.TestCase):
    def test_health_needs_no_credential(self) -> None:
        app, _ = build_application(CopilotSettings())
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)

    def test_health_reports_the_layer_as_absent_when_disabled(self) -> None:
        app, _ = build_application(CopilotSettings())
        with TestClient(app) as client:
            body = client.get("/health").json()
        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["intelligence"], {"enabled": False})

    def test_health_reports_worker_state_and_counters(self) -> None:
        app, service = build_application(
            CopilotSettings(intelligence_enabled=True, intelligence_provider="deterministic")
        )
        cid = "health-1"
        with TestClient(app) as client:
            client.post(
                f"{API_PREFIX}/sessions/{cid}/events",
                json=utterance(cid, 1, CARDIAC[0]),
                headers=headers(cid),
            )
            self.assertTrue(service.wait_until_idle(20))
            body = client.get("/health").json()

        intelligence = body["intelligence"]
        self.assertTrue(intelligence["enabled"])
        self.assertTrue(intelligence["worker_running"])
        self.assertEqual(intelligence["provider"], "deterministic")
        self.assertGreaterEqual(intelligence["submitted"], 1)
        self.assertGreaterEqual(intelligence["completed"], 1)

    def test_a_dead_model_does_not_make_the_service_unhealthy(self) -> None:
        """
        An orchestrator must not restart a container that is serving clinical
        output correctly because a model is down.
        """
        with no_model_configured():
            app, service = build_application(
                CopilotSettings(
                    intelligence_enabled=True, intelligence_provider="langgraph"
                )
            )
        cid = "health-2"
        with TestClient(app) as client:
            client.post(
                f"{API_PREFIX}/sessions/{cid}/events",
                json=utterance(cid, 1, CARDIAC[0]),
                headers=headers(cid),
            )
            service.wait_until_idle(20)
            body = client.get("/health").json()

        self.assertEqual(body["status"], "healthy")
        self.assertTrue(body["intelligence"]["enabled"])


if __name__ == "__main__":
    unittest.main()
