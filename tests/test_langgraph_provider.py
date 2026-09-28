"""
The LangGraph intelligence provider.

Two things are under test. The first is that the graph works: four nodes, real
`langgraph` execution, proposals that survive the validation boundary. The
second matters more — that a model behaving badly cannot do damage. Most of
this file is the second.

No live calls anywhere. Every node reaches a `ScriptedModelClient`, so the
graph runs to completion offline and deterministically.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("CLINICAL_COPILOT_ENABLE_GRAPH", "False")
os.environ.setdefault("CLINICAL_COPILOT_ENABLE_LLM", "False")

from fastapi.testclient import TestClient  # noqa: E402

from app.utils.jwt_utils import generate_jwt_token  # noqa: E402
from copilot_api.main import API_PREFIX, create_app  # noqa: E402
from integration.copilot_event import (  # noqa: E402
    CopilotEvent,
    Provenance,
    SpeakerRole,
    UtterancePayload,
)
from integration.copilot_runtime import CopilotResult, CopilotRuntime  # noqa: E402
from integration.intelligence import (  # noqa: E402
    CopilotContext,
    IntelligenceGateway,
    IntelligenceProvider,
    IntelligenceStatus,
    LangGraphIntelligenceProvider,
    ModelUnavailable,
    ProposalKind,
    ScriptedModelClient,
    UnavailableModelClient,
    build_context,
)

CARDIAC = [
    "I am a 62 year old male. I have severe chest pain and I am sweating.",
    "I also feel short of breath.",
    "I have high blood pressure and diabetes.",
]


def utterance(consultation_id: str, sequence: int, text: str) -> CopilotEvent:
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
    )


def cardiac_result(consultation_id: str = "lg-1") -> CopilotResult:
    runtime = CopilotRuntime()
    for sequence, text in enumerate(CARDIAC, start=1):
        runtime.ingest(utterance(consultation_id, sequence, text))
    return runtime.get_session(consultation_id).latest_result


def good_script() -> Dict[str, Any]:
    """A well-behaved model: grounded, in scope, citing real references."""
    return {
        "synthesis": {
            "summary": "62-year-old man with severe chest pain, sweating and "
            "breathlessness on a background of hypertension and diabetes.",
            "refs": ["symptom:symptom_chest_pain", "dx:unstable_angina"],
        },
        "unresolved": {
            "items": [
                {"text": "What is the blood pressure now?", "refs": ["missing:blood_pressure"]},
                {"text": "Does the pain radiate anywhere?", "refs": ["symptom:symptom_chest_pain"]},
            ]
        },
        "explanations": {
            "items": [
                {
                    "text": "Unstable angina fits the pain and risk profile; an ECG "
                    "has not yet excluded infarction.",
                    "refs": ["dx:unstable_angina", "symptom:symptom_chest_pain"],
                }
            ]
        },
        "investigation_notes": {
            "items": [
                {
                    "text": "A normal first ECG does not exclude ACS; repeat if the "
                    "pain continues.",
                    "refs": ["inv:ecg_12lead"],
                }
            ]
        },
    }


def consult(script: Dict[str, Any] | None = None, **kwargs):
    model = ScriptedModelClient(script if script is not None else good_script(), **kwargs)
    result = cardiac_result()
    outcome = IntelligenceGateway(LangGraphIntelligenceProvider(model)).consult(result)
    return outcome, model, result


# ══════════════════════════════════════════════════════════════════════════
# The graph runs
# ══════════════════════════════════════════════════════════════════════════


class TestTheGraphRuns(unittest.TestCase):
    def test_it_is_a_real_langgraph(self) -> None:
        from langgraph.graph.state import CompiledStateGraph

        provider = LangGraphIntelligenceProvider(ScriptedModelClient(good_script()))
        self.assertIsInstance(provider._graph, CompiledStateGraph)

    def test_it_satisfies_the_provider_protocol(self) -> None:
        self.assertIsInstance(
            LangGraphIntelligenceProvider(UnavailableModelClient()), IntelligenceProvider
        )

    def test_all_four_nodes_contribute(self) -> None:
        outcome, model, _ = consult()

        self.assertEqual(outcome.status, IntelligenceStatus.OK)
        self.assertEqual(
            {c["purpose"] for c in model.calls},
            {"synthesis", "unresolved", "explanations", "investigation_notes"},
        )
        kinds = {p.kind for p in outcome.proposals}
        self.assertIn(ProposalKind.SUMMARY, kinds)
        self.assertIn(ProposalKind.QUESTION, kinds)
        self.assertIn(ProposalKind.INVESTIGATION_NOTE, kinds)

    def test_nothing_is_rejected_by_the_boundary(self) -> None:
        """
        The graph constructs grounding rather than asking for it.

        Each node is handed the refs it may cite and filters the rest, so the
        validator should never see an unknown reference from this provider.
        """
        outcome, _, _ = consult()
        self.assertEqual(outcome.rejected, [])

    def test_nodes_run_in_order(self) -> None:
        _, model, _ = consult()
        self.assertEqual(
            [c["purpose"] for c in model.calls],
            ["synthesis", "unresolved", "explanations", "investigation_notes"],
        )

    def test_the_prompt_carries_the_guardrail_and_the_evidence(self) -> None:
        _, model, _ = consult()
        prompt = model.prompt_for("synthesis")

        self.assertIn("remains", prompt)
        self.assertIn("authoritative", prompt)
        self.assertIn("symptom:symptom_chest_pain", prompt)
        # The patient's own words reach the model, not just canonical ids.
        self.assertIn("chest pain", prompt)

    def test_an_empty_consultation_short_circuits(self) -> None:
        runtime = CopilotRuntime()
        runtime.ingest(utterance("empty-1", 1, "Good morning."))
        result = runtime.get_session("empty-1").latest_result

        model = ScriptedModelClient(good_script())
        outcome = IntelligenceGateway(LangGraphIntelligenceProvider(model)).consult(result)

        self.assertEqual(model.calls, [], "no model call for an empty consultation")
        self.assertEqual(outcome.proposals, [])

    def test_proposal_ids_are_replay_stable(self) -> None:
        first, _, _ = consult()
        second, _, _ = consult()
        self.assertEqual(
            [p.proposal_id for p in first.proposals],
            [p.proposal_id for p in second.proposals],
        )


# ══════════════════════════════════════════════════════════════════════════
# A badly behaved model
# ══════════════════════════════════════════════════════════════════════════


class TestAMisbehavingModelCannotDoDamage(unittest.TestCase):
    def test_an_invented_reference_is_dropped_not_propagated(self) -> None:
        script = good_script()
        script["unresolved"] = {
            "items": [
                {
                    "text": "Has the patient had a recent PET scan?",
                    "refs": ["symptom:symptom_haemoptysis", "missing:blood_pressure"],
                }
            ]
        }
        outcome, _, _ = consult(script)

        questions = [p for p in outcome.proposals if p.kind is ProposalKind.QUESTION]
        self.assertEqual(len(questions), 1)
        # The invented ref is gone; the real one survives.
        self.assertEqual(questions[0].grounded_in, ["missing:blood_pressure"])
        self.assertEqual(outcome.rejected, [])

    def test_a_wholly_ungrounded_item_produces_no_proposal(self) -> None:
        script = good_script()
        script["unresolved"] = {
            "items": [{"text": "Ask about their holiday.", "refs": ["symptom:invented"]}]
        }
        outcome, _, _ = consult(script)
        self.assertEqual(
            [p for p in outcome.proposals if p.kind is ProposalKind.QUESTION], []
        )

    def test_a_model_cannot_introduce_an_investigation(self) -> None:
        """
        The reconciliation rule, enforced twice.

        The node is only shown recommended investigations, and a note that
        names anything else has nothing to attach to.
        """
        script = good_script()
        script["investigation_notes"] = {
            "items": [
                {"text": "Arrange a whole-body MRI.", "refs": ["inv:whole_body_mri"]}
            ]
        }
        outcome, _, _ = consult(script)

        notes = [p for p in outcome.proposals if p.kind is ProposalKind.INVESTIGATION_NOTE]
        self.assertEqual(notes, [])
        self.assertNotIn("whole_body_mri", json.dumps(outcome.model_dump(), default=str))

    def test_a_model_cannot_change_a_priority_or_a_rank(self) -> None:
        script = good_script()
        script["investigation_notes"] = {
            "items": [
                {
                    "text": "Downgrade this to routine.",
                    "refs": ["inv:ecg_12lead"],
                    "priority": "ROUTINE",
                    "rank": 1,
                }
            ]
        }
        outcome, _, result = consult(script)

        # The extra keys are simply not part of a Proposal, so they vanish.
        note = next(p for p in outcome.proposals if p.kind is ProposalKind.INVESTIGATION_NOTE)
        self.assertFalse(hasattr(note, "priority"))
        ecg = next(
            i for i in result.reasoning.recommended_investigations
            if i.investigation_id == "ecg_12lead"
        )
        self.assertEqual(ecg.priority, "URGENT")

    def test_a_runaway_generation_is_refused_by_the_boundary(self) -> None:
        script = good_script()
        script["synthesis"] = {
            "summary": "x" * 5000,
            "refs": ["symptom:symptom_chest_pain"],
        }
        outcome, _, _ = consult(script)
        self.assertIn("CONTENT_TOO_LONG", [r.code for r in outcome.rejected])

    def test_malformed_json_from_a_node_costs_only_that_node(self) -> None:
        script = good_script()
        script["explanations"] = {"unexpected": "shape"}
        outcome, _, _ = consult(script)

        self.assertEqual(outcome.status, IntelligenceStatus.OK)
        self.assertTrue(outcome.proposals)
        self.assertEqual(
            [p for p in outcome.proposals if p.metadata.get("node") == "explanations"], []
        )

    def test_a_node_whose_model_call_raises_costs_only_that_node(self) -> None:
        outcome, _, _ = consult(
            good_script(), fail_with={"synthesis": ModelUnavailable("no key")}
        )

        self.assertEqual(outcome.status, IntelligenceStatus.OK)
        self.assertTrue(outcome.proposals)
        self.assertEqual(
            [p for p in outcome.proposals if p.metadata.get("node") == "synthesis"], []
        )
        self.assertIn("synthesis", str(outcome.telemetry["node_notes"]))

    def test_an_unexpected_exception_in_a_node_is_contained(self) -> None:
        outcome, _, _ = consult(
            good_script(), fail_with={"unresolved": ValueError("bad state")}
        )
        self.assertEqual(outcome.status, IntelligenceStatus.OK)
        self.assertTrue(outcome.proposals)

    def test_every_node_failing_reports_unavailable_not_ok(self) -> None:
        """Silence must not read as 'the model had nothing to add'."""
        outcome = IntelligenceGateway(
            LangGraphIntelligenceProvider(UnavailableModelClient())
        ).consult(cardiac_result())

        self.assertEqual(outcome.status, IntelligenceStatus.UNAVAILABLE)
        self.assertEqual(outcome.proposals, [])
        self.assertTrue(outcome.message)


# ══════════════════════════════════════════════════════════════════════════
# The clinical core stays authoritative
# ══════════════════════════════════════════════════════════════════════════


class TestTheClinicalCoreIsUntouched(unittest.TestCase):
    def test_running_the_graph_changes_nothing_clinical(self) -> None:
        result = cardiac_result("untouched-1")
        before = result.model_dump_json()
        IntelligenceGateway(
            LangGraphIntelligenceProvider(ScriptedModelClient(good_script()))
        ).consult(result)
        self.assertEqual(result.model_dump_json(), before)

    def test_the_graph_receives_a_frozen_context(self) -> None:
        captured: List[CopilotContext] = []

        class Capturing(ScriptedModelClient):
            pass

        provider = LangGraphIntelligenceProvider(ScriptedModelClient(good_script()))
        original = provider._synthesis

        def spy(state):
            captured.append(state["context"])
            return original(state)

        provider._synthesis = spy  # type: ignore[assignment]
        provider._graph = provider._build_graph()
        IntelligenceGateway(provider).consult(cardiac_result("frozen-1"))

        self.assertTrue(captured)
        with self.assertRaises(Exception):
            captured[0].consultation_id = "elsewhere"  # type: ignore[misc]

    def test_the_provider_is_swappable(self) -> None:
        from integration.intelligence import (
            DeterministicIntelligenceProvider,
            NullIntelligenceProvider,
        )

        result = cardiac_result("swap-1")
        for provider in (
            NullIntelligenceProvider(),
            DeterministicIntelligenceProvider(),
            LangGraphIntelligenceProvider(ScriptedModelClient(good_script())),
        ):
            outcome = IntelligenceGateway(provider).consult(result)
            self.assertEqual(outcome.consultation_id, result.consultation_id)
            self.assertEqual(outcome.fingerprint, result.fingerprint)


# ══════════════════════════════════════════════════════════════════════════
# The complete path
# ══════════════════════════════════════════════════════════════════════════


class TestTranscriptToFastApi(unittest.TestCase):
    """
    Transcript → ClinicalState → CopilotResult → CopilotContext → LangGraph →
    validated proposals, alongside the unchanged FastAPI clinical surface.
    """

    def setUp(self) -> None:
        self.runtime = CopilotRuntime()
        self.client = TestClient(create_app(self.runtime))
        self.cid = "path-1"
        self.headers = {"Authorization": f"Bearer {generate_jwt_token(self.cid, 'doctor')}"}

    def test_the_whole_path(self) -> None:
        for sequence, text in enumerate(CARDIAC, start=1):
            response = self.client.post(
                f"{API_PREFIX}/sessions/{self.cid}/events",
                json=utterance(self.cid, sequence, text).model_dump(mode="json"),
                headers=self.headers,
            )
            self.assertEqual(response.status_code, 200, response.text)

        # The clinical surface is unchanged by the existence of a graph.
        insights = self.client.get(
            f"{API_PREFIX}/sessions/{self.cid}/insights", headers=self.headers
        ).json()
        self.assertTrue(insights["differential"])
        self.assertTrue(insights["red_flags"])

        # The same consultation, through the intelligence boundary.
        result = self.runtime.get_session(self.cid).latest_result
        outcome = IntelligenceGateway(
            LangGraphIntelligenceProvider(ScriptedModelClient(good_script()))
        ).consult(result)

        self.assertEqual(outcome.status, IntelligenceStatus.OK)
        self.assertTrue(outcome.proposals)
        self.assertEqual(outcome.rejected, [])

        # Every proposal cites something that is genuinely in this consultation.
        context_refs = set(build_context(result).references)
        for p in outcome.proposals:
            self.assertTrue(set(p.grounded_in) <= context_refs)

    def test_the_api_still_serves_clinical_output_when_the_model_is_dead(self) -> None:
        for sequence, text in enumerate(CARDIAC, start=1):
            self.client.post(
                f"{API_PREFIX}/sessions/{self.cid}/events",
                json=utterance(self.cid, sequence, text).model_dump(mode="json"),
                headers=self.headers,
            )

        outcome = IntelligenceGateway(
            LangGraphIntelligenceProvider(UnavailableModelClient())
        ).consult(self.runtime.get_session(self.cid).latest_result)
        self.assertEqual(outcome.status, IntelligenceStatus.UNAVAILABLE)

        insights = self.client.get(
            f"{API_PREFIX}/sessions/{self.cid}/insights", headers=self.headers
        )
        self.assertEqual(insights.status_code, 200)
        self.assertTrue(insights.json()["red_flags"])


if __name__ == "__main__":
    unittest.main()
