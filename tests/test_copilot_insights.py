"""
The clinician-facing surfaces, end to end.

Each test drives the whole chain — transcript event, clinical state,
deterministic reasoning, `CopilotResult`, projection, HTTP response — because
the thing worth protecting is that a clinician sees what the engine actually
concluded, unchanged and attributable.

The projection is assertable in a way the engine is not: every value on a
surface must be present in the result it came from. Several tests below check
exactly that, so a future change that quietly synthesises a field fails here.

Offline: the runtime never enables enrichment, and no network is touched.
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
from integration.copilot_insights import (  # noqa: E402
    ADVISORY,
    Basis,
    CopilotInsights,
    to_insights,
)
from integration.copilot_runtime import CopilotResult, CopilotRuntime  # noqa: E402
from integration.doctor_adapter import utterance_to_copilot_event  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "golden_consultation.json"


def load_golden() -> Dict[str, Any]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_golden(consultation_id: str) -> tuple[List[CopilotResult], CopilotRuntime]:
    emitted: List[CopilotResult] = []
    runtime = CopilotRuntime(emitted.append)
    for utterance in sorted(load_golden()["utterances"], key=lambda u: u["sequence"]):
        runtime.ingest(
            utterance_to_copilot_event(utterance, consultation_id=consultation_id)
        )
    return emitted, runtime


# ══════════════════════════════════════════════════════════════════════════
# The six surfaces
# ══════════════════════════════════════════════════════════════════════════


class TestInsightSurfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        emitted, _ = run_golden("insights-1")
        cls.result = emitted[-1]
        cls.insights: CopilotInsights = to_insights(cls.result)

    # ── 1. clinical summary ───────────────────────────────────────────────
    def test_the_summary_carries_the_observed_presentation(self) -> None:
        summary = self.insights.summary
        self.assertEqual(summary.basis, Basis.OBSERVED)
        self.assertEqual(summary.age, 62)
        self.assertEqual(summary.gender, "Male")
        self.assertIn("symptom_chest_pain", summary.chief_complaint)
        self.assertEqual(summary.severity, "Severe")

    def test_the_summary_carries_history_and_medications(self) -> None:
        summary = self.insights.summary
        self.assertIn("Hypertension", summary.history)
        self.assertIn("Diabetes", summary.history)
        self.assertIn("Aspirin", summary.medications)

    def test_symptoms_keep_their_attributes_and_their_origin(self) -> None:
        names = {s.name for s in self.insights.summary.symptoms}
        self.assertIn("symptom_chest_pain", names)

        chest = next(s for s in self.insights.summary.symptoms if s.name == "symptom_chest_pain")
        self.assertEqual(chest.severity, "severe")
        # Traceable to the utterance that produced it.
        self.assertTrue(chest.utterance_id)
        self.assertTrue(chest.source_text)

    def test_a_negated_symptom_is_not_shown_as_present(self) -> None:
        fever = [s for s in self.insights.summary.symptoms if s.name == "symptom_fever"]
        for symptom in fever:
            self.assertNotEqual(symptom.present, "True")

    def test_only_recorded_vitals_are_shown(self) -> None:
        # None were taken in this consultation, so the panel is empty rather
        # than a row of blanks.
        self.assertEqual(self.insights.summary.vitals, {})

    # ── 2. differential ───────────────────────────────────────────────────
    def test_the_differential_is_ranked_and_marked_inferred(self) -> None:
        differential = self.insights.differential
        self.assertGreater(len(differential), 0)
        self.assertEqual([d.rank for d in differential], list(range(1, len(differential) + 1)))
        for item in differential:
            self.assertEqual(item.basis, Basis.INFERRED)

    def test_the_leading_candidate_is_coronary_and_evidenced(self) -> None:
        top = self.insights.differential[0]
        self.assertIn(top.disease_id, {"unstable_angina", "acs"})
        self.assertIn("chest pain", [e.lower() for e in top.supporting_evidence])

    def test_confidence_is_the_engines_own_number(self) -> None:
        """Not rescaled, not bucketed — presentation must not re-score."""
        for projected, source in zip(
            self.insights.differential, self.result.reasoning.candidate_diagnoses
        ):
            self.assertEqual(projected.confidence, source.confidence)
            self.assertEqual(projected.rank, source.rank)

    # ── 3. red flags ──────────────────────────────────────────────────────
    def test_red_flags_say_why_they_fired(self) -> None:
        self.assertGreater(len(self.insights.red_flags), 0)
        self.assertIn("CRITICAL", {f.severity for f in self.insights.red_flags})
        for flag in self.insights.red_flags:
            self.assertTrue(flag.recommended_action)
            self.assertTrue(flag.supporting_evidence or flag.triggered_by)

    # ── 4. investigations ─────────────────────────────────────────────────
    def test_investigations_carry_reason_and_priority(self) -> None:
        investigations = self.insights.investigations
        ecg = next(i for i in investigations if i.investigation_id == "ecg_12lead")
        self.assertEqual(ecg.priority, "URGENT")
        self.assertTrue(ecg.reason)
        for item in investigations:
            self.assertIn(item.priority, {"URGENT", "HIGH", "ROUTINE"})

    def test_investigations_are_the_rule_engines_set_exactly(self) -> None:
        """
        The deterministic set is authoritative.

        No model contributes here, and the projection neither adds nor drops.
        """
        self.assertEqual(
            [i.investigation_id for i in self.insights.investigations],
            [i.investigation_id for i in self.result.reasoning.recommended_investigations],
        )
        self.assertEqual(self.result.reasoning.llm_recommended_investigations, [])

    # ── 5. missing information ────────────────────────────────────────────
    def test_missing_information_is_asked_as_questions(self) -> None:
        missing = self.insights.missing_information
        self.assertGreater(len(missing), 0)
        self.assertIn("blood_pressure", {m.field for m in missing})
        for item in missing:
            self.assertTrue(item.question.strip().endswith("?"))

    # ── 6. safety ─────────────────────────────────────────────────────────
    def test_safety_carries_contraindications_and_risk_factors(self) -> None:
        safety = self.insights.safety
        self.assertTrue(safety.contraindications)
        for item in safety.contraindications:
            self.assertIn(item.severity, {"ABSOLUTE", "RELATIVE"})
            self.assertTrue(item.reason)

        factor_ids = {r.factor_id for r in safety.risk_factors}
        self.assertIn("hypertension", factor_ids)
        self.assertIn("diabetes", factor_ids)

    # ── assistive posture ─────────────────────────────────────────────────
    def test_the_payload_states_that_it_is_assistive(self) -> None:
        self.assertEqual(self.insights.advisory, ADVISORY)
        self.assertIn("not diagnoses", self.insights.advisory)

    def test_observed_and_inferred_are_distinguishable_everywhere(self) -> None:
        self.assertEqual(self.insights.summary.basis, Basis.OBSERVED)
        inferred = (
            list(self.insights.differential)
            + list(self.insights.red_flags)
            + list(self.insights.investigations)
            + list(self.insights.missing_information)
            + list(self.insights.safety.contraindications)
            + list(self.insights.safety.risk_factors)
        )
        self.assertTrue(inferred)
        for item in inferred:
            self.assertEqual(item.basis, Basis.INFERRED)

    def test_provenance_survives_into_the_clinician_payload(self) -> None:
        self.assertTrue(self.insights.provenance)
        self.assertTrue(
            any(e["event_id"].startswith("utt-") for e in self.insights.provenance.values())
        )

    def test_the_payload_names_the_utterance_and_state_it_reflects(self) -> None:
        self.assertEqual(self.insights.event_id, self.result.event_id)
        self.assertEqual(self.insights.sequence, self.result.sequence)
        self.assertEqual(self.insights.state_version, self.result.clinical_state.version)


# ══════════════════════════════════════════════════════════════════════════
# Nothing is invented
# ══════════════════════════════════════════════════════════════════════════


class TestProjectionAddsNothing(unittest.TestCase):
    """
    Every value on a surface must exist in the result it came from.

    This is the guard against the projection quietly becoming a second source
    of clinical content.
    """

    @classmethod
    def setUpClass(cls) -> None:
        emitted, _ = run_golden("insights-2")
        cls.result = emitted[-1]
        cls.insights = to_insights(cls.result)

    def test_every_differential_item_comes_from_the_engine(self) -> None:
        source = {c.disease_id: c for c in self.result.reasoning.candidate_diagnoses}
        for item in self.insights.differential:
            self.assertIn(item.disease_id, source)
            self.assertEqual(item.supporting_evidence, list(source[item.disease_id].supporting_symptoms))
            self.assertEqual(item.not_yet_assessed, list(source[item.disease_id].missing_symptoms))

    def test_every_red_flag_comes_from_the_engine(self) -> None:
        source = {f.flag_id for f in self.result.reasoning.red_flags}
        self.assertEqual({f.flag_id for f in self.insights.red_flags}, source)

    def test_every_missing_question_comes_from_the_engine(self) -> None:
        source = {m.field: m.question for m in self.result.reasoning.missing_information}
        for item in self.insights.missing_information:
            self.assertEqual(item.question, source[item.field])

    def test_every_summary_symptom_comes_from_the_clinical_state(self) -> None:
        source = {s.get("name") for s in self.result.clinical_state.symptoms}
        self.assertEqual({s.name for s in self.insights.summary.symptoms}, source)

    def test_the_projection_is_deterministic(self) -> None:
        self.assertEqual(
            to_insights(self.result).model_dump(),
            to_insights(self.result).model_dump(),
        )

    def test_an_empty_consultation_produces_empty_surfaces_not_filler(self) -> None:
        runtime = CopilotRuntime()
        golden = load_golden()
        # The doctor's opening greeting alone: no clinical fact stated.
        runtime.ingest(
            utterance_to_copilot_event(golden["utterances"][0], consultation_id="empty-1")
        )
        insights = to_insights(runtime.get_session("empty-1").latest_result)

        self.assertEqual(insights.differential, [])
        self.assertEqual(insights.red_flags, [])
        self.assertEqual(insights.summary.symptoms, [])
        # The advisory is still present: an empty panel is still assistive output.
        self.assertEqual(insights.advisory, ADVISORY)


# ══════════════════════════════════════════════════════════════════════════
# Over HTTP
# ══════════════════════════════════════════════════════════════════════════


class TestInsightsOverHttp(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = CopilotRuntime.__new__(CopilotRuntime)  # placeholder, replaced below
        self.runtime = CopilotRuntime()
        self.client = TestClient(create_app(self.runtime))

    def headers(self, consultation_id: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {generate_jwt_token(consultation_id, 'doctor')}"}

    def post_utterance(self, consultation_id: str, utterance: Dict[str, Any]):
        event = utterance_to_copilot_event(utterance, consultation_id=consultation_id)
        return self.client.post(
            f"{API_PREFIX}/sessions/{consultation_id}/events",
            json=event.model_dump(mode="json"),
            headers=self.headers(consultation_id),
        )

    def test_the_full_flow_reaches_the_api_response(self) -> None:
        """transcript event → state → reasoning → result → insights → HTTP."""
        cid = "http-1"
        golden = load_golden()
        last = None
        for utterance in sorted(golden["utterances"], key=lambda u: u["sequence"]):
            response = self.post_utterance(cid, utterance)
            self.assertEqual(response.status_code, 200, response.text)
            if response.json()["insights"] is not None:
                last = response.json()["insights"]

        self.assertIsNotNone(last)
        self.assertEqual(last["summary"]["age"], 62)
        self.assertEqual(last["summary"]["basis"], "OBSERVED")
        self.assertTrue(last["differential"])
        self.assertTrue(last["red_flags"])
        self.assertTrue(last["investigations"])
        self.assertTrue(last["missing_information"])
        self.assertTrue(last["safety"]["risk_factors"])
        self.assertIn("not diagnoses", last["advisory"])

    def test_the_insights_endpoint_serves_the_current_picture(self) -> None:
        cid = "http-2"
        for utterance in sorted(load_golden()["utterances"], key=lambda u: u["sequence"]):
            self.post_utterance(cid, utterance)

        response = self.client.get(
            f"{API_PREFIX}/sessions/{cid}/insights", headers=self.headers(cid)
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["consultation_id"], cid)
        self.assertEqual(
            [d["disease_id"] for d in body["differential"]][:1],
            [self.runtime.get_session(cid).latest_result.reasoning.candidate_diagnoses[0].disease_id],
        )

    def test_insights_reflect_the_latest_state_even_when_nothing_was_emitted(self) -> None:
        """
        A panel opened mid-consultation wants the current picture.

        Sequence 4 of the golden consultation is suppressed — the conclusion
        did not change — but the endpoint must still answer with the state as
        it stands, not with nothing.
        """
        cid = "http-3"
        utterances = sorted(load_golden()["utterances"], key=lambda u: u["sequence"])
        for utterance in utterances[:4]:  # ends on the suppressed turn
            self.post_utterance(cid, utterance)

        self.assertEqual(self.runtime.get_session(cid).suppressed_count, 1)
        response = self.client.get(
            f"{API_PREFIX}/sessions/{cid}/insights", headers=self.headers(cid)
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sequence"], 4)

    def test_insights_before_any_settled_utterance_is_404(self) -> None:
        cid = "http-4"
        self.client.post(
            f"{API_PREFIX}/sessions",
            json={"consultation_id": cid},
            headers=self.headers(cid),
        )
        response = self.client.get(
            f"{API_PREFIX}/sessions/{cid}/insights", headers=self.headers(cid)
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "NO_RESULT_YET")

    def test_insights_for_an_unknown_consultation_is_404(self) -> None:
        response = self.client.get(
            f"{API_PREFIX}/sessions/nope/insights", headers=self.headers("nope")
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "SESSION_NOT_FOUND")

    def test_insights_require_a_credential_for_that_consultation(self) -> None:
        cid = "http-5"
        self.post_utterance(cid, load_golden()["utterances"][1])

        self.assertEqual(
            self.client.get(f"{API_PREFIX}/sessions/{cid}/insights").status_code, 401
        )
        self.assertEqual(
            self.client.get(
                f"{API_PREFIX}/sessions/{cid}/insights", headers=self.headers("someone-else")
            ).status_code,
            403,
        )

    def test_a_partial_utterance_yields_no_insights_on_the_response(self) -> None:
        cid = "http-6"
        utterance = dict(load_golden()["utterances"][1], status="PARTIAL", finalizedAt=None)
        response = self.post_utterance(cid, utterance)

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["insights"])


if __name__ == "__main__":
    unittest.main()
