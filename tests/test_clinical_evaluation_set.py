"""
The golden clinical evaluation set.

Six consultations, driven through the whole chain — transcript, extraction,
clinical state, deterministic reasoning, throttling, insights — and checked
against expectations that were measured, not predicted.

Its job is the one thing unit tests cannot do: catch a change that is locally
correct and clinically wrong. A rule tweak that makes one detector test pass
while silencing stroke, or a dictionary edit that makes a cardiac case shout
sepsis, fails here.

Two expectation kinds carry most of the weight:

* `red_flags` — must be raised. A missed emergency is the worst failure mode.
* `forbid_red_flags` — must not be raised. This is the alert-fatigue guard, and
  it is what the demographics defect would have tripped.

Offline throughout.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("CLINICAL_COPILOT_ENABLE_GRAPH", "False")
os.environ.setdefault("CLINICAL_COPILOT_ENABLE_LLM", "False")

from integration.copilot_event import (  # noqa: E402
    CopilotEvent,
    Provenance,
    SpeakerRole,
    UtterancePayload,
)
from integration.copilot_insights import CopilotInsights, to_insights  # noqa: E402
from integration.copilot_runtime import CopilotRuntime  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "clinical_evaluation_set.json"


def load_cases() -> List[Dict[str, Any]]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)["cases"]


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
            started_at="2026-09-27T09:00:00.000Z",
            finalized_at="2026-09-27T09:00:04.000Z",
        ),
    )


def run_case(case: Dict[str, Any]) -> CopilotInsights:
    runtime = CopilotRuntime()
    consultation_id = f"eval-{case['id']}"
    for sequence, text in enumerate(case["utterances"], start=1):
        runtime.ingest(utterance(consultation_id, sequence, text))
    return to_insights(runtime.get_session(consultation_id).latest_result)


class TestClinicalEvaluationSet(unittest.TestCase):
    """Every case, every expectation, with the case id in each failure."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases()
        cls.results = {case["id"]: run_case(case) for case in cls.cases}

    def _for(self, case: Dict[str, Any]) -> CopilotInsights:
        return self.results[case["id"]]

    def test_the_set_covers_every_clinical_class(self) -> None:
        categories = {case["category"] for case in self.cases}
        self.assertEqual(categories, {"normal", "high_risk", "ambiguous", "negative"})

    def test_demographics(self) -> None:
        for case in self.cases:
            expect, insights = case["expect"], self._for(case)
            with self.subTest(case=case["id"]):
                self.assertEqual(insights.summary.age, expect["age"])
                self.assertEqual(insights.summary.gender, expect["gender"])

    def test_symptoms_present(self) -> None:
        for case in self.cases:
            expect, insights = case["expect"], self._for(case)
            present = {s.name for s in insights.summary.symptoms if s.present == "True"}
            with self.subTest(case=case["id"]):
                self.assertEqual(present, set(expect["symptoms_present"]))

    def test_negated_symptoms_are_not_present(self) -> None:
        for case in self.cases:
            expect, insights = case["expect"], self._for(case)
            recorded = {s.name: s for s in insights.summary.symptoms}
            with self.subTest(case=case["id"]):
                for name in expect["symptoms_absent"]:
                    self.assertIn(name, recorded, f"{name} was not recorded at all")
                    self.assertNotEqual(recorded[name].present, "True")

    def test_expected_red_flags_are_raised(self) -> None:
        """A missed emergency is the failure that matters most."""
        for case in self.cases:
            expect, insights = case["expect"], self._for(case)
            raised = {f.flag_id for f in insights.red_flags}
            with self.subTest(case=case["id"]):
                for flag_id in expect["red_flags"]:
                    self.assertIn(flag_id, raised, f"{flag_id} was not raised")

    def test_forbidden_red_flags_are_not_raised(self) -> None:
        """The alert-fatigue guard."""
        for case in self.cases:
            expect, insights = case["expect"], self._for(case)
            raised = {f.flag_id for f in insights.red_flags}
            with self.subTest(case=case["id"]):
                for flag_id in expect["forbid_red_flags"]:
                    self.assertNotIn(
                        flag_id,
                        raised,
                        f"{flag_id} fired on a case where it does not belong",
                    )

    def test_the_right_flags_lead_the_panel(self) -> None:
        for case in self.cases:
            expect, insights = case["expect"], self._for(case)
            primary = {f.flag_id for f in insights.red_flags if f.is_primary}
            with self.subTest(case=case["id"]):
                self.assertEqual(primary, set(expect["primary_red_flags"]))

    def test_urgent_investigations(self) -> None:
        for case in self.cases:
            expect, insights = case["expect"], self._for(case)
            urgent = {i.investigation_id for i in insights.investigations if i.priority == "URGENT"}
            with self.subTest(case=case["id"]):
                for investigation_id in expect["urgent_investigations"]:
                    self.assertIn(investigation_id, urgent)
                if not expect["urgent_investigations"]:
                    self.assertEqual(urgent, set(), "nothing should be urgent here")

    def test_optional_expectations(self) -> None:
        for case in self.cases:
            expect, insights = case["expect"], self._for(case)
            with self.subTest(case=case["id"]):
                for disease_id in expect.get("differential_contains", []):
                    self.assertIn(
                        disease_id, {c.disease_id for c in insights.differential}
                    )
                for item in expect.get("history", []):
                    self.assertIn(item, insights.summary.history)
                for item in expect.get("medications", []):
                    self.assertIn(item, insights.summary.medications)

    # ── properties that must hold for every case ──────────────────────────
    def test_every_flag_is_ranked_and_evidenced(self) -> None:
        for case in self.cases:
            insights = self._for(case)
            with self.subTest(case=case["id"]):
                ranks = [f.priority_rank for f in insights.red_flags]
                self.assertEqual(ranks, sorted(ranks))
                self.assertEqual(len(set(ranks)), len(ranks), "ranks must be unique")
                for flag in insights.red_flags:
                    self.assertTrue(
                        flag.clinical_evidence,
                        f"{flag.flag_id} fired with no clinical evidence",
                    )
                    self.assertTrue(flag.recommended_action)

    def test_a_quiet_consultation_raises_nothing_at_all(self) -> None:
        """Normal and negative cases must produce no alarm of any kind."""
        for case in self.cases:
            if case["category"] not in ("normal", "negative"):
                continue
            insights = self._for(case)
            with self.subTest(case=case["id"]):
                self.assertEqual(insights.red_flags, [])
                self.assertEqual(
                    [i for i in insights.investigations if i.priority == "URGENT"], []
                )

    def test_every_case_carries_the_advisory_and_provenance(self) -> None:
        for case in self.cases:
            insights = self._for(case)
            with self.subTest(case=case["id"]):
                self.assertIn("not diagnoses", insights.advisory)
                if insights.summary.symptoms:
                    self.assertTrue(insights.provenance)

    def test_observed_evidence_is_traceable_to_an_utterance(self) -> None:
        for case in self.cases:
            insights = self._for(case)
            with self.subTest(case=case["id"]):
                for flag in insights.red_flags:
                    for ref in flag.evidence_trail:
                        if ref.kind == "OBSERVED":
                            self.assertTrue(ref.utterance_id)
                            self.assertTrue(ref.quote)
                            # The quote is what was said, not the canonical id.
                            self.assertNotIn("symptom_", ref.quote)

    def test_the_whole_set_is_reproducible(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertEqual(
                    run_case(case).model_dump(exclude={"consultation_id"}),
                    self._for(case).model_dump(exclude={"consultation_id"}),
                )


if __name__ == "__main__":
    unittest.main()
