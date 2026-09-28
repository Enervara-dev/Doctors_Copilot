"""
Result change detection.

Reasoning still runs after every settled utterance; what changes here is
whether the conclusion is *sent*. These tests pin the two halves of that:
a semantic view that ignores clocks, counters and trace metadata, and a runtime
that emits only when the clinical picture has actually moved.

The category tests drive the real engine with real utterances rather than
hand-assembling a `ClinicalReasoningResult`. A fabricated result would test the
fingerprint against itself; a real one proves that saying "I have diabetes"
genuinely reaches `risk_factors` and causes an emission.

Offline throughout — the runtime never enables enrichment.
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
from integration.copilot_runtime import CopilotResult, CopilotRuntime  # noqa: E402
from integration.semantic_fingerprint import (  # noqa: E402
    SEMANTIC_CATEGORIES,
    changed_categories,
    fingerprint,
    semantic_view,
)
from integration.doctor_adapter import utterance_to_copilot_event  # noqa: E402
from clinical_intelligence.rule_engine.engine import ClinicalReasoningEngine  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "golden_consultation.json"


def load_golden() -> Dict[str, Any]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)


def say(
    consultation_id: str,
    sequence: int,
    text: str,
    *,
    role: SpeakerRole = SpeakerRole.PATIENT,
    event_id: str | None = None,
) -> CopilotEvent:
    return CopilotEvent(
        event_id=event_id or f"utt-{sequence:03d}",
        consultation_id=consultation_id,
        sequence=sequence,
        provenance=Provenance.FIXTURE,
        payload=UtterancePayload(
            text=text,
            speaker_role=role,
            speaker_label=role.value.title(),
            is_final=True,
            confidence=0.95,
            started_at="2026-09-27T09:00:00.000Z",
            finalized_at="2026-09-27T09:00:04.000Z",
        ),
    )


def drive(consultation_id: str, utterances: List[str]) -> tuple[List[CopilotResult], Any]:
    """Runs a consultation and returns (emitted results, session)."""
    emitted: List[CopilotResult] = []
    runtime = CopilotRuntime(emitted.append)
    for i, text in enumerate(utterances, start=1):
        runtime.ingest(say(consultation_id, i, text))
    return emitted, runtime.get_session(consultation_id)


# ══════════════════════════════════════════════════════════════════════════
# The semantic view itself
# ══════════════════════════════════════════════════════════════════════════


class TestSemanticView(unittest.TestCase):
    """The view must be blind to everything that is not clinical."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.emitted, cls.session = drive(
            "view-1",
            [
                "I am a 62 year old male with severe chest pain and sweating.",
                "I also feel short of breath and nauseous.",
                "I have high blood pressure and diabetes.",
            ],
        )
        cls.result = cls.emitted[-1]

    def test_the_view_covers_every_declared_category(self) -> None:
        view = semantic_view(self.result.clinical_state, self.result.reasoning)
        self.assertEqual(set(view), set(SEMANTIC_CATEGORIES))

    def test_a_regenerated_reasoning_timestamp_is_not_a_change(self) -> None:
        """
        The reason model equality cannot be used.

        `reasoning_timestamp` is `default_factory=datetime.utcnow`, so two
        identical passes are never equal as models.
        """
        reasoning = self.result.reasoning
        before = fingerprint(self.result.clinical_state, reasoning)

        mutated = reasoning.model_copy(deep=True)
        mutated.reasoning_timestamp = mutated.reasoning_timestamp.replace(year=2030)
        mutated.explainability.reasoning_duration_ms = 999_999.0
        mutated.explainability.rules_evaluated = 123_456

        self.assertNotEqual(reasoning, mutated)
        self.assertEqual(before, fingerprint(self.result.clinical_state, mutated))

    def test_symptom_bookkeeping_is_not_a_change(self) -> None:
        """`entity_id`, `updated_at` and `utterance_id` are not clinical facts."""
        state = self.result.clinical_state.model_copy(deep=True)
        before = fingerprint(state, self.result.reasoning)

        for symptom in state.symptoms:
            symptom["entity_id"] = "symptom_999"
            symptom["updated_at"] = "1999999999.0"
            symptom["utterance_id"] = "utt-rewritten"
            symptom["source_text"] = "a differently worded transcript"

        self.assertEqual(before, fingerprint(state, self.result.reasoning))

    def test_a_negated_symptom_is_a_different_picture(self) -> None:
        state = self.result.clinical_state.model_copy(deep=True)
        before = fingerprint(state, self.result.reasoning)
        state.symptoms[0]["present"] = "False"
        self.assertNotEqual(before, fingerprint(state, self.result.reasoning))

    def test_candidate_order_is_meaning_and_is_compared(self) -> None:
        reasoning = self.result.reasoning.model_copy(deep=True)
        if len(reasoning.candidate_diagnoses) < 2:
            self.skipTest("needs at least two candidates to reorder")
        before = fingerprint(self.result.clinical_state, reasoning)
        reasoning.candidate_diagnoses.reverse()
        self.assertNotEqual(before, fingerprint(self.result.clinical_state, reasoning))

    def test_investigation_order_is_not_meaning_and_is_ignored(self) -> None:
        reasoning = self.result.reasoning.model_copy(deep=True)
        if len(reasoning.recommended_investigations) < 2:
            self.skipTest("needs at least two investigations to reorder")
        before = fingerprint(self.result.clinical_state, reasoning)
        reasoning.recommended_investigations.reverse()
        self.assertEqual(before, fingerprint(self.result.clinical_state, reasoning))

    def test_the_fingerprint_is_stable_across_processes(self) -> None:
        """
        sha256 over canonical JSON — no object identity, no PYTHONHASHSEED.

        Recomputing it must give the same digest, which is what makes replay
        after a restart produce the same emission pattern.
        """
        digest = fingerprint(self.result.clinical_state, self.result.reasoning)
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            digest, fingerprint(self.result.clinical_state, self.result.reasoning)
        )

    def test_changed_categories_ignores_empty_ones_on_the_first_result(self) -> None:
        empty = {c: None for c in SEMANTIC_CATEGORIES}
        empty["demographics"] = {"age": None, "gender": None}
        empty["symptoms"] = [{"name": "symptom_x", "present": "True"}]
        self.assertEqual(changed_categories(None, empty), ["symptoms"])


# ══════════════════════════════════════════════════════════════════════════
# Emission behaviour
# ══════════════════════════════════════════════════════════════════════════


class TestFirstResultAlwaysEmits(unittest.TestCase):
    def test_the_first_result_emits_even_when_thin(self) -> None:
        emitted, session = drive("first-1", ["Good morning."])
        self.assertEqual(len(emitted), 1)
        self.assertEqual(session.reasoning_runs, 1)
        self.assertEqual(session.suppressed_count, 0)

    def test_the_first_result_emits_with_no_clinical_content_at_all(self) -> None:
        emitted, _ = drive("first-2", ["Mm."])
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0].is_empty)


class TestNoChangeIsSuppressed(unittest.TestCase):
    def test_repeating_the_same_clinical_content_emits_once(self) -> None:
        """reasoning calls > 1, emissions == 1."""
        emitted, session = drive(
            "same-1",
            [
                "I have severe chest pain.",
                "I have severe chest pain.",
                "I have severe chest pain.",
            ],
        )
        self.assertEqual(session.final_count, 3)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(session.suppressed_count, 2)
        # The state never moved, so the conclusion was computed once and reused.
        self.assertEqual(session.reasoning_runs, 1)
        self.assertEqual(session.reasoning_skipped, 2)

    def test_conversational_filler_does_not_emit(self) -> None:
        emitted, session = drive(
            "same-2",
            ["I have severe chest pain.", "Yes.", "Mhm.", "Okay."],
        )
        self.assertEqual(len(emitted), 1)
        self.assertEqual(session.suppressed_count, 3)

    def test_a_suppressed_result_is_still_kept_as_the_latest(self) -> None:
        """No emission must never mean the computation was thrown away."""
        emitted, session = drive(
            "same-3", ["I have severe chest pain.", "I have severe chest pain."]
        )
        self.assertEqual(len(emitted), 1)
        self.assertIsNotNone(session.latest_result)
        self.assertEqual(session.latest_result.sequence, 2)
        self.assertEqual(session.latest_result.fingerprint, emitted[0].fingerprint)

    def test_the_fingerprint_survives_suppression_for_the_next_comparison(self) -> None:
        emitted, session = drive(
            "same-4",
            [
                "I have severe chest pain.",
                "I have severe chest pain.",
                "I also feel short of breath.",
            ],
        )
        self.assertEqual([r.sequence for r in emitted], [1, 3])
        self.assertEqual(session.suppressed_count, 1)


class TestMeaningfulChangeEmits(unittest.TestCase):
    """
    Each clinical category, moved independently by a real utterance.

    The assertion is both that an emission happened and that the expected
    category is named in `changed` — otherwise a test could pass because some
    unrelated field moved.
    """

    def _advance(self, cid: str, opening: List[str], trigger: str):
        emitted, session = drive(cid, opening + [trigger])
        return emitted, session

    def test_a_new_symptom_changes_the_picture(self) -> None:
        emitted, _ = self._advance(
            "chg-sym", ["I have severe chest pain."], "I also feel short of breath."
        )
        self.assertEqual(len(emitted), 2)
        self.assertIn("symptoms", emitted[-1].changed)

    def test_a_new_symptom_changes_the_diagnosis_candidates(self) -> None:
        emitted, _ = self._advance(
            "chg-dx", ["I have severe chest pain."], "I also feel short of breath and nauseous."
        )
        self.assertIn("candidate_diagnoses", emitted[-1].changed)

    def test_stated_history_changes_the_risk_factors(self) -> None:
        emitted, _ = self._advance(
            "chg-risk",
            ["I have severe chest pain."],
            "I have high blood pressure and diabetes.",
        )
        self.assertEqual(len(emitted), 2)
        self.assertIn("risk_factors", emitted[-1].changed)
        self.assertIn("history", emitted[-1].changed)

    def test_stated_history_changes_the_contraindications(self) -> None:
        emitted, _ = self._advance(
            "chg-ci", ["I have severe chest pain."], "I also have diabetes."
        )
        self.assertIn("contraindications", emitted[-1].changed)

    def test_a_medication_changes_the_picture(self) -> None:
        emitted, _ = self._advance(
            "chg-med", ["I have severe chest pain."], "I take aspirin daily."
        )
        self.assertEqual(len(emitted), 2)
        self.assertIn("medications", emitted[-1].changed)

    def test_symptoms_change_the_red_flags(self) -> None:
        emitted, _ = self._advance(
            "chg-rf", ["I have a mild headache."], "I have severe chest pain and I am sweating."
        )
        self.assertIn("red_flags", emitted[-1].changed)

    def test_symptoms_change_the_investigations(self) -> None:
        emitted, _ = self._advance(
            "chg-inv", ["I have a mild headache."], "I have severe chest pain."
        )
        self.assertIn("investigations", emitted[-1].changed)

    def test_new_findings_change_the_missing_information(self) -> None:
        # A symptom that opens a new differential changes what still needs
        # asking. Adding shortness of breath to chest pain does not — verified
        # — because the outstanding questions are already the same ones.
        emitted, _ = self._advance(
            "chg-mi", ["I have severe chest pain."], "I have a fever and a cough."
        )
        self.assertEqual(len(emitted), 2)
        self.assertIn("missing_information", emitted[-1].changed)

    def test_demographics_change_the_picture(self) -> None:
        emitted, _ = self._advance(
            "chg-demo", ["I have severe chest pain."], "I am a 62 year old male."
        )
        self.assertEqual(len(emitted), 2)
        self.assertIn("demographics", emitted[-1].changed)

    def test_every_category_is_exercised_by_the_golden_consultation(self) -> None:
        """A guard against a category nothing in the suite can move."""
        emitted: List[CopilotResult] = []
        runtime = CopilotRuntime(emitted.append)
        golden = load_golden()
        for u in sorted(golden["utterances"], key=lambda u: u["sequence"]):
            runtime.ingest(utterance_to_copilot_event(u, consultation_id="cat-cover"))

        moved = {c for r in emitted for c in r.changed}
        for expected in (
            "demographics",
            "symptoms",
            "history",
            "medications",
            "candidate_diagnoses",
            "red_flags",
            "risk_factors",
            "investigations",
            "missing_information",
            "contraindications",
        ):
            self.assertIn(expected, moved, f"{expected} was never exercised")


class TestMetadataOnlyChangeDoesNotEmit(unittest.TestCase):
    def test_a_later_event_with_the_same_content_is_metadata_only(self) -> None:
        """
        Different event id, sequence and timestamps; identical clinical content.

        This is the metadata-only case: everything the transport cares about
        changed, nothing the clinician cares about did.
        """
        emitted: List[CopilotResult] = []
        runtime = CopilotRuntime(emitted.append)
        runtime.ingest(say("meta-1", 1, "I have severe chest pain.", event_id="a"))
        runtime.ingest(say("meta-1", 2, "I have severe chest pain.", event_id="b"))

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].event_id, "a")
        self.assertEqual(runtime.get_session("meta-1").suppressed_count, 1)


class TestOrderingOfEmissions(unittest.TestCase):
    def test_a_b_c_where_a_equals_b_emits_a_then_c(self) -> None:
        """The consumer must never receive B."""
        emitted, session = drive(
            "abc-1",
            [
                "I have severe chest pain.",   # A
                "I have severe chest pain.",   # B — identical
                "I also have diabetes.",       # C
            ],
        )
        self.assertEqual([r.sequence for r in emitted], [1, 3])
        self.assertEqual(session.suppressed_count, 1)
        # B added no clinical fact, so no second pass was needed for it.
        self.assertEqual(session.reasoning_runs, 2)
        self.assertEqual(session.reasoning_skipped, 1)

    def test_the_emitted_sequence_names_the_event_that_caused_the_change(self) -> None:
        emitted, _ = drive(
            "abc-2",
            ["I have severe chest pain.", "Yes.", "Mhm.", "I also have diabetes."],
        )
        # Not sequence 2 or 3 — the change arrived with 4.
        self.assertEqual([r.sequence for r in emitted], [1, 4])
        self.assertEqual(emitted[-1].event_id, "utt-004")


class TestSessionIsolation(unittest.TestCase):
    def test_both_consultations_emit_their_first_identical_result(self) -> None:
        emitted: List[CopilotResult] = []
        runtime = CopilotRuntime(emitted.append)
        runtime.ingest(say("A", 1, "I have severe chest pain."))
        runtime.ingest(say("B", 1, "I have severe chest pain."))

        self.assertEqual(len(emitted), 2)
        self.assertEqual([r.consultation_id for r in emitted], ["A", "B"])
        # Same clinical picture, so the same digest — and both still emitted,
        # because a fingerprint is only ever compared within its own session.
        self.assertEqual(emitted[0].fingerprint, emitted[1].fingerprint)

    def test_one_session_suppressing_does_not_suppress_the_other(self) -> None:
        emitted: List[CopilotResult] = []
        runtime = CopilotRuntime(emitted.append)
        runtime.ingest(say("A", 1, "I have severe chest pain."))
        runtime.ingest(say("A", 2, "I have severe chest pain."))  # suppressed in A
        runtime.ingest(say("B", 1, "I have severe chest pain."))  # first in B

        self.assertEqual([r.consultation_id for r in emitted], ["A", "B"])
        self.assertEqual(runtime.get_session("A").suppressed_count, 1)
        self.assertEqual(runtime.get_session("B").suppressed_count, 0)


class TestConsumerFailureDoesNotCorruptState(unittest.TestCase):
    def test_a_throwing_consumer_leaves_change_detection_intact(self) -> None:
        seen: List[CopilotResult] = []

        def flaky(result: CopilotResult) -> None:
            seen.append(result)
            if result.sequence == 1:
                raise RuntimeError("consumer blew up on the first result")

        runtime = CopilotRuntime(flaky)
        runtime.ingest(say("fail-1", 1, "I have severe chest pain."))
        runtime.ingest(say("fail-1", 2, "I have severe chest pain."))  # unchanged
        runtime.ingest(say("fail-1", 3, "I also have diabetes."))      # changed

        session = runtime.get_session("fail-1")
        self.assertEqual([r.sequence for r in seen], [1, 3])
        self.assertEqual(session.suppressed_count, 1)
        self.assertEqual(session.emitted_count, 2)


class TestReasoningIsSkippedWhenTheStateHasNotMoved(unittest.TestCase):
    """
    The guard, and the property it rests on.

    Reasoning is a pure function of the clinical state, so an unchanged state
    digest means the previous conclusion still stands. The first test is the
    evidence for that; the rest are the behaviour it licenses.
    """

    def test_reasoning_is_deterministic_from_the_clinical_state(self) -> None:
        """
        The precondition. If this fails, the guard is unsound and must go.

        Five passes over one state, plus a freshly constructed engine, must all
        reach the same semantic conclusion, and a pass must not mutate the
        state it was handed.
        """
        _, session = drive(
            "det-1",
            [
                "I am a 62 year old male with severe chest pain and sweating.",
                "I have high blood pressure and diabetes.",
            ],
        )
        state = session.current_state()

        engine = ClinicalReasoningEngine()
        digests = {fingerprint(state, engine.reason(state)) for _ in range(5)}
        self.assertEqual(len(digests), 1)
        self.assertIn(
            fingerprint(state, ClinicalReasoningEngine().reason(state)), digests
        )

        before = state.model_dump_json()
        engine.reason(state)
        self.assertEqual(state.model_dump_json(), before, "reason() mutated its input")

    def test_the_first_final_always_runs_reasoning(self) -> None:
        _, session = drive("guard-first", ["Good morning."])
        self.assertEqual(session.reasoning_runs, 1)
        self.assertEqual(session.reasoning_skipped, 0)

    def test_a_changed_state_runs_reasoning(self) -> None:
        _, session = drive(
            "guard-changed", ["I have severe chest pain.", "I also have diabetes."]
        )
        self.assertEqual(session.reasoning_runs, 2)
        self.assertEqual(session.reasoning_skipped, 0)

    def test_an_unchanged_state_skips_reasoning(self) -> None:
        _, session = drive(
            "guard-unchanged", ["I have severe chest pain.", "Yes.", "Mhm."]
        )
        self.assertEqual(session.final_count, 3)
        self.assertEqual(session.reasoning_runs, 1)
        self.assertEqual(session.reasoning_skipped, 2)

    def test_the_previous_conclusion_is_preserved_when_a_pass_is_skipped(self) -> None:
        emitted, session = drive("guard-preserve", ["I have severe chest pain.", "Yes."])
        self.assertEqual(session.reasoning_skipped, 1)
        latest = session.latest_result
        self.assertIsNotNone(latest)
        # The reused conclusion is the emitted one, not an empty stand-in.
        self.assertEqual(latest.fingerprint, emitted[0].fingerprint)
        self.assertEqual(
            [c.disease_id for c in latest.reasoning.candidate_diagnoses],
            [c.disease_id for c in emitted[0].reasoning.candidate_diagnoses],
        )

    def test_reasoning_resumes_after_a_skip(self) -> None:
        """A skip must not latch: the next real change is still reasoned about."""
        emitted, session = drive(
            "guard-resume",
            ["I have severe chest pain.", "Yes.", "I also have diabetes."],
        )
        self.assertEqual(session.reasoning_runs, 2)
        self.assertEqual(session.reasoning_skipped, 1)
        self.assertEqual([r.sequence for r in emitted], [1, 3])

    def test_skipping_changes_nothing_a_consumer_sees(self) -> None:
        """
        The optimization must be invisible.

        The emitted stream is compared against the pattern recorded before the
        guard existed: same sequences, same count, same suppression.
        """
        emitted: List[CopilotResult] = []
        runtime = CopilotRuntime(emitted.append)
        golden = load_golden()
        for u in sorted(golden["utterances"], key=lambda x: x["sequence"]):
            runtime.ingest(utterance_to_copilot_event(u, consultation_id="guard-golden"))

        session = runtime.get_session("guard-golden")
        self.assertEqual([r.sequence for r in emitted], [1, 2, 3, 5, 6])
        self.assertEqual(session.final_count, 6)
        self.assertEqual(session.reasoning_runs, 5)
        self.assertEqual(session.reasoning_skipped, 1)
        # The skipped pass is the same turn whose result was suppressed: the
        # doctor asking a question, which carries no clinical fact.
        self.assertEqual(session.suppressed_count, 1)

    def test_a_skipped_pass_is_isolated_to_its_own_consultation(self) -> None:
        runtime = CopilotRuntime()
        runtime.ingest(say("iso-A", 1, "I have severe chest pain."))
        runtime.ingest(say("iso-A", 2, "I have severe chest pain."))
        runtime.ingest(say("iso-B", 1, "I have severe chest pain."))

        self.assertEqual(runtime.get_session("iso-A").reasoning_skipped, 1)
        self.assertEqual(runtime.get_session("iso-B").reasoning_skipped, 0)
        self.assertEqual(runtime.get_session("iso-B").reasoning_runs, 1)


class TestReplayProducesTheSameEmissionPattern(unittest.TestCase):
    @staticmethod
    def _golden_run(consultation_id: str) -> List[CopilotResult]:
        emitted: List[CopilotResult] = []
        runtime = CopilotRuntime(emitted.append)
        golden = load_golden()
        for u in sorted(golden["utterances"], key=lambda u: u["sequence"]):
            runtime.ingest(
                utterance_to_copilot_event(u, consultation_id=consultation_id)
            )
        return emitted

    def test_the_same_events_produce_the_same_emissions(self) -> None:
        a = self._golden_run("replay-t1")
        b = self._golden_run("replay-t2")

        self.assertEqual([r.sequence for r in a], [r.sequence for r in b])
        self.assertEqual([r.event_id for r in a], [r.event_id for r in b])
        self.assertEqual([r.fingerprint for r in a], [r.fingerprint for r in b])
        self.assertEqual([r.changed for r in a], [r.changed for r in b])

    def test_the_golden_consultation_suppresses_exactly_the_silent_turn(self) -> None:
        emitted = self._golden_run("replay-t3")
        # Sequence 4 is the doctor asking about medication and conditions: a
        # question, carrying no clinical fact of its own.
        self.assertEqual([r.sequence for r in emitted], [1, 2, 3, 5, 6])


if __name__ == "__main__":
    unittest.main()
