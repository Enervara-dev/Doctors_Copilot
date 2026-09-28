"""
The contract between the Doctor platform and this clinical core.

This is the first test that drives the existing Python pipeline with the Doctor
platform's own event shape, end to end:

    Doctor TranscriptUtterance -> CopilotEvent -> TranscriptEvent
        -> ClinicalStateEngine -> ClinicalReasoningEngine -> structured output

It is hermetic by construction rather than by mocking. `engine.reason()` takes
`run_llm=False` by default, so no Gemini call is made; `ENABLE_GRAPH` is turned
off before the engine is imported, so no Neo4j connection is attempted. Nothing
here touches ElevenLabs, Pinecone or HuggingFace. Retrieval stays on, because
it is local TF-IDF over a checked-in YAML file with no network behind it.

Assertions are on structure and identifiers — `disease_id`, `priority`,
`severity`, counts and membership — never on model wording, which is neither
stable nor owned by this repository.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Must precede the engine import: these flags are read once, at module import.
os.environ.setdefault("CLINICAL_COPILOT_ENABLE_GRAPH", "False")
os.environ.setdefault("CLINICAL_COPILOT_ENABLE_LLM", "False")

from app.services.clinical.state_engine import ClinicalStateEngine  # noqa: E402
from clinical_intelligence.rule_engine.engine import (  # noqa: E402
    ClinicalReasoningEngine,
)
from integration.copilot_event import (  # noqa: E402
    SCHEMA_VERSION,
    CopilotEvent,
    EventType,
    SpeakerRole,
)
from integration.doctor_adapter import (  # noqa: E402
    UnmappableSpeakerError,
    copilot_event_to_transcript_event,
    normalize_speaker,
    utterance_to_copilot_event,
    utterances_to_transcript_events,
)

FIXTURE = Path(__file__).parent / "fixtures" / "golden_consultation.json"


def load_golden() -> Dict[str, Any]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_consultation(
    utterances: List[Dict[str, Any]],
    consultation_id: str,
    *,
    strict_speaker: bool = True,
):
    """
    Drives one consultation through the real pipeline.

    A fresh `ClinicalStateEngine` per call rather than the module singleton, so
    one test's state can never leak into another's.
    """
    engine = ClinicalStateEngine()
    events = utterances_to_transcript_events(
        utterances, consultation_id=consultation_id, strict_speaker=strict_speaker
    )
    state = None
    for event in events:
        state = engine.update_state_from_event(event)
    return engine, state


# ══════════════════════════════════════════════════════════════════════════
# 1. The mapping itself
# ══════════════════════════════════════════════════════════════════════════


class TestDoctorUtteranceMapping(unittest.TestCase):
    """Field-by-field, Doctor TranscriptUtterance -> CopilotEvent -> TranscriptEvent."""

    def setUp(self) -> None:
        self.golden = load_golden()
        self.utterance = self.golden["utterances"][1]  # the patient's first turn

    def test_every_doctor_field_survives_into_the_envelope(self) -> None:
        event = utterance_to_copilot_event(
            self.utterance, consultation_id="c-1", offset_seconds=4.0, duration_seconds=8.0
        )

        self.assertEqual(event.schema_version, SCHEMA_VERSION)
        self.assertEqual(event.event_id, "utt-002")
        self.assertEqual(event.consultation_id, "c-1")
        self.assertEqual(event.sequence, 2)
        self.assertEqual(event.event_type, EventType.UTTERANCE)
        self.assertEqual(event.provenance.value, "SERVICE")

        payload = event.payload
        self.assertEqual(payload.speaker_role, SpeakerRole.PATIENT)
        self.assertEqual(payload.speaker_label, "Patient")
        self.assertTrue(payload.is_final)
        self.assertEqual(payload.confidence, 0.94)
        self.assertEqual(payload.started_at, "2026-09-27T09:00:04.000Z")
        self.assertEqual(payload.finalized_at, "2026-09-27T09:00:12.000Z")
        self.assertIn("62 year old male", payload.text)

    def test_the_pipeline_receives_a_canonical_speaker_id(self) -> None:
        event = utterance_to_copilot_event(self.utterance, consultation_id="c-1")
        transcript_event = copilot_event_to_transcript_event(event)

        # The field the clinical pipeline actually branches on.
        self.assertEqual(transcript_event.speaker_id, "patient")
        self.assertEqual(transcript_event.role, "patient")
        self.assertEqual(transcript_event.session_id, "c-1")
        self.assertEqual(transcript_event.event_id, "utt-002")
        self.assertEqual(transcript_event.sequence_number, 2)
        self.assertFalse(transcript_event.is_partial)
        self.assertTrue(transcript_event.is_final)

    def test_text_is_carried_in_both_fields_the_pipeline_reads(self) -> None:
        event = utterance_to_copilot_event(self.utterance, consultation_id="c-1")
        te = copilot_event_to_transcript_event(event)
        self.assertEqual(te.transcript, te.text)
        self.assertTrue(te.transcript)

    def test_iso_timestamps_become_epoch_seconds(self) -> None:
        event = utterance_to_copilot_event(self.utterance, consultation_id="c-1")
        te = copilot_event_to_transcript_event(event)
        self.assertIsInstance(te.timestamp, float)
        self.assertGreater(te.timestamp, 1_600_000_000)

    def test_a_partial_utterance_is_marked_partial(self) -> None:
        partial = dict(self.utterance, status="PARTIAL", finalizedAt=None)
        event = utterance_to_copilot_event(partial, consultation_id="c-1")
        te = copilot_event_to_transcript_event(event)
        self.assertTrue(te.is_partial)
        self.assertFalse(te.is_final)

    def test_a_status_outside_the_contract_is_refused(self) -> None:
        broken = dict(self.utterance, status="SETTLING")
        with self.assertRaises(ValueError):
            utterance_to_copilot_event(broken, consultation_id="c-1")

    def test_events_are_ordered_by_sequence_not_arrival(self) -> None:
        shuffled = list(reversed(self.golden["utterances"]))
        events = utterances_to_transcript_events(shuffled, consultation_id="c-1")
        self.assertEqual([e.sequence_number for e in events], [1, 2, 3, 4, 5, 6])

    def test_a_repeated_utterance_is_dropped(self) -> None:
        doubled = self.golden["utterances"] + [self.golden["utterances"][1]]
        events = utterances_to_transcript_events(doubled, consultation_id="c-1")
        self.assertEqual(len(events), len(self.golden["utterances"]))

    def test_a_partial_and_its_final_are_not_duplicates_of_each_other(self) -> None:
        partial = dict(self.utterance, status="PARTIAL", finalizedAt=None)
        events = utterances_to_transcript_events(
            [partial, self.utterance], consultation_id="c-1"
        )
        # Same id, two states of one utterance: both must cross the boundary.
        self.assertEqual(len(events), 2)
        self.assertEqual({e.event_id for e in events}, {"utt-002"})
        self.assertEqual([e.is_final for e in events], [False, True])


# ══════════════════════════════════════════════════════════════════════════
# 2. Speaker normalization — the verified silent-failure boundary
# ══════════════════════════════════════════════════════════════════════════


class TestSpeakerNormalization(unittest.TestCase):
    """
    Case A and Case B from the audit.

    The pipeline compares `speaker_id` against lowercase literals in
    `state_engine.py:126` and eight other sites. A raw diarization label
    matches none of them, so every patient-reported fact is dropped in silence.
    """

    def setUp(self) -> None:
        self.golden = load_golden()

    def test_case_a_doctor_platform_roles_map_to_canonical_roles(self) -> None:
        self.assertEqual(normalize_speaker("PATIENT"), SpeakerRole.PATIENT)
        self.assertEqual(normalize_speaker("DOCTOR"), SpeakerRole.DOCTOR)
        self.assertEqual(normalize_speaker("ATTENDER"), SpeakerRole.ATTENDER)
        self.assertEqual(normalize_speaker("UNKNOWN"), SpeakerRole.UNKNOWN)
        # Already-canonical input is accepted unchanged.
        self.assertEqual(normalize_speaker("patient"), SpeakerRole.PATIENT)

    def test_case_b_a_raw_diarization_label_is_refused_loudly(self) -> None:
        for label in ("SPEAKER_00", "SPEAKER_01", "speaker_00", "SPK_1"):
            with self.assertRaises(UnmappableSpeakerError) as caught:
                normalize_speaker(label)
            self.assertIn("diarization label", str(caught.exception))

    def test_case_b_regression_the_silent_empty_state_cannot_recur(self) -> None:
        """
        The exact failure found in the audit.

        Feeding SPEAKER_00 straight through used to produce a clinical state
        with no symptoms, no error and no log line. The boundary now refuses it
        before a single utterance reaches the pipeline.
        """
        raw = [dict(u, speaker=f"SPEAKER_0{i % 2}") for i, u in enumerate(self.golden["utterances"])]
        with self.assertRaises(UnmappableSpeakerError):
            utterances_to_transcript_events(raw, consultation_id="c-raw")

    def test_an_unmappable_speaker_can_be_downgraded_deliberately(self) -> None:
        """Opting out is allowed, but it has to be asked for."""
        self.assertEqual(
            normalize_speaker("SPEAKER_00", strict=False), SpeakerRole.UNKNOWN
        )

    def test_case_a_and_case_b_on_the_same_transcript(self) -> None:
        """
        The comparison the audit asked for, as one assertion.

        Case A (canonical roles) extracts clinical facts. Case B (raw labels),
        forced through with strict=False, extracts none — which is exactly why
        strict mode is the default.
        """
        _, state_a = run_consultation(self.golden["utterances"], "case-a")
        self.assertGreater(len(state_a.symptoms), 0)
        self.assertEqual(state_a.patient_info.age, 62)

        raw = [dict(u, speaker="SPEAKER_00") for u in self.golden["utterances"]]
        _, state_b = run_consultation(raw, "case-b", strict_speaker=False)
        self.assertEqual(len(state_b.symptoms), 0)
        self.assertIsNone(state_b.patient_info.age)


# ══════════════════════════════════════════════════════════════════════════
# 3. The golden consultation, end to end
# ══════════════════════════════════════════════════════════════════════════


class TestGoldenConsultation(unittest.TestCase):
    """
    The reference case. An ACS presentation, driven entirely by Doctor events.

    Every assertion is on structure or identifiers. Confidence values are
    bounded rather than pinned, so rule tuning shows up as a ranking change
    instead of a spurious failure.
    """

    @classmethod
    def setUpClass(cls) -> None:
        golden = load_golden()
        cls.engine, cls.state = run_consultation(
            golden["utterances"], golden["consultationId"]
        )
        cls.reasoning = ClinicalReasoningEngine().reason(cls.state)
        cls.consultation_id = golden["consultationId"]

    # ── extraction ────────────────────────────────────────────────────────
    def test_demographics_are_extracted_from_the_patient_turn(self) -> None:
        self.assertEqual(self.state.patient_info.age, 62)
        self.assertEqual(self.state.patient_info.gender, "Male")

    def test_the_chief_complaint_is_the_presenting_symptom(self) -> None:
        self.assertIn("symptom_chest_pain", self.state.chief_complaint)

    def test_symptoms_are_extracted_with_severity(self) -> None:
        # NOTE: ClinicalState.symptoms is declared List[Dict[str, Optional[Any]]],
        # so entries are plain dicts here even though extraction produced typed
        # SymptomEntity objects. Asserting on the real shape, not the nicer one.
        names = {s["name"] for s in self.state.symptoms}
        self.assertIn("symptom_chest_pain", names)
        self.assertIn("symptom_shortness_of_breath", names)
        self.assertEqual(self.state.severity, "Severe")

    def test_a_negated_symptom_is_not_recorded_as_present(self) -> None:
        # "I do not have any fever" is the last patient turn.
        present = {
            s["name"] for s in self.state.symptoms if str(s.get("present")) == "True"
        }
        self.assertNotIn("symptom_fever", present)

    def test_history_and_medication_come_from_the_patient(self) -> None:
        self.assertIn("Hypertension", self.state.medical_history)
        self.assertIn("Diabetes", self.state.medical_history)
        self.assertIn("Aspirin", self.state.current_medications)

    def test_state_is_versioned_across_the_consultation(self) -> None:
        self.assertGreater(self.state.version, 1)

    def test_every_fact_carries_provenance(self) -> None:
        provenance = self.engine.get_provenance(self.consultation_id)
        self.assertTrue(provenance)
        entry = next(iter(provenance.values()))
        for key in ("event_id", "session_id", "sequence_number", "speaker_id", "version"):
            self.assertIn(key, entry)
        # Provenance points back at a Doctor utterance id, so a fact can be
        # traced to the exact thing that was said.
        self.assertTrue(
            any(e["event_id"].startswith("utt-") for e in provenance.values())
        )

    # ── reasoning ─────────────────────────────────────────────────────────
    def test_candidate_diagnoses_are_ranked_and_cardiac(self) -> None:
        candidates = self.reasoning.candidate_diagnoses
        self.assertGreater(len(candidates), 0)
        ids = [c.disease_id for c in candidates]
        self.assertTrue(
            {"unstable_angina", "acs"} & set(ids),
            f"expected a coronary candidate, got {ids}",
        )
        # Ranked by confidence, densely, starting at 1.
        self.assertEqual([c.rank for c in candidates], list(range(1, len(candidates) + 1)))
        confidences = [c.confidence for c in candidates]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        for c in candidates:
            self.assertGreaterEqual(c.confidence, 0.0)
            self.assertLessEqual(c.confidence, 1.0)

    def test_the_leading_candidate_is_supported_by_extracted_symptoms(self) -> None:
        top = self.reasoning.candidate_diagnoses[0]
        self.assertTrue(top.supporting_symptoms)
        self.assertIn("chest pain", [s.lower() for s in top.supporting_symptoms])

    def test_a_critical_red_flag_is_raised(self) -> None:
        severities = {f.severity for f in self.reasoning.red_flags}
        self.assertIn("CRITICAL", severities)
        for flag in self.reasoning.red_flags:
            self.assertTrue(flag.recommended_action)
            self.assertTrue(flag.supporting_evidence)

    def test_risk_factors_come_from_the_stated_history(self) -> None:
        ids = {r.factor_id for r in self.reasoning.risk_factors}
        self.assertIn("hypertension", ids)
        self.assertIn("diabetes", ids)
        for factor in self.reasoning.risk_factors:
            self.assertEqual(factor.source, "medical_history")

    def test_an_urgent_ecg_is_recommended(self) -> None:
        investigations = self.reasoning.recommended_investigations
        self.assertGreater(len(investigations), 0)
        ecg = next(
            (i for i in investigations if i.investigation_id == "ecg_12lead"), None
        )
        self.assertIsNotNone(ecg, "a 12-lead ECG is the defining investigation here")
        self.assertEqual(ecg.priority, "URGENT")
        self.assertTrue(ecg.reason)
        for inv in investigations:
            self.assertIn(inv.priority, ("URGENT", "HIGH", "ROUTINE"))

    def test_investigations_are_not_duplicated(self) -> None:
        ids = [i.investigation_id for i in self.reasoning.recommended_investigations]
        self.assertEqual(len(ids), len(set(ids)))

    def test_missing_information_is_asked_as_questions(self) -> None:
        missing = self.reasoning.missing_information
        self.assertGreater(len(missing), 0)
        fields = {m.field for m in missing}
        self.assertIn("blood_pressure", fields)
        for item in missing:
            self.assertTrue(item.question.strip().endswith("?"))

    def test_a_contraindication_is_derived_from_the_history(self) -> None:
        # Diabetes is stated, so steroid cover should be flagged.
        self.assertTrue(self.reasoning.contraindications)

    def test_no_llm_fields_are_populated_in_the_hermetic_run(self) -> None:
        """
        The whole point of the hermeticity.

        `run_llm` defaults to False, so the enrichment layer never runs and the
        deterministic output stands alone. If this ever fails, the test has
        started calling a paid API.
        """
        self.assertEqual(self.reasoning.llm_recommended_investigations, [])

    def test_the_deterministic_pass_is_reproducible(self) -> None:
        """Same input, same structured output — no clock, no randomness, no model."""
        golden = load_golden()
        _, state = run_consultation(golden["utterances"], "repeat-run")
        again = ClinicalReasoningEngine().reason(state)

        self.assertEqual(
            [(c.disease_id, round(c.confidence, 6)) for c in again.candidate_diagnoses],
            [
                (c.disease_id, round(c.confidence, 6))
                for c in self.reasoning.candidate_diagnoses
            ],
        )
        self.assertEqual(
            sorted(i.investigation_id for i in again.recommended_investigations),
            sorted(
                i.investigation_id for i in self.reasoning.recommended_investigations
            ),
        )


if __name__ == "__main__":
    unittest.main()
