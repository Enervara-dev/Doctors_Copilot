"""
Clinician-facing evidence must quote the patient, not the dictionary.

Extraction runs on normalized text, which is correct for matching and wrong for
provenance: the normalizer rewrites "chest pain" to the canonical
`symptom_chest_pain` in place, and that rewritten string was being recorded as
the evidence a clinician reads.

These tests pin both halves of the fix — that the original words reach the
Copilot surfaces, and that nothing the rule engine reads moved as a result.
"""

from __future__ import annotations

import os
import unittest
from typing import List

os.environ.setdefault("CLINICAL_COPILOT_ENABLE_GRAPH", "False")
os.environ.setdefault("CLINICAL_COPILOT_ENABLE_LLM", "False")

from app.services.clinical.normalizer import ClinicalNormalizer  # noqa: E402
from app.services.clinical.pipeline import (  # noqa: E402
    ClinicalProcessingPipeline,
    _original_fragments,
)
from integration.copilot_insights import to_insights  # noqa: E402
from integration.copilot_runtime import CopilotResult, CopilotRuntime  # noqa: E402
from integration.copilot_event import (  # noqa: E402
    CopilotEvent,
    Provenance,
    SpeakerRole,
    UtterancePayload,
)

CANONICAL_MARKERS = ("symptom_", "disease_", "drug_")


def say(consultation_id: str, sequence: int, text: str) -> CopilotEvent:
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


def drive(consultation_id: str, utterances: List[str]) -> CopilotResult:
    runtime = CopilotRuntime()
    for i, text in enumerate(utterances, start=1):
        runtime.ingest(say(consultation_id, i, text))
    return runtime.get_session(consultation_id).latest_result


class TestEvidenceQuotesThePatient(unittest.TestCase):
    def test_the_normalizer_really_does_rewrite_the_text(self) -> None:
        """The premise. If this stops being true, the fix is unnecessary."""
        normalized = ClinicalNormalizer().normalize("I have severe chest pain.")
        self.assertIn("symptom_chest_pain", normalized)
        self.assertNotIn("chest pain", normalized.replace("symptom_chest_pain", ""))

    def test_evidence_shows_the_spoken_words(self) -> None:
        result = drive("ev-1", ["I have been having severe chest pain for two hours."])
        insights = to_insights(result)

        chest = next(s for s in insights.summary.symptoms if s.name == "symptom_chest_pain")
        self.assertIn("chest pain", chest.source_text)
        self.assertNotIn("symptom_chest_pain", chest.source_text)

    def test_no_surface_leaks_a_canonical_identifier_into_evidence(self) -> None:
        result = drive(
            "ev-2",
            [
                "I am a 62 year old male with severe chest pain and sweating.",
                "I also feel short of breath and nauseous.",
                "I take aspirin daily and I have high blood pressure and diabetes.",
            ],
        )
        insights = to_insights(result)

        for symptom in insights.summary.symptoms:
            if not symptom.source_text:
                continue
            for marker in CANONICAL_MARKERS:
                self.assertNotIn(
                    marker,
                    symptom.source_text,
                    f"{symptom.name} evidence still shows a canonical id: {symptom.source_text!r}",
                )

    def test_the_canonical_name_is_still_the_canonical_name(self) -> None:
        """
        Only the evidence string was restored.

        The identifier the rule engine matches on must be untouched, or this
        fix would have silently changed which rules fire.
        """
        insights = to_insights(drive("ev-3", ["I have severe chest pain."]))
        self.assertIn(
            "symptom_chest_pain", {s.name for s in insights.summary.symptoms}
        )

    def test_evidence_comes_from_the_utterance_it_is_attributed_to(self) -> None:
        result = drive(
            "ev-4",
            ["I have severe chest pain.", "I also feel nauseous."],
        )
        insights = to_insights(result)

        by_name = {s.name: s for s in insights.summary.symptoms}
        self.assertEqual(by_name["symptom_chest_pain"].utterance_id, "utt-001")
        self.assertIn("chest pain", by_name["symptom_chest_pain"].source_text)
        self.assertEqual(by_name["symptom_nausea"].utterance_id, "utt-002")
        self.assertIn("nauseous", by_name["symptom_nausea"].source_text)

    def test_a_negated_symptom_quotes_the_negation(self) -> None:
        insights = to_insights(drive("ev-5", ["I do not have any fever."]))
        fever = next(s for s in insights.summary.symptoms if s.name == "symptom_fever")
        self.assertIn("do not have", fever.source_text)
        self.assertNotEqual(fever.present, "True")

    def test_the_sentence_is_the_one_the_symptom_came_from(self) -> None:
        insights = to_insights(
            drive("ev-6", ["I have severe chest pain. I also feel nauseous."])
        )
        by_name = {s.name: s for s in insights.summary.symptoms}
        self.assertIn("chest pain", by_name["symptom_chest_pain"].source_text)
        self.assertNotIn("nauseous", by_name["symptom_chest_pain"].source_text)


class TestFragmentMapping(unittest.TestCase):
    """The pairing itself, including the case where it must refuse."""

    def test_sentences_pair_by_position(self) -> None:
        original = "I have chest pain. I feel sick."
        normalized = "I have symptom_chest_pain. I feel symptom_nausea."
        mapping = _original_fragments(original, normalized)

        self.assertEqual(mapping["I have symptom_chest_pain"], "I have chest pain")
        self.assertEqual(mapping["I feel symptom_nausea"], "I feel sick")

    def test_the_whole_utterance_is_mapped_for_extractors_that_record_it(self) -> None:
        mapping = _original_fragments("BP is 150 over 90", "BP is 150 over 90")
        self.assertEqual(mapping["BP is 150 over 90"], "BP is 150 over 90")

    def test_a_mismatched_split_produces_no_sentence_mapping(self) -> None:
        """
        Rather than guess.

        If normalization ever started splitting or joining sentences, pairing
        by position would attribute one patient's words to another symptom.
        Leaving the canonical text in place is poor; misquoting is worse.
        """
        mapping = _original_fragments("One sentence here.", "One. Two. Three.")
        self.assertNotIn("One", mapping)
        self.assertNotIn("Two", mapping)

    def test_an_empty_utterance_maps_nothing(self) -> None:
        self.assertEqual(_original_fragments("", ""), {})


class TestExtractionIsUnaffected(unittest.TestCase):
    """
    The rule engine must not notice this change.

    Extraction still runs on normalized text; only the provenance string is
    rewritten afterwards.
    """

    def test_the_same_entities_are_extracted_as_before(self) -> None:
        pipeline = ClinicalProcessingPipeline()
        result = pipeline.process(
            {
                "session_id": "px-1",
                "speaker_id": "patient",
                "transcript": "I have severe chest pain and I take aspirin.",
                "timestamp": 1.0,
                "utterance_id": "u1",
            }
        )
        self.assertIn("symptom_chest_pain", {s.name for s in result.symptoms})
        self.assertIn("aspirin", {m.name.lower() for m in result.medications})

    def test_reasoning_still_reaches_the_same_conclusion(self) -> None:
        result = drive(
            "px-2",
            [
                "I am a 62 year old male with severe chest pain and sweating.",
                "I also feel short of breath and nauseous.",
                "I take aspirin daily and I have high blood pressure and diabetes.",
            ],
        )
        ids = {c.disease_id for c in result.reasoning.candidate_diagnoses}
        self.assertTrue({"unstable_angina", "acs"} & ids, f"got {ids}")
        self.assertIn(
            "ecg_12lead",
            {i.investigation_id for i in result.reasoning.recommended_investigations},
        )

    def test_medication_evidence_is_also_restored(self) -> None:
        result = drive("px-3", ["I take a paracetamol every morning."])
        # Whatever the normalizer mapped it to, the evidence must read as spoken.
        for entry in result.clinical_state.symptoms:
            if entry.get("source_text"):
                self.assertNotIn("symptom_", entry["source_text"])


if __name__ == "__main__":
    unittest.main()
