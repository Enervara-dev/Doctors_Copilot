"""
Test suite for clinical_pipeline_integration

Purpose:
Defines automated unit and integration tests to verify code coverage and correctness.

Module/Component: Test Suites
"""

import unittest
import time
from app.services.providers.models import TranscriptEvent
from app.services.clinical import clinical_state_engine
from app.services.clinical.models import ClinicalState, VitalObservation
from app.services.speaker_alignment import speaker_alignment_service
from clinical_intelligence.rule_engine.matcher import SymptomMatcher
from clinical_intelligence.rule_engine.scorer import DiseaseScorer
from clinical_intelligence.rule_engine.explainability import ExplainabilityTracer
from app.services.clinical.dictionary.loader import dictionary_manager

class TestClinicalPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.session_id = "integration-test-session-999"
        clinical_state_engine.clear_state(self.session_id)
        speaker_alignment_service.clear_session(self.session_id)

    def tearDown(self):
        clinical_state_engine.clear_state(self.session_id)
        speaker_alignment_service.clear_session(self.session_id)

    def test_strict_name_extractor(self):
        # 1. Valid patient introductions
        event1 = TranscriptEvent(
            session_id=self.session_id,
            sequence_number=1,
            role="patient",
            speaker_id="patient",
            transcript="Hello, my name is John.",
            is_final=True,
            is_partial=False,
            event_id="e_name_1",
            timestamp=100.0
        )
        state = clinical_state_engine.update_state_from_event(event1)
        self.assertEqual(state.patient_info.patient_name, "John")

        # 2. Blocklisted and non-introductions must not extract name
        clinical_state_engine.clear_state(self.session_id)
        event2 = TranscriptEvent(
            session_id=self.session_id,
            sequence_number=2,
            role="patient",
            speaker_id="patient",
            transcript="I am sick today.",
            is_final=True,
            is_partial=False,
            event_id="e_name_2",
            timestamp=102.0
        )
        state = clinical_state_engine.update_state_from_event(event2)
        self.assertNotEqual(state.patient_info.patient_name, "Sick")
        self.assertIsNone(state.patient_info.patient_name)

    def test_speaker_gating_rules(self):
        # Patient turn: symptoms MUST populate, vitals MUST NOT
        event_pat = TranscriptEvent(
            session_id=self.session_id,
            sequence_number=1,
            role="patient",
            speaker_id="patient",
            transcript="I have a terrible sore throat and a high fever.",
            is_final=True,
            is_partial=False,
            event_id="e_gate_1",
            timestamp=100.0
        )
        state = clinical_state_engine.update_state_from_event(event_pat)
        symptom_names = [s["name"] for s in state.symptoms]
        self.assertIn("symptom_sore_throat", symptom_names)
        self.assertIn("symptom_fever", symptom_names)
        self.assertIsNone(state.vital_signs.temperature)

        # Doctor turn: vitals and treatment MUST populate, symptoms MUST NOT (unless quoting)
        event_doc = TranscriptEvent(
            session_id=self.session_id,
            sequence_number=2,
            role="doctor",
            speaker_id="doctor",
            transcript="Your temperature is 101 degrees F and I'll prescribe you paracetamol.",
            is_final=True,
            is_partial=False,
            event_id="e_gate_2",
            timestamp=102.0
        )
        state = clinical_state_engine.update_state_from_event(event_doc)
        self.assertIsNotNone(state.vital_signs.temperature)
        self.assertEqual(state.vital_signs.temperature.value, "101")
        self.assertEqual(len(state.treatment_plan.medicines), 1)
        self.assertEqual(state.treatment_plan.medicines[0]["name"], "Paracetamol")

    def test_modifiers_distribution_and_validation(self):
        # Distribute a single duration to multiple symptoms in the same clause
        event = TranscriptEvent(
            session_id=self.session_id,
            sequence_number=1,
            role="patient",
            speaker_id="patient",
            transcript="I have had a runny nose and dry throat since yesterday.",
            is_final=True,
            is_partial=False,
            event_id="e_mod_1",
            timestamp=100.0
        )
        state = clinical_state_engine.update_state_from_event(event)
        symptoms = {s["name"]: s for s in state.symptoms}
        self.assertIn("symptom_rhinorrhea", symptoms)
        self.assertIn("symptom_throat_dryness", symptoms)
        self.assertEqual(symptoms["symptom_rhinorrhea"]["duration"], "yesterday")
        self.assertEqual(symptoms["symptom_throat_dryness"]["duration"], "yesterday")

    def test_qualitative_vitals_extraction(self):
        event = TranscriptEvent(
            session_id=self.session_id,
            sequence_number=1,
            role="doctor",
            speaker_id="doctor",
            transcript="The patient is tachycardic, afebrile, and hypotensive.",
            is_final=True,
            is_partial=False,
            event_id="e_vitals_1",
            timestamp=100.0
        )
        state = clinical_state_engine.update_state_from_event(event)
        self.assertEqual(state.vital_signs.pulse.qualitative, "tachycardic")
        self.assertEqual(state.vital_signs.temperature.qualitative, "normal")
        self.assertEqual(state.vital_signs.bp.qualitative, "hypotensive")

        # Confirm qualitative vital flags propagate to examination findings
        findings = [f.finding for f in state.examination]
        self.assertIn("tachycardia", findings)

    def test_exclusionary_symptoms_tension_headache(self):
        from pathlib import Path
        from clinical_intelligence.rule_engine.loader import RuleLoader
        rules_dir = Path("clinical_intelligence/rules")
        loader = RuleLoader(rules_dir=rules_dir)
        rules = loader.get()
        matcher = SymptomMatcher(rules.diseases, rules.symptoms, rule_set=rules)
        scorer = DiseaseScorer()
        tracer = ExplainabilityTracer()

        # 1. Headache without fever (No exclusion)
        present = [{"name": "symptom_headache", "confidence": "High", "status": "Active"}]
        results = matcher.match(present_symptoms=present, negated_symptoms=[], tracer=tracer)
        tension_match = next(r for r in results if r.disease_id == "tension_headache")
        self.assertFalse(tension_match.exclusion_triggered)
        self.assertGreater(tension_match.raw_score, 0.0)

        # 2. Headache with fever (Triggers exclusion)
        present_with_fever = [
            {"name": "symptom_headache", "confidence": "High", "status": "Active"},
            {"name": "symptom_fever", "confidence": "High", "status": "Active"}
        ]
        results_ex = matcher.match(present_symptoms=present_with_fever, negated_symptoms=[], tracer=tracer)
        tension_ex_match = next(r for r in results_ex if r.disease_id == "tension_headache")
        self.assertTrue(tension_ex_match.exclusion_triggered)
        self.assertEqual(tension_ex_match.raw_score, 0.0)

    def test_deterministic_scoring_tie_breaking(self):
        disease_rules = [
            {"id": "disease_b", "name": "Disease B", "category": "test", "cardinal_symptoms": [{"name": "symptom_cough", "weight": 0.5}], "min_diagnosis_threshold": 0.0},
            {"id": "disease_a", "name": "Disease A", "category": "test", "cardinal_symptoms": [{"name": "symptom_cough", "weight": 0.5}], "min_diagnosis_threshold": 0.0}
        ]
        symptom_rules = []
        matcher = SymptomMatcher(disease_rules, symptom_rules)
        scorer = DiseaseScorer()
        tracer = ExplainabilityTracer()

        present = [{"name": "symptom_cough", "confidence": "High", "status": "Active"}]
        matches = matcher.match(present_symptoms=present, negated_symptoms=[], tracer=tracer)
        
        # Confirm sorting by disease_id ascending if raw scores are equal
        self.assertEqual(matches[0].disease_id, "disease_a")
        self.assertEqual(matches[1].disease_id, "disease_b")

        candidates = scorer.score(
            match_results=matches,
            vital_signs={},
            age=30,
            gender="Male",
            identified_risk_factor_ids=[],
            tracer=tracer
        )
        self.assertEqual(candidates[0].disease_id, "disease_a")
        self.assertEqual(candidates[1].disease_id, "disease_b")

    def test_audit_trail_and_validation(self):
        # 1. Invalid age must trigger audit trail logging
        event = TranscriptEvent(
            session_id=self.session_id,
            sequence_number=1,
            role="patient",
            speaker_id="patient",
            transcript="I am 250 years old.",
            is_final=True,
            is_partial=False,
            event_id="e_audit_1",
            timestamp=100.0
        )
        state = clinical_state_engine.update_state_from_event(event)
        self.assertIsNone(state.patient_info.age)
        self.assertTrue(len(state.audit_trail) > 0)
        action_types = [a["action"] for a in state.audit_trail]
        self.assertIn("invalidate_age", action_types)
