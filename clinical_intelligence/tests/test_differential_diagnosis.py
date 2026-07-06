from __future__ import annotations

"""
Test suite for differential_diagnosis

Purpose:
Defines automated unit and integration tests to verify code coverage and correctness.

Module/Component: Test Suites
"""

import unittest
from clinical_intelligence.rule_engine.engine import ClinicalReasoningEngine
from clinical_intelligence.rule_engine.models import ClinicalReasoningResult

class TestDifferentialDiagnosis(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ClinicalReasoningEngine()

    def test_viral_uri_scenario(self) -> None:
        # Create a mock clinical state for Viral URI
        state = {
            "session_id": "test_session_uri",
            "chief_complaint": ["cough"],
            "symptoms": [
                {"name": "cough", "status": "Active", "present": "True", "confidence": "High"},
                {"name": "fever", "status": "Active", "present": "True", "confidence": "High"},
                {"name": "sore throat", "status": "Active", "present": "True", "confidence": "High"}
            ],
            "vital_signs": {
                "temperature": 38.5
            },
            "medical_history": [],
            "current_medications": [],
            "allergies": []
        }

        result = self.engine.reason(state)
        self.assertIsInstance(result, ClinicalReasoningResult)
        self.assertTrue(len(result.candidate_diagnoses) > 0)
        
        # Viral URI should be among the top candidate diagnoses
        candidate_ids = [c.disease_id for c in result.candidate_diagnoses]
        self.assertIn("viral_uri", candidate_ids)

    def test_stroke_scenario(self) -> None:
        # Stroke scenario
        state = {
            "session_id": "test_session_stroke",
            "chief_complaint": ["focal neurological deficit"],
            "symptoms": [
                {"name": "focal neurological deficit", "status": "Active", "present": "True", "confidence": "High"},
                {"name": "facial droop", "status": "Active", "present": "True", "confidence": "High"},
                {"name": "unilateral weakness", "status": "Active", "present": "True", "confidence": "High"}
            ],
            "vital_signs": {
                "bp_systolic": 165
            },
            "medical_history": ["hypertension", "atrial_fibrillation"],
            "current_medications": [],
            "allergies": []
        }

        result = self.engine.reason(state)
        self.assertIsInstance(result, ClinicalReasoningResult)
        self.assertTrue(len(result.candidate_diagnoses) > 0)

        # Stroke should be ranked high
        top_disease = result.candidate_diagnoses[0]
        self.assertEqual(top_disease.disease_id, "ischemic_stroke")
        self.assertGreaterEqual(top_disease.confidence, 0.39)

if __name__ == "__main__":
    unittest.main()
