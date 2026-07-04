from __future__ import annotations
import unittest
from typing import Any, Dict, Set, List
from pathlib import Path

# Import score evaluators
from clinical_intelligence.rule_engine.scores.news2 import NEWS2Evaluator
from clinical_intelligence.rule_engine.scores.qsofa import qSOFAEvaluator
from clinical_intelligence.rule_engine.scores.curb65 import CURB65Evaluator
from clinical_intelligence.rule_engine.scores.heart import HEARTEvaluator
from clinical_intelligence.rule_engine.scores.wells import WellsEvaluator
from clinical_intelligence.rule_engine.scores.chads2vasc import CHA2DS2VAScEvaluator
from clinical_intelligence.rule_engine.scores.timi import TIMIEvaluator
from clinical_intelligence.rule_engine.scores import evaluate_all

# Import validator/exceptions/models
from clinical_intelligence.rule_engine.exceptions import RuleValidationError
from clinical_intelligence.rule_engine.validator import validate_rules
from clinical_intelligence.rule_engine.loader import RuleSet
from clinical_intelligence.rule_engine.models import ClinicalScoreResult, DebugDiagnostics
from clinical_intelligence.rule_engine.engine import ClinicalReasoningEngine
from clinical_intelligence.rule_engine.loader import RuleLoader


class TestNEWS2Evaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = NEWS2Evaluator()

    def test_news2_normal_vitals(self):
        # NEWS2 should be 0 for perfect vitals, SpO2 scale 1
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 98, "temperature": 37.0, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(),
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=45,
            gender="male"
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.score, 0)
        self.assertEqual(res.interpretation, "Low Risk (Score 0)")

    def test_news2_severe_resp_rate(self):
        # resp_rate >= 25 gives 3 points
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 26, "spo2": 97, "temperature": 37.0, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(),
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=45,
            gender="male"
        )
        self.assertEqual(res.score, 3)

    def test_news2_oxygen_scale2_copd(self):
        # SpO2 scale 2 (COPD target 88-92%). Let's check with hypercapnic respiratory failure
        res = self.evaluator.calculate(
            state_data={"medical_history": ["copd"]},
            vitals={"respiratory_rate": 15, "spo2": 90, "temperature": 37.0, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(),
            negated_symptoms=set(),
            medical_history=["copd"],
            current_medications=[],
            allergies=[],
            age=45,
            gender="male"
        )
        self.assertEqual(res.score, 0) # 90% is normal on scale 2

    def test_news2_temperature_extreme(self):
        # temp <= 35 gives 3 points
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 98, "temperature": 34.5, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(),
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=45,
            gender="male"
        )
        self.assertEqual(res.score, 3)

    def test_news2_pulse_bradycardia(self):
        # pulse <= 40 gives 3 points
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 98, "temperature": 37.0, "bp_systolic": 120, "pulse": 35},
            present_symptoms=set(),
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=45,
            gender="male"
        )
        self.assertEqual(res.score, 3)

    def test_news2_avpu_confusion(self):
        # Confusion symptom present gives 3 points
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 98, "temperature": 37.0, "bp_systolic": 120, "pulse": 75},
            present_symptoms={"confusion"},
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=45,
            gender="male"
        )
        self.assertEqual(res.score, 3)


class TestqSOFAEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = qSOFAEvaluator()

    def test_qsofa_zero(self):
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "bp_systolic": 120},
            present_symptoms=set(),
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=45,
            gender="male"
        )
        self.assertEqual(res.score, 0)
        self.assertEqual(res.interpretation, "Low Sepsis Risk (Score 0)")

    def test_qsofa_max_points(self):
        # resp rate >= 22 (+1), bp_systolic <= 100 (+1), confusion/altered mental status (+1)
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 24, "bp_systolic": 90},
            present_symptoms={"altered mental status"},
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=45,
            gender="male"
        )
        self.assertEqual(res.score, 3)
        self.assertEqual(res.interpretation, "High Sepsis Risk (Score 3)")

    def test_qsofa_missing_vitals(self):
        res = self.evaluator.calculate(
            state_data={},
            vitals={},
            present_symptoms=set(),
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=45,
            gender="male"
        )
        self.assertIsNone(res)


class TestCURB65Evaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = CURB65Evaluator()

    def test_curb65_young_normal(self):
        res = self.evaluator.calculate(
            state_data={"lab_results": {"urea": 5.0}},
            vitals={"respiratory_rate": 15, "bp_systolic": 120, "bp_diastolic": 80},
            present_symptoms=set(),
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=30,
            gender="male"
        )
        self.assertEqual(res.score, 0)
        self.assertEqual(res.interpretation, "Low Mortality Risk (Score 0)")

    def test_curb65_elderly_all_symptoms(self):
        # age >= 65 (+1), resp_rate >= 30 (+1), bp_systolic < 90 (+1), confusion (+1), urea (+1 from state_data)
        res = self.evaluator.calculate(
            state_data={"lab_results": {"urea": 8.5}},
            vitals={"respiratory_rate": 32, "bp_systolic": 85, "bp_diastolic": 55},
            present_symptoms={"confusion"},
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=70,
            gender="female"
        )
        self.assertEqual(res.score, 5)
        self.assertEqual(res.interpretation, "High Mortality Risk (Score 5)")


class TestHEARTEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = HEARTEvaluator()

    def test_heart_zero(self):
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "Normal", "lab_results": {"troponin": 0.01}},
            vitals={},
            present_symptoms={"chest pain"},
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=35,
            gender="male"
        )
        # Highly suspicious history: No. ECG: Normal (0). Age < 45 (0). Troponin < 1x (0). RFs: 0 (0).
        # Total score should be 0 or 1
        self.assertTrue(res.score <= 1)

    def test_heart_high_risk(self):
        # Age 70 (2 pts), positive troponin (2 pts), history highly suspicious (2 pts), ECG ST-deviation (2 pts), 3+ risk factors (2 pts)
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "ST depression", "lab_results": {"troponin": "highly elevated 3x normal"}},
            vitals={},
            present_symptoms={"crushing chest pain", "sweating"},
            negated_symptoms=set(),
            medical_history=["hypertension", "diabetes", "hyperlipidemia"],
            current_medications=[],
            allergies=[],
            age=70,
            gender="female"
        )
        self.assertEqual(res.score, 10)
        self.assertEqual(res.interpretation, "High Risk (Score 10)")


class TestWellsEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = WellsEvaluator()

    def test_wells_low(self):
        res = self.evaluator.calculate(
            state_data={},
            vitals={"pulse": 80},
            present_symptoms={"dyspnea"},
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=40,
            gender="male"
        )
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.interpretation, "Low Clinical Probability (Score 0.0)")

    def test_wells_high(self):
        # Clinically suspected DVT (+3), heart rate > 100 (+1.5), previous DVT (+1.5), malignancy (+1)
        res = self.evaluator.calculate(
            state_data={},
            vitals={"pulse": 110},
            present_symptoms={"leg pain", "leg warmth", "hemoptysis", "dyspnea"},
            negated_symptoms=set(),
            medical_history=["previous dvt", "active_malignancy"],
            current_medications=[],
            allergies=[],
            age=55,
            gender="female"
        )
        self.assertEqual(res.score, 8.0)
        self.assertEqual(res.interpretation, "High Clinical Probability (Score 8.0)")


class TestCHA2DS2VAScEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = CHA2DS2VAScEvaluator()

    def test_chads_young_female_no_factors(self):
        res = self.evaluator.calculate(
            state_data={},
            vitals={},
            present_symptoms=set(),
            negated_symptoms=set(),
            medical_history=["atrial fibrillation"],
            current_medications=[],
            allergies=[],
            age=30,
            gender="female"
        )
        self.assertEqual(res.score, 1) # 1 pt for female sex
        self.assertEqual(res.interpretation, "Low Stroke Risk (Score 1)")

    def test_chads_elderly_stroke_hypertension(self):
        # Age 76 (+2), prior stroke (+2), hypertension (+1), diabetes (+1), female (+1)
        res = self.evaluator.calculate(
            state_data={},
            vitals={},
            present_symptoms=set(),
            negated_symptoms=set(),
            medical_history=["atrial fibrillation", "stroke", "hypertension", "diabetes"],
            current_medications=[],
            allergies=[],
            age=76,
            gender="female"
        )
        self.assertEqual(res.score, 7)
        self.assertEqual(res.interpretation, "High Stroke Risk (Score 7)")


class TestTIMIEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = TIMIEvaluator()

    def test_timi_zero(self):
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "Normal", "lab_results": {"troponin": 0.01}},
            vitals={},
            present_symptoms={"chest pain"},
            negated_symptoms=set(),
            medical_history=[],
            current_medications=[],
            allergies=[],
            age=40,
            gender="male"
        )
        self.assertEqual(res.score, 0)
        self.assertEqual(res.interpretation, "Low Risk (Score 0)")

    def test_timi_high(self):
        # Age >= 65 (+1), 3+ CAD risk factors (+1), aspirin use (+1), severe angina (+1), ST changes (+1), troponin positive (+1)
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "ST deviation", "lab_results": {"troponin": 0.2}},
            vitals={},
            present_symptoms={"chest pain", "angina"},
            medical_history=["hypertension", "diabetes", "hyperlipidemia", "cad"],
            current_medications=["aspirin"],
            allergies=[],
            age=70,
            gender="male",
            negated_symptoms=set()
        )
        self.assertEqual(res.score, 6)
        self.assertEqual(res.interpretation, "High Risk (Score 6)")


class TestRuleValidator(unittest.TestCase):
    def test_valid_minimal_ruleset(self):
        rules = RuleSet(
            diseases=[{
                "id": "my_disease",
                "name": "My Disease",
                "category": "cardio",
                "cardinal_symptoms": [{"name": "chest pain", "weight": 0.5}],
                "supportive_symptoms": [],
                "risk_factor_modifiers": {"smoking": 0.1}
            }],
            symptoms=[{"canonical": "chest pain", "aliases": []}],
            investigations=[],
            red_flags=[],
            risk_factors=[{"id": "smoking", "name": "Smoking", "category": "social"}],
            contraindications=[],
            drug_interactions=[]
        )
        try:
            validate_rules(rules)
        except RuleValidationError as exc:
            self.fail(f"validate_rules failed on valid ruleset: {exc}")

    def test_validator_detects_duplicate_disease_id(self):
        rules = RuleSet(
            diseases=[
                {"id": "d1", "name": "D1", "category": "c"},
                {"id": "d1", "name": "D2", "category": "c"}
            ],
            symptoms=[],
            investigations=[],
            red_flags=[],
            risk_factors=[],
            contraindications=[],
            drug_interactions=[]
        )
        with self.assertRaises(RuleValidationError):
            validate_rules(rules)

    def test_validator_detects_invalid_weights(self):
        rules = RuleSet(
            diseases=[{
                "id": "d1",
                "name": "D1",
                "category": "c",
                "cardinal_symptoms": [{"name": "chest pain", "weight": -0.5}],
            }],
            symptoms=[],
            investigations=[],
            red_flags=[],
            risk_factors=[],
            contraindications=[],
            drug_interactions=[]
        )
        with self.assertRaises(RuleValidationError):
            validate_rules(rules)

    def test_validator_detects_circular_dependencies(self):
        # d1 has self loop modifier, which forms a cycle
        rules = RuleSet(
            diseases=[{
                "id": "d1",
                "name": "D1",
                "category": "c",
                "risk_factor_modifiers": {"d1": 0.1}
            }],
            symptoms=[],
            investigations=[],
            red_flags=[],
            risk_factors=[],
            contraindications=[],
            drug_interactions=[]
        )
        with self.assertRaises(RuleValidationError):
            validate_rules(rules)


class TestDiagnosticsAndIntegration(unittest.TestCase):
    def test_reasoning_engine_with_debug_mode(self):
        # Create a sample loader
        project_dir = Path(__file__).parent.parent.parent
        loader = RuleLoader(rules_dir=project_dir / "clinical_intelligence" / "rules")
        engine = ClinicalReasoningEngine(loader=loader)

        # Build a valid clinical state payload
        state = {
            "session_id": "debug-test-session",
            "patient_info": {"age": 58, "gender": "male"},
            "chief_complaint": ["chest pain"],
            "symptoms": [
                {"name": "chest pain", "status": "Active", "confidence": "High"},
                {"name": "shortness of breath", "status": "Active", "confidence": "Medium"}
            ],
            "vital_signs": {"respiratory_rate": "18", "spo2": "97", "temperature": "37.1", "bp_systolic": "142", "pulse": "88"},
            "medical_history": ["hypertension", "diabetes"],
            "current_medications": [],
            "allergies": [],
            "family_history": [],
            "ecg_findings": "Normal",
            "lab_results": {"troponin": 0.01}
        }

        # Reason in debug mode
        res = engine.reason(state, debug=True)

        self.assertIsNotNone(res.debug_diagnostics)
        self.assertGreater(res.debug_diagnostics.time_taken_ms, 0.0)
        self.assertTrue(len(res.debug_diagnostics.triggered_rules) > 0)
        self.assertTrue(len(res.debug_diagnostics.rule_scores) > 0)
        self.assertIn("HEART Score", res.debug_diagnostics.clinical_scores)
        self.assertTrue(res.debug_diagnostics.memory_used_bytes > 0)

        # Check candidate calibration explainability fields
        for cand in res.candidate_diagnoses:
            self.assertIsNotNone(cand.why_suggested)
            self.assertIsNotNone(cand.why_competing_lower)
            self.assertIsNotNone(cand.rules_fired)
            self.assertIsNotNone(cand.rules_not_fired)


if __name__ == "__main__":
    unittest.main()
