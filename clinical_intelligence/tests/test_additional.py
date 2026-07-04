from __future__ import annotations

"""
Test suite for additional

Purpose:
Defines automated unit and integration tests to verify code coverage and correctness.

Module/Component: Test Suites
"""

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

# Import validator/exceptions/models
from clinical_intelligence.rule_engine.exceptions import RuleValidationError
from clinical_intelligence.rule_engine.validator import validate_rules
from clinical_intelligence.rule_engine.loader import RuleSet


class TestNEWS2Granular(unittest.TestCase):
    def setUp(self):
        self.evaluator = NEWS2Evaluator()

    def test_news2_spo2_scale1_93(self):
        # SpO2 93% on Scale 1 gives 2 points
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 93, "temperature": 37.0, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 2)

    def test_news2_spo2_scale1_95(self):
        # SpO2 95% on Scale 1 gives 1 point
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 95, "temperature": 37.0, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_news2_spo2_scale2_85(self):
        # SpO2 85% on Scale 2 (COPD) gives 3 points
        res = self.evaluator.calculate(
            state_data={"medical_history": ["copd"]},
            vitals={"respiratory_rate": 15, "spo2": 85, "temperature": 37.0, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=["copd"], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 2) # Scale 2: 84-85% = 2 points

    def test_news2_spo2_scale2_89(self):
        # SpO2 89% on Scale 2 (COPD) gives 0 points (normal range 88-92%)
        res = self.evaluator.calculate(
            state_data={"medical_history": ["copd"]},
            vitals={"respiratory_rate": 15, "spo2": 89, "temperature": 37.0, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=["copd"], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 0)

    def test_news2_spo2_scale2_95(self):
        # SpO2 95% on Scale 2 with supplemental oxygen gives 2 points
        res = self.evaluator.calculate(
            state_data={"medical_history": ["copd"]},
            vitals={"respiratory_rate": 15, "spo2": 95, "temperature": 37.0, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=["copd"], current_medications=["nasal cannula oxygen"], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 4) # Scale 2: 93-96% (2 pts) + supplemental oxygen (2 pts) = 4 points

    def test_news2_bp_systolic_le_90(self):
        # Systolic BP <= 90 gives 3 points
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 98, "temperature": 37.0, "bp_systolic": 85, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 3)

    def test_news2_bp_systolic_95(self):
        # Systolic BP 91-100 gives 2 points
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 98, "temperature": 37.0, "bp_systolic": 95, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 2)

    def test_news2_bp_systolic_105(self):
        # Systolic BP 101-110 gives 1 point
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 98, "temperature": 37.0, "bp_systolic": 105, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_news2_bp_systolic_225(self):
        # Systolic BP >= 220 gives 3 points
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 98, "temperature": 37.0, "bp_systolic": 225, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 3)

    def test_news2_temp_hypothermia_mild(self):
        # Temp 35.1-36.0 gives 1 point
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 98, "temperature": 35.5, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_news2_temp_fever_mild(self):
        # Temp 38.1-39.0 gives 1 point
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 98, "temperature": 38.5, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_news2_temp_fever_severe(self):
        # Temp >= 39.1 gives 2 points
        res = self.evaluator.calculate(
            state_data={},
            vitals={"respiratory_rate": 15, "spo2": 98, "temperature": 39.5, "bp_systolic": 120, "pulse": 75},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 2)


class TestqSOFAGranular(unittest.TestCase):
    def setUp(self):
        self.evaluator = qSOFAEvaluator()

    def test_qsofa_somnolence(self):
        res = self.evaluator.calculate(
            state_data={}, vitals={"respiratory_rate": 15, "bp_systolic": 120},
            present_symptoms={"somnolence"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_qsofa_coma(self):
        res = self.evaluator.calculate(
            state_data={}, vitals={"respiratory_rate": 15, "bp_systolic": 120},
            present_symptoms={"coma"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_qsofa_lethargy(self):
        res = self.evaluator.calculate(
            state_data={}, vitals={"respiratory_rate": 15, "bp_systolic": 120},
            present_symptoms={"lethargy"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_qsofa_stupor(self):
        res = self.evaluator.calculate(
            state_data={}, vitals={"respiratory_rate": 15, "bp_systolic": 120},
            present_symptoms={"stupor"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_qsofa_unresponsive(self):
        res = self.evaluator.calculate(
            state_data={}, vitals={"respiratory_rate": 15, "bp_systolic": 120},
            present_symptoms={"unresponsive"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=45, gender="male"
        )
        self.assertEqual(res.score, 1)


class TestCURB65Granular(unittest.TestCase):
    def setUp(self):
        self.evaluator = CURB65Evaluator()

    def test_curb65_score_1(self):
        # age 70 (+1)
        res = self.evaluator.calculate(
            state_data={"lab_results": {"urea": 5.0}}, vitals={"respiratory_rate": 15, "bp_systolic": 120, "bp_diastolic": 80},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=70, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_curb65_score_2(self):
        # age 70 (+1), respiration 32 (+1)
        res = self.evaluator.calculate(
            state_data={"lab_results": {"urea": 5.0}}, vitals={"respiratory_rate": 32, "bp_systolic": 120, "bp_diastolic": 80},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=70, gender="male"
        )
        self.assertEqual(res.score, 2)

    def test_curb65_score_3(self):
        # age 70 (+1), respiration 32 (+1), bp 80/50 (+1)
        res = self.evaluator.calculate(
            state_data={"lab_results": {"urea": 5.0}}, vitals={"respiratory_rate": 32, "bp_systolic": 80, "bp_diastolic": 50},
            present_symptoms=set(), negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=70, gender="male"
        )
        self.assertEqual(res.score, 3)

    def test_curb65_score_4(self):
        # age 70 (+1), respiration 32 (+1), bp 80/50 (+1), confusion (+1)
        res = self.evaluator.calculate(
            state_data={"lab_results": {"urea": 5.0}}, vitals={"respiratory_rate": 32, "bp_systolic": 80, "bp_diastolic": 50},
            present_symptoms={"confusion"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=70, gender="male"
        )
        self.assertEqual(res.score, 4)


class TestHEARTGranular(unittest.TestCase):
    def setUp(self):
        self.evaluator = HEARTEvaluator()

    def test_heart_history_moderate(self):
        # moderate chest pain characteristics (+1 pt)
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "Normal", "lab_results": {"troponin": 0.01}},
            vitals={}, present_symptoms={"chest pain", "sweating"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=35, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_heart_ecg_nonspecific(self):
        # nonspecific ST/T changes (+1 pt)
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "nonspecific repolarization disturbance", "lab_results": {"troponin": 0.01}},
            vitals={}, present_symptoms={"chest pain", "sweating"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=35, gender="male"
        )
        self.assertTrue(res.score >= 2)

    def test_heart_risk_factors_1(self):
        # 1 risk factor (+1 pt)
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "Normal", "lab_results": {"troponin": 0.01}},
            vitals={}, present_symptoms={"chest pain", "sweating"}, negated_symptoms=set(), medical_history=["hypertension"], current_medications=[], allergies=[], age=35, gender="male"
        )
        self.assertTrue(res.score >= 2)

    def test_heart_age_middle(self):
        # Age 50 (+1 pt)
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "Normal", "lab_results": {"troponin": 0.01}},
            vitals={}, present_symptoms={"chest pain", "sweating"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=50, gender="male"
        )
        self.assertTrue(res.score >= 2)

    def test_heart_troponin_borderline(self):
        # Troponin borderline elevated (+1 pt)
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "Normal", "lab_results": {"troponin": "elevated borderline"}},
            vitals={}, present_symptoms={"chest pain", "sweating"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=35, gender="male"
        )
        self.assertTrue(res.score >= 2)


class TestWellsGranular(unittest.TestCase):
    def setUp(self):
        self.evaluator = WellsEvaluator()

    def test_wells_alternative_diagnosis(self):
        # PE is #1 diagnosis or equally likely (+3 pts)
        res = self.evaluator.calculate(
            state_data={"diagnosis_tentative": ["pulmonary embolism"]},
            vitals={"pulse": 80},
            present_symptoms={"dyspnea"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=40, gender="male"
        )
        self.assertEqual(res.score, 3.0)

    def test_wells_immobilization(self):
        # Immobilization (+1.5 pts)
        res = self.evaluator.calculate(
            state_data={}, vitals={"pulse": 80},
            present_symptoms={"dyspnea"}, negated_symptoms=set(), medical_history=["immobilization"], current_medications=[], allergies=[], age=40, gender="male"
        )
        self.assertEqual(res.score, 1.5)

    def test_wells_hemoptysis(self):
        # Hemoptysis (+1.0 pt)
        res = self.evaluator.calculate(
            state_data={}, vitals={"pulse": 80},
            present_symptoms={"dyspnea", "hemoptysis"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=40, gender="male"
        )
        self.assertEqual(res.score, 1.0)

    def test_wells_malignancy(self):
        # Malignancy active (+1.0 pt)
        res = self.evaluator.calculate(
            state_data={}, vitals={"pulse": 80},
            present_symptoms={"dyspnea"}, negated_symptoms=set(), medical_history=["active cancer"], current_medications=[], allergies=[], age=40, gender="male"
        )
        self.assertEqual(res.score, 1.0)


class TestCHADS2Granular(unittest.TestCase):
    def setUp(self):
        self.evaluator = CHA2DS2VAScEvaluator()

    def test_chads_age_category_1(self):
        # Age 68 (+1 pt)
        res = self.evaluator.calculate(
            state_data={}, vitals={}, present_symptoms=set(), negated_symptoms=set(),
            medical_history=["atrial fibrillation"], current_medications=[], allergies=[], age=68, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_chads_age_category_2(self):
        # Age 78 (+2 pts)
        res = self.evaluator.calculate(
            state_data={}, vitals={}, present_symptoms=set(), negated_symptoms=set(),
            medical_history=["atrial fibrillation"], current_medications=[], allergies=[], age=78, gender="male"
        )
        self.assertEqual(res.score, 2)

    def test_chads_vascular_disease(self):
        # Vascular disease (prior MI) (+1 pt)
        res = self.evaluator.calculate(
            state_data={}, vitals={}, present_symptoms=set(), negated_symptoms=set(),
            medical_history=["atrial fibrillation", "myocardial infarction"], current_medications=[], allergies=[], age=40, gender="male"
        )
        self.assertEqual(res.score, 1)


class TestTIMIGranular(unittest.TestCase):
    def setUp(self):
        self.evaluator = TIMIEvaluator()

    def test_timi_score_1(self):
        # Age 70 (+1)
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "Normal", "lab_results": {"troponin": 0.01}},
            vitals={}, present_symptoms={"chest pain"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=70, gender="male"
        )
        self.assertEqual(res.score, 1)

    def test_timi_score_2(self):
        # Age 70 (+1), ST changes (+1)
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "ST deviation", "lab_results": {"troponin": 0.01}},
            vitals={}, present_symptoms={"chest pain"}, negated_symptoms=set(), medical_history=[], current_medications=[], allergies=[], age=70, gender="male"
        )
        self.assertEqual(res.score, 2)

    def test_timi_score_3(self):
        # Age 70 (+1), ST changes (+1), aspirin use (+1)
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "ST deviation", "lab_results": {"troponin": 0.01}},
            vitals={}, present_symptoms={"chest pain"}, negated_symptoms=set(), medical_history=[], current_medications=["aspirin"], allergies=[], age=70, gender="male"
        )
        self.assertEqual(res.score, 3)

    def test_timi_score_4(self):
        # Age 70 (+1), ST changes (+1), aspirin use (+1), 3 CAD risk factors (+1)
        res = self.evaluator.calculate(
            state_data={"ecg_findings": "ST deviation", "lab_results": {"troponin": 0.01}},
            vitals={}, present_symptoms={"chest pain"}, negated_symptoms=set(), medical_history=["hypertension", "diabetes", "hyperlipidemia"], current_medications=["aspirin"], allergies=[], age=70, gender="male"
        )
        self.assertEqual(res.score, 4)


class TestValidatorComprehensive(unittest.TestCase):
    def test_missing_disease_name(self):
        rules = RuleSet(
            diseases=[{"id": "d1", "category": "c"}],
            symptoms=[], investigations=[], red_flags=[], risk_factors=[], contraindications=[], drug_interactions=[]
        )
        with self.assertRaises(RuleValidationError):
            validate_rules(rules)

    def test_missing_disease_category(self):
        rules = RuleSet(
            diseases=[{"id": "d1", "name": "D1"}],
            symptoms=[], investigations=[], red_flags=[], risk_factors=[], contraindications=[], drug_interactions=[]
        )
        with self.assertRaises(RuleValidationError):
            validate_rules(rules)

    def test_duplicate_symptom_canonical(self):
        rules = RuleSet(
            diseases=[],
            symptoms=[{"canonical": "cough"}, {"canonical": "cough"}],
            investigations=[], red_flags=[], risk_factors=[], contraindications=[], drug_interactions=[]
        )
        with self.assertRaises(RuleValidationError):
            validate_rules(rules)

    def test_missing_red_flag_condition(self):
        rules = RuleSet(
            diseases=[], symptoms=[], investigations=[],
            red_flags=[{"id": "rf1", "severity": "CRITICAL"}],
            risk_factors=[], contraindications=[], drug_interactions=[]
        )
        with self.assertRaises(RuleValidationError):
            validate_rules(rules)

    def test_missing_contraindication_fields(self):
        rules = RuleSet(
            diseases=[], symptoms=[], investigations=[], red_flags=[], risk_factors=[],
            contraindications=[{"id": "c1", "severity": "ABSOLUTE"}],
            drug_interactions=[]
        )
        with self.assertRaises(RuleValidationError):
            validate_rules(rules)

    def test_duplicate_contraindication_id(self):
        rules = RuleSet(
            diseases=[], symptoms=[], investigations=[], red_flags=[], risk_factors=[],
            contraindications=[
                {"id": "c1", "drug_or_procedure": "d", "reason": "r", "severity": "ABSOLUTE"},
                {"id": "c1", "drug_or_procedure": "d2", "reason": "r2", "severity": "ABSOLUTE"}
            ],
            drug_interactions=[]
        )
        with self.assertRaises(RuleValidationError):
            validate_rules(rules)

    def test_duplicate_drug_interaction_id(self):
        rules = RuleSet(
            diseases=[], symptoms=[], investigations=[], red_flags=[], risk_factors=[], contraindications=[],
            drug_interactions=[
                {"id": "di1", "drug_a": "a", "drug_b": "b", "severity": "MAJOR"},
                {"id": "di1", "drug_a": "a", "drug_b": "b", "severity": "MAJOR"}
            ]
        )
        with self.assertRaises(RuleValidationError):
            validate_rules(rules)


if __name__ == "__main__":
    unittest.main()
