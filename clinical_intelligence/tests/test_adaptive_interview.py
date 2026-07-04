from __future__ import annotations

"""
Test suite for adaptive_interview

Purpose:
Defines automated unit and integration tests to verify code coverage and correctness.

Module/Component: Test Suites
"""

import unittest
import asyncio
from typing import Any, Dict, List

from clinical_intelligence.interview.evidence_gap_analyzer import EvidenceGapAnalyzer, DiseaseGap
from clinical_intelligence.interview.completeness import CompletenessPlanner
from clinical_intelligence.interview.question_memory import QuestionMemoryManager
from clinical_intelligence.interview.question_planner import QuestionPlanner
from clinical_intelligence.interview.examination_planner import ExaminationPlanner
from clinical_intelligence.interview.investigation_planner import InvestigationPlanner
from clinical_intelligence.interview.stage_planner import StagePlanner
from clinical_intelligence.interview.decision_policy import DecisionPolicy
from clinical_intelligence.interview.conversation_manager import ConversationManager
from clinical_intelligence.rule_engine.models import (
    ClinicalDecisionRecommendation,
    DecisionAction,
    ActionPriority,
    ConsultationStage,
    CandidateDiagnosis,
    ClinicalReasoningResult
)
from clinical_intelligence.rule_engine.engine import ClinicalReasoningEngine

"""
Unit Tests for Adaptive Consultation Intelligence & CDSS.

Created for Module 7 (Adaptive Clinical Decision Support & Consultation Intelligence).
This module contains unit tests verifying:
- Evidence gap analyzer (positive, negative, and missing symptoms classification).
- Completeness scoring for subjective history, PMH, vitals, and physical examinations.
- Question Memory manager (refused/skipped question tracking).
- Question Planner clinical utility calculation logic.
- Red flag bypass overrides (cardiac red-flag diagnostic elevation).
- Workflow stage constraints (blocking exams and investigations during history-taking).
"""


class MockSession:
    def __init__(self, session_id: str = "mock-sess-123"):
        self.session_id = session_id
        self.consultation_stage = "Greeting"
        self.status = "ACTIVE"
        self.clinical_state = {
            "version": 1,
            "chief_complaint": [],
            "symptoms": [],
            "medical_history": [],
            "current_medications": [],
            "allergies": [],
            "vital_signs": {},
            "examination": []
        }
        self.question_history = []
        self.consultation_memory = {}
        self.timeline = []
        self.cached_subsystem_outputs = {}


class TestAdaptiveInterview(unittest.TestCase):
    """Unit tests validating the CDSS and Adaptive Consultation Intelligence planners."""

    def setUp(self) -> None:
        self.engine = ClinicalReasoningEngine()
        self.registry = self.engine._profile_registry
        self.gap_analyzer = EvidenceGapAnalyzer()
        self.completeness_planner = CompletenessPlanner()
        self.memory_manager = QuestionMemoryManager()
        self.question_planner = QuestionPlanner(self.memory_manager)
        self.examination_planner = ExaminationPlanner(self.memory_manager)
        self.investigation_planner = InvestigationPlanner(self.memory_manager)
        self.stage_planner = StagePlanner()
        self.decision_policy = DecisionPolicy()
        self.conv_manager = ConversationManager()

    def test_evidence_gap_analysis(self) -> None:
        """Verify that positive, negative, and missing findings are extracted correctly."""
        state = {
            "version": 1,
            "symptoms": [
                {"name": "cough", "present": True, "confidence": 1.0},
                {"name": "fever", "present": False, "confidence": 1.0}
            ],
            "vital_signs": {
                "temperature": {"value": "101.5", "qualitative": "High"}
            },
            "examination": [
                {"finding": "throat congestion", "body_system": "ENT"}
            ],
            "medical_history": ["diabetes"]
        }
        
        candidates = [
            {"disease_id": "influenza_like_illness", "name": "Influenza-Like Illness", "confidence": 0.85}
        ]
        
        gaps = self.gap_analyzer.analyze(state, candidates, self.registry)
        self.assertTrue(len(gaps) > 0)
        gap = gaps[0]
        self.assertEqual(gap.disease_id, "influenza_like_illness")
        # Cough should be positive
        self.assertIn("cough", [f.lower() for f in gap.positive_findings] + [f.replace("symptom_", "").lower() for f in gap.positive_findings])
        # Fever was negated
        self.assertIn("fever", [f.lower() for f in gap.negative_findings] + [f.replace("symptom_", "").lower() for f in gap.negative_findings])
        # Body aches / fatigue should be missing
        self.assertIn("fatigue", [f.lower() for f in gap.missing_findings] + [f.replace("symptom_", "").lower() for f in gap.missing_findings])
        self.assertTrue(gap.completeness > 0.0)

    def test_completeness_calculation(self) -> None:
        """Verify completeness Aggregation."""
        gaps = [
            DiseaseGap(
                disease_id="influenza_like_illness",
                disease_name="Influenza-Like Illness",
                positive_findings=["cough"],
                negative_findings=["fever"],
                missing_findings=["chills", "fatigue"],
                completeness=0.5
            )
        ]
        state = {"version": 1}
        comp = self.completeness_planner.compute_completeness(gaps, state)
        self.assertEqual(comp["overall"], 50.0)

    def test_question_memory(self) -> None:
        """Verify that already asked or skipped/refused questions are blocked."""
        session = MockSession()
        session.question_history = ["cough"]
        session.consultation_memory = {"refused": ["fatigue"]}
        
        self.assertTrue(self.memory_manager.is_blocked("cough", session))
        self.assertTrue(self.memory_manager.is_blocked("fatigue", session))
        self.assertFalse(self.memory_manager.is_blocked("fever", session))

    def test_question_planner_clinical_utility(self) -> None:
        """Verify question utility ranking."""
        gaps = [
            DiseaseGap(
                disease_id="influenza_like_illness",
                disease_name="Influenza-Like Illness",
                positive_findings=[],
                negative_findings=[],
                missing_findings=["chills", "fever", "cough"],
                completeness=0.0
            )
        ]
        session = MockSession()
        candidates = [{"disease_id": "influenza_like_illness", "name": "Influenza-Like Illness"}]
        recs = self.question_planner.plan_recommendations(gaps, session, candidates, "HPI")
        
        self.assertTrue(len(recs) > 0)
        # Should prioritize critical symptoms or fever higher
        self.assertEqual(recs[0].action, DecisionAction.ASK_QUESTION)
        self.assertIn("chills", [r.target_finding for r in recs] + [r.target_finding.lower() for r in recs])

    def test_red_flag_overrides(self) -> None:
        """Verify that DecisionPolicy prioritizes critical red flag recommendations."""
        gaps = [
            DiseaseGap(
                disease_id="acs",
                disease_name="Acute Coronary Syndrome",
                positive_findings=["chest pain"],
                negative_findings=[],
                missing_findings=["spo2", "ecg"],
                completeness=0.5
            )
        ]
        session = MockSession()
        candidates = [{"disease_id": "acs", "name": "Acute Coronary Syndrome"}]
        red_flags = [{"flag_id": "cardiac_chest_pain", "condition": "ACS", "severity": "CRITICAL"}]
        recommended_investigations = [{"investigation_id": "ecg_12_lead", "name": "12-Lead ECG", "priority": "URGENT", "reason": "Rule out myocardial infarction."}]
        
        recs, next_rec, telemetry = self.decision_policy.determine_next_action(
            gaps=gaps,
            session=session,
            candidate_diagnoses=candidates,
            red_flags=red_flags,
            recommended_investigations=recommended_investigations,
            current_stage=ConsultationStage.HPI
        )
        
        self.assertTrue(telemetry.red_flag_override)
        self.assertIsNotNone(next_rec)
        self.assertEqual(next_rec.action, DecisionAction.RECOMMEND_INVESTIGATION)
        self.assertEqual(next_rec.priority, ActionPriority.CRITICAL)
        self.assertIn("ECG", next_rec.target_finding)

    def test_workflow_constraints(self) -> None:
        """Verify that early history stage constrains ordering investigations unless overridden."""
        gaps = [
            DiseaseGap(
                disease_id="influenza_like_illness",
                disease_name="Influenza-Like Illness",
                positive_findings=[],
                negative_findings=[],
                missing_findings=["fever", "throat culture"],
                completeness=0.0
            )
        ]
        session = MockSession()
        candidates = [{"disease_id": "influenza_like_illness", "name": "Influenza-Like Illness"}]
        red_flags = []
        recommended_investigations = [{"investigation_id": "rapid_flu", "name": "Rapid Flu Test", "priority": "ROUTINE", "reason": "Confirm Influenza."}]
        
        recs, next_rec, telemetry = self.decision_policy.determine_next_action(
            gaps=gaps,
            session=session,
            candidate_diagnoses=candidates,
            red_flags=red_flags,
            recommended_investigations=recommended_investigations,
            current_stage=ConsultationStage.HPI
        )
        
        self.assertTrue(telemetry.workflow_constraint_triggered)
        # Next action should be a question, not the test, due to constraint
        self.assertEqual(next_rec.action, DecisionAction.ASK_QUESTION)
