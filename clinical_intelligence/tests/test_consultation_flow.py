from __future__ import annotations

"""
Test suite for consultation_flow

Purpose:
Defines automated unit and integration tests to verify code coverage and correctness.

Module/Component: Test Suites
"""

import unittest
import asyncio
from typing import Any, Dict, List

from clinical_intelligence.workflow.reasoning_coordinator import ReasoningCoordinator
from clinical_intelligence.rule_engine.models import DecisionAction, ActionPriority, ConsultationStage
from clinical_intelligence.session.session_manager import ConsultationSessionManager
from clinical_intelligence.storage.repository import MemorySessionRepository
from clinical_intelligence.core.container import container
from clinical_intelligence.workflow.protocols import ISessionManager, ISessionRepository
from clinical_intelligence.events.event_bus import EventBus
from clinical_intelligence.workflow.protocols import IEventBus

"""
Integration Tests for End-to-End CDSS Consultation Flows.

Created for Module 7 (Adaptive Clinical Decision Support & Consultation Intelligence).
This file contains integration tests simulating clinical consultation flows:
1. Chest pain emergency red flag override: Checks that a patient presenting with high-confidence
   chest symptoms immediately gets recommended for a 12-lead ECG, bypassing history constraints.
2. Standard viral URI progression: Simulates a multi-turn history-taking flow that sequentially
   moves stages from Chief Complaint to HPI, and then HPI to PMH as symptoms are populated.
"""


class TestConsultationFlow(unittest.TestCase):
    """Integration test suite verifying multi-turn consultation flows, constraints, and progress."""

    def setUp(self) -> None:
        self.coordinator = ReasoningCoordinator()
        self.registry = self.coordinator.engine._profile_registry
        self.repository = MemorySessionRepository()
        self.session_manager = ConsultationSessionManager(self.repository)
        self.event_bus = EventBus()
        container.register_instance(ISessionRepository, self.repository)
        container.register_instance(ISessionManager, self.session_manager)
        container.register_instance(IEventBus, self.event_bus)

    def test_chest_pain_red_flag_flow(self) -> None:
        """Verify red flag override triggers immediate ECG recommendation instead of HPI questions."""
        session = self.session_manager.create_session("sess_redflag", "p_1", "c_1")
        session.consultation_stage = "HPI"
        session.clinical_state = {
            "version": 1,
            "chief_complaint": ["chest pain"],
            "symptoms": [
                {"name": "chest pain", "present": True, "status": "Active", "confidence": "High"},
                {"name": "sweating", "present": True, "status": "Active", "confidence": "High"},
                {"name": "radiating arm pain", "present": True, "status": "Active", "confidence": "High"}
            ]
        }

        # 1. Run coordinator reasoning pass
        result = self.coordinator.coordinate_reasoning(session, run_llm=False)
        
        self.assertIsNotNone(result.decision_support)
        ds = result.decision_support
        
        # Red flags check
        self.assertTrue(len(result.red_flags) > 0)
        self.assertTrue(ds.telemetry.red_flag_override)
        
        # Next recommendation should be critical ECG
        self.assertIsNotNone(ds.next_recommendation)
        self.assertEqual(ds.next_recommendation.action, DecisionAction.RECOMMEND_INVESTIGATION)
        self.assertEqual(ds.next_recommendation.priority, ActionPriority.CRITICAL)
        self.assertTrue(any("ECG" in r.target_finding or "ecg" in r.target_finding.lower() for r in ds.recommendations))

    def test_viral_uri_standard_progression(self) -> None:
        """Verify normal workflow transitions from Chief Complaint -> HPI -> PMH as symptoms are populated."""
        session = self.session_manager.create_session("sess_viral", "p_1", "c_1")
        session.consultation_stage = "Chief Complaint"
        session.clinical_state = {
            "version": 1,
            "chief_complaint": ["cough"],
            "symptoms": [
                {"name": "symptom_cough", "present": True, "status": "Active", "confidence": "High"}
            ]
        }

        # 1. First turn: Chief Complaint to HPI
        result = self.coordinator.coordinate_reasoning(session, run_llm=False)
        self.assertEqual(session.consultation_stage, "HPI")
        self.assertEqual(result.decision_support.next_recommendation.action, DecisionAction.ADVANCE_STAGE)

        # 2. Add HPI details
        session.clinical_state["symptoms"].append({"name": "symptom_nasal_congestion", "present": True, "status": "Active", "confidence": "High"})
        session.clinical_state["symptoms"].append({"name": "symptom_fever", "present": False, "status": "Resolved", "confidence": "High"})
        
        # 3. Second turn: trigger HPI to PMH transition recommendation
        result = self.coordinator.coordinate_reasoning(session, run_llm=False)
        self.assertEqual(session.consultation_stage, "PMH")
