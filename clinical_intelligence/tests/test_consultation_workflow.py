from __future__ import annotations
import unittest
import asyncio
import time
from typing import Any, Dict, List

from clinical_intelligence.core.container import container
from clinical_intelligence.workflow.protocols import (
    IEventBus,
    ISessionManager,
    IWorkflowEngine,
    IQuestionController,
    IReasoningCoordinator,
    ITimelineEngine,
    ISessionRepository,
)
from clinical_intelligence.events.event_bus import EventBus
from clinical_intelligence.storage.repository import MemorySessionRepository
from clinical_intelligence.session.session_manager import ConsultationSessionManager, ConsultationSession
from clinical_intelligence.session.memory import ConsultationMemory
from clinical_intelligence.session.version_tracker import StateVersionTracker
from clinical_intelligence.workflow.workflow_engine import ConsultationWorkflowEngine
from clinical_intelligence.workflow.state_machine import ConsultationStateMachine, InvalidTransitionError
from clinical_intelligence.workflow.question_controller import IntelligentQuestionController
from clinical_intelligence.workflow.reasoning_coordinator import ReasoningCoordinator
from clinical_intelligence.timeline.timeline import TimelineEngine
from clinical_intelligence.events.events import (
    SessionStarted,
    QuestionAsked,
    QuestionAnswered,
    ClinicalStateUpdated,
)


class TestConsultationWorkflow(unittest.IsolatedAsyncioTestCase):
    """Verifies complete Module 4 session lifecycle, workflow DAGs, and event logs."""

    def setUp(self) -> None:
        # Re-register fresh DI instances to isolate runs
        self.event_bus = EventBus()
        self.repository = MemorySessionRepository()
        self.session_manager = ConsultationSessionManager(self.repository)
        self.workflow_engine = ConsultationWorkflowEngine()
        self.state_machine = ConsultationStateMachine()
        self.question_controller = IntelligentQuestionController()
        self.reasoning_coordinator = ReasoningCoordinator()
        self.timeline_engine = TimelineEngine()

        container.register_instance(IEventBus, self.event_bus)
        container.register_instance(ISessionRepository, self.repository)
        container.register_instance(ISessionManager, self.session_manager)
        container.register_instance(IWorkflowEngine, self.workflow_engine)
        container.register_instance(IQuestionController, self.question_controller)
        container.register_instance(IReasoningCoordinator, self.reasoning_coordinator)
        container.register_instance(ITimelineEngine, self.timeline_engine)

    def test_session_lifecycle_and_isolation(self) -> None:
        """Verify session creation, retrievals, locks, and deep-copy data isolation."""
        session1 = self.session_manager.create_session("sess_001", "p_1", "c_1")
        self.assertEqual(session1.session_id, "sess_001")
        self.assertEqual(session1.status, "CREATED")

        session2 = self.session_manager.create_session("sess_002", "p_2", "c_2")
        
        # Verify locks exist
        self.assertIsNotNone(self.session_manager.get_lock("sess_001"))
        
        # Verify isolation: updates to session1 shouldn't mutate session2 or repo version until saved
        session1.clinical_state["symptoms"] = [{"name": "Pain", "status": "Active"}]
        self.session_manager.save_session(session1)
        
        loaded2 = self.session_manager.load_session("sess_002")
        self.assertEqual(loaded2.clinical_state, {})

        # Test heartbeat expire
        session1.expires_at = time.time() - 100.0
        self.repository.save(session1)
        cleaned = self.session_manager.cleanup_expired_sessions()
        self.assertEqual(cleaned, 1)

    def test_state_machine_stage_transitions(self) -> None:
        """Verify deterministic state progress validation rules and exceptions checks."""
        session = self.session_manager.create_session("sess_003", "p_3", "c_3")
        self.assertEqual(session.consultation_stage, "Greeting")

        # Legal transition
        self.state_machine.transition_to(session, "Chief Complaint")
        self.assertEqual(session.consultation_stage, "Chief Complaint")

        # Illegal transition: greeting directly to Assessment is illegal
        with self.assertRaises(InvalidTransitionError):
            self.state_machine.transition_to(session, "Assessment")

        # Rollback check
        self.state_machine.rollback(session, "Greeting")
        self.assertEqual(session.consultation_stage, "Greeting")

    def test_consultation_memory_undo(self) -> None:
        """Verify memory findings, timeline tracking, and push-down undo snapshots."""
        session = self.session_manager.create_session("sess_004", "p_4", "c_4")
        memory = ConsultationMemory(session.consultation_memory)
        
        memory.add_confirmed_finding("chest_pain")
        self.assertIn("chest_pain", session.consultation_memory["confirmed_findings"])

        memory.add_negated_finding("fever")
        self.assertIn("fever", session.consultation_memory["negated_findings"])

        # Undo change
        ok = memory.undo()
        self.assertTrue(ok)
        self.assertNotIn("fever", session.consultation_memory["negated_findings"])

    def test_state_version_tracker_skips(self) -> None:
        """Verify MD5 hash trackers determine execution skipping flags correctly."""
        session = self.session_manager.create_session("sess_005", "p_5", "c_5")
        tracker = StateVersionTracker(session.version_tracker, session)

        changes = tracker.update_hashes(session.clinical_state, [], [])
        self.assertTrue(tracker.needs_reasoning(changes))

        # Re-verify immediately without state changes (hashes match)
        changes2 = tracker.update_hashes(session.clinical_state, [], [])
        self.assertFalse(tracker.needs_reasoning(changes2))
        self.assertFalse(tracker.needs_llm(changes2))

    def test_event_bus_pub_sub_replay(self) -> None:
        """Verify event publications, async/sync subscribers, and event replay lists."""
        events_received = []

        def event_handler(event: SessionStarted) -> None:
            events_received.append(event)

        self.event_bus.subscribe(SessionStarted, event_handler)

        event = SessionStarted(session_id="sess_006", patient_id="p_6", clinician_id="c_6")
        self.event_bus.publish(event)

        self.assertEqual(len(events_received), 1)
        self.assertEqual(events_received[0].patient_id, "p_6")

        # Test Replay
        history = self.event_bus.replay("sess_006")
        self.assertEqual(len(history), 1)
        self.assertTrue(isinstance(history[0], SessionStarted))

    def test_timeline_stage_snapshots(self) -> None:
        """Verify timeline engine captures snapshots on stage boundaries."""
        session = self.session_manager.create_session("sess_007", "p_7", "c_7")
        self.timeline_engine.take_snapshot(session, "Greeting")
        
        snap = self.timeline_engine.get_snapshot("sess_007", "Greeting")
        self.assertIsNotNone(snap)
        self.assertEqual(snap["session_id"], "sess_007")
        self.assertEqual(snap["stage"], "Greeting")

    def test_question_deduplication_priorities(self) -> None:
        """Verify controller filters answered questions and ranks high-gain items first."""
        session = self.session_manager.create_session("sess_008", "p_8", "c_8")
        
        # Setup mock questions inside cache
        from clinical_intelligence.rule_engine.models import AdaptiveQuestionItem
        q1 = AdaptiveQuestionItem(question="Is it severe?", purpose="Check pain", related_diagnosis="ACS", priority="HIGH", clinical_importance="H")
        q2 = AdaptiveQuestionItem(question="Do you smoke?", purpose="Check habit", related_diagnosis="COPD", priority="ROUTINE", clinical_importance="L")
        session.cached_subsystem_outputs["llm_adaptive_questions"] = [q1, q2]

        questions = self.question_controller.get_next_questions(session, limit=2)
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0].question, "Is it severe?") # High priority ranks first

        # Answer the first question and re-verify deduplication
        session.consultation_memory["adaptive_questions"] = ["Is it severe?"]
        session.question_history.clear()
        questions2 = self.question_controller.get_next_questions(session, limit=2)
        self.assertEqual(len(questions2), 1)
        self.assertEqual(questions2[0].question, "Do you smoke?")

    async def test_session_locks_concurrency(self) -> None:
        """Verify concurrent locking isolates and schedules session executions."""
        session = self.session_manager.create_session("sess_009", "p_9", "c_9")
        lock = self.session_manager.get_lock("sess_009")

        execution_order = []

        async def worker(task_name: str, duration: float) -> None:
            async with lock:
                execution_order.append(f"{task_name}_start")
                await asyncio.sleep(duration)
                execution_order.append(f"{task_name}_end")

        # Run tasks concurrently
        await asyncio.gather(
            worker("taskA", 0.05),
            worker("taskB", 0.01)
        )

        # Due to lock synchronization, taskB must wait for taskA to finish completely
        self.assertEqual(execution_order, ["taskA_start", "taskA_end", "taskB_start", "taskB_end"])
