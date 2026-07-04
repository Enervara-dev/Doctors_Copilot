from __future__ import annotations
import sys
import os
import time

# Add root folder to sys path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from clinical_intelligence.session.session_manager import ConsultationSessionManager
from clinical_intelligence.workflow.workflow_engine import ConsultationWorkflowEngine
from clinical_intelligence.workflow.state_machine import ConsultationStateMachine
from clinical_intelligence.workflow.question_controller import IntelligentQuestionController
from clinical_intelligence.workflow.reasoning_coordinator import ReasoningCoordinator
from clinical_intelligence.timeline.timeline import TimelineEngine
from clinical_intelligence.session.memory import ConsultationMemory


def main():
    print("=== START CONFLICT-FREE CONSULTATION WORKFLOW VERIFICATION ===")
    
    # 1. Initialize Registry DI Container
    eb = EventBus()
    repo = MemorySessionRepository()
    sm = ConsultationSessionManager(repo)
    we = ConsultationWorkflowEngine()
    qc = IntelligentQuestionController()
    rc = ReasoningCoordinator()
    te = TimelineEngine()
    
    container.register_instance(IEventBus, eb)
    container.register_instance(ISessionRepository, repo)
    container.register_instance(ISessionManager, sm)
    container.register_instance(IWorkflowEngine, we)
    container.register_instance(IQuestionController, qc)
    container.register_instance(IReasoningCoordinator, rc)
    container.register_instance(ITimelineEngine, te)
    
    state_machine = ConsultationStateMachine()
    
    # Create session
    session_id = "test_consultation_flow_verification"
    session = sm.create_session(session_id, "patient_abc", "doctor_xyz")
    
    # Verify stage Greeting
    print(f"Initial Consultation Stage: {session.consultation_stage} (expected: Greeting)")
    assert session.consultation_stage == "Greeting"
    
    # Update state: add symptoms to trigger HPI stage
    session.clinical_state = {
        "symptoms": [
            {"name": "Chest Pain", "status": "Active", "present": True},
            {"name": "Sweating", "status": "Active", "present": True}
        ]
    }
    sm.save_session(session)
    
    # Deterministic Stage Inference
    inferred = we.determine_stage(session)
    print(f"Inferred Stage based on state facts: {inferred} (expected: HPI)")
    assert inferred == "HPI"
    
    # Transition State Machine
    state_machine.transition_to(session, "Chief Complaint")
    state_machine.transition_to(session, "HPI")
    print(f"Consultation stage transitioned to: {session.consultation_stage}")
    assert session.consultation_stage == "HPI"
    
    # Coordinate reasoning 1st pass
    res1 = rc.coordinate_reasoning(session, run_llm=True, debug=True)
    print("First reasoning pass completed.")
    
    # Coordinate reasoning 2nd pass: state is unchanged, reasoning should be skipped!
    rc.coordinate_reasoning(session, run_llm=True, debug=True)
    print(f"Reasoning invoked: {rc.reasoning_invoked} | reasoning skipped: {rc.reasoning_skipped}")
    assert rc.reasoning_skipped == 1
    
    # Replay events
    events = eb.replay(session_id)
    print(f"Emitted Pub/Sub Event History count: {len(events)}")
    
    # Replay timeline entries
    timeline = te.get_timeline(session_id)
    print(f"Timeline Log Audits count: {len(timeline)}")
    
    # Archive session
    sm.archive_session(session_id)
    stats = sm.get_statistics()
    print(f"Session statistics summary: {stats}")
    
    print("=== ALL CONFLICT-FREE MODULE 4 ORCHESTRATION VERIFIED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
