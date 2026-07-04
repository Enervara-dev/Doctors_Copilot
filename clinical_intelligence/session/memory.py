from __future__ import annotations
import copy
import time
from typing import Any, Dict, List, Optional


class ConsultationMemory:
    """Tracks findings, history states, timeline amendments, and push-down undo snapshots."""

    def __init__(self, memory_dict: Dict[str, Any]) -> None:
        self.memory = memory_dict
        # Initialize memory dict fields if missing
        if "confirmed_findings" not in self.memory:
            self.memory["confirmed_findings"] = []
        if "negated_findings" not in self.memory:
            self.memory["negated_findings"] = []
        if "uncertain_findings" not in self.memory:
            self.memory["uncertain_findings"] = []
        if "medications" not in self.memory:
            self.memory["medications"] = []
        if "allergies" not in self.memory:
            self.memory["allergies"] = []
        if "investigations" not in self.memory:
            self.memory["investigations"] = []
        if "soap_history" not in self.memory:
            self.memory["soap_history"] = []
        if "adaptive_questions" not in self.memory:
            self.memory["adaptive_questions"] = []
        if "patient_answers" not in self.memory:
            self.memory["patient_answers"] = []
        if "reasoning_snapshots" not in self.memory:
            self.memory["reasoning_snapshots"] = []
        if "milestones" not in self.memory:
            self.memory["milestones"] = []
        if "undo_stack" not in self.memory:
            self.memory["undo_stack"] = []

    def save_state_to_undo(self) -> None:
        """Push a deep copy of the current memory fields to the undo stack."""
        state_snap = {
            "confirmed_findings": copy.deepcopy(self.memory["confirmed_findings"]),
            "negated_findings": copy.deepcopy(self.memory["negated_findings"]),
            "uncertain_findings": copy.deepcopy(self.memory["uncertain_findings"]),
            "medications": copy.deepcopy(self.memory["medications"]),
            "allergies": copy.deepcopy(self.memory["allergies"]),
            "investigations": copy.deepcopy(self.memory["investigations"]),
            "soap_history": copy.deepcopy(self.memory["soap_history"]),
            "adaptive_questions": copy.deepcopy(self.memory["adaptive_questions"]),
            "patient_answers": copy.deepcopy(self.memory["patient_answers"]),
            "reasoning_snapshots": copy.deepcopy(self.memory["reasoning_snapshots"]),
            "milestones": copy.deepcopy(self.memory["milestones"]),
        }
        self.memory["undo_stack"].append(state_snap)

    def undo(self) -> bool:
        """Restore the previous memory state from the undo stack."""
        if not self.memory["undo_stack"]:
            return False
        previous_state = self.memory["undo_stack"].pop()
        for k, v in previous_state.items():
            self.memory[k] = v
        return True

    def add_confirmed_finding(self, finding: str) -> None:
        self.save_state_to_undo()
        if finding not in self.memory["confirmed_findings"]:
            self.memory["confirmed_findings"].append(finding)
        if finding in self.memory["negated_findings"]:
            self.memory["negated_findings"].remove(finding)
        if finding in self.memory["uncertain_findings"]:
            self.memory["uncertain_findings"].remove(finding)

    def add_negated_finding(self, finding: str) -> None:
        self.save_state_to_undo()
        if finding not in self.memory["negated_findings"]:
            self.memory["negated_findings"].append(finding)
        if finding in self.memory["confirmed_findings"]:
            self.memory["confirmed_findings"].remove(finding)
        if finding in self.memory["uncertain_findings"]:
            self.memory["uncertain_findings"].remove(finding)

    def add_uncertain_finding(self, finding: str) -> None:
        self.save_state_to_undo()
        if finding not in self.memory["uncertain_findings"]:
            self.memory["uncertain_findings"].append(finding)
        if finding in self.memory["confirmed_findings"]:
            self.memory["confirmed_findings"].remove(finding)
        if finding in self.memory["negated_findings"]:
            self.memory["negated_findings"].remove(finding)

    def add_medication(self, med: str) -> None:
        self.save_state_to_undo()
        if med not in self.memory["medications"]:
            self.memory["medications"].append(med)

    def add_allergy(self, allergy: str) -> None:
        self.save_state_to_undo()
        if allergy not in self.memory["allergies"]:
            self.memory["allergies"].append(allergy)

    def add_investigation(self, inv: str) -> None:
        self.save_state_to_undo()
        if inv not in self.memory["investigations"]:
            self.memory["investigations"].append(inv)

    def record_soap_update(self, soap_draft: Dict[str, Any]) -> None:
        self.save_state_to_undo()
        self.memory["soap_history"].append({
            "timestamp": time.time(),
            "soap": copy.deepcopy(soap_draft)
        })

    def record_question_and_answer(self, question: str, answer: str) -> None:
        self.save_state_to_undo()
        self.memory["adaptive_questions"].append(question)
        self.memory["patient_answers"].append({
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })

    def record_reasoning_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.memory["reasoning_snapshots"].append({
            "timestamp": time.time(),
            "snapshot": copy.deepcopy(snapshot)
        })

    def record_milestone(self, stage: str, description: str) -> None:
        self.memory["milestones"].append({
            "timestamp": time.time(),
            "stage": stage,
            "description": description
        })
