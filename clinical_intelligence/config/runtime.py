from __future__ import annotations

"""
runtime.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

from typing import List
import os

# Feature flags toggling subsystems execution
ENABLE_RAG = os.getenv("CLINICAL_COPILOT_ENABLE_RAG", "True").lower() in ("true", "1", "yes")
ENABLE_GRAPH = os.getenv("CLINICAL_COPILOT_ENABLE_GRAPH", "True").lower() in ("true", "1", "yes")
ENABLE_LLM = os.getenv("CLINICAL_COPILOT_ENABLE_LLM", "True").lower() in ("true", "1", "yes")
ENABLE_SOAP = os.getenv("CLINICAL_COPILOT_ENABLE_SOAP", "True").lower() in ("true", "1", "yes")
ENABLE_PATIENT_EXPLANATION = os.getenv("CLINICAL_COPILOT_ENABLE_PATIENT_EXPLANATION", "True").lower() in ("true", "1", "yes")
ENABLE_ADAPTIVE_QUESTIONS = os.getenv("CLINICAL_COPILOT_ENABLE_ADAPTIVE_QUESTIONS", "True").lower() in ("true", "1", "yes")
ENABLE_EMERGENCY_WORKFLOW = os.getenv("CLINICAL_COPILOT_ENABLE_EMERGENCY_WORKFLOW", "True").lower() in ("true", "1", "yes")
ENABLE_RED_FLAG_ALERTS = os.getenv("CLINICAL_COPILOT_ENABLE_RED_FLAG_ALERTS", "True").lower() in ("true", "1", "yes")

# Execution retry limits and timeouts
PROVIDER_TIMEOUT_SECONDS = 30.0
RETRY_LIMIT = 3
MAX_PARALLEL_TASKS = 4

# Context budget constraints
MAX_CONTEXT_TOKENS = 4096
MAX_RESPONSE_TOKENS = 1024

# Circuit Breaker resilience parameters
CIRCUIT_BREAKER = {
    "failure_threshold": 3,
    "cooldown_seconds": 30.0,
    "half_open_requests": 1,
}


class TaskDefinition:
    """Represents a configuration-driven execution task block in the LLM pipeline."""

    def __init__(
        self,
        task_id: str,
        priority: int,
        dependencies: List[str],
        timeout_seconds: float,
        enabled: bool,
    ) -> None:
        self.task_id = task_id
        self.priority = priority
        self.dependencies = dependencies
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled


# Task definitions configuration mapping
TASK_DEFINITIONS = [
    TaskDefinition(
        task_id="EmergencyEscalation",
        priority=1,
        dependencies=[],
        timeout_seconds=5.0,
        enabled=ENABLE_EMERGENCY_WORKFLOW,
    ),
    TaskDefinition(
        task_id="AdaptiveQuestions",
        priority=2,
        dependencies=[],
        timeout_seconds=5.0,
        enabled=ENABLE_ADAPTIVE_QUESTIONS,
    ),
    TaskDefinition(
        task_id="InvestigationRecommendations",
        priority=3,
        dependencies=["AdaptiveQuestions"],
        timeout_seconds=5.0,
        enabled=True,
    ),
    TaskDefinition(
        task_id="DifferentialExplanation",
        priority=4,
        dependencies=["AdaptiveQuestions"],
        timeout_seconds=5.0,
        enabled=True,
    ),
    TaskDefinition(
        task_id="ClinicalSummary",
        priority=5,
        dependencies=["InvestigationRecommendations", "DifferentialExplanation"],
        timeout_seconds=5.0,
        enabled=True,
    ),
    TaskDefinition(
        task_id="SOAPGeneration",
        priority=6,
        dependencies=["ClinicalSummary"],
        timeout_seconds=5.0,
        enabled=ENABLE_SOAP,
    ),
    TaskDefinition(
        task_id="PatientExplanation",
        priority=7,
        dependencies=["SOAPGeneration"],
        timeout_seconds=5.0,
        enabled=ENABLE_PATIENT_EXPLANATION,
    ),
]
