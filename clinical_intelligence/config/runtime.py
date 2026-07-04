from __future__ import annotations
from typing import List

# Feature flags toggling subsystems execution
ENABLE_RAG = True
ENABLE_GRAPH = True
ENABLE_LLM = True
ENABLE_SOAP = True
ENABLE_PATIENT_EXPLANATION = True
ENABLE_ADAPTIVE_QUESTIONS = True
ENABLE_EMERGENCY_WORKFLOW = True

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
