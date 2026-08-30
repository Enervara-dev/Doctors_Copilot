from __future__ import annotations

"""
models.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

"""
Pydantic output models for the Rule-Based Clinical Reasoning Engine.

All models are fully typed and validated with Pydantic v2.
The top-level output is `ClinicalReasoningResult`.
"""

from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

class AlertType(str, Enum):
    RED_FLAG = "red_flag"
    CONTRAINDICATION = "contraindication"
    INTERACTION = "interaction"
    MISSING_INFO = "missing_info"
    RISK_FACTOR = "risk_factor"

from clinical_intelligence.knowledge_graph.graph_models import (
    GraphEvidence,
    GraphMetadata,
)


class ClinicalGuideline(BaseModel):
    """Metadata representing the clinical guideline a rule references."""
    organization: str
    guideline_name: str = Field(alias="guideline", default="")
    version: str
    recommendation: Optional[str] = None
    strength: Optional[Literal["Strong", "Moderate", "Weak"]] = None
    evidence_level: Optional[Literal["A", "B", "C"]] = None

    model_config = ConfigDict(populate_by_name=True)


class RuleMetadata(BaseModel):
    """Metadata representing rule authoring information."""
    rule_id: str
    rule_version: str = "1.0.0"
    created_date: str = "2026-01-01"
    updated_date: str = "2026-07-03"


class ClinicalScoreResult(BaseModel):
    """Outputs for a single clinical score calculated by the engine."""
    name: str
    score: float
    interpretation: str
    evidence: List[str]
    recommendations: List[str]



# ─────────────────────────────────────────────
# Telemetry and Citation models (added for Milestone 2 Module 3)
# ─────────────────────────────────────────────

# Strongly typed output models for LLM clinical statements
class ClinicalSummaryModel(BaseModel):
    assessment: str
    key_findings: List[str] = Field(default_factory=list)
    unresolved_issues: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)

class SOAPModel(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str

class LLMInvestigationRecommendation(BaseModel):
    name: str
    reason: str
    priority: str
    guideline_reference: str

class PatientExplanationModel(BaseModel):
    explanation_text: str
    reassurance: str
    warnings: List[str] = Field(default_factory=list)

class EmergencyEscalationModel(BaseModel):
    alert_header: str
    first_responder_actions: List[str] = Field(default_factory=list)
    stabilization_notes: str

class DifferentialExplanationModel(BaseModel):
    explanations: Dict[str, str] = Field(default_factory=dict)


class EvidenceCitation(BaseModel):
    rule_id: Optional[str] = None
    graph_path_id: Optional[str] = None
    guideline_id: Optional[str] = None
    guideline_version: Optional[str] = None
    clinical_score: Optional[str] = None
    confidence: float = 1.0
    evidence_type: str  # E.g. "Rule", "Graph", "Guideline", "Score"

class AdaptiveQuestionItem(BaseModel):
    question: str
    purpose: str
    related_diagnosis: str
    priority: str
    clinical_importance: str

class LLMTelemetry(BaseModel):
    prompt_execution_id: str
    model: str
    provider: str
    prompt_id: str
    prompt_version: str
    provider_name: Optional[str] = None
    provider_version: Optional[str] = None
    retriever_version: Optional[str] = None
    graph_version: Optional[str] = None
    context_version: Optional[str] = None
    retrieval_time_ms: Optional[float] = None
    validation_time_ms: Optional[float] = None
    formatting_time_ms: Optional[float] = None
    total_llm_pipeline_time_ms: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    retry_count: int = 0
    cache_hit: bool = False
    validation_failed: bool = False
    execution_timestamp: str

class RetrieverTelemetry(BaseModel):
    retrieved_documents: int = 0
    filtered_documents: int = 0
    average_similarity: float = 0.0
    retrieval_latency_ms: float = 0.0

class ProviderTelemetry(BaseModel):
    provider_name: str
    is_available: bool = True
    error_rate: float = 0.0
    timeout_count: int = 0
    rate_limit_count: int = 0
    average_latency_ms: float = 0.0

class TaskTelemetry(BaseModel):
    task_name: str
    priority: int
    execution_order: int
    latency_ms: float = 0.0
    status: str  # "success", "failed", "cancelled", "skipped"

class LLMTelemetrySummary(BaseModel):
    total_requests: int = 0
    successful_requests: int = 0
    retries: int = 0
    failures: int = 0
    provider_switches: int = 0
    cache_hit_ratio: float = 0.0
    average_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    tasks: List[TaskTelemetry] = Field(default_factory=list)
    providers: List[ProviderTelemetry] = Field(default_factory=list)
    retriever: Optional[RetrieverTelemetry] = None
    llm_calls: List[LLMTelemetry] = Field(default_factory=list)

class LLMExplainability(BaseModel):
    generated_from_prompt_id: str
    prompt_version: str
    citations: List[EvidenceCitation] = Field(default_factory=list)
    reasoning_summary: str
    confidence_rating: str  # "High", "Medium", "Low"
    needs_human_review: bool = False


class WorkflowTelemetry(BaseModel):
    """Workflow execution times, transition stats, and skipped modules metrics."""
    workflow_stage: str
    execution_duration: float
    session_version: str
    workflow_transitions: int
    pause_count: int = 0
    resume_count: int = 0
    rollback_count: int = 0
    skipped_reasoning: int = 0
    skipped_graph: int = 0
    skipped_retrieval: int = 0
    skipped_llm: int = 0
    emitted_events: int = 0
    timeline_entries: int = 0
    active_session_count: int = 0
    idle_session_count: int = 0
    archived_session_count: int = 0
    request_id: Optional[str] = None


class DebugDiagnostics(BaseModel):
    """Diagnostics details populated when debug mode is enabled."""
    triggered_rules: List[str]
    rule_scores: Dict[str, float]
    clinical_scores: Dict[str, float]
    evidence: List[str]
    contraindications: List[str]
    guidelines_used: List[str]
    time_taken_ms: float
    memory_used_bytes: int
    request_id: Optional[str] = None

    # Graph diagnostic telemetry (added for Milestone 2 Module 2)
    graph_query_count: int = 0
    graph_traversal_time_ms: float = 0.0
    graph_cache_hits: int = 0
    graph_cache_misses: int = 0
    nodes_traversed: int = 0
    relationships_traversed: int = 0
    neo4j_execution_time_ms: float = 0.0

    # LLM diagnostic telemetry (added for Milestone 2 Module 3)
    llm_telemetry: Optional[LLMTelemetrySummary] = None
    
    # Workflow diagnostic telemetry (added for Milestone 2 Module 4)
    workflow_telemetry: Optional[WorkflowTelemetry] = None


# ─────────────────────────────────────────────
# Reasoning trace
# ─────────────────────────────────────────────

class ReasoningStep(BaseModel):
    """One atomic step in the reasoning trace."""

    step_id: str = Field(description="Unique identifier for this reasoning step")
    component: str = Field(description="Engine component that produced this step")
    description: str = Field(description="Human-readable description of the reasoning")
    input_facts: List[str] = Field(
        default_factory=list,
        description="Clinical facts used as input to this step",
    )
    output: str = Field(description="Conclusion reached by this step")
    confidence_delta: float = Field(
        default=0.0,
        description="How much this step changed the candidate's confidence score",
    )


class ExplainabilityMetadata(BaseModel):
    """Full reasoning audit trail for the entire engine run."""

    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    rules_evaluated: int = Field(default=0, description="Total rule evaluations performed")
    rules_matched: int = Field(default=0, description="Rules that contributed to output")
    reasoning_duration_ms: float = Field(
        default=0.0, description="Wall-clock time for the full reasoning pass (ms)"
    )
    rule_files_loaded: List[str] = Field(
        default_factory=list, description="YAML rule files active during this run"
    )


# ─────────────────────────────────────────────
# Candidate diagnoses
# ─────────────────────────────────────────────

class CandidateDiagnosis(BaseModel):
    """A single candidate diagnosis produced by the reasoning engine."""

    disease_id: str = Field(description="Unique identifier matching diseases.yaml")
    name: str = Field(description="Human-readable disease name")
    category: str = Field(description="Clinical category, e.g. cardiovascular")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Normalised confidence score [0, 1]"
    )
    rank: int = Field(description="Rank by confidence (1 = highest)")
    supporting_symptoms: List[str] = Field(
        default_factory=list, description="Present symptoms that support this diagnosis"
    )
    missing_symptoms: List[str] = Field(
        default_factory=list,
        description="Cardinal symptoms that are absent or unasked",
    )
    negated_symptoms: List[str] = Field(
        default_factory=list, description="Explicitly negated symptoms"
    )
    matched_rules: List[str] = Field(
        default_factory=list, description="Rule IDs that fired for this diagnosis"
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="Active risk factors that raised confidence"
    )
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Free-text evidence items (vitals, history, demographics)",
    )
    reasoning_trace: List[ReasoningStep] = Field(
        default_factory=list, description="Step-by-step reasoning for this candidate"
    )
    guideline: Optional[ClinicalGuideline] = None
    rule_metadata: Optional[RuleMetadata] = None
    why_suggested: Optional[str] = None
    why_competing_lower: Optional[str] = None
    rules_fired: List[str] = Field(default_factory=list)
    rules_not_fired: List[str] = Field(default_factory=list)
    clinical_scores_used: List[str] = Field(default_factory=list)
    confidence_percent: Optional[float] = None
    graph_evidence: Optional[GraphEvidence] = None


# ─────────────────────────────────────────────
# Red flags
# ─────────────────────────────────────────────

class RedFlagAlert(BaseModel):
    """A life-threatening clinical scenario detected by the engine."""

    flag_id: str
    condition: str
    severity: Literal["CRITICAL", "HIGH", "MODERATE"]
    supporting_evidence: List[str] = Field(default_factory=list)
    recommended_action: str
    triggered_by: List[str] = Field(
        default_factory=list,
        description="Symptoms / vitals / history items that triggered this flag",
    )
    guideline: Optional[ClinicalGuideline] = None
    rule_metadata: Optional[RuleMetadata] = None


# ─────────────────────────────────────────────
# Risk factors
# ─────────────────────────────────────────────

class IdentifiedRiskFactor(BaseModel):
    """A clinical risk factor identified in the patient's state."""

    factor_id: str
    name: str
    category: str
    present: bool
    source: str = Field(description="Where this was found, e.g. medical_history, vitals")
    confidence_modifier: float = Field(
        description="How much this factor adjusts disease confidence scores"
    )


# ─────────────────────────────────────────────
# Investigations
# ─────────────────────────────────────────────

class InvestigationRecommendation(BaseModel):
    """A recommended investigation with priority and rationale."""

    investigation_id: str
    name: str
    priority: Literal["URGENT", "HIGH", "ROUTINE"]
    reason: str
    for_conditions: List[str] = Field(
        default_factory=list,
        description="Disease names that this investigation targets",
    )
    escalated_by_red_flag: bool = Field(
        default=False,
        description="True if priority was elevated because a red flag is active",
    )
    guideline: Optional[ClinicalGuideline] = None
    rule_metadata: Optional[RuleMetadata] = None


# ─────────────────────────────────────────────
# Missing information
# ─────────────────────────────────────────────

class MissingInfoItem(BaseModel):
    """A clinically important piece of information not yet collected."""

    field: str = Field(description="Logical field name, e.g. 'pain_radiation'")
    question: str = Field(description="Suggested follow-up question for the clinician")
    priority: int = Field(
        ge=1, le=5,
        description="Priority 1 (most urgent) to 5 (nice to have)",
    )
    relevant_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions for which this info would change confidence",
    )


# ─────────────────────────────────────────────
# Contraindications
# ─────────────────────────────────────────────

class Contraindication(BaseModel):
    """A detected contraindication."""

    contraindication_id: str
    drug_or_procedure: str = Field(
        description="The treatment or investigation that is contraindicated"
    )
    reason: str = Field(description="Why it is contraindicated")
    trigger: str = Field(
        description="The allergy, history item, or medication that triggers this"
    )
    severity: Literal["ABSOLUTE", "RELATIVE"]
    recommendation: str
    guideline: Optional[ClinicalGuideline] = None
    rule_metadata: Optional[RuleMetadata] = None


# ─────────────────────────────────────────────
# Drug interactions
# ─────────────────────────────────────────────

class DrugInteraction(BaseModel):
    """A clinically significant drug–drug interaction."""

    interaction_id: str
    drug_a: str
    drug_b: str
    severity: Literal["MAJOR", "MODERATE", "MINOR"]
    mechanism: str
    clinical_effect: str
    recommendation: str
    guideline: Optional[ClinicalGuideline] = None
    rule_metadata: Optional[RuleMetadata] = None


# ─────────────────────────────────────────────
# Clinical alerts
# ─────────────────────────────────────────────

class ClinicalAlert(BaseModel):
    """A general-purpose clinical alert generated by the engine."""

    alert_id: str
    alert_type: AlertType = Field(
        description="Category: red_flag | contraindication | interaction | missing_info | risk_factor"
    )
    severity: Literal["CRITICAL", "HIGH", "MODERATE", "LOW", "INFO"]
    title: str
    message: str
    supporting_evidence: List[str] = Field(default_factory=list)
    recommended_action: Optional[str] = None
    guideline: Optional[ClinicalGuideline] = None
    rule_metadata: Optional[RuleMetadata] = None


# ─────────────────────────────────────────────
# Top-level result
# ─────────────────────────────────────────────

class ClinicalReasoningResult(BaseModel):
    """
    The complete output of a single clinical reasoning pass.

    This is the sole public output type of ClinicalReasoningEngine.reason().
    Every field is fully explainable and deterministically derived from
    rule evaluations — no LLM is involved.
    """

    session_id: str
    reasoning_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Diagnoses ranked by confidence
    candidate_diagnoses: List[CandidateDiagnosis] = Field(default_factory=list)

    # Life-threatening alerts (highest priority)
    red_flags: List[RedFlagAlert] = Field(default_factory=list)

    # Risk factor analysis
    risk_factors: List[IdentifiedRiskFactor] = Field(default_factory=list)

    # Recommended investigations
    recommended_investigations: List[InvestigationRecommendation] = Field(
        default_factory=list
    )

    # Missing clinical information + follow-up questions
    missing_information: List[MissingInfoItem] = Field(default_factory=list)

    # Safety checks
    contraindications: List[Contraindication] = Field(default_factory=list)
    drug_interactions: List[DrugInteraction] = Field(default_factory=list)

    # Actionable alerts (aggregated)
    clinical_alerts: List[ClinicalAlert] = Field(default_factory=list)

    # Full reasoning audit trail
    explainability: ExplainabilityMetadata = Field(
        default_factory=ExplainabilityMetadata
    )

    # Engine metadata
    engine_version: str = Field(default="1.0.0")
    rule_set_version: Dict[str, Any] = Field(default_factory=dict)

    # Added features for Milestone 2
    clinical_scores: List[ClinicalScoreResult] = Field(default_factory=list)
    debug_diagnostics: Optional[DebugDiagnostics] = None
    graph_metadata: Optional[GraphMetadata] = None
    graph_status: Optional[Dict[str, Any]] = None

    # Hybrid Intelligence features (added for Milestone 2 Module 3)
    llm_adaptive_questions: List[AdaptiveQuestionItem] = Field(default_factory=list)
    llm_differential_explanations: Dict[str, str] = Field(default_factory=dict)
    llm_recommended_investigations: List[LLMInvestigationRecommendation] = Field(default_factory=list)
    llm_clinical_summary: Optional[ClinicalSummaryModel] = None
    llm_soap_notes: Optional[SOAPModel] = None
    llm_patient_explanation: Optional[PatientExplanationModel] = None
    llm_emergency_escalation: Optional[EmergencyEscalationModel] = None
    llm_explainability: Optional[LLMExplainability] = None
    # "ok" when the enrichment ran, "unavailable" when no provider answered, and
    # "disabled" when the layer is switched off. The llm_* fields above are empty in
    # both non-ok cases: there is no templated stand-in any more, so a consumer can
    # tell "the model said nothing" from "the model said this".
    llm_status: str = "ok"
    needs_human_review: bool = False

    # Decision support recommendations (added for Milestone 2 Module 7)
    decision_support: Optional[ConsultationDecisionResult] = None

    pipeline_version: str = "1.0.0"
    retriever_version: str = "1.0.0"
    prompt_version: str = "1.0.0"
    provider_version: str = "1.0.0"

    model_config = ConfigDict()


# ─────────────────────────────────────────────
# Planners and Interview stage models (Milestone 2 Module 7)
# ─────────────────────────────────────────────

from enum import Enum

class ConsultationStage(str, Enum):
    GREETING = "Greeting"
    CHIEF_COMPLAINT = "Chief Complaint"
    HPI = "HPI"
    PMH = "PMH"
    MEDICATION_REVIEW = "Medication Review"
    ALLERGY_REVIEW = "Allergy Review"
    EXAMINATION = "Examination"
    INVESTIGATIONS = "Investigations"
    ASSESSMENT = "Assessment"
    PLAN = "Plan"
    FOLLOW_UP = "Follow-up"
    COMPLETED = "Completed"

class DecisionAction(str, Enum):
    ASK_QUESTION = "ask_question"
    ADVANCE_STAGE = "advance_stage"
    COMPLETE_CONSULTATION = "complete_consultation"
    RECOMMEND_INVESTIGATION = "recommend_investigation"
    RECOMMEND_EXAM = "recommend_exam"
    NO_ACTION = "no_action"

class ActionPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

class ClinicalDecisionRecommendation(BaseModel):
    action: DecisionAction
    priority: ActionPriority
    confidence: float
    rationale: str
    affected_diagnoses: List[str] = Field(default_factory=list)
    expected_information_gain: float = 0.0
    target_finding: str = "none"
    rule_sources: List[str] = Field(default_factory=list)

class DecisionTelemetry(BaseModel):
    candidate_count: int
    red_flag_override: bool = False
    workflow_constraint_triggered: bool = False
    decision_time_ms: float = 0.0
    planner_selected: str = ""
    utility_score: float = 0.0

class DiseaseExplanation(BaseModel):
    disease: str
    confidence: float
    supporting_evidence: List[str] = Field(default_factory=list)
    conflicting_evidence: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    guideline_references: List[str] = Field(default_factory=list)
    graph_paths: List[str] = Field(default_factory=list)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    recommended_questions: List[str] = Field(default_factory=list)

class ConsultationDecisionResult(BaseModel):
    recommendations: List[ClinicalDecisionRecommendation] = Field(default_factory=list)
    next_recommendation: Optional[ClinicalDecisionRecommendation] = None
    workflow_stage: ConsultationStage
    consultation_progress: float = 0.0
    telemetry: DecisionTelemetry
