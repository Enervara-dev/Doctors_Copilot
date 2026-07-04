from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from clinical_intelligence.rule_engine.models import (
    ClinicalSummaryModel,
    SOAPModel,
    LLMInvestigationRecommendation,
    PatientExplanationModel,
    EmergencyEscalationModel,
)

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Converts raw validated JSON dictionaries into strongly-typed Pydantic models."""

    @staticmethod
    def format_clinical_summary(data: Optional[Dict[str, Any]]) -> Tuple[Optional[ClinicalSummaryModel], float]:
        """Convert to ClinicalSummaryModel. Returns (model, formatting_time_ms)."""
        start_t = time.perf_counter()
        if not data:
            return None, 0.0
        try:
            model = ClinicalSummaryModel(
                assessment=data.get("assessment", ""),
                key_findings=data.get("key_findings", []),
                unresolved_issues=data.get("unresolved_issues", []),
                next_steps=data.get("next_steps", []),
            )
            return model, (time.perf_counter() - start_t) * 1000.0
        except Exception as e:
            logger.warning(f"ResponseFormatter: Failed to format ClinicalSummaryModel: {e}")
            return None, (time.perf_counter() - start_t) * 1000.0

    @staticmethod
    def format_soap(data: Optional[Dict[str, Any]]) -> Tuple[Optional[SOAPModel], float]:
        """Convert to SOAPModel."""
        start_t = time.perf_counter()
        if not data:
            return None, 0.0
        try:
            model = SOAPModel(
                subjective=data.get("subjective", ""),
                objective=data.get("objective", ""),
                assessment=data.get("assessment", ""),
                plan=data.get("plan", ""),
            )
            return model, (time.perf_counter() - start_t) * 1000.0
        except Exception as e:
            logger.warning(f"ResponseFormatter: Failed to format SOAPModel: {e}")
            return None, (time.perf_counter() - start_t) * 1000.0

    @staticmethod
    def format_investigations(data: Optional[Dict[str, Any]]) -> Tuple[List[LLMInvestigationRecommendation], float]:
        """Convert list of recommendations to LLMInvestigationRecommendation models."""
        start_t = time.perf_counter()
        if not data or "investigations" not in data:
            return [], 0.0
        try:
            models = []
            for item in data.get("investigations", []):
                models.append(
                    LLMInvestigationRecommendation(
                        name=item.get("name", ""),
                        reason=item.get("reason", ""),
                        priority=item.get("priority", "ROUTINE"),
                        guideline_reference=item.get("guideline_reference", "General Clinical Guidelines"),
                    )
                )
            return models, (time.perf_counter() - start_t) * 1000.0
        except Exception as e:
            logger.warning(f"ResponseFormatter: Failed to format LLMInvestigationRecommendation list: {e}")
            return [], (time.perf_counter() - start_t) * 1000.0

    @staticmethod
    def format_patient_explanation(data: Optional[Dict[str, Any]]) -> Tuple[Optional[PatientExplanationModel], float]:
        """Convert to PatientExplanationModel."""
        start_t = time.perf_counter()
        if not data:
            return None, 0.0
        try:
            model = PatientExplanationModel(
                explanation_text=data.get("explanation_text", ""),
                reassurance=data.get("reassurance", ""),
                warnings=data.get("warnings", []),
            )
            return model, (time.perf_counter() - start_t) * 1000.0
        except Exception as e:
            logger.warning(f"ResponseFormatter: Failed to format PatientExplanationModel: {e}")
            return None, (time.perf_counter() - start_t) * 1000.0

    @staticmethod
    def format_emergency_escalation(data: Optional[Dict[str, Any]]) -> Tuple[Optional[EmergencyEscalationModel], float]:
        """Convert to EmergencyEscalationModel."""
        start_t = time.perf_counter()
        if not data:
            return None, 0.0
        try:
            model = EmergencyEscalationModel(
                alert_header=data.get("alert_header", "EMERGENCY CLINICAL ESCALATION REQUIRED"),
                first_responder_actions=data.get("first_responder_actions", []),
                stabilization_notes=data.get("stabilization_notes", ""),
            )
            return model, (time.perf_counter() - start_t) * 1000.0
        except Exception as e:
            logger.warning(f"ResponseFormatter: Failed to format EmergencyEscalationModel: {e}")
            return None, (time.perf_counter() - start_t) * 1000.0
