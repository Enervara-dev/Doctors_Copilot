from __future__ import annotations

"""
registry.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PromptDefinition:
    """Represents a versioned LLM prompt template and its limits."""

    def __init__(
        self,
        prompt_id: str,
        version: str,
        description: str,
        expected_json_schema: Dict[str, Any],
        max_context_size: int,
        max_response_size: int,
        template_string: str,
    ) -> None:
        self.prompt_id = prompt_id
        self.version = version
        self.description = description
        self.expected_json_schema = expected_json_schema
        self.max_context_size = max_context_size
        self.max_response_size = max_response_size
        self.template_string = template_string


class PromptRegistry:
    """Central registry maintaining versioned system prompt templates."""

    def __init__(self) -> None:
        self._prompts: Dict[str, PromptDefinition] = {}
        self._register_default_prompts()

    def register(self, definition: PromptDefinition) -> None:
        self._prompts[definition.prompt_id] = definition

    def get(self, prompt_id: str) -> Optional[PromptDefinition]:
        return self._prompts.get(prompt_id)

    def _register_default_prompts(self) -> None:
        # 1. Adaptive Questions prompt template
        self.register(
            PromptDefinition(
                prompt_id="adaptive_questions_v1",
                version="1.0.0",
                description="Generates targeted, high-information-gain follow-up questions.",
                max_context_size=3000,
                max_response_size=800,
                expected_json_schema={
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "purpose": {"type": "string"},
                                    "related_diagnosis": {"type": "string"},
                                    "priority": {"type": "string"},
                                    "clinical_importance": {"type": "string"},
                                },
                                "required": ["question", "purpose", "related_diagnosis", "priority", "clinical_importance"],
                            },
                        }
                    },
                    "required": ["questions"],
                },
                template_string=(
                    "You are a Senior Physician. Based on the patient clinical context, generate up to {question_limit} follow-up questions.\n"
                    "Do NOT ask questions that have already been answered or asked.\n"
                    "Consultation Stage: {stage}\n"
                    "Progress Score: {progress_percent}%\n"
                    "\n--- Context ---\n"
                    "{context}\n"
                    "\nReturn output strictly in JSON format matching the schema."
                ),
            )
        )

        # 2. SOAP Generation prompt template
        self.register(
            PromptDefinition(
                prompt_id="soap_v1",
                version="1.0.0",
                description="Generates standard Subjective, Objective, Assessment, Plan notes.",
                max_context_size=3000,
                max_response_size=1000,
                expected_json_schema={
                    "type": "object",
                    "properties": {
                        "subjective": {"type": "string"},
                        "objective": {"type": "string"},
                        "assessment": {"type": "string"},
                        "plan": {"type": "string"},
                    },
                    "required": ["subjective", "objective", "assessment", "plan"],
                },
                template_string=(
                    "Generate structured SOAP consultation notes based on the clinical evidence below.\n"
                    "\n--- Context ---\n"
                    "{context}\n"
                    "\nReturn output strictly in JSON format matching the schema."
                ),
            )
        )

        # 3. Clinical Summary prompt template
        self.register(
            PromptDefinition(
                prompt_id="clinical_summary_v1",
                version="1.0.0",
                description="Generates high-level clinical assessment summary.",
                max_context_size=3000,
                max_response_size=600,
                expected_json_schema={
                    "type": "object",
                    "properties": {
                        "assessment": {"type": "string"},
                        "key_findings": {"type": "array", "items": {"type": "string"}},
                        "unresolved_issues": {"type": "array", "items": {"type": "string"}},
                        "next_steps": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["assessment", "key_findings", "unresolved_issues", "next_steps"],
                },
                template_string=(
                    "Generate a concise clinical summary assessment.\n"
                    "\n--- Context ---\n"
                    "{context}\n"
                    "\nReturn output strictly in JSON format matching the schema."
                ),
            )
        )

        # 4. Investigation Recommendation prompt template
        self.register(
            PromptDefinition(
                prompt_id="investigation_v1",
                version="1.0.0",
                description="Generates guideline-backed test recommendations.",
                max_context_size=3000,
                max_response_size=800,
                expected_json_schema={
                    "type": "object",
                    "properties": {
                        "investigations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "reason": {"type": "string"},
                                    "priority": {"type": "string"},
                                    "guideline_reference": {"type": "string"},
                                },
                                "required": ["name", "reason", "priority", "guideline_reference"],
                            },
                        }
                    },
                    "required": ["investigations"],
                },
                template_string=(
                    "Recommend appropriate clinical investigations based on the active findings and guideline metadata.\n"
                    "\n--- Context ---\n"
                    "{context}\n"
                    "\nReturn output strictly in JSON format matching the schema."
                ),
            )
        )

        # 5. Differential Diagnosis Explanation prompt template
        self.register(
            PromptDefinition(
                prompt_id="differential_explanation_v1",
                version="1.0.0",
                description="Explains ranking comparisons between candidate diagnoses.",
                max_context_size=3000,
                max_response_size=800,
                expected_json_schema={
                    "type": "object",
                    "properties": {
                        "explanations": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        }
                    },
                    "required": ["explanations"],
                },
                template_string=(
                    "Explain why candidates are ranked in the current order based on rule weights and graph paths.\n"
                    "\n--- Context ---\n"
                    "{context}\n"
                    "\nReturn output strictly in JSON format matching the schema."
                ),
            )
        )

        # 6. Patient Explanation prompt template
        self.register(
            PromptDefinition(
                prompt_id="patient_explanation_v1",
                version="1.0.0",
                description="Generates patient-friendly explanations in simple language.",
                max_context_size=3000,
                max_response_size=800,
                expected_json_schema={
                    "type": "object",
                    "properties": {
                        "explanation_text": {"type": "string"},
                        "reassurance": {"type": "string"},
                        "warnings": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["explanation_text", "reassurance", "warnings"],
                },
                template_string=(
                    "Translate the suspected clinical findings into simple, jargon-free patient advice.\n"
                    "\n--- Context ---\n"
                    "{context}\n"
                    "\nReturn output strictly in JSON format matching the schema."
                ),
            )
        )

        # 7. Emergency Escalation prompt template
        self.register(
            PromptDefinition(
                prompt_id="emergency_escalation_v1",
                version="1.0.0",
                description="Bypasses normal flow for immediate red-flag response guidance.",
                max_context_size=2000,
                max_response_size=500,
                expected_json_schema={
                    "type": "object",
                    "properties": {
                        "alert_header": {"type": "string"},
                        "first_responder_actions": {"type": "array", "items": {"type": "string"}},
                        "stabilization_notes": {"type": "string"},
                    },
                    "required": ["alert_header", "first_responder_actions", "stabilization_notes"],
                },
                template_string=(
                    "ALERT: Active emergency red flag detected! Generate stabilization and first-responder clinical actions.\n"
                    "\n--- Context ---\n"
                    "{context}\n"
                    "\nReturn output strictly in JSON format matching the schema."
                ),
            )
        )
