from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ClinicalResponseValidator:
    """Validates generated clinical content formats (SOAP, Questions, Investigations)."""

    @staticmethod
    def validate_soap(soap: Dict[str, Any]) -> bool:
        """Verify that SOAP dict contains Subjective, Objective, Assessment, and Plan."""
        required = ["subjective", "objective", "assessment", "plan"]
        for field in required:
            if not soap.get(field):
                logger.warning(f"SOAP validation failed: missing or empty '{field}' section.")
                return False
        return True

    @staticmethod
    def validate_adaptive_questions(questions: List[Dict[str, Any]], answered_questions: List[str]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Deduplicates, checks priority formats, and filters out questions already asked.
        Returns:
            (is_valid, cleaned_questions_list)
        """
        seen_texts = set()
        cleaned = []

        for q in questions:
            q_text = q.get("question", "").strip()
            if not q_text:
                continue
            
            # Check duplicates or already answered
            if q_text.lower() in seen_texts or q_text.lower() in [aq.lower() for aq in answered_questions]:
                continue
            
            seen_texts.add(q_text.lower())
            
            # Ensure priority and diagnosis are set
            q["priority"] = q.get("priority", "ROUTINE")
            q["related_diagnosis"] = q.get("related_diagnosis", "General")
            q["purpose"] = q.get("purpose", "Gather clinical information")
            q["clinical_importance"] = q.get("clinical_importance", "Low")
            
            cleaned.append(q)

        return len(cleaned) > 0, cleaned

    @staticmethod
    def validate_recommended_investigations(investigations: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Checks for valid priorities, formats, and eliminates duplicate recommendations.
        """
        seen_names = set()
        cleaned = []
        
        for inv in investigations:
            name = inv.get("name", "").strip()
            if not name:
                continue
            
            if name.lower() in seen_names:
                continue
                
            seen_names.add(name.lower())
            inv["priority"] = inv.get("priority", "ROUTINE")
            inv["reason"] = inv.get("reason", "Indicated by clinical guidelines")
            
            cleaned.append(inv)
            
        return len(cleaned) > 0, cleaned
