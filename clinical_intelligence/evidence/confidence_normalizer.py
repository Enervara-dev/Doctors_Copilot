from __future__ import annotations

"""
confidence_normalizer.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging

logger = logging.getLogger(__name__)

class ConfidenceNormalizer:
    """
    Normalizes numerical scores into structured clinical confidence brackets
    (e.g., High, Medium, Low Confidence).
    """

    def normalize_confidence_level(self, score: float) -> str:
        """Map raw score to clinical confidence category."""
        if score >= 0.75:
            return "High"
        elif score >= 0.40:
            return "Medium"
        else:
            return "Low"
