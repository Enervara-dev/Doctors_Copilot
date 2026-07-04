from __future__ import annotations

"""
normalizer.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging
from typing import List, Dict
from clinical_intelligence.evidence.evidence_models import EvidenceContribution

logger = logging.getLogger(__name__)

class EvidenceNormalizer:
    """
    Cleans, deduplicates, and resolves conflicts within aggregated clinical evidence contributions
    prior to scoring.
    """

    def normalize(self, contributions: List[EvidenceContribution]) -> List[EvidenceContribution]:
        if not contributions:
            return []

        # 1. Deduplicate/merge by finding keys
        merged: Dict[str, EvidenceContribution] = {}

        for contrib in contributions:
            key = contrib.finding.lower().strip()
            if key not in merged:
                merged[key] = contrib
            else:
                existing = merged[key]
                
                # Polarity Conflict Resolution: "negative" (exclusionary) overrides "positive"
                if contrib.polarity == "negative" or existing.polarity == "negative":
                    existing.polarity = "negative"
                    existing.weight = max(existing.weight, contrib.weight)
                    existing.explanation = f"[CONFLICT RESOLVED] Exclusionary/negated finding overrides positive match for: '{existing.finding}'"
                else:
                    # Positive merges: Keep highest confidence and update properties
                    existing.confidence = max(existing.confidence, contrib.confidence)
                    existing.weight = max(existing.weight, contrib.weight)

        return list(merged.values())
