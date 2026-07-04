from __future__ import annotations

"""
hybrid_scorer.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

from typing import Dict, List, Any
from pydantic import BaseModel
from clinical_intelligence.evidence.evidence_models import EvidenceContribution
from clinical_intelligence.config.evidence import MAX_PROVIDER_CONTRIBUTIONS

class DiseaseScore(BaseModel):
    disease_id: str
    confidence: float
    score_breakdown: Dict[str, Any]


class HybridEvidenceScorer:
    """
    Scorer computing candidate disease confidence scores from aggregated, normalized EvidenceContributions.
    Enforces hard provider contribution caps to balance medical sources.
    """

    def score(self, disease_id: str, contributions: List[EvidenceContribution]) -> DiseaseScore:
        # Group contributions by source
        grouped: Dict[str, List[EvidenceContribution]] = {}
        for c in contributions:
            grouped.setdefault(c.source, []).append(c)

        breakdown = {
            "symptoms": 0.0,
            "vitals": 0.0,
            "graph": 0.0,
            "guidelines": 0.0,
            "history": 0.0,
            "demographics": 0.0,
            "penalty": 0.0,
        }

        # Calculate subtotals for each source
        for source, items in grouped.items():
            subtotal = 0.0
            for c in items:
                direction = 1.0 if c.polarity == "positive" else -1.0
                subtotal += c.weight * c.confidence * direction

            # Map the engine source ID to the output breakdown categories
            if source == "symptom":
                breakdown_key = "symptoms"
            elif source == "vital":
                breakdown_key = "vitals"
            elif source == "knowledge_graph":
                breakdown_key = "graph"
            elif source == "guidelines":
                breakdown_key = "guidelines"
            elif source in ["history", "risk_factor"]:
                breakdown_key = "history"
            elif source == "demographics":
                breakdown_key = "demographics"
            elif source == "exclusion":
                breakdown_key = "penalty"
            else:
                continue

            # Accumulate subtotal for the mapped category
            breakdown[breakdown_key] += subtotal

        # Apply hard upper/lower bounds defined in MAX_PROVIDER_CONTRIBUTIONS
        symptoms_cap = MAX_PROVIDER_CONTRIBUTIONS.get("symptom", 0.50)
        breakdown["symptoms"] = max(0.0, min(symptoms_cap, breakdown["symptoms"]))

        vitals_cap = MAX_PROVIDER_CONTRIBUTIONS.get("vital", 0.15)
        breakdown["vitals"] = max(0.0, min(vitals_cap, breakdown["vitals"]))

        graph_cap = MAX_PROVIDER_CONTRIBUTIONS.get("knowledge_graph", 0.03)
        breakdown["graph"] = max(0.0, min(graph_cap, breakdown["graph"]))

        guidelines_cap = MAX_PROVIDER_CONTRIBUTIONS.get("guidelines", 0.02)
        breakdown["guidelines"] = max(0.0, min(guidelines_cap, breakdown["guidelines"]))

        history_cap = MAX_PROVIDER_CONTRIBUTIONS.get("history", 0.10) + MAX_PROVIDER_CONTRIBUTIONS.get("risk_factor", 0.10)
        breakdown["history"] = max(0.0, min(history_cap, breakdown["history"]))

        demographics_cap = MAX_PROVIDER_CONTRIBUTIONS.get("demographics", 0.05)
        breakdown["demographics"] = max(0.0, min(demographics_cap, breakdown["demographics"]))

        # Exclusions contribute only as negative penalty
        breakdown["penalty"] = min(0.0, breakdown["penalty"])

        # Compute total sum and clamp final confidence between [0.0, 1.0]
        total = sum(breakdown.values())
        confidence = max(0.0, min(1.0, total))
        breakdown["total"] = round(confidence, 4)

        # Round breakdown details for clean serialized formats
        rounded_breakdown = {k: round(v, 4) for k, v in breakdown.items()}

        return DiseaseScore(
            disease_id=disease_id,
            confidence=round(confidence, 4),
            score_breakdown=rounded_breakdown
        )
