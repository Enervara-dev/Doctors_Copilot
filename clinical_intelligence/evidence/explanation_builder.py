from __future__ import annotations

"""
explanation_builder.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging
from typing import Dict, List
from clinical_intelligence.evidence.evidence_models import DiseaseProfile, EvidenceResult
from clinical_intelligence.evidence.hybrid_scorer import DiseaseScore
from clinical_intelligence.rule_engine.models import DiseaseExplanation

logger = logging.getLogger(__name__)

class ExplanationBuilder:
    """
    Decoupled Explanation Builder.
    Deterministically compiles structured diagnostic explanations reading
    ONLY DiseaseScore and EvidenceResult data.
    """

    def build_explanations(
        self,
        scores: List[DiseaseScore],
        evidence_by_disease: Dict[str, List[EvidenceResult]],
        disease_profiles: Dict[str, DiseaseProfile],
        questions_by_disease: Dict[str, List[str]] = None
    ) -> Dict[str, DiseaseExplanation]:
        explanations: Dict[str, DiseaseExplanation] = {}

        for score in scores:
            disease_id = score.disease_id
            profile = disease_profiles.get(disease_id)
            if not profile:
                continue

            results = evidence_by_disease.get(disease_id) or []
            supporting = []
            conflicting = []
            missing = []
            guidelines = []
            graph_paths = []

            for res in results:
                for c in res.contributions:
                    if c.polarity == "positive":
                        supporting.append(c.explanation)
                        if c.source == "guidelines":
                            guidelines.append(c.explanation)
                        elif c.source == "knowledge_graph":
                            graph_paths.append(c.explanation)
                    else:
                        conflicting.append(c.explanation)

            # Build missing information list
            present_findings = [c.finding.lower().strip() for res in results for c in res.contributions if c.polarity == "positive"]
            for s_def in profile.cardinal_symptoms:
                s_name = s_def["name"].lower().strip()
                if s_name not in present_findings:
                    missing.append(f"Missing cardinal symptom: {s_def['name']}")

            # Questions mapping
            disease_questions = (questions_by_disease or {}).get(disease_id, [])

            explanation = DiseaseExplanation(
                disease=profile.name,
                confidence=score.confidence,
                supporting_evidence=supporting,
                conflicting_evidence=conflicting,
                missing_information=missing,
                guideline_references=guidelines,
                graph_paths=graph_paths,
                score_breakdown=score.score_breakdown,
                recommended_questions=disease_questions
            )
            explanations[disease_id] = explanation

        return explanations
