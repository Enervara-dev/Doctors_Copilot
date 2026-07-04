from __future__ import annotations

"""
differential_ranker.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging
from typing import Any, Dict, List
from clinical_intelligence.evidence.evidence_models import DiseaseProfile, EvidenceResult
from clinical_intelligence.evidence.hybrid_scorer import DiseaseScore
from clinical_intelligence.rule_engine.models import CandidateDiagnosis, ClinicalGuideline, RuleMetadata, ReasoningStep
from clinical_intelligence.config.evidence import MIN_DIAGNOSIS_SCORE, MAX_DIFFERENTIALS

logger = logging.getLogger(__name__)

class DifferentialRanker:
    """
    Ranks differential candidate diagnoses deterministically using confidence score,
    completeness, guideline support, graph consistency, and alphabetical tie-breaking.
    """

    def rank(
        self,
        scores: List[DiseaseScore],
        disease_profiles: Dict[str, DiseaseProfile],
        evidence_by_disease: Dict[str, List[EvidenceResult]]
    ) -> List[CandidateDiagnosis]:
        # Filter by threshold
        valid_scores = [s for s in scores if s.confidence >= MIN_DIAGNOSIS_SCORE]

        # Calculate auxiliary sorting features for each scored candidate
        sort_keys = []
        for s in valid_scores:
            disease_id = s.disease_id
            profile = disease_profiles.get(disease_id)
            results = evidence_by_disease.get(disease_id) or []

            # 1. Evidence completeness: count of positive polarity contributions
            pos_contrib_count = 0
            # 2. Guideline support indicator
            has_guideline_support = 0
            # 3. Graph consistency indicator
            has_graph_consistency = 0

            for res in results:
                for c in res.contributions:
                    if c.polarity == "positive":
                        pos_contrib_count += 1
                        if c.source == "guidelines":
                            has_guideline_support = 1
                        if c.source == "knowledge_graph":
                            has_graph_consistency = 1

            # Sort parameters tuple:
            # - confidence (descending)
            # - pos_contrib_count (descending)
            # - has_guideline_support (descending)
            # - has_graph_consistency (descending)
            # - disease_id (ascending) for deterministic tie-breaking
            key = (-s.confidence, -pos_contrib_count, -has_guideline_support, -has_graph_consistency, disease_id)
            sort_keys.append((key, s))

        # Sort candidate scores
        sort_keys.sort(key=lambda x: x[0])
        ranked_scores = [item[1] for item in sort_keys[:MAX_DIFFERENTIALS]]

        ranked_candidates = []
        for rank_idx, s in enumerate(ranked_scores, start=1):
            disease_id = s.disease_id
            profile = disease_profiles[disease_id]
            results = evidence_by_disease.get(disease_id) or []

            supporting_syms = []
            missing_syms = []
            negated_syms = []
            supporting_evidence = []
            risk_factors = []
            reasoning_trace = []
            graph_evidence = None

            # Compile details from EvidenceResults
            for res in results:
                # Store GraphEvidence model if available
                if res.provider == "KnowledgeGraphProvider" and "relationship_paths" in res.metadata:
                    from clinical_intelligence.knowledge_graph.graph_models import GraphEvidence
                    graph_evidence = GraphEvidence(
                        connected_symptoms=res.metadata.get("connected_symptoms", []),
                        related_risk_factors=res.metadata.get("related_risk_factors", []),
                        related_guidelines=res.metadata.get("related_guidelines", []),
                        connected_investigations=res.metadata.get("connected_investigations", []),
                        related_treatments=res.metadata.get("related_treatments", []),
                        relationship_paths=res.metadata.get("relationship_paths", [])
                    )

                for c in res.contributions:
                    if c.polarity == "positive":
                        supporting_evidence.append(c.explanation)
                        if c.source == "symptom":
                            supporting_syms.append(c.finding)
                        elif c.source == "risk_factor":
                            risk_factors.append(c.finding)
                        elif c.source == "history":
                            risk_factors.append(c.finding)
                    else:
                        if c.source == "symptom" or c.source == "exclusion":
                            negated_syms.append(c.finding)

            # Compile missing symptoms
            # Any cardinal symptom from the profile not present in supporting or negated list is missing
            present_syms_lower = [x.lower().strip() for x in supporting_syms]
            negated_syms_lower = [x.lower().strip() for x in negated_syms]
            for s_def in profile.cardinal_symptoms:
                s_name = s_def["name"]
                s_name_lower = s_name.lower().strip()
                if s_name_lower not in present_syms_lower and s_name_lower not in negated_syms_lower:
                    missing_syms.append(s_name)

            # Compile rules fired / not fired list for metadata checks
            rules_fired = []
            rules_not_fired = []
            for s_def in profile.cardinal_symptoms:
                s_name = s_def["name"]
                s_name_lower = s_name.lower().strip()
                rule_id = f"{disease_id}_cardinal_{s_name_lower.replace(' ', '_')}"
                if s_name_lower in present_syms_lower:
                    rules_fired.append(rule_id)
                else:
                    rules_not_fired.append(rule_id)

            for s_def in profile.supportive_symptoms:
                s_name = s_def["name"]
                s_name_lower = s_name.lower().strip()
                rule_id = f"{disease_id}_supportive_{s_name_lower.replace(' ', '_')}"
                if s_name_lower in present_syms_lower:
                    rules_fired.append(rule_id)
                else:
                    rules_not_fired.append(rule_id)

            # Map ClinicalGuideline
            guideline = None
            if profile.guideline_tags:
                guideline = ClinicalGuideline(
                    organization=profile.guideline_tags[0] if len(profile.guideline_tags) > 0 else "General",
                    guideline=profile.guideline_tags[1] if len(profile.guideline_tags) > 1 else "General",
                    version="1.0.0"
                )

            rule_meta = RuleMetadata(
                rule_id=disease_id,
                rule_version=profile.profile_version,
                created_date="2026-01-01",
                updated_date="2026-07-04"
            )

            # Map ReasoningSteps into trace
            step_idx = 1
            for res in results:
                for c in res.contributions:
                    reasoning_trace.append(ReasoningStep(
                        step_id=f"step_{disease_id}_{step_idx}",
                        component=res.provider,
                        description=c.explanation,
                        input_facts=[c.finding],
                        output=f"Score contribution: {c.weight * c.confidence * (1.0 if c.polarity == 'positive' else -1.0):.3f}",
                        confidence_delta=c.weight * c.confidence * (1.0 if c.polarity == 'positive' else -1.0)
                    ))
                    step_idx += 1

            candidate = CandidateDiagnosis(
                disease_id=disease_id,
                name=profile.name,
                category=profile.category,
                confidence=s.confidence,
                rank=rank_idx,
                supporting_symptoms=supporting_syms,
                missing_symptoms=missing_syms,
                negated_symptoms=negated_syms,
                matched_rules=rules_fired,
                risk_factors=risk_factors,
                supporting_evidence=supporting_evidence,
                reasoning_trace=reasoning_trace,
                guideline=guideline,
                rule_metadata=rule_meta,
                rules_fired=rules_fired,
                rules_not_fired=rules_not_fired,
                confidence_percent=round(s.confidence * 100.0, 2),
                graph_evidence=graph_evidence
            )
            ranked_candidates.append(candidate)

        # Generate deterministic explanations (why_suggested, why_competing_lower)
        for i, c in enumerate(ranked_candidates):
            sym_summary = f"{len(c.supporting_symptoms)} matching symptoms ({', '.join(c.supporting_symptoms[:3])})" if c.supporting_symptoms else "no supporting symptoms"
            rf_summary = f", risk factors ({', '.join(c.risk_factors)})" if c.risk_factors else ""
            c.why_suggested = f"Suggested because of {sym_summary}{rf_summary}. Calibrated confidence is {c.confidence_percent}%."

            if i == 0:
                c.why_competing_lower = "This is the primary candidate with the highest overall symptom alignment and risk profile."
            else:
                better_cand = ranked_candidates[i - 1]
                c.why_competing_lower = (
                    f"Scored lower than competing diagnosis '{better_cand.name}' (confidence {better_cand.confidence_percent}%) "
                    f"due to lower symptom weights or fewer matching risk factor modifiers."
                )

        return ranked_candidates
