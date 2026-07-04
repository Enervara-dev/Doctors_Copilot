from __future__ import annotations

"""
Evidence Integration Broker

Purpose:
Fuses Neo4j Knowledge Graph and guideline guidelines onto candidate reasoning scores.

Module/Component: Rule Engine Core
"""

"""
Evidence Integration Broker

Purpose:
Fuses Neo4j Knowledge Graph and guideline guidelines onto candidate reasoning scores.

Module/Component: Rule Engine Core
"""

import logging
from typing import Any, Dict, List
from clinical_intelligence.rule_engine.models import CandidateDiagnosis
from clinical_intelligence.knowledge_graph.service import KnowledgeGraphService
from clinical_intelligence.knowledge_graph.graph_models import GraphEvidence

logger = logging.getLogger(__name__)


class EvidenceProvider:
    """Abstract base class representing a clinical evidence retrieval channel."""

    def get_evidence(
        self, diseases: List[str], symptoms: List[str]
    ) -> Dict[str, Any]:
        """
        Fetch clinical evidence for a list of candidate diseases and present symptoms.
        Returns a dict mapping: disease_id -> evidence payload
        """
        raise NotImplementedError


class GraphEvidenceProvider(EvidenceProvider):
    """Retrieves clinical evidence from the Neo4j Knowledge Graph database."""

    def __init__(self, service: KnowledgeGraphService) -> None:
        self.service = service

    def get_evidence(
        self, diseases: List[str], symptoms: List[str]
    ) -> Dict[str, GraphEvidence]:
        """Fetch batched graph relationship paths and map them to GraphEvidence models."""
        if not self.service.is_active():
            return {}

        raw_evidence = self.service.get_batch_diagnoses_evidence(symptoms, diseases)
        mapped_evidence = {}

        for dis_id, data in raw_evidence.items():
            try:
                mapped_evidence[dis_id] = GraphEvidence(
                    connected_symptoms=data.get("connected_symptoms", []),
                    related_risk_factors=data.get("related_risk_factors", []),
                    related_guidelines=data.get("related_guidelines", []),
                    connected_investigations=data.get("connected_investigations", []),
                    related_treatments=data.get("related_treatments", []),
                    relationship_paths=data.get("relationship_paths", []),
                )
            except Exception as e:
                logger.warning(
                    f"GraphEvidenceProvider failed to map evidence for disease '{dis_id}': {e}"
                )

        return mapped_evidence


from clinical_intelligence.rag.retriever import ClinicalGuidelinesRetriever

class VectorEvidenceProvider(EvidenceProvider):
    """Retrieves clinical guideline citations from the Vector Store retriever."""

    def __init__(self, retriever: ClinicalGuidelinesRetriever, state: Dict[str, Any]) -> None:
        self.retriever = retriever
        self.state = state

    def get_evidence(
        self, diseases: List[str], symptoms: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch ranked guidelines for candidates and present symptoms."""
        candidates_input = [{"disease_id": d, "category": ""} for d in diseases]
        scores_input = {} # Vitals and scores matched dynamically inside the retriever

        guidelines = self.retriever.retrieve_explainable_guidelines(
            symptoms=symptoms,
            candidates=candidates_input,
            risk_factors=self.state.get("medical_history", []) or [],
            medications=self.state.get("current_medications", []) or [],
            age=float(self.state.get("patient_info", {}).get("age", 0.0) or 0.0),
            gender=str(self.state.get("patient_info", {}).get("gender", "") or ""),
            red_flags=[],
            scores=scores_input,
            stage="Assessment"
        )

        mapped_evidence: Dict[str, List[Dict[str, Any]]] = {}
        for gl in guidelines:
            # Match matching_entities list
            for ent in gl.get("matching_entities", []) or []:
                ent_id = str(ent).lower().strip()
                mapped_evidence.setdefault(ent_id, []).append(gl)
            # Safe fallback: if matches disease_id directly
            for d in diseases:
                d_clean = d.lower().strip()
                if d_clean in gl.get("text_content", "").lower():
                    mapped_evidence.setdefault(d_clean, []).append(gl)

        return mapped_evidence


class EvidenceAggregator:
    """Coordinates multiple pluggable EvidenceProviders to enrich candidate diagnoses."""

    def __init__(self, providers: List[EvidenceProvider]) -> None:
        self.providers = providers

    def aggregate_evidence(
        self, candidates: List[CandidateDiagnosis], present_symptoms: List[str]
    ) -> List[CandidateDiagnosis]:
        """
        Queries all registered providers in batch and merges retrieved evidence
        into the candidates. Does NOT modify rules rankings or confidences.
        """
        if not candidates:
            return candidates

        diseases = [c.disease_id for c in candidates]

        for provider in self.providers:
            try:
                evidence_map = provider.get_evidence(diseases, present_symptoms)
                
                # Merge evidence into candidate diagnoses
                for cand in candidates:
                    dis_id = cand.disease_id.lower().strip()
                    if dis_id in evidence_map:
                        evidence_item = evidence_map[dis_id]
                        if isinstance(evidence_item, GraphEvidence):
                            cand.graph_evidence = evidence_item
                        elif isinstance(evidence_item, list):
                            # Append guideline text references to supporting_evidence
                            for gl in evidence_item:
                                ref = f"Guideline Recommendation: {gl.get('title')} ({gl.get('organization')}) - {gl.get('text_content')}"
                                if ref not in cand.supporting_evidence:
                                    cand.supporting_evidence.append(ref)
            except Exception as e:
                logger.warning(
                    f"EvidenceAggregator failed to execute provider {provider.__class__.__name__}: {e}"
                )

        return candidates
