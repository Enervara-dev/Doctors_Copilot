from __future__ import annotations

"""
graph_reasoner.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging
from typing import Dict, List
from clinical_intelligence.knowledge_graph.service import KnowledgeGraphService
from clinical_intelligence.knowledge_graph.graph_models import GraphEvidence

logger = logging.getLogger(__name__)

class GraphReasoner:
    """
    Retrieves and compiles matching nodes and relationships from Neo4j paths
    into structured GraphEvidence for the Reasoning Engine.
    Degrades gracefully if database is inactive.
    """

    def __init__(self, service: KnowledgeGraphService) -> None:
        self.service = service

    def reason_evidence(self, diseases: List[str], symptoms: List[str]) -> Dict[str, GraphEvidence]:
        if not self.service.is_active():
            return {}

        try:
            raw_evidence = self.service.get_batch_diagnoses_evidence(symptoms, diseases)
        except Exception as e:
            logger.warning(f"GraphReasoner query failed: {e}")
            return {}

        mapped_evidence: Dict[str, GraphEvidence] = {}

        for dis_id, data in raw_evidence.items():
            try:
                paths = data.get("relationship_paths", [])
                matched_nodes = []
                matched_rels = []

                for path in paths:
                    for node in path.nodes:
                        if node.id not in matched_nodes:
                            matched_nodes.append(node.id)
                    for rel in path.relationships:
                        rel_id = f"{rel.source_id}->{rel.rel_type}->{rel.target_id}"
                        if rel_id not in matched_rels:
                            matched_rels.append(rel_id)

                # Compute graph confidence based on matches
                confidence = 1.0 if paths else 0.0

                mapped_evidence[dis_id] = GraphEvidence(
                    connected_symptoms=data.get("connected_symptoms", []),
                    related_risk_factors=data.get("related_risk_factors", []),
                    related_guidelines=data.get("related_guidelines", []),
                    connected_investigations=data.get("connected_investigations", []),
                    related_treatments=data.get("related_treatments", []),
                    relationship_paths=paths,
                    matched_nodes=matched_nodes,
                    matched_relationships=matched_rels,
                    confidence=confidence
                )
            except Exception as e:
                logger.warning(f"GraphReasoner: failed mapping evidence for '{dis_id}': {e}")

        return mapped_evidence
