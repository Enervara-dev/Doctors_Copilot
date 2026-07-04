from __future__ import annotations

"""
Test suite for knowledge_graph

Purpose:
Defines automated unit and integration tests to verify code coverage and correctness.

Module/Component: Test Suites
"""

import unittest
from unittest.mock import MagicMock, patch
import time

from clinical_intelligence.knowledge_graph.graph_models import (
    GraphMetadata,
    GraphNode,
    GraphRelationship,
    GraphPath,
    GraphEvidence,
)
from clinical_intelligence.knowledge_graph.graph_manager import Neo4jConnectionManager
from clinical_intelligence.knowledge_graph.graph_queries import GraphQueryEngine
from clinical_intelligence.knowledge_graph.cache import GraphQueryCache
from clinical_intelligence.knowledge_graph.service import KnowledgeGraphService
from clinical_intelligence.knowledge_graph.importer import validate_graph_data
from clinical_intelligence.rule_engine.aggregator import EvidenceAggregator, GraphEvidenceProvider, EvidenceProvider
from clinical_intelligence.rule_engine.models import CandidateDiagnosis


class TestGraphModels(unittest.TestCase):
    def test_graph_node_and_rel_models(self):
        node = GraphNode(id="acs", label="Disease", properties={"name": "Acute Coronary Syndrome"})
        self.assertEqual(node.id, "acs")
        self.assertEqual(node.label, "Disease")

        rel = GraphRelationship(source_id="chest_pain", target_id="acs", rel_type="SUGGESTS")
        self.assertEqual(rel.source_id, "chest_pain")
        self.assertEqual(rel.rel_type, "SUGGESTS")

    def test_graph_path_model(self):
        n1 = GraphNode(id="chest_pain", label="Symptom", properties={"name": "chest pain"})
        n2 = GraphNode(id="acs", label="Disease", properties={"name": "ACS"})
        rel = GraphRelationship(source_id="chest_pain", target_id="acs", rel_type="SUGGESTS")
        
        path = GraphPath(nodes=[n1, n2], relationships=[rel], depth=1, evidence_source="rules")
        self.assertEqual(len(path.nodes), 2)
        self.assertEqual(path.depth, 1)


class TestGraphQueryCache(unittest.TestCase):
    def test_cache_ttl_and_telemetry(self):
        cache = GraphQueryCache(default_ttl_seconds=0.1)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)

        # Set value
        cache.set("key1", "value1")
        
        # Hit check
        val = cache.get("key1")
        self.assertEqual(val, "value1")
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 0)

        # Miss check
        val_miss = cache.get("non_existent")
        self.assertIsNone(val_miss)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

        # Expiration check
        time.sleep(0.15)
        val_expired = cache.get("key1")
        self.assertIsNone(val_expired)
        self.assertEqual(cache.misses, 2)
        self.assertEqual(cache.hit_ratio, 1.0 / 3.0)


class TestImporterValidation(unittest.TestCase):
    def test_validation_detects_duplicate_node_ids(self):
        nodes = [
            {"id": "acs", "label": "Disease", "properties": {"name": "ACS"}},
            {"id": "acs", "label": "Disease", "properties": {"name": "ACS Dupe"}},
        ]
        errors = validate_graph_data(nodes, [])
        self.assertTrue(any("Duplicate node ID" in err for err in errors))

    def test_validation_detects_invalid_labels(self):
        nodes = [
            {"id": "sym1", "label": "InvalidLabel", "properties": {"name": "Symptom 1"}}
        ]
        errors = validate_graph_data(nodes, [])
        self.assertTrue(any("invalid clinical label" in err for err in errors))

    def test_validation_detects_dangling_edges(self):
        nodes = [
            {"id": "acs", "label": "Disease", "properties": {"name": "ACS"}}
        ]
        relationships = [
            {"source_id": "chest_pain", "target_id": "acs", "rel_type": "SUGGESTS"}
        ]
        errors = validate_graph_data(nodes, relationships)
        self.assertTrue(any("Dangling edge" in err for err in errors))

    def test_validation_detects_invalid_relationship_types(self):
        nodes = [
            {"id": "chest_pain", "label": "Symptom", "properties": {"name": "chest pain"}},
            {"id": "acs", "label": "Disease", "properties": {"name": "ACS"}}
        ]
        relationships = [
            {"source_id": "chest_pain", "target_id": "acs", "rel_type": "INVALID_REL_TYPE"}
        ]
        errors = validate_graph_data(nodes, relationships)
        self.assertTrue(any("invalid type" in err for err in errors))

    def test_validation_detects_orphan_nodes(self):
        nodes = [
            {"id": "chest_pain", "label": "Symptom", "properties": {"name": "chest pain"}},
            {"id": "acs", "label": "Disease", "properties": {"name": "ACS"}},
            {"id": "orphan_node", "label": "RiskFactor", "properties": {"name": "Orphan"}}
        ]
        relationships = [
            {"source_id": "chest_pain", "target_id": "acs", "rel_type": "SUGGESTS"}
        ]
        errors = validate_graph_data(nodes, relationships)
        self.assertTrue(any("Orphan node" in err for err in errors))


class TestGracefulDegradationAndManager(unittest.TestCase):
    @patch("clinical_intelligence.knowledge_graph.graph_manager.NEO4J_AVAILABLE", False)
    def test_manager_inactive_when_library_missing(self):
        manager = Neo4jConnectionManager()
        self.assertFalse(manager.is_active)
        self.assertFalse(manager.is_connected())
        self.assertEqual(manager.run_query("MATCH (n) RETURN n"), [])

    @patch("clinical_intelligence.knowledge_graph.graph_manager.GraphDatabase")
    def test_manager_graceful_when_db_offline(self, mock_db):
        # Force driver creation to raise connection error
        mock_db.driver.side_effect = Exception("Bolt connection timeout")
        manager = Neo4jConnectionManager()
        self.assertFalse(manager.is_active)
        self.assertEqual(manager.run_query("MATCH (n) RETURN n"), [])
        manager.create_constraints_and_indexes() # Should return immediately without crash

    @patch("clinical_intelligence.knowledge_graph.graph_manager.GraphDatabase")
    def test_manager_active_when_online(self, mock_db):
        mock_driver = MagicMock()
        mock_db.driver.return_value = mock_driver
        manager = Neo4jConnectionManager()
        self.assertTrue(manager.is_active)
        manager.close()
        mock_driver.close.assert_called_once()


class TestKnowledgeGraphService(unittest.TestCase):
    def setUp(self):
        self.mock_manager = MagicMock()
        self.mock_manager.is_active = True
        self.mock_manager.is_connected.return_value = True
        self.cache = GraphQueryCache()
        self.service = KnowledgeGraphService(self.mock_manager, self.cache)

    def test_health_check_up(self):
        # Setup mock returns for health check counts
        self.mock_manager.run_query.side_effect = [
            [{"cnt": 15}], # nodes count
            [{"cnt": 25}], # relationships count
            [{"time": "2026-07-04T00:00:00Z"}] # last import metadata
        ]
        h = self.service.health()
        self.assertEqual(h["status"], "UP")
        self.assertEqual(h["node_count"], 15)
        self.assertEqual(h["relationship_count"], 25)
        self.assertEqual(h["last_import"], "2026-07-04T00:00:00Z")

    def test_health_check_offline(self):
        self.mock_manager.is_active = False
        h = self.service.health()
        self.assertEqual(h["status"], "DOWN")
        self.assertEqual(h["node_count"], 0)

    def test_query_timeout_graceful_recovery(self):
        # Force service batch call to raise transient exception
        self.mock_manager.run_query.side_effect = Exception("Neo4j Session Transaction Timeout")
        res = self.service.get_batch_diagnoses_evidence(["chest_pain"], ["acs"])
        self.assertIn("acs", res)
        # Verify fallback empty lists returned instead of crash
        self.assertEqual(res["acs"]["connected_symptoms"], [])


class TestGraphQueriesAndAggregator(unittest.TestCase):
    def setUp(self):
        self.mock_manager = MagicMock()
        self.mock_manager.is_active = True
        self.cache = GraphQueryCache()
        self.service = KnowledgeGraphService(self.mock_manager, self.cache)
        provider = GraphEvidenceProvider(self.service)
        self.aggregator = EvidenceAggregator([provider])

    def test_traversal_queries_telemetry(self):
        self.mock_manager.run_query.return_value = [
            {"disease_id": "acs", "matched_symptoms": ["chest_pain"]}
        ]
        res = self.service.query_engine.find_possible_diagnoses(["chest_pain"])
        self.assertEqual(len(res), 1)
        self.assertEqual(self.service.query_engine.query_count, 1)
        self.assertEqual(self.service.query_engine.nodes_traversed, 2)
        self.assertEqual(self.service.query_engine.relationships_traversed, 1)

    def test_batch_find_relationship_paths_mapping(self):
        self.mock_manager.run_query.return_value = [
            {
                "disease_id": "acs",
                "nodes": [
                    {"id": "chest_pain", "label": "Symptom", "properties": {"name": "chest pain"}},
                    {"id": "acs", "label": "Disease", "properties": {"name": "ACS"}}
                ],
                "relationships": [
                    {"source_id": "chest_pain", "target_id": "acs", "rel_type": "SUGGESTS", "properties": {}}
                ]
            }
        ]
        paths_map = self.service.query_engine.batch_find_relationship_paths(["chest_pain"], ["acs"])
        self.assertIn("acs", paths_map)
        paths = paths_map["acs"]
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].depth, 1)
        self.assertEqual(paths[0].nodes[0].id, "chest_pain")
        self.assertEqual(paths[0].relationships[0].rel_type, "SUGGESTS")

    def test_aggregator_enrichment_not_altering_scores(self):
        # Setup batch queries returns
        self.mock_manager.run_query.side_effect = [
            [{"disease_id": "acs", "symptoms": ["chest pain"]}], # batch symptoms
            [{"disease_id": "acs", "tests": [{"id": "ecg", "name": "ECG", "priority": "HIGH"}]}], # batch tests
            [{"disease_id": "acs", "treatments": ["aspirin"]}], # batch treatments
            [], # batch guidelines
            [] # batch paths
        ]
        candidates = [
            CandidateDiagnosis(
                disease_id="acs",
                name="ACS",
                category="cardio",
                confidence=0.85,
                rank=1,
                supporting_symptoms=[],
                missing_symptoms=[],
                negated_symptoms=[],
                matched_rules=[],
                risk_factors=[],
                supporting_evidence=[],
                reasoning_trace=[]
            )
        ]
        
        # Run aggregator
        enriched = self.aggregator.aggregate_evidence(candidates, ["chest_pain"])
        
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].confidence, 0.85) # Rank/Score unchanged!
        self.assertEqual(enriched[0].rank, 1)
        self.assertIsNotNone(enriched[0].graph_evidence)
        self.assertEqual(enriched[0].graph_evidence.connected_symptoms, ["chest pain"])
        self.assertEqual(enriched[0].graph_evidence.connected_investigations, ["ECG"])

    def test_custom_evidence_provider_registration(self):
        class MockVectorProvider(EvidenceProvider):
            def get_evidence(self, diseases, symptoms):
                return {"acs": "vector_guideline_evidence_text"}

        candidates = [
            CandidateDiagnosis(
                disease_id="acs", name="ACS", category="c", confidence=0.9, rank=1,
                supporting_symptoms=[], missing_symptoms=[], negated_symptoms=[],
                matched_rules=[], risk_factors=[], supporting_evidence=[], reasoning_trace=[]
            )
        ]
        
        # Register both graph and vector providers
        graph_provider = GraphEvidenceProvider(self.service)
        vector_provider = MockVectorProvider()
        aggregator = EvidenceAggregator([graph_provider, vector_provider])
        
        enriched = aggregator.aggregate_evidence(candidates, ["chest_pain"])
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
