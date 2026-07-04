from __future__ import annotations
import time
import logging
from typing import Any, Dict, List, Optional, Set
from clinical_intelligence.knowledge_graph.graph_manager import Neo4jConnectionManager
from clinical_intelligence.knowledge_graph.graph_models import (
    GraphNode,
    GraphRelationship,
    GraphPath,
)

logger = logging.getLogger(__name__)


class GraphQueryEngine:
    """Handles parameterized Cypher query traversals with telemetry tracking and caching support."""

    def __init__(self, manager: Neo4jConnectionManager, cache: Optional[Any] = None) -> None:
        self.manager = manager
        self.cache = cache

        # Diagnostics Performance Telemetry
        self.query_count = 0
        self.total_traversal_time_ms = 0.0
        self.nodes_traversed = 0
        self.relationships_traversed = 0
        self.neo4j_execution_time_ms = 0.0

    def reset_metrics(self) -> None:
        """Reset diagnostics metrics before a new reasoning pass."""
        self.query_count = 0
        self.total_traversal_time_ms = 0.0
        self.nodes_traversed = 0
        self.relationships_traversed = 0
        self.neo4j_execution_time_ms = 0.0

    def _execute_query(
        self, cypher: str, parameters: Optional[Dict[str, Any]] = None, cache_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Internal helper to execute cypher query, handle metrics, and caching."""
        self.query_count += 1
        start_traversal = time.perf_counter()

        # Check Cache if available
        if self.cache and cache_key:
            cached_val = self.cache.get(cache_key)
            if cached_val is not None:
                elapsed = (time.perf_counter() - start_traversal) * 1000.0
                self.total_traversal_time_ms += elapsed
                return cached_val

        # Execute Neo4j query
        start_neo4j = time.perf_counter()
        results = self.manager.run_query(cypher, parameters)
        neo4j_time = (time.perf_counter() - start_neo4j) * 1000.0
        self.neo4j_execution_time_ms += neo4j_time

        # Update cache
        if self.cache and cache_key:
            self.cache.set(cache_key, results)

        elapsed = (time.perf_counter() - start_traversal) * 1000.0
        self.total_traversal_time_ms += elapsed

        return results

    def find_possible_diagnoses(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """Identify candidate diseases matching the specified symptoms list."""
        if not self.manager.is_active or not symptoms:
            return []

        clean_syms = [s.lower().strip() for s in symptoms]
        cypher = """
        MATCH (s:Symptom)-[:SUGGESTS|HAS_SYMPTOM]-(d:Disease)
        WHERE toLower(s.id) IN $symptoms OR toLower(s.name) IN $symptoms
        RETURN d.id as disease_id, collect(s.id) as matched_symptoms
        """
        cache_key = f"possible_diagnoses:{','.join(clean_syms)}"
        records = self._execute_query(cypher, {"symptoms": clean_syms}, cache_key)

        # Update node/rel traversal counts based on results
        for r in records:
            self.nodes_traversed += len(r.get("matched_symptoms", [])) + 1
            self.relationships_traversed += len(r.get("matched_symptoms", []))

        return records

    def find_related_symptoms(self, disease: str) -> List[str]:
        """Retrieve symptom names connected to the disease via graph links."""
        if not self.manager.is_active or not disease:
            return []

        cypher = """
        MATCH (d:Disease)-[:HAS_SYMPTOM]-(s:Symptom)
        WHERE toLower(d.id) = toLower($disease)
        RETURN s.name as name
        """
        records = self._execute_query(cypher, {"disease": disease}, f"related_symptoms:{disease}")
        self.nodes_traversed += len(records) + 1
        self.relationships_traversed += len(records)
        return [r["name"] for r in records if "name" in r]

    def find_recommended_tests(self, disease: str) -> List[Dict[str, Any]]:
        """Get recommended medical investigations connected to this disease."""
        if not self.manager.is_active or not disease:
            return []

        cypher = """
        MATCH (d:Disease)-[:INVESTIGATED_BY]-(i:Investigation)
        WHERE toLower(d.id) = toLower($disease)
        RETURN i.id as id, i.name as name, i.priority as priority
        """
        records = self._execute_query(cypher, {"disease": disease}, f"recommended_tests:{disease}")
        self.nodes_traversed += len(records) + 1
        self.relationships_traversed += len(records)
        return records

    def find_drug_interactions(self, medications: List[str]) -> List[Dict[str, Any]]:
        """Find drug-drug interactions within a list of medications."""
        if not self.manager.is_active or len(medications) < 2:
            return []

        clean_meds = [m.lower().strip() for m in medications]
        cypher = """
        MATCH (m1:Medication)-[r:INTERACTS_WITH]-(m2:Medication)
        WHERE toLower(m1.id) IN $meds AND toLower(m2.id) IN $meds AND m1.id < m2.id
        RETURN m1.id as drug_a, m2.id as drug_b, r.severity as severity, r.description as description
        """
        cache_key = f"drug_interactions:{','.join(sorted(clean_meds))}"
        records = self._execute_query(cypher, {"meds": clean_meds}, cache_key)
        self.nodes_traversed += len(records) * 2
        self.relationships_traversed += len(records)
        return records

    def find_contraindications(self, disease: str) -> List[Dict[str, Any]]:
        """Identify medications contraindicated for a given disease."""
        if not self.manager.is_active or not disease:
            return []

        cypher = """
        MATCH (d:Disease)-[r:CONTRAINDICATED_WITH]-(m:Medication)
        WHERE toLower(d.id) = toLower($disease)
        RETURN m.id as medication_id, r.severity as severity, r.reason as reason
        """
        records = self._execute_query(cypher, {"disease": disease}, f"contraindications:{disease}")
        self.nodes_traversed += len(records) + 1
        self.relationships_traversed += len(records)
        return records

    def find_emergency_conditions(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """Find emergency/red flags associated with matching symptoms."""
        if not self.manager.is_active or not symptoms:
            return []

        clean_syms = [s.lower().strip() for s in symptoms]
        cypher = """
        MATCH (s:Symptom)-[:ASSOCIATED_WITH]-(e:EmergencyCondition)
        WHERE toLower(s.id) IN $symptoms
        RETURN e.id as emergency_id, e.severity as severity, e.recommended_action as recommended_action
        """
        cache_key = f"emergencies:{','.join(clean_syms)}"
        records = self._execute_query(cypher, {"symptoms": clean_syms}, cache_key)
        self.nodes_traversed += len(records) + len(clean_syms)
        self.relationships_traversed += len(records)
        return records

    def find_related_guidelines(self, disease: str) -> List[Dict[str, Any]]:
        """Find guidelines linked to a disease."""
        if not self.manager.is_active or not disease:
            return []

        cypher = """
        MATCH (d:Disease)-[:GUIDED_BY]-(g:ClinicalGuideline)
        WHERE toLower(d.id) = toLower($disease)
        RETURN g.organization as organization, g.recommendation as recommendation, g.guideline_name as guideline_name
        """
        records = self._execute_query(cypher, {"disease": disease}, f"guidelines:{disease}")
        self.nodes_traversed += len(records) + 1
        self.relationships_traversed += len(records)
        return records

    def find_high_risk_combinations(self) -> List[Dict[str, Any]]:
        """Retrieve list of high risk disease modifiers/risk factors relationships."""
        if not self.manager.is_active:
            return []

        cypher = """
        MATCH (d:Disease)-[:RISK_FACTOR_FOR]-(rf:RiskFactor)
        RETURN d.id as disease_id, rf.id as risk_factor_id
        """
        records = self._execute_query(cypher, {}, "high_risk_combinations")
        self.nodes_traversed += len(records) * 2
        self.relationships_traversed += len(records)
        return records

    def find_treatments(self, disease: str) -> List[str]:
        """Find medications or treatments linked to a disease."""
        if not self.manager.is_active or not disease:
            return []

        cypher = """
        MATCH (d:Disease)-[:TREATED_BY]-(m:Medication)
        WHERE toLower(d.id) = toLower($disease)
        RETURN m.id as id
        """
        records = self._execute_query(cypher, {"disease": disease}, f"treatments:{disease}")
        self.nodes_traversed += len(records) + 1
        self.relationships_traversed += len(records)
        return [r["id"] for r in records if "id" in r]

    def find_relationship_paths(self, symptoms: List[str], disease: str) -> List[GraphPath]:
        """Perform path traversals to build structured GraphPath evidence linking symptoms to disease."""
        if not self.manager.is_active or not symptoms or not disease:
            return []

        clean_syms = [s.lower().strip() for s in symptoms]
        cypher = """
        MATCH path = (s:Symptom)-[r:HAS_SYMPTOM|SUGGESTS]-(d:Disease)
        WHERE (toLower(s.id) IN $symptoms OR toLower(s.name) IN $symptoms) AND toLower(d.id) = toLower($disease)
        RETURN [node in nodes(path) | {id: node.id, label: labels(node)[0], properties: properties(node)}] as nodes,
               [rel in relationships(path) | {source_id: startNode(rel).id, target_id: endNode(rel).id, rel_type: type(rel), properties: properties(rel)}] as relationships
        """
        cache_key = f"paths:{disease}:{','.join(clean_syms)}"
        records = self._execute_query(cypher, {"symptoms": clean_syms, "disease": disease}, cache_key)

        paths = []
        for r in records:
            try:
                nodes_list = [
                    GraphNode(id=n["id"], label=n["label"], properties=n.get("properties", {}))
                    for n in r.get("nodes", [])
                ]
                rels_list = [
                    GraphRelationship(
                        source_id=rel["source_id"],
                        target_id=rel["target_id"],
                        rel_type=rel["rel_type"],
                        properties=rel.get("properties", {}),
                    )
                    for rel in r.get("relationships", [])
                ]
                paths.append(
                    GraphPath(
                        nodes=nodes_list,
                        relationships=rels_list,
                        depth=len(rels_list),
                        evidence_source="rules_knowledge_graph",
                    )
                )
                self.nodes_traversed += len(nodes_list)
                self.relationships_traversed += len(rels_list)
            except Exception as ex:
                logger.warning(f"Error parsing GraphPath traversal record: {ex}")

        return paths

    # ── Batch Query Extensions (added for Milestone 2 refinements) ──────

    def batch_find_related_symptoms(self, diseases: List[str]) -> Dict[str, List[str]]:
        """Fetch symptom names connected to multiple diseases in a single query pass."""
        if not self.manager.is_active or not diseases:
            return {}

        clean_dis = [d.lower().strip() for d in diseases]
        cypher = """
        MATCH (d:Disease)-[:HAS_SYMPTOM]-(s:Symptom)
        WHERE toLower(d.id) IN $diseases
        RETURN d.id as disease_id, collect(s.name) as symptoms
        """
        cache_key = f"batch_symptoms:{','.join(sorted(clean_dis))}"
        records = self._execute_query(cypher, {"diseases": clean_dis}, cache_key)
        
        result = {}
        for r in records:
            dis_id = r["disease_id"]
            syms = r["symptoms"]
            result[dis_id] = syms
            self.nodes_traversed += len(syms) + 1
            self.relationships_traversed += len(syms)
        return result

    def batch_find_recommended_tests(self, diseases: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Get recommended investigations for multiple diseases in a single query pass."""
        if not self.manager.is_active or not diseases:
            return {}

        clean_dis = [d.lower().strip() for d in diseases]
        cypher = """
        MATCH (d:Disease)-[:INVESTIGATED_BY]-(i:Investigation)
        WHERE toLower(d.id) IN $diseases
        RETURN d.id as disease_id, collect({id: i.id, name: i.name, priority: i.priority}) as tests
        """
        cache_key = f"batch_tests:{','.join(sorted(clean_dis))}"
        records = self._execute_query(cypher, {"diseases": clean_dis}, cache_key)
        
        result = {}
        for r in records:
            dis_id = r["disease_id"]
            tests = r["tests"]
            result[dis_id] = tests
            self.nodes_traversed += len(tests) + 1
            self.relationships_traversed += len(tests)
        return result

    def batch_find_relationship_paths(self, symptoms: List[str], diseases: List[str]) -> Dict[str, List[GraphPath]]:
        """Retrieve structured relationship paths between active symptoms and all candidate diseases in batch."""
        if not self.manager.is_active or not symptoms or not diseases:
            return {}

        clean_syms = [s.lower().strip() for s in symptoms]
        clean_dis = [d.lower().strip() for d in diseases]
        cypher = """
        MATCH path = (s:Symptom)-[r:HAS_SYMPTOM|SUGGESTS]-(d:Disease)
        WHERE (toLower(s.id) IN $symptoms OR toLower(s.name) IN $symptoms) AND toLower(d.id) IN $diseases
        RETURN d.id as disease_id,
               [node in nodes(path) | {id: node.id, label: labels(node)[0], properties: properties(node)}] as nodes,
               [rel in relationships(path) | {source_id: startNode(rel).id, target_id: endNode(rel).id, rel_type: type(rel), properties: properties(rel)}] as relationships
        """
        cache_key = f"batch_paths:{','.join(sorted(clean_dis))}:{','.join(clean_syms)}"
        records = self._execute_query(cypher, {"symptoms": clean_syms, "diseases": clean_dis}, cache_key)
        
        result: Dict[str, List[GraphPath]] = {}
        for r in records:
            dis_id = r["disease_id"]
            try:
                nodes_list = [
                    GraphNode(id=n["id"], label=n["label"], properties=n.get("properties", {}))
                    for n in r.get("nodes", [])
                ]
                rels_list = [
                    GraphRelationship(
                        source_id=rel["source_id"],
                        target_id=rel["target_id"],
                        rel_type=rel["rel_type"],
                        properties=rel.get("properties", {}),
                    )
                    for rel in r.get("relationships", [])
                ]
                path = GraphPath(
                    nodes=nodes_list,
                    relationships=rels_list,
                    depth=len(rels_list),
                    evidence_source="rules_knowledge_graph",
                )
                result.setdefault(dis_id, []).append(path)
                self.nodes_traversed += len(nodes_list)
                self.relationships_traversed += len(rels_list)
            except Exception as ex:
                logger.warning(f"Error parsing batch GraphPath traversal record: {ex}")
        return result

    def batch_find_guidelines(self, diseases: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve clinical guidelines for multiple diseases in a single query pass."""
        if not self.manager.is_active or not diseases:
            return {}

        clean_dis = [d.lower().strip() for d in diseases]
        cypher = """
        MATCH (d:Disease)-[:GUIDED_BY]-(g:ClinicalGuideline)
        WHERE toLower(d.id) IN $diseases
        RETURN d.id as disease_id, collect({organization: g.organization, recommendation: g.recommendation, guideline_name: g.guideline_name}) as guidelines
        """
        cache_key = f"batch_guidelines:{','.join(sorted(clean_dis))}"
        records = self._execute_query(cypher, {"diseases": clean_dis}, cache_key)
        
        result = {}
        for r in records:
            dis_id = r["disease_id"]
            gls = r["guidelines"]
            result[dis_id] = gls
            self.nodes_traversed += len(gls) + 1
            self.relationships_traversed += len(gls)
        return result

    def batch_find_treatments(self, diseases: List[str]) -> Dict[str, List[str]]:
        """Fetch medications connected to multiple diseases in a single query pass."""
        if not self.manager.is_active or not diseases:
            return {}

        clean_dis = [d.lower().strip() for d in diseases]
        cypher = """
        MATCH (d:Disease)-[:TREATED_BY]-(m:Medication)
        WHERE toLower(d.id) IN $diseases
        RETURN d.id as disease_id, collect(m.id) as treatments
        """
        cache_key = f"batch_treatments:{','.join(sorted(clean_dis))}"
        records = self._execute_query(cypher, {"diseases": clean_dis}, cache_key)
        
        result = {}
        for r in records:
            dis_id = r["disease_id"]
            treats = r["treatments"]
            result[dis_id] = treats
            self.nodes_traversed += len(treats) + 1
            self.relationships_traversed += len(treats)
        return result
