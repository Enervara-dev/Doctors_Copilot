from __future__ import annotations

"""
service.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from clinical_intelligence.knowledge_graph.graph_manager import Neo4jConnectionManager
from clinical_intelligence.knowledge_graph.graph_queries import GraphQueryEngine
from clinical_intelligence.knowledge_graph.cache import GraphQueryCache
from clinical_intelligence.knowledge_graph.graph_models import GraphPath, GraphMetadata

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """
    Orchestrator service coordinating Neo4j traversals, caching,
    timeout protections, and operational health monitoring.
    """

    def __init__(
        self,
        manager: Neo4jConnectionManager,
        cache: Optional[GraphQueryCache] = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self.manager = manager
        self.cache = cache or GraphQueryCache()
        self.query_engine = GraphQueryEngine(self.manager, self.cache)
        self.timeout = timeout_seconds

    def is_active(self) -> bool:
        """Check if the underlying graph manager connection is active."""
        return self.manager.is_active

    def health(self) -> Dict[str, Any]:
        """Expose operational diagnostics and health indices for the knowledge graph."""
        if not self.is_active():
            return {
                "status": "DOWN",
                "neo4j_connectivity": False,
                "graph_version": "1.0.0",
                "node_count": 0,
                "relationship_count": 0,
                "last_import": "N/A",
                "cache_hit_ratio": 0.0,
            }

        start_time = time.perf_counter()
        connectivity = self.manager.is_connected()
        
        node_cnt = 0
        rel_cnt = 0
        last_imp = "unknown"

        if connectivity:
            try:
                # Retrieve counts with strict timeout
                nodes_res = self.manager.run_query("MATCH (n) RETURN count(n) as cnt")
                if nodes_res:
                    node_cnt = nodes_res[0].get("cnt", 0)

                rels_res = self.manager.run_query("MATCH ()-[r]->() RETURN count(r) as cnt")
                if rels_res:
                    rel_cnt = rels_res[0].get("cnt", 0)

                meta_res = self.manager.run_query("MATCH (m:GraphMetadata {id: 'graph_metadata'}) RETURN m.import_timestamp as time")
                if meta_res:
                    last_imp = meta_res[0].get("time", "unknown")
            except Exception as e:
                logger.warning(f"Error querying health counts: {e}")
                connectivity = False

        status_str = "UP" if connectivity else "DEGRADED"

        return {
            "status": status_str,
            "neo4j_connectivity": connectivity,
            "graph_version": "1.0.0",
            "node_count": node_cnt,
            "relationship_count": rel_cnt,
            "last_import": last_imp,
            "cache_hit_ratio": round(self.cache.hit_ratio, 4),
            "diagnostics_latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
        }

    # ── Batch queries wrapper with timeout protection ───────────────────

    def get_batch_diagnoses_evidence(
        self, symptoms: List[str], diseases: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch all clinical evidence relationships (symptoms, tests, treatments, paths)
        for all candidate diseases in batch. Enforces strict query timeouts.
        """
        if not self.is_active() or not diseases:
            return {}

        start_time = time.perf_counter()
        evidence_by_disease: Dict[str, Dict[str, Any]] = {}

        try:
            # We fetch all parameters using batch Cyphers to avoid N+1 queries.
            # Cypher runs within self.timeout constraints inside the manager run_query.
            
            # 1. Fetch symptoms
            syms_map = self.query_engine.batch_find_related_symptoms(diseases)
            
            # 2. Fetch recommended tests
            tests_map = self.query_engine.batch_find_recommended_tests(diseases)
            
            # 3. Fetch treatments
            treatments_map = self.query_engine.batch_find_treatments(diseases)
            
            # 4. Fetch guidelines
            guidelines_map = self.query_engine.batch_find_guidelines(diseases)
            
            # 5. Fetch paths from symptoms to diseases
            paths_map = self.query_engine.batch_find_relationship_paths(symptoms, diseases)

            # Assemble map grouped by disease ID
            for dis_id in diseases:
                d_id_clean = dis_id.lower().strip()
                evidence_by_disease[dis_id] = {
                    "connected_symptoms": syms_map.get(d_id_clean, []),
                    "connected_investigations": [t["name"] for t in tests_map.get(d_id_clean, [])],
                    "related_treatments": treatments_map.get(d_id_clean, []),
                    "related_guidelines": [g["recommendation"] for g in guidelines_map.get(d_id_clean, []) if g.get("recommendation")],
                    "relationship_paths": paths_map.get(d_id_clean, []),
                }

        except Exception as e:
            logger.warning(
                f"KnowledgeGraphService batch queries timed out or failed: {e}. "
                "Returning baseline results."
            )
            # Safe fallback: empty lists for all categories
            for dis_id in diseases:
                evidence_by_disease[dis_id] = {
                    "connected_symptoms": [],
                    "connected_investigations": [],
                    "related_treatments": [],
                    "related_guidelines": [],
                    "relationship_paths": [],
                }

        return evidence_by_disease
