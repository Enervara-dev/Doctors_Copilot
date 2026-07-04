from __future__ import annotations

"""
Neo4j Connection Manager

Purpose:
Manages the lifecycle of Neo4j Aura connections, verifying constraints and recovering from offline states.

Module/Component: Knowledge Graph
"""

"""
Neo4j Connection Manager

Purpose:
Manages the lifecycle of Neo4j Aura connections, verifying constraints and recovering from offline states.

Module/Component: Knowledge Graph
"""

import os
import logging
from typing import Any, Dict, List, Optional

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logger = logging.getLogger(__name__)


class Neo4jConnectionManager:
    """Manages the lifecycle, indexes, and sessions for the Neo4j Knowledge Graph."""

    def __init__(self) -> None:
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver: Optional[Driver] = None
        self.is_active = False

        if not NEO4J_AVAILABLE:
            logger.warning(
                "Neo4j library not installed. Gracefully disabling Knowledge Graph enrichment."
            )
            return

        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            # Verify connectivity eagerly on startup
            self.driver.verify_connectivity()
            self.is_active = True
            logger.info("Successfully connected to Neo4j database.")
        except Exception as e:
            logger.warning(
                f"Failed to connect to Neo4j at {self.uri}: {e}. "
                "Disabling Knowledge Graph enrichment."
            )
            self.is_active = False

    def close(self) -> None:
        """Close driver connection pools."""
        if self.driver:
            try:
                self.driver.close()
            except Exception as e:
                logger.warning(f"Error closing Neo4j driver: {e}")

    def is_connected(self) -> bool:
        """Check if driver connection is verified and active."""
        if not self.is_active or not self.driver:
            return False
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def create_constraints_and_indexes(self) -> None:
        """Automatically construct required Neo4j Unique Constraints and Indexes."""
        if not self.is_active or not self.driver:
            return

        from clinical_intelligence.knowledge_graph.schema import CONSTRAINTS, INDEXES

        with self.driver.session() as session:
            # Create unique constraints
            for label, constraint_query in CONSTRAINTS.items():
                try:
                    session.run(constraint_query)
                    logger.debug(f"Ensured constraint for label: {label}")
                except Exception as e:
                    logger.warning(f"Could not create constraint for label {label}: {e}")

            # Create indexes
            for label, index_query in INDEXES.items():
                try:
                    session.run(index_query)
                    logger.debug(f"Ensured index for label: {label}")
                except Exception as e:
                    logger.warning(f"Could not create index for label {label}: {e}")

    def run_query(
        self, cypher: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a parameterized Cypher query and return results as lists of dicts.
        If Neo4j is offline, returns an empty list gracefully.
        """
        if not self.is_active or not self.driver:
            return []

        try:
            with self.driver.session() as session:
                result = session.run(cypher, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.warning(f"Neo4j query execution failed: {e}")
            return []


# Global Singleton Manager instance
_graph_manager: Optional[Neo4jConnectionManager] = None


def get_graph_manager() -> Neo4jConnectionManager:
    """Retrieve or construct the global Neo4j connection coordinator."""
    global _graph_manager
    if _graph_manager is None:
        _graph_manager = Neo4jConnectionManager()
    return _graph_manager
