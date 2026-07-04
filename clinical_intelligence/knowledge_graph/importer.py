from __future__ import annotations
import sys
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple
from pathlib import Path

from clinical_intelligence.knowledge_graph.graph_manager import get_graph_manager
from clinical_intelligence.knowledge_graph.graph_builder import extract_graph_elements
from clinical_intelligence.rule_engine.loader import RuleLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from clinical_intelligence.knowledge_graph.schema import NODE_LABELS as VALID_LABELS, RELATIONSHIP_TYPES as VALID_RELATIONSHIPS


def validate_graph_data(
    nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
) -> List[str]:
    """
    Perform pre-import validation of graph nodes and relationships.
    Identifies duplicates, orphans, dangling edges, invalid labels/relationships, or missing properties.
    """
    errors: List[str] = []
    node_ids: Set[str] = set()

    # 1. Validate Nodes
    for idx, node in enumerate(nodes):
        nid = node.get("id")
        label = node.get("label")
        props = node.get("properties", {})

        if not nid:
            errors.append(f"Node at index {idx} is missing a required 'id'")
            continue

        if nid in node_ids:
            errors.append(f"Duplicate node ID detected in input lists: '{nid}'")
        node_ids.add(nid)

        if label not in VALID_LABELS:
            errors.append(f"Node '{nid}' has an invalid clinical label: '{label}'")

        if not props.get("name"):
            errors.append(f"Node '{nid}' is missing the required 'name' property")

    # 2. Validate Relationships
    rel_keys: Set[Tuple[str, str, str]] = set()
    connected_nodes: Set[str] = set()

    for idx, rel in enumerate(relationships):
        src = rel.get("source_id")
        tgt = rel.get("target_id")
        rtype = rel.get("rel_type")
        props = rel.get("properties", {})

        if not src or not tgt:
            errors.append(f"Relationship at index {idx} has missing source_id or target_id")
            continue

        # Check for dangling edges
        if src not in node_ids:
            errors.append(f"Dangling edge: source node '{src}' does not exist in nodes list")
        if tgt not in node_ids:
            errors.append(f"Dangling edge: target node '{tgt}' does not exist in nodes list")

        if rtype not in VALID_RELATIONSHIPS:
            errors.append(
                f"Relationship between '{src}' and '{tgt}' has invalid type: '{rtype}'"
            )

        # Check for duplicates
        rkey = (src, tgt, rtype)
        if rkey in rel_keys:
            errors.append(f"Duplicate relationship detected: {src} -[{rtype}]-> {tgt}")
        rel_keys.add(rkey)

        connected_nodes.add(src)
        connected_nodes.add(tgt)

    # 3. Validate Orphans
    for nid in node_ids:
        # ClinicalGuidelines might be connected only outgoing, which is fine
        if nid not in connected_nodes:
            errors.append(f"Orphan node detected: '{nid}' has no connected relationships")

    return errors


def import_rules_to_graph() -> bool:
    """
    Load active rule YAML files, validate the generated graph structures,
    and bulk import them into Neo4j. Returns True on success, False otherwise.
    """
    logger.info("Initializing rule loader...")
    project_dir = Path(__file__).parent.parent.parent
    rules_dir = project_dir / "clinical_intelligence" / "rules"
    
    loader = RuleLoader(rules_dir=rules_dir)
    rules = loader.get()

    logger.info("Extracting graph elements from RuleSet...")
    nodes, relationships = extract_graph_elements(rules)

    logger.info("Running pre-import validations...")
    validation_errors = validate_graph_data(nodes, relationships)
    if validation_errors:
        logger.error(
            f"Graph Import Aborted! Found {len(validation_errors)} validation violations:"
        )
        for err in validation_errors[:10]:
            logger.error(f" - {err}")
        if len(validation_errors) > 10:
            logger.error(f" ... and {len(validation_errors) - 10} more errors.")
        return False

    logger.info("Pre-import validation passed successfully.")

    manager = get_graph_manager()
    if not manager.is_active:
        logger.error("Neo4j database connection is offline. Aborting import.")
        return False

    try:
        logger.info("Clearing existing clinical graph database...")
        manager.run_query("MATCH (n) DETACH DELETE n")

        logger.info("Ensuring schema indexes and constraints are created...")
        manager.create_constraints_and_indexes()

        # Batch creation of nodes by label
        logger.info("Batch creating graph nodes...")
        label_groups: Dict[str, List[Dict[str, Any]]] = {}
        for node in nodes:
            label = node["label"]
            label_groups.setdefault(label, []).append(
                {"id": node["id"], **node["properties"]}
            )

        for label, props_list in label_groups.items():
            cypher = f"UNWIND $batch as p CREATE (n:{label}) SET n = p"
            manager.run_query(cypher, {"batch": props_list})

        # Batch creation of relationships by relationship type
        logger.info("Batch creating graph relationships...")
        rel_groups: Dict[str, List[Dict[str, Any]]] = {}
        for rel in relationships:
            rtype = rel["rel_type"]
            rel_groups.setdefault(rtype, []).append(
                {
                    "source_id": rel["source_id"],
                    "target_id": rel["target_id"],
                    "properties": rel["properties"],
                }
            )

        for rtype, rels_list in rel_groups.items():
            cypher = (
                f"UNWIND $batch as r "
                f"MATCH (a {{id: r.source_id}}) "
                f"MATCH (b {{id: r.target_id}}) "
                f"MERGE (a)-[rel:{rtype}]->(b) "
                f"SET rel = r.properties"
            )
            manager.run_query(cypher, {"batch": rels_list})

        # Write Import Metadata node
        logger.info("Writing graph lifecycle metadata...")
        import_time = datetime.utcnow().isoformat()
        metadata_cypher = """
        CREATE (m:GraphMetadata {
            id: "graph_metadata",
            graph_version: $graph_version,
            knowledge_version: $knowledge_version,
            import_timestamp: $import_timestamp,
            rule_database_version: $rule_database_version
        })
        """
        # Form database version string summaries
        db_summary = {
            "disease_rules": len(rules.diseases),
            "symptom_rules": len(rules.symptoms),
            "red_flags": len(rules.red_flags),
            "contraindications": len(rules.contraindications),
            "drug_interactions": len(rules.drug_interactions),
        }
        manager.run_query(
            metadata_cypher,
            {
                "graph_version": "1.0.0",
                "knowledge_version": f"ruleset-v{len(rules.diseases)}",
                "import_timestamp": import_time,
                "rule_database_version": str(db_summary),
            },
        )

        logger.info("Knowledge Graph Sync Completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Knowledge Graph import failed with exception: {e}")
        return False
    finally:
        manager.close()


if __name__ == "__main__":
    success = import_rules_to_graph()
    sys.exit(0 if success else 1)
