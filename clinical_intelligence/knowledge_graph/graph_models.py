from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GraphMetadata(BaseModel):
    """Metadata detailing the Knowledge Graph versions and sync lifecycle."""
    graph_version: str = Field(description="Code version of the knowledge graph schema")
    knowledge_version: str = Field(description="Content version of the loaded rules database")
    import_timestamp: str = Field(description="ISO timestamp when the import was run")
    rule_database_version: Dict[str, int] = Field(
        default_factory=dict,
        description="Summary of active rule files mtimes or counts at sync time"
    )


class GraphNode(BaseModel):
    """A single node inside the Clinical Knowledge Graph."""
    id: str = Field(description="Unique string identifier for the node")
    label: str = Field(description="Neo4j label representing node type (e.g. Disease, Symptom)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Custom node properties")


class GraphRelationship(BaseModel):
    """A directed edge connecting two clinical nodes."""
    source_id: str = Field(description="ID of the starting/source node")
    target_id: str = Field(description="ID of the ending/target node")
    rel_type: str = Field(description="Type of the clinical relationship (e.g. HAS_SYMPTOM)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Custom edge properties")


class GraphPath(BaseModel):
    """A structured traversal path detailing clinical reasoning evidence from the graph."""
    nodes: List[GraphNode] = Field(
        default_factory=list,
        description="Chronological list of clinical entities traversed in this path"
    )
    relationships: List[GraphRelationship] = Field(
        default_factory=list,
        description="List of relationship edges connecting the nodes in order"
    )
    depth: int = Field(default=1, description="Length of the traversal path")
    evidence_source: str = Field(
        default="knowledge_graph",
        description="Origin of the underlying clinical relation (e.g. guidelines or rule database)"
    )
    guideline_reference: Optional[str] = Field(
        default=None,
        description="Optional clinical recommendation/guideline text justifying this pathway"
    )


class GraphEvidence(BaseModel):
    """Unified container for all knowledge graph discoveries related to a candidate diagnosis."""
    connected_symptoms: List[str] = Field(
        default_factory=list,
        description="Names of symptoms connected to the disease via graph links"
    )
    related_risk_factors: List[str] = Field(
        default_factory=list,
        description="Names of risk factors connected to the disease"
    )
    related_guidelines: List[str] = Field(
        default_factory=list,
        description="Guideline instructions or reference names retrieved from the graph"
    )
    connected_investigations: List[str] = Field(
        default_factory=list,
        description="Investigations recommended by graph-based disease links"
    )
    related_treatments: List[str] = Field(
        default_factory=list,
        description="Medications or procedures linked to treating this disease"
    )
    relationship_paths: List[GraphPath] = Field(
        default_factory=list,
        description="Structured traversal pathways linking symptoms/facts to this candidate"
    )
