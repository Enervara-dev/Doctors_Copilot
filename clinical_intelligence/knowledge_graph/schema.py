from __future__ import annotations

"""
schema.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

# Centralized graph node labels representing clinical entities
NODE_LABELS = {
    "Disease",
    "Symptom",
    "Medication",
    "RiskFactor",
    "Investigation",
    "Procedure",
    "BodySystem",
    "ClinicalGuideline",
    "EmergencyCondition",
    "DrugInteraction",
    "Contraindication",
    "RedFlag",
}

# Centralized clinical relationship edges connecting schema nodes
RELATIONSHIP_TYPES = {
    "HAS_SYMPTOM",
    "SUGGESTS",
    "TREATED_BY",
    "INVESTIGATED_BY",
    "CONTRAINDICATED_WITH",
    "INTERACTS_WITH",
    "RISK_FACTOR_FOR",
    "PART_OF",
    "GUIDED_BY",
    "CAUSES",
    "ASSOCIATED_WITH",
}

# Auto-generation of uniqueness constraints on ID properties (1-to-1 per label)
CONSTRAINTS = {
    label: f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
    for label in NODE_LABELS
}

# Auto-generation of name-property indexes (1-to-1 per label)
INDEXES = {
    label: f"CREATE INDEX {label.lower()}_name_idx IF NOT EXISTS FOR (n:{label}) ON (n.name)"
    for label in NODE_LABELS
}
