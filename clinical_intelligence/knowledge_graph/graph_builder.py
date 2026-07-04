from __future__ import annotations

"""
Knowledge Graph Schema Builder

Purpose:
Builds Neo4j schema definitions and imports database node types.

Module/Component: Knowledge Graph
"""

"""
Knowledge Graph Schema Builder

Purpose:
Builds Neo4j schema definitions and imports database node types.

Module/Component: Knowledge Graph
"""

import logging
from typing import Any, Dict, List, Set, Tuple
from clinical_intelligence.rule_engine.loader import RuleSet

logger = logging.getLogger(__name__)


def extract_graph_elements(rules: RuleSet) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract nodes and relationships from a RuleSet rules database.
    Returns:
        (nodes_list, relationships_list)
        where nodes_list has dicts like: {"id": str, "label": str, "properties": dict}
        and relationships_list has dicts like: {"source_id": str, "target_id": str, "rel_type": str, "properties": dict}
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    relationships: List[Dict[str, Any]] = []

    # Map of disease IDs to easily check label roles
    all_disease_ids = {dis.get("id") for dis in rules.diseases if dis.get("id")}
    all_symptom_ids = {sym.get("canonical") for sym in rules.symptoms if sym.get("canonical")}

    def add_node(node_id: str, label: str, properties: Dict[str, Any]) -> None:
        nid = str(node_id).strip()
        if not nid:
            return
        
        # Resolve Label Conflict (primary role check)
        # If this ID belongs to a Disease, its label must always be Disease
        resolved_label = label
        if nid in all_disease_ids:
            resolved_label = "Disease"
        elif nid in all_symptom_ids and label not in ("Disease", "RedFlag", "EmergencyCondition"):
            resolved_label = "Symptom"

        if nid in nodes:
            nodes[nid]["properties"].update(properties)
            # If label resolved to a stronger type, update it
            if resolved_label != nodes[nid]["label"]:
                # Upgrade to stronger label (e.g. RiskFactor -> Disease)
                if nodes[nid]["label"] == "RiskFactor" and resolved_label == "Disease":
                    nodes[nid]["label"] = "Disease"
                elif nodes[nid]["label"] == "Symptom" and resolved_label == "Disease":
                    nodes[nid]["label"] = "Disease"
        else:
            nodes[nid] = {"id": nid, "label": resolved_label, "properties": properties}

    def add_relationship(
        source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any]
    ) -> None:
        sid = str(source_id).strip()
        tid = str(target_id).strip()
        if not sid or not tid:
            return
        relationships.append(
            {
                "source_id": sid,
                "target_id": tid,
                "rel_type": rel_type,
                "properties": properties,
            }
        )

    # 1. Symptoms
    for sym in rules.symptoms:
        sym_id = sym.get("canonical", "")
        add_node(
            sym_id,
            "Symptom",
            {"name": sym_id, "aliases": sym.get("aliases", [])},
        )

    # 2. Risk Factors
    for rf in rules.risk_factors:
        rf_id = rf.get("id", "")
        add_node(
            rf_id,
            "RiskFactor",
            {"name": rf.get("name", rf_id), "category": rf.get("category", "clinical")},
        )

    # 3. Investigations
    for inv in rules.investigations:
        inv_id = inv.get("id", "")
        add_node(
            inv_id,
            "Investigation",
            {"name": inv.get("name", inv_id)},
        )
        # Link dynamically to for_diseases
        for dis_id in inv.get("for_diseases", []):
            add_node(dis_id, "Disease", {"name": dis_id})
            add_relationship(
                dis_id,
                inv_id,
                "INVESTIGATED_BY",
                {"priority": inv.get("priority", "ROUTINE")},
            )

    # 4. Diseases
    for dis in rules.diseases:
        dis_id = dis.get("id", "")
        add_node(
            dis_id,
            "Disease",
            {"name": dis.get("name", dis_id), "category": dis.get("category", "")},
        )

        # Connect Guideline
        guideline = dis.get("clinical_guideline")
        if guideline:
            g_id = f"guideline_{dis_id}"
            add_node(
                g_id,
                "ClinicalGuideline",
                {
                    "organization": guideline.get("organization", ""),
                    "guideline_name": guideline.get("guideline_name", guideline.get("guideline", "")),
                    "version": guideline.get("version", ""),
                    "recommendation": guideline.get("recommendation", ""),
                    "strength": guideline.get("strength", "Strong"),
                    "evidence_level": guideline.get("evidence_level", "A"),
                    "name": guideline.get("guideline_name", guideline.get("guideline", ""))
                },
            )
            add_relationship(dis_id, g_id, "GUIDED_BY", {})

        # Connect Symptoms (Cardinal)
        for cs in dis.get("cardinal_symptoms", []):
            cs_name = cs.get("name", "")
            add_node(cs_name, "Symptom", {"name": cs_name})
            add_relationship(
                dis_id,
                cs_name,
                "HAS_SYMPTOM",
                {"weight": float(cs.get("weight", 0.5)), "type": "cardinal"},
            )
            add_relationship(cs_name, dis_id, "SUGGESTS", {})

        # Connect Symptoms (Supportive)
        for ss in dis.get("supportive_symptoms", []):
            ss_name = ss.get("name", "")
            add_node(ss_name, "Symptom", {"name": ss_name})
            add_relationship(
                dis_id,
                ss_name,
                "HAS_SYMPTOM",
                {"weight": float(ss.get("weight", 0.2)), "type": "supportive"},
            )

        # Connect Risk Factors
        for rf_mod, weight in dis.get("risk_factor_modifiers", {}).items():
            rf_label = "Disease" if rf_mod in all_disease_ids else "RiskFactor"
            add_node(rf_mod, rf_label, {"name": rf_mod})
            add_relationship(
                dis_id,
                rf_mod,
                "RISK_FACTOR_FOR",
                {"modifier": float(weight)},
            )

    # 5. Red Flags / Emergency Conditions
    for rf_alert in rules.red_flags:
        rf_id = rf_alert.get("id", "")
        add_node(
            rf_id,
            "EmergencyCondition",
            {
                "name": rf_alert.get("condition", rf_id),
                "severity": rf_alert.get("severity", "CRITICAL"),
                "recommended_action": rf_alert.get("recommended_action", ""),
            },
        )
        
        # Link symptoms connected to the Red Flag
        connected_symptom_list = (rf_alert.get("required_symptoms", []) or []) + (rf_alert.get("any_of_symptoms", []) or [])
        for sym in connected_symptom_list:
            add_node(sym, "Symptom", {"name": sym})
            add_relationship(sym, rf_id, "ASSOCIATED_WITH", {})

        # If condition matches a disease, link them
        matched_disease = rf_alert.get("condition", "")
        for dis in rules.diseases:
            if dis.get("name") == matched_disease or dis.get("id") == rf_alert.get("id", "").replace("_red_flag", ""):
                add_relationship(dis.get("id"), rf_id, "ASSOCIATED_WITH", {})

    # 6. Contraindications
    for contra in rules.contraindications:
        c_id = contra.get("id", "")
        drug = contra.get("drug_or_procedure", "")
        add_node(c_id, "Contraindication", {"name": c_id, "reason": contra.get("reason", "")})
        add_node(drug, "Medication", {"name": drug})
        
        # Link from Contraindication to Medication
        add_relationship(c_id, drug, "ASSOCIATED_WITH", {})

        # If a disease matches the condition, create a direct contraindicated relationship
        reason = contra.get("reason", "").lower()
        for dis in rules.diseases:
            dis_id = dis.get("id", "")
            dis_name = dis.get("name", "").lower()
            if dis_id in reason or dis_name in reason:
                add_relationship(
                    dis_id,
                    drug,
                    "CONTRAINDICATED_WITH",
                    {
                        "severity": contra.get("severity", "ABSOLUTE"),
                        "reason": contra.get("reason", ""),
                    },
                )
                # Ensure the Disease node is connected to the Contraindication
                add_relationship(dis_id, c_id, "ASSOCIATED_WITH", {})

    # 7. Drug Interactions
    for di in rules.drug_interactions:
        di_id = di.get("id", "")
        drug_a = di.get("drug_a", "")
        drug_b = di.get("drug_b", "")
        
        add_node(di_id, "RedFlag", {"name": di_id, "description": di.get("description", "")})
        add_node(drug_a, "Medication", {"name": drug_a})
        add_node(drug_b, "Medication", {"name": drug_b})
        
        add_relationship(di_id, drug_a, "ASSOCIATED_WITH", {})
        add_relationship(di_id, drug_b, "ASSOCIATED_WITH", {})
        
        add_relationship(
            drug_a,
            drug_b,
            "INTERACTS_WITH",
            {
                "severity": di.get("severity", "MAJOR"),
                "description": di.get("description", ""),
            },
        )

    return list(nodes.values()), relationships
