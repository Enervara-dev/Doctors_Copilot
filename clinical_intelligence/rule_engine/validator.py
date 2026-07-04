from __future__ import annotations

"""
validator.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

from typing import Any, Dict, List, Set
from clinical_intelligence.rule_engine.exceptions import RuleValidationError

class RuleSet:
    diseases: List[Dict[str, Any]]
    symptoms: List[Dict[str, Any]]
    investigations: List[Dict[str, Any]]
    red_flags: List[Dict[str, Any]]
    risk_factors: List[Dict[str, Any]]
    contraindications: List[Dict[str, Any]]
    drug_interactions: List[Dict[str, Any]]


def validate_rules(rules: Any) -> None:
    """
    Perform deep validation on loaded RuleSet.
    Fails fast by raising RuleValidationError if rules are invalid.
    """
    symptom_canonicals: Set[str] = set()
    disease_ids: Set[str] = set()
    red_flag_ids: Set[str] = set()
    risk_factor_ids: Set[str] = set()
    contraindication_ids: Set[str] = set()
    drug_interaction_ids: Set[str] = set()
    investigation_ids: Set[str] = set()

    # ── 1. Symptoms Validation ──────────────────────────────────
    for idx, sym in enumerate(rules.symptoms):
        canonical = sym.get("canonical")
        if not canonical:
            raise RuleValidationError(f"symptom_idx_{idx}", "canonical", "Missing canonical symptom name")
        canonical_clean = canonical.lower().strip()
        if canonical_clean in symptom_canonicals:
            raise RuleValidationError(canonical, "canonical", f"Duplicate canonical symptom: '{canonical}'")
        symptom_canonicals.add(canonical_clean)
        
        aliases = sym.get("aliases", [])
        if not isinstance(aliases, list):
            raise RuleValidationError(canonical, "aliases", "Aliases must be a list")

    # ── 2. Risk Factors Validation ──────────────────────────────
    for idx, rf in enumerate(rules.risk_factors):
        rf_id = rf.get("id")
        if not rf_id:
            raise RuleValidationError(f"risk_factor_idx_{idx}", "id", "Missing risk factor ID")
        if rf_id in risk_factor_ids:
            raise RuleValidationError(rf_id, "id", f"Duplicate risk factor ID: '{rf_id}'")
        risk_factor_ids.add(rf_id)
        if not rf.get("name"):
            raise RuleValidationError(rf_id, "name", "Missing risk factor name")
        if not rf.get("category"):
            raise RuleValidationError(rf_id, "category", "Missing risk factor category")

    # ── 3. Diseases Validation ──────────────────────────────────
    for idx, dis in enumerate(rules.diseases):
        dis_id = dis.get("id")
        if not dis_id:
            raise RuleValidationError(f"disease_idx_{idx}", "id", "Missing disease ID")
        if dis_id in disease_ids:
            raise RuleValidationError(dis_id, "id", f"Duplicate disease ID: '{dis_id}'")
        disease_ids.add(dis_id)

        if not dis.get("name"):
            raise RuleValidationError(dis_id, "name", "Missing disease name")
        if not dis.get("category"):
            raise RuleValidationError(dis_id, "category", "Missing disease category")

        # Validate cardinal symptoms
        cardinals = dis.get("cardinal_symptoms", [])
        if not isinstance(cardinals, list):
            raise RuleValidationError(dis_id, "cardinal_symptoms", "Must be a list")
        for sym_entry in cardinals:
            sym_name = sym_entry.get("name")
            if not sym_name or not isinstance(sym_name, str) or not sym_name.strip():
                raise RuleValidationError(dis_id, "cardinal_symptoms.name", "Missing or invalid symptom name")
            weight = sym_entry.get("weight")
            if weight is None:
                raise RuleValidationError(dis_id, "cardinal_symptoms.weight", "Missing symptom weight")
            try:
                if float(weight) <= 0.0:
                    raise RuleValidationError(dis_id, "cardinal_symptoms.weight", f"Weight must be positive: {weight}")
            except (ValueError, TypeError):
                raise RuleValidationError(dis_id, "cardinal_symptoms.weight", f"Invalid weight value: {weight}")

        # Validate supportive symptoms
        supportives = dis.get("supportive_symptoms", [])
        if not isinstance(supportives, list):
            raise RuleValidationError(dis_id, "supportive_symptoms", "Must be a list")
        for sym_entry in supportives:
            sym_name = sym_entry.get("name")
            if not sym_name or not isinstance(sym_name, str) or not sym_name.strip():
                raise RuleValidationError(dis_id, "supportive_symptoms.name", "Missing or invalid symptom name")
            weight = sym_entry.get("weight")
            if weight is None:
                raise RuleValidationError(dis_id, "supportive_symptoms.weight", "Missing symptom weight")
            try:
                if float(weight) <= 0.0:
                    raise RuleValidationError(dis_id, "supportive_symptoms.weight", f"Weight must be positive: {weight}")
            except (ValueError, TypeError):
                raise RuleValidationError(dis_id, "supportive_symptoms.weight", f"Invalid weight value: {weight}")

        # Validate age_range
        age_range = dis.get("age_range")
        if age_range:
            if not isinstance(age_range, dict):
                raise RuleValidationError(dis_id, "age_range", "Must be a dictionary")
            if "modifier" not in age_range:
                raise RuleValidationError(dis_id, "age_range.modifier", "Missing age modifier")

        # Validate gender_modifier
        gender_mod = dis.get("gender_modifier")
        if gender_mod:
            if not isinstance(gender_mod, dict):
                raise RuleValidationError(dis_id, "gender_modifier", "Must be a dictionary")
            if "gender" not in gender_mod or "modifier" not in gender_mod:
                raise RuleValidationError(dis_id, "gender_modifier", "Missing gender or modifier")

        # Validate risk factor modifiers
        rf_mods = dis.get("risk_factor_modifiers", {})
        if not isinstance(rf_mods, dict):
            raise RuleValidationError(dis_id, "risk_factor_modifiers", "Must be a dictionary")
        for rf_mod_id in rf_mods.keys():
            if not rf_mod_id or not isinstance(rf_mod_id, str) or not rf_mod_id.strip():
                raise RuleValidationError(
                    dis_id,
                    "risk_factor_modifiers",
                    "Risk factor ID must be a non-empty string"
                )

    # ── 4. Red Flags Validation ─────────────────────────────────
    for idx, rf in enumerate(rules.red_flags):
        rf_id = rf.get("id")
        if not rf_id:
            raise RuleValidationError(f"red_flag_idx_{idx}", "id", "Missing red flag ID")
        if rf_id in red_flag_ids:
            raise RuleValidationError(rf_id, "id", f"Duplicate red flag ID: '{rf_id}'")
        red_flag_ids.add(rf_id)

        for key in ["condition", "severity", "recommended_action"]:
            if not rf.get(key):
                raise RuleValidationError(rf_id, key, f"Missing {key}")

        # Symptoms checks
        for sym in rf.get("required_symptoms", []):
            if not sym or not isinstance(sym, str) or not sym.strip():
                raise RuleValidationError(rf_id, "required_symptoms", "Symptom name must be a non-empty string")

        for sym in rf.get("any_of_symptoms", []):
            if not sym or not isinstance(sym, str) or not sym.strip():
                raise RuleValidationError(rf_id, "any_of_symptoms", "Symptom name must be a non-empty string")

        # Risk factors checks
        for rf_crit in rf.get("risk_factor_criteria", []):
            if not rf_crit or not isinstance(rf_crit, str) or not rf_crit.strip():
                raise RuleValidationError(rf_id, "risk_factor_criteria", "Risk factor criteria must be non-empty string")

    # ── 5. Contraindications Validation ──────────────────────────
    for idx, ci in enumerate(rules.contraindications):
        ci_id = ci.get("id")
        if not ci_id:
            raise RuleValidationError(f"contraindication_idx_{idx}", "id", "Missing contraindication ID")
        if ci_id in contraindication_ids:
            raise RuleValidationError(ci_id, "id", f"Duplicate contraindication ID: '{ci_id}'")
        contraindication_ids.add(ci_id)

        for key in ["drug_or_procedure", "reason", "severity"]:
            if not ci.get(key):
                raise RuleValidationError(ci_id, key, f"Missing {key}")

    # ── 6. Drug Interactions Validation ──────────────────────────
    for idx, di in enumerate(rules.drug_interactions):
        di_id = di.get("id")
        if not di_id:
            raise RuleValidationError(f"drug_interaction_idx_{idx}", "id", "Missing drug interaction ID")
        if di_id in drug_interaction_ids:
            raise RuleValidationError(di_id, "id", f"Duplicate drug interaction ID: '{di_id}'")
        drug_interaction_ids.add(di_id)

        for key in ["drug_a", "drug_b", "severity"]:
            if not di.get(key):
                raise RuleValidationError(di_id, key, f"Missing {key}")

    # ── 7. Investigations Validation ─────────────────────────────
    for idx, inv in enumerate(rules.investigations):
        inv_id = inv.get("id")
        if not inv_id:
            raise RuleValidationError(f"investigation_idx_{idx}", "id", "Missing investigation ID")
        if inv_id in investigation_ids:
            raise RuleValidationError(inv_id, "id", f"Duplicate investigation ID: '{inv_id}'")
        investigation_ids.add(inv_id)

        for key in ["name", "priority"]:
            if not inv.get(key):
                raise RuleValidationError(inv_id, key, f"Missing {key}")

        for cond in inv.get("for_conditions", []):
            if cond not in disease_ids:
                # Warning or validate. Let's make it a strict validation error to satisfy "invalid diagnoses"
                raise RuleValidationError(inv_id, "for_conditions", f"Disease '{cond}' not defined in diseases.yaml")

    # ── 8. Dependency / Circular References Validation ───────────
    # We build a dependency graph of rules. A rule X depends on Y if X references Y.
    # We verify that there are no circular dependencies.
    adj: Dict[str, List[str]] = {}
    for dis in rules.diseases:
        dis_id = dis["id"]
        adj[dis_id] = list(dis.get("risk_factor_modifiers", {}).keys())

    for rf_id in risk_factor_ids:
        # Check if risk factor references diseases or other risk factors
        if rf_id not in adj:
            adj[rf_id] = []

    # Cycle detection using DFS
    visited: Dict[str, int] = {} # 0=unvisited, 1=visiting, 2=visited
    for node in adj:
        visited[node] = 0

    def dfs(u: str) -> bool:
        visited[u] = 1
        for v in adj.get(u, []):
            if v not in visited:
                continue
            if visited[v] == 1:
                return True # Cycle detected
            if visited[v] == 0:
                if dfs(v):
                    return True
        visited[u] = 2
        return False

    for node in adj:
        if visited[node] == 0:
            if dfs(node):
                raise RuleValidationError(node, "circular_reference", f"Circular dependency cycle detected starting at '{node}'")
