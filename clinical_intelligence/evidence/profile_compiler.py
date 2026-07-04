from __future__ import annotations

"""
profile_compiler.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import hashlib
import time
from typing import Any, Dict, List
import yaml
from pathlib import Path

from clinical_intelligence.evidence.evidence_models import DiseaseProfile
from clinical_intelligence.evidence.disease_profile_registry import DiseaseProfileRegistry

class ProfileCompiler:
    """
    Compiles diseases.yaml rules into validated, versioned and hashed DiseaseProfile objects.
    Constructs inverse lookup indexes for candidate retrieval.
    """

    @staticmethod
    def compile_file(yaml_path: Path) -> DiseaseProfileRegistry:
        if not yaml_path.exists():
            raise FileNotFoundError(f"Disease rule file not found at {yaml_path}")

        import re
        symptoms_path = yaml_path.parent / "symptoms.yaml"
        symptom_data = []
        if symptoms_path.exists():
            with open(symptoms_path, "r", encoding="utf-8") as sf:
                symptom_data = yaml.safe_load(sf) or []

        alias_map: Dict[str, str] = {}
        compiled_regexes: Dict[str, Any] = {}
        for entry in symptom_data:
            canonical = entry.get("canonical", "").lower().strip()
            if not canonical:
                continue
            alias_map[canonical] = canonical
            compiled_regexes[canonical] = re.compile(r'\b' + re.escape(canonical) + r'\b')
            for alias in entry.get("aliases", []):
                alias_clean = alias.lower().strip()
                alias_map[alias_clean] = canonical
                compiled_regexes[alias_clean] = re.compile(r'\b' + re.escape(alias_clean) + r'\b')

        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
            data = yaml.safe_load(content) or []

        source_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        compiled_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        profiles: Dict[str, DiseaseProfile] = {}
        symptom_index: Dict[str, List[str]] = {}
        system_index: Dict[str, List[str]] = {}
        history_index: Dict[str, List[str]] = {}
        risk_index: Dict[str, List[str]] = {}

        for entry in data:
            if not isinstance(entry, dict):
                continue
            dis_id = entry.get("id", "unknown_disease")
            dis_name = entry.get("name", dis_id)
            dis_category = entry.get("category", "unknown")

            # 1. Normalize Symptoms (convert string lists to dictionaries)
            cardinal_normalized = []
            for s in entry.get("cardinal_symptoms", []):
                if isinstance(s, dict):
                    cardinal_normalized.append(s)
                else:
                    cardinal_normalized.append({"name": str(s), "weight": 0.25})

            supportive_normalized = []
            for s in entry.get("supportive_symptoms", []):
                if isinstance(s, dict):
                    supportive_normalized.append(s)
                else:
                    supportive_normalized.append({"name": str(s), "weight": 0.20})

            # Exclude, exam, vitals, tags, risk factors
            exclusionary = [str(x) for x in entry.get("exclusionary_symptoms", [])]
            examination = [str(x) for x in entry.get("examination", [])]
            vitals = entry.get("vital_sign_criteria", []) or []
            
            # Guidelines tags
            guideline_tags = []
            gl_def = entry.get("guideline", {})
            if gl_def:
                guideline_tags.append(gl_def.get("organization", "General"))
                guideline_tags.append(gl_def.get("guideline", "General"))
            guideline_tags.extend(entry.get("guideline_tags", []) or [])

            # Modifiers
            risk_modifiers = entry.get("risk_factor_modifiers", {}) or {}
            age_range = entry.get("age_range")
            gender_modifier = entry.get("gender_modifier")

            profile = DiseaseProfile(
                id=dis_id,
                name=dis_name,
                category=dis_category,
                profile_version="1.0.0",
                compiled_at=compiled_at,
                source_hash=source_hash,
                cardinal_symptoms=cardinal_normalized,
                supportive_symptoms=supportive_normalized,
                exclusionary_symptoms=exclusionary,
                examination=examination,
                vitals=vitals,
                risk_factor_modifiers=risk_modifiers,
                age_range=age_range,
                gender_modifier=gender_modifier,
                guideline_tags=guideline_tags
            )
            profiles[dis_id] = profile

            # 2. Build Inverse Indexes for rapid lookups
            # Symptom Index
            for s in cardinal_normalized + supportive_normalized:
                s_name = s["name"].lower().strip()
                symptom_index.setdefault(s_name, []).append(dis_id)
            
            # System (Category) Index
            cat_clean = dis_category.lower().strip()
            system_index.setdefault(cat_clean, []).append(dis_id)

            # History (Risk factors) Index
            for rf_id in risk_modifiers.keys():
                rf_clean = rf_id.lower().strip()
                risk_index.setdefault(rf_clean, []).append(dis_id)

        return DiseaseProfileRegistry(
            profiles=profiles,
            symptom_index=symptom_index,
            system_index=system_index,
            risk_index=risk_index,
            source_hash=source_hash,
            alias_map=alias_map,
            compiled_regexes=compiled_regexes
        )
