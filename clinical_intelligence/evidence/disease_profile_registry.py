from __future__ import annotations

"""
disease_profile_registry.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

from typing import Any, Dict, List, Optional
from clinical_intelligence.evidence.evidence_models import DiseaseProfile

class DiseaseProfileRegistry:
    """
    Registry caching DiseaseProfiles and their inverse lookup indexes.
    All engine reasoning lookups fetch compiled data from here.
    """

    def __init__(
        self,
        profiles: Dict[str, DiseaseProfile],
        symptom_index: Dict[str, List[str]],
        system_index: Dict[str, List[str]],
        risk_index: Dict[str, List[str]],
        source_hash: str,
        alias_map: Dict[str, str] = None,
        compiled_regexes: Dict[str, Any] = None
    ) -> None:
        self.profiles = profiles
        self.symptom_index = symptom_index
        self.system_index = system_index
        self.risk_index = risk_index
        self.source_hash = source_hash
        self.alias_map = alias_map or {}
        self.compiled_regexes = compiled_regexes or {}

    def get_profile(self, disease_id: str) -> Optional[DiseaseProfile]:
        """Fetch a specific disease profile by ID."""
        return self.profiles.get(disease_id)

    def get_all_profiles(self) -> List[DiseaseProfile]:
        """Fetch all compiled disease profiles."""
        return list(self.profiles.values())

    def get_by_symptom(self, symptom_name: str) -> List[str]:
        """Lookup disease IDs that contain this symptom."""
        return self.symptom_index.get(symptom_name.lower().strip(), [])

    def get_by_risk_factor(self, risk_factor_id: str) -> List[str]:
        """Lookup disease IDs linked to this risk factor modifier."""
        return self.risk_index.get(risk_factor_id.lower().strip(), [])
