from __future__ import annotations

"""
evidence_context.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from clinical_intelligence.evidence.evidence_models import DiseaseProfile

logger = logging.getLogger(__name__)

class EvidenceContext(BaseModel):
    """
    EvidenceContext binds the ClinicalState, compiled disease profiles,
    Knowledge Graph pre-fetched queries, and a transient caching layer.
    """
    state: Dict[str, Any]
    disease_profiles: Dict[str, DiseaseProfile] = Field(default_factory=dict)
    graph_prefetch: Dict[str, Any] = Field(default_factory=dict)
    guideline_prefetch: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)
    alias_map: Dict[str, str] = Field(default_factory=dict)
    compiled_regexes: Dict[str, Any] = Field(default_factory=dict)

    def get_cached_result(self, provider_name: str, disease_id: str) -> Optional[Any]:
        """Fetch cached provider result if available for this state version."""
        state_version = self.state.get("version", 0)
        cache_key = f"{provider_name}:{disease_id}:{state_version}"
        return self.cache.get(cache_key)

    def set_cached_result(self, provider_name: str, disease_id: str, result: Any) -> None:
        """Cache a provider result stamped with the current state version."""
        state_version = self.state.get("version", 0)
        cache_key = f"{provider_name}:{disease_id}:{state_version}"
        self.cache[cache_key] = result
