from __future__ import annotations

"""
version_tracker.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import hashlib
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class StateVersionTracker:
    """Computes deterministic hashes of clinical inputs to verify re-execution requirements."""

    def __init__(self, tracker_dict: Dict[str, str], session: Any) -> None:
        self.tracker = tracker_dict
        self.session = session

    def _get_hash(self, data: Any) -> str:
        """Compute MD5 hash of any serializable object."""
        try:
            serialized = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(serialized.encode("utf-8")).hexdigest()
        except Exception as e:
            logger.warning(f"StateVersionTracker: Hash serialization failed: {e}")
            return "unknown-hash"

    def update_hashes(
        self,
        current_clinical_state: Dict[str, Any],
        current_guidelines: List[Any],
        current_graph: List[Any]
    ) -> Dict[str, bool]:
        """
        Compare current input states with previous hashes.
        Returns a dict of bool changes indicating what needs updating.
        """
        changes = {}
        
        keys_to_track = {
            "symptoms": current_clinical_state.get("symptoms", []),
            "medications": current_clinical_state.get("current_medications", []),
            "allergies": current_clinical_state.get("allergies", []),
            "history": current_clinical_state.get("medical_history", []),
            "investigations": current_clinical_state.get("investigations", []),
            "risk_factors": current_clinical_state.get("risk_factors", []),
            "graph": current_graph,
            "guidelines": current_guidelines,
            "stage": self.session.consultation_stage,
        }

        for key, value in keys_to_track.items():
            current_hash = self._get_hash(value)
            prev_hash = self.tracker.get(key)
            if prev_hash != current_hash:
                changes[key] = True
                self.tracker[key] = current_hash
            else:
                changes[key] = False

        return changes

    def needs_reasoning(self, changes: Dict[str, bool]) -> bool:
        """Rule engine needs to run if clinical facts (symptoms, meds, history, etc.) changed."""
        return (
            changes.get("symptoms", False)
            or changes.get("medications", False)
            or changes.get("allergies", False)
            or changes.get("history", False)
            or changes.get("investigations", False)
            or changes.get("risk_factors", False)
        )

    def needs_graph(self, changes: Dict[str, bool]) -> bool:
        """Knowledge Graph needs to query if symptoms changed."""
        return changes.get("symptoms", False)

    def needs_retrieval(self, changes: Dict[str, bool]) -> bool:
        """Guidelines retriever needs to search if symptoms or risk factors changed."""
        return (
            changes.get("symptoms", False)
            or changes.get("risk_factors", False)
            or changes.get("history", False)
        )

    def needs_llm(self, changes: Dict[str, bool]) -> bool:
        """LLM needs to run if clinical facts, graph, guidelines, or workflow stage changed."""
        return any(changes.values())
