from __future__ import annotations
import logging
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class HallucinationGuard:
    """
    Cross-checks LLM responses against deterministic sources of truth
    (Rule Engine candidates, Knowledge Graph nodes, and RAG guidelines).
    """

    def __init__(
        self,
        rule_diseases: Set[str],
        graph_investigations: Set[str],
        guidelines_tests: Set[str]
    ) -> None:
        self.rule_diseases = {d.lower().strip() for d in rule_diseases}
        self.valid_investigations = {i.lower().strip() for i in graph_investigations}.union(
            {t.lower().strip() for t in guidelines_tests}
        )

    def verify_diagnoses(self, diagnoses: List[str]) -> Tuple[bool, List[str]]:
        """
        Verify that every generated diagnosis exists in the Rule Engine list.
        Returns:
            (is_clean, verified_diagnoses_list)
        """
        verified = []
        for d in diagnoses:
            d_clean = d.lower().strip()
            if d_clean in self.rule_diseases or any(rd in d_clean or d_clean in rd for rd in self.rule_diseases):
                # Find matching primary rule ID
                matched = next((rd for rd in self.rule_diseases if rd in d_clean or d_clean in rd), d)
                verified.append(matched)
            else:
                logger.warning(f"HallucinationGuard: Discarded unsupported diagnosis '{d}'")
                
        return len(verified) == len(diagnoses), verified

    def verify_investigations(self, investigations: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Verify that every recommended test exists in the Guidelines or Knowledge Graph.
        """
        verified = []
        for inv in investigations:
            name = inv.get("name", "").strip()
            name_clean = name.lower().strip()
            
            # Simple substring matching for investigations validation
            matched = False
            for val_inv in self.valid_investigations:
                if val_inv in name_clean or name_clean in val_inv:
                    matched = True
                    break
            
            if matched or not self.valid_investigations:
                verified.append(inv)
            else:
                logger.warning(f"HallucinationGuard: Discarded unsupported investigation recommendation '{name}'")
                
        return len(verified) == len(investigations), verified
