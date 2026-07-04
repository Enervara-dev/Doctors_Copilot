from __future__ import annotations

"""
schema_validator.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class JSONSchemaValidator:
    """Validates if raw output string is valid JSON and conforms to a list of required keys."""

    @staticmethod
    def validate(raw_output: str, required_keys: Optional[List[str]] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Parses JSON and verifies that all required_keys are present.
        Returns:
            (is_valid, parsed_dict_or_none)
        """
        clean_text = raw_output.strip()
        
        # Strip potential markdown formatting wraps (e.g. ```json ... ```)
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()

        try:
            parsed = json.loads(clean_text)
            if not isinstance(parsed, dict):
                logger.warning("Parsed JSON is not a dictionary.")
                return False, None
            
            if required_keys:
                for k in required_keys:
                    if k not in parsed:
                        logger.warning(f"Schema violation: required key '{k}' missing.")
                        return False, parsed
            return True, parsed
        except Exception as e:
            logger.warning(f"JSON validation failed: {e}")
            return False, None
