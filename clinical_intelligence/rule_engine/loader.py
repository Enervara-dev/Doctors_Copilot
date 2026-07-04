from __future__ import annotations

"""
Hot-Reloadable Ruleset Loader

Purpose:
Parses and loads YAML rule definitions, caching them in memory and hot-reloading file changes automatically.

Module/Component: Rule Engine Core
"""

"""
Hot-Reloadable Ruleset Loader

Purpose:
Parses and loads YAML rule definitions, caching them in memory and hot-reloading file changes automatically.

Module/Component: Rule Engine Core
"""

"""
Rule Loader — YAML rule file loading with hot-reload support.

Loads all rule categories from the rules/ directory. Checks file mtimes on
every `get()` call and silently reloads changed files. Thread-safe via RLock.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from clinical_intelligence.rule_engine.exceptions import RuleLoadError
from clinical_intelligence.rule_engine.validator import validate_rules

logger = logging.getLogger(__name__)

# Locate the rules directory relative to this file.
_RULES_DIR = Path(__file__).parent.parent / "rules"

_RULE_FILES = [
    "diseases.yaml",
    "symptoms.yaml",
    "investigations.yaml",
    "red_flags.yaml",
    "risk_factors.yaml",
    "contraindications.yaml",
    "drug_interactions.yaml",
]


class RuleSet:
    """Container for all loaded rule categories."""

    def __init__(
        self,
        diseases: List[Dict[str, Any]],
        symptoms: List[Dict[str, Any]],
        investigations: List[Dict[str, Any]],
        red_flags: List[Dict[str, Any]],
        risk_factors: List[Dict[str, Any]],
        contraindications: List[Dict[str, Any]],
        drug_interactions: List[Dict[str, Any]],
    ) -> None:
        self.diseases = diseases
        self.symptoms = symptoms
        self.investigations = investigations
        self.red_flags = red_flags
        self.risk_factors = risk_factors
        self.contraindications = contraindications
        self.drug_interactions = drug_interactions

        # Pre-compile regular expressions and cache alias lookups and weights
        self.compiled_aliases: Dict[str, str] = {}
        self.compiled_regexes: Dict[str, Any] = {}
        self.cached_weights: Dict[str, Dict[str, float]] = {}  # disease_id -> symptom -> weight
        self._compile_caches()

    def _compile_caches(self) -> None:
        import re
        # 1. Compile symptoms aliases
        for entry in self.symptoms:
            canonical = entry.get("canonical", "").lower().strip()
            if not canonical:
                continue
            self.compiled_aliases[canonical] = canonical
            self.compiled_regexes[canonical] = re.compile(r'\b' + re.escape(canonical) + r'\b')
            for alias in entry.get("aliases", []):
                alias_clean = alias.lower().strip()
                self.compiled_aliases[alias_clean] = canonical
                self.compiled_regexes[alias_clean] = re.compile(r'\b' + re.escape(alias_clean) + r'\b')
        
        # 2. Cache weights
        for dis in self.diseases:
            dis_id = dis.get("id", "")
            self.cached_weights[dis_id] = {}
            for s in dis.get("cardinal_symptoms", []) + dis.get("supportive_symptoms", []):
                name = s.get("name", "").lower().strip()
                weight = float(s.get("weight", 0.0))
                self.cached_weights[dis_id][name] = weight

    def version_summary(self) -> Dict[str, Any]:
        """Return a lightweight dict describing rule counts."""
        return {
            "diseases": len(self.diseases),
            "symptoms": len(self.symptoms),
            "investigations": len(self.investigations),
            "red_flags": len(self.red_flags),
            "risk_factors": len(self.risk_factors),
            "contraindications": len(self.contraindications),
            "drug_interactions": len(self.drug_interactions),
        }


class RuleLoader:
    """
    Thread-safe YAML rule loader with automatic hot-reload.

    Rules are loaded once and cached. On each `get()` call the loader checks
    file modification times and reloads any files that have changed since the
    last successful load.

    Usage::

        loader = RuleLoader()
        rules = loader.get()   # always returns the freshest rule set
    """

    def __init__(self, rules_dir: Optional[Path] = None) -> None:
        self._rules_dir: Path = rules_dir or _RULES_DIR
        self._lock = threading.RLock()
        self._rule_set: Optional[RuleSet] = None
        # path → last known mtime
        self._mtimes: Dict[str, float] = {}
        self._load_all()

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def get(self) -> RuleSet:
        """Return the current rule set, reloading from disk if any file has changed."""
        with self._lock:
            if self._needs_reload():
                logger.info("RuleLoader: Rule file change detected — reloading rules.")
                self._load_all()
            return self._rule_set  # type: ignore[return-value]

    def loaded_files(self) -> List[str]:
        """Return the list of YAML files that were successfully loaded."""
        return list(self._mtimes.keys())

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _needs_reload(self) -> bool:
        """Return True if any rule file mtime has changed since last load."""
        for filename in _RULE_FILES:
            path = self._rules_dir / filename
            try:
                current_mtime = path.stat().st_mtime
            except OSError:
                continue
            if self._mtimes.get(str(path)) != current_mtime:
                return True
        return False

    def _load_all(self) -> None:
        """Load all YAML rule files into the in-memory rule set."""
        loaded: Dict[str, List[Dict[str, Any]]] = {}
        new_mtimes: Dict[str, float] = {}

        for filename in _RULE_FILES:
            path = self._rules_dir / filename
            data = self._load_file(path)
            key = filename.replace(".yaml", "")
            loaded[key] = data
            try:
                new_mtimes[str(path)] = path.stat().st_mtime
            except OSError:
                pass

        rule_set = RuleSet(
            diseases=loaded.get("diseases", []),
            symptoms=loaded.get("symptoms", []),
            investigations=loaded.get("investigations", []),
            red_flags=loaded.get("red_flags", []),
            risk_factors=loaded.get("risk_factors", []),
            contraindications=loaded.get("contraindications", []),
            drug_interactions=loaded.get("drug_interactions", []),
        )

        # Fail fast: Validate every loaded rule.
        validate_rules(rule_set)

        self._rule_set = rule_set
        self._mtimes = new_mtimes
        disease_names = [d.get("name", d.get("id")) for d in loaded.get("diseases", [])]
        logger.info(f"RuleLoader: Loaded diseases: {', '.join(disease_names)}")
        logger.info(
            "RuleLoader: Rules loaded successfully.",
            extra=self._rule_set.version_summary(),
        )

    @staticmethod
    def _load_file(path: Path) -> List[Dict[str, Any]]:
        """Parse a single YAML rule file, returning its list of entries."""
        if not path.exists():
            raise RuleLoadError(str(path), "file not found")
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise RuleLoadError(str(path), f"YAML parse error: {exc}") from exc

        if data is None:
            logger.warning("RuleLoader: %s is empty — returning empty list.", path.name)
            return []
        if not isinstance(data, list):
            raise RuleLoadError(
                str(path), f"expected a YAML list at top level, got {type(data).__name__}"
            )

        # Expose rule_id, rule_version, created_date, and updated_date on every rule entry.
        for entry in data:
            if isinstance(entry, dict):
                # Ensure rule_id is present
                if "id" in entry:
                    entry.setdefault("rule_id", entry["id"])
                elif "canonical" in entry:
                    entry.setdefault("rule_id", entry["canonical"].lower().replace(" ", "_"))
                else:
                    entry.setdefault("rule_id", "unknown_rule_id")
                
                entry.setdefault("rule_version", entry.get("version", "1.0.0"))
                entry.setdefault("created_date", "2026-01-01")
                entry.setdefault("updated_date", "2026-07-03")

        return data


# Module-level singleton — shared across the engine components.
_loader: Optional[RuleLoader] = None
_loader_lock = threading.Lock()


def get_loader(rules_dir: Optional[Path] = None) -> RuleLoader:
    """Return the module-level singleton RuleLoader, creating it on first call."""
    global _loader
    if _loader is None:
        with _loader_lock:
            if _loader is None:
                _loader = RuleLoader(rules_dir=rules_dir)
    return _loader
