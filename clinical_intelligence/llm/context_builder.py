from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple

from clinical_intelligence.config.runtime import MAX_CONTEXT_TOKENS

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Constructs minimal, compressed, and token-budgeted prompt contexts.
    Infers consultation stages and progress percentages automatically.
    """

    @staticmethod
    def infer_stage_and_progress(
        symptoms: List[Dict[str, Any]],
        history: List[str],
        medications: List[str],
        candidates: List[Any],
        has_vitals: bool,
        is_complete: bool = False
    ) -> Tuple[str, int]:
        """
        Infer consultation stage and progress score dynamically.
        Returns:
            (stage_name, progress_percent)
        """
        if is_complete:
            return "Consultation Complete", 100

        active_syms = [s for s in symptoms if s.get("status") == "Active"]

        if not active_syms:
            return "Greeting", 5
        if len(active_syms) <= 1 and not history and not medications:
            return "Chief Complaint", 20
        if active_syms and not history:
            return "History of Present Illness", 40
        if history and not medications:
            return "Past Medical History", 55
        if history and medications and not has_vitals and not candidates:
            return "Medication Review", 65
        if has_vitals and not candidates:
            return "Examination", 75
        if candidates and not is_complete:
            return "Assessment", 85
        return "Treatment Planning", 95

    @classmethod
    def build_minimal_context(
        self,
        state: Dict[str, Any],
        candidates: List[Any],
        red_flags: List[Any],
        risk_factors: List[Any],
        medications: List[str],
        missing_info: List[Any],
        guidelines: List[Any],
        graph_paths: List[Any],
        is_complete: bool = False
    ) -> Tuple[str, str, int]:
        """
        Constructs a prioritized, compressed context string respecting context budgets.
        Returns:
            (context_string, inferred_stage, progress_percent)
        """
        # 1. Infer Stage & Progress
        symptoms = state.get("symptoms", [])
        history = state.get("medical_history", [])
        has_vitals = bool(state.get("vital_signs"))
        
        stage, progress = self.infer_stage_and_progress(
            symptoms, history, medications, candidates, has_vitals, is_complete
        )

        # 2. Build context categories with strict prioritization (Active/Red Flags first)
        lines = []

        # Category 1: Red Flags (High Priority)
        if red_flags:
            rf_strs = [f"- {rf.condition} (Severity: {rf.severity})" for rf in red_flags]
            lines.append("Active Red Flags:\n" + "\n".join(rf_strs))

        # Category 2: Top Candidates
        if candidates:
            cand_strs = [f"- {c.name} (Conf: {c.confidence:.2f})" for c in candidates[:3]]
            lines.append("Top Candidates:\n" + "\n".join(cand_strs))

        # Category 3: Active Symptoms (Cleaned/Compressed)
        active_sym_names = list(set([s.get("name") for s in symptoms if s.get("status") == "Active" and s.get("name")]))
        if active_sym_names:
            lines.append(f"Active Symptoms: {', '.join(active_sym_names)}")

        # Category 4: Missing Information
        if missing_info:
            mi_strs = [f"- {mi.question} (targets: {', '.join(mi.relevant_conditions)})" for mi in missing_info[:3]]
            lines.append("Missing Evidence Gaps:\n" + "\n".join(mi_strs))

        # Category 5: Risk Factors & Vitals
        vitals = state.get("vital_signs", {})
        vital_strs = [f"{k}={v}" for k, v in vitals.items() if v]
        if vital_strs:
            lines.append(f"Vitals: {', '.join(vital_strs)}")

        if risk_factors:
            rf_names = list(set([rf.name for rf in risk_factors if rf.present]))
            if rf_names:
                lines.append(f"Risk Factors: {', '.join(rf_names)}")

        # Category 6: Current Medications
        if medications:
            lines.append(f"Current Medications: {', '.join(medications)}")

        # Category 7: Retrieved Guidelines & Graph Evidence
        if guidelines:
            gl_strs = [f"- [{gl.get('organization')}] {gl.get('title')}: {gl.get('text_content')[:120]}..." for gl in guidelines[:3]]
            lines.append("Retrieved Guidelines:\n" + "\n".join(gl_strs))

        # 3. Concatenate and budget tokens
        context_str = "\n\n".join(lines)
        
        # Safe character budget fallback approximation (1 token ~= 4 chars)
        char_limit = MAX_CONTEXT_TOKENS * 4
        if len(context_str) > char_limit:
            context_str = context_str[:char_limit] + "\n...[Context Budget Truncated]..."
            logger.warning("Prompt context exceeded budget and was truncated.")

        return context_str, stage, progress
Class = ContextBuilder
