from __future__ import annotations
import logging
from typing import Any, List, Optional

from clinical_intelligence.workflow.protocols import IQuestionController

logger = logging.getLogger(__name__)


class IntelligentQuestionController(IQuestionController):
    """Prioritizes, filters, and formats adaptive questions deterministically."""

    def __init__(self, max_questions: int = 5) -> None:
        self.max_questions = max_questions

    def get_next_questions(self, session: Any, limit: Optional[int] = None) -> List[Any]:
        """
        Processes and filters raw LLM questions to match priority configurations.
        Ordering:
          1. Emergency clarification
          2. Missing diagnostic evidence
          3. Differential discrimination
          4. Guideline clarification
          5. Optional refinement
        """
        max_q = limit or self.max_questions
        
        raw_questions = session.cached_subsystem_outputs.get("llm_adaptive_questions", [])
        
        memory = session.consultation_memory
        answered_set = set(memory.get("adaptive_questions", []))
        skipped_set = set(session.question_history)
        
        filtered_questions = []
        seen_question_texts = set()

        for q in raw_questions:
            # support both object and dict format
            q_text = getattr(q, "question", "").strip() if hasattr(q, "question") else q.get("question", "").strip()
            if not q_text:
                continue
            if q_text in answered_set or q_text in skipped_set or q_text in seen_question_texts:
                continue
                
            seen_question_texts.add(q_text)
            filtered_questions.append(q)

        # Priority mapping
        def get_question_priority(q: Any) -> int:
            p_upper = getattr(q, "priority", "ROUTINE").upper() if hasattr(q, "priority") else q.get("priority", "ROUTINE").upper()
            purpose_lower = getattr(q, "purpose", "").lower() if hasattr(q, "purpose") else q.get("purpose", "").lower()
            diag_lower = getattr(q, "related_diagnosis", "").lower() if hasattr(q, "related_diagnosis") else q.get("related_diagnosis", "").lower()

            if p_upper == "CRITICAL" or ("emergency" in purpose_lower or "red flag" in purpose_lower or "infarction" in diag_lower or "acs" in diag_lower):
                return 1
            if p_upper == "HIGH":
                return 2
            if p_upper == "MEDIUM":
                return 3
            if "guideline" in purpose_lower or "protocol" in purpose_lower:
                return 4
            return 5

        # Sort deterministically
        def sort_key(q: Any):
            text = getattr(q, "question", "") if hasattr(q, "question") else q.get("question", "")
            return (get_question_priority(q), text)

        filtered_questions.sort(key=sort_key)
        selected_questions = filtered_questions[:max_q]
        
        # Resolve EventBus
        try:
            from clinical_intelligence.core.container import container
            from clinical_intelligence.workflow.protocols import IEventBus
            from clinical_intelligence.events.events import QuestionAsked
            eb = container.resolve(IEventBus)
        except Exception:
            eb = None

        for sq in selected_questions:
            text = getattr(sq, "question", "") if hasattr(sq, "question") else sq.get("question", "")
            purpose = getattr(sq, "purpose", "") if hasattr(sq, "purpose") else sq.get("purpose", "")
            session.question_history.append(text)
            if eb:
                eb.publish(QuestionAsked(session_id=session.session_id, question=text, purpose=purpose))
            
        return selected_questions
