from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
from clinical_intelligence.rule_engine.models import ClinicalScoreResult

class ClinicalScoreEvaluator:
    name: str

    def calculate(
        self,
        *,
        state_data: Dict[str, Any],
        vitals: Dict[str, Optional[float]],
        present_symptoms: Set[str],
        negated_symptoms: Set[str],
        medical_history: List[str],
        current_medications: List[str],
        allergies: List[str],
        age: Optional[int],
        gender: Optional[str]
    ) -> Optional[ClinicalScoreResult]:
        raise NotImplementedError


# Import evaluators lazily to prevent circular imports
def evaluate_all(
    *,
    state_data: Dict[str, Any],
    vitals: Dict[str, Optional[float]],
    present_symptoms: Set[str],
    negated_symptoms: Set[str],
    medical_history: List[str],
    current_medications: List[str],
    allergies: List[str],
    age: Optional[int],
    gender: Optional[str]
) -> List[ClinicalScoreResult]:
    from clinical_intelligence.rule_engine.scores.news2 import NEWS2Evaluator
    from clinical_intelligence.rule_engine.scores.qsofa import qSOFAEvaluator
    from clinical_intelligence.rule_engine.scores.curb65 import CURB65Evaluator
    from clinical_intelligence.rule_engine.scores.heart import HEARTEvaluator
    from clinical_intelligence.rule_engine.scores.wells import WellsEvaluator
    from clinical_intelligence.rule_engine.scores.chads2vasc import CHA2DS2VAScEvaluator
    from clinical_intelligence.rule_engine.scores.timi import TIMIEvaluator

    evaluators: List[ClinicalScoreEvaluator] = [
        NEWS2Evaluator(),
        qSOFAEvaluator(),
        CURB65Evaluator(),
        HEARTEvaluator(),
        WellsEvaluator(),
        CHA2DS2VAScEvaluator(),
        TIMIEvaluator()
    ]

    results: List[ClinicalScoreResult] = []
    for evaluator in evaluators:
        try:
            res = evaluator.calculate(
                state_data=state_data,
                vitals=vitals,
                present_symptoms=present_symptoms,
                negated_symptoms=negated_symptoms,
                medical_history=medical_history,
                current_medications=current_medications,
                allergies=allergies,
                age=age,
                gender=gender
            )
            if res is not None:
                results.append(res)
        except Exception:
            # Squelch and log or skip to prevent crashing engine if a score has bad data formats
            pass

    return results
