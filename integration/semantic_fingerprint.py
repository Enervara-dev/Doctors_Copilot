"""
What counts as a change in the Copilot's clinical opinion.

Reasoning runs after every settled utterance, but most utterances do not change
what the Copilot concludes. Emitting a full result each time makes a consumer
re-render, re-diff and re-alert for nothing, and it buries the moments that do
matter — a new red flag arriving — in a stream of identical payloads.

Model equality cannot answer this. `ClinicalReasoningResult.reasoning_timestamp`
is `default_factory=datetime.utcnow`, so two identical reasoning passes are
never equal; `explainability.reasoning_duration_ms` is wall-clock; a symptom
carries `updated_at` and the id of the utterance that last touched it. Comparing
the models directly would report a change on every single event.

So the semantic view is built explicitly: the clinical conclusions and the
clinical facts behind them, and nothing else. Two rules govern what goes in.

* **A field is included when a clinician would act differently if it changed.**
  A new candidate, a re-ranking, an escalated investigation priority, a symptom
  appearing or being negated — all clinical. A duration in milliseconds is not.

* **Order is preserved only where order is meaning.** Candidate diagnoses are
  ranked, so their order is compared. Investigations, red flags, risk factors
  and missing-information items are sets with their own priorities, so they are
  sorted by identifier before comparison — the engine emitting them in a
  different order is not a clinical event.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from app.services.clinical.models import ClinicalState
from clinical_intelligence.rule_engine.models import ClinicalReasoningResult

#: Confidence is compared to four decimal places. The engine is deterministic so
#: exact equality would hold today, but a difference of 1e-5 is not a clinical
#: event and should never wake a consumer.
_CONFIDENCE_PRECISION = 4

#: What the patient presented with — the engine's input.
STATE_CATEGORIES = (
    "demographics",
    "presentation",
    "symptoms",
    "history",
    "medications",
    "allergies",
    "vitals",
)

#: What the engine concluded from it.
REASONING_CATEGORIES = (
    "candidate_diagnoses",
    "red_flags",
    "risk_factors",
    "investigations",
    "missing_information",
    "contraindications",
    "drug_interactions",
    "clinical_scores",
)

#: The categories the view is divided into. Exposed so a result can say which
#: ones moved, and so a test can name one without restating the structure.
SEMANTIC_CATEGORIES = STATE_CATEGORIES + REASONING_CATEGORIES


def _symptom_view(symptom: Dict[str, Any]) -> Dict[str, Any]:
    """
    One symptom, reduced to what is clinically true about it.

    `entity_id` is a running counter (`symptom_1`), `updated_at` is a
    timestamp and `utterance_id` records which utterance last touched it.
    None of the three changes what is wrong with the patient.
    """
    return {
        "name": symptom.get("name"),
        "present": str(symptom.get("present")),
        "severity": symptom.get("severity"),
        "duration": symptom.get("duration"),
        "status": symptom.get("status"),
        "onset": symptom.get("onset"),
        "progression": symptom.get("progression"),
        "character": symptom.get("character"),
        "frequency": symptom.get("frequency"),
    }


def clinical_state_view(state: ClinicalState) -> Dict[str, Any]:
    """
    The engine's input, reduced to what is clinically true.

    Split out from the full view because reasoning is a pure function of this:
    verified empirically — five passes over one state produce one semantic
    digest, a freshly constructed engine agrees, and `reason()` does not mutate
    what it is given. The only randomness in the engine is a `uuid4` step id in
    the explainability trace, which no view includes.

    So an unchanged digest here means the conclusion cannot have moved, and the
    reasoning pass can be skipped. The one way that could be wrong is a rule
    YAML being edited mid-consultation — the loader hot-reloads by mtime — which
    does not happen during a live consultation.
    """
    vitals = state.vital_signs.model_dump() if state.vital_signs else {}

    return {
        "demographics": {
            "age": state.patient_info.age,
            "gender": state.patient_info.gender,
        },
        "presentation": {
            "chief_complaint": sorted(state.chief_complaint or []),
            "severity": state.severity,
            "duration": state.duration,
        },
        "symptoms": sorted(
            (_symptom_view(s) for s in state.symptoms),
            key=lambda s: (s["name"] or "", s["present"]),
        ),
        "history": sorted(state.medical_history or []),
        "medications": sorted(state.current_medications or []),
        "allergies": sorted(state.allergies or []),
        # Vitals are clinical input, and several scores key off them.
        "vitals": {k: v for k, v in sorted(vitals.items()) if v is not None},
    }


def reasoning_view(reasoning: ClinicalReasoningResult) -> Dict[str, Any]:
    """What the engine concluded, with nothing time- or trace-dependent."""
    return {
        # Rank order IS the meaning here, so this list is not sorted.
        "candidate_diagnoses": [
            {
                "disease_id": c.disease_id,
                "rank": c.rank,
                "confidence": round(c.confidence, _CONFIDENCE_PRECISION),
                "supporting_symptoms": sorted(c.supporting_symptoms),
                "missing_symptoms": sorted(c.missing_symptoms),
            }
            for c in reasoning.candidate_diagnoses
        ],
        "red_flags": sorted(
            (
                {
                    "flag_id": f.flag_id,
                    "severity": f.severity,
                    "supporting_evidence": sorted(f.supporting_evidence),
                }
                for f in reasoning.red_flags
            ),
            key=lambda f: f["flag_id"],
        ),
        "risk_factors": sorted(
            (
                {"factor_id": r.factor_id, "present": r.present, "source": r.source}
                for r in reasoning.risk_factors
            ),
            key=lambda r: r["factor_id"],
        ),
        "investigations": sorted(
            (
                {
                    "investigation_id": i.investigation_id,
                    "priority": i.priority,
                    "escalated_by_red_flag": i.escalated_by_red_flag,
                    "for_conditions": sorted(i.for_conditions),
                }
                for i in reasoning.recommended_investigations
            ),
            key=lambda i: i["investigation_id"],
        ),
        "missing_information": sorted(
            (
                {"field": m.field, "priority": m.priority}
                for m in reasoning.missing_information
            ),
            key=lambda m: m["field"],
        ),
        "contraindications": sorted(
            c.contraindication_id for c in reasoning.contraindications
        ),
        "drug_interactions": sorted(
            getattr(d, "interaction_id", repr(d)) for d in reasoning.drug_interactions
        ),
        "clinical_scores": sorted(
            (
                {
                    "score_id": getattr(s, "score_id", None) or getattr(s, "name", ""),
                    "value": getattr(s, "score", None),
                    "risk": getattr(s, "risk_level", None),
                }
                for s in reasoning.clinical_scores
            ),
            key=lambda s: str(s["score_id"]),
        ),
    }


def semantic_view(state: ClinicalState, reasoning: ClinicalReasoningResult) -> Dict[str, Any]:
    """
    The clinically meaningful projection of one reasoning pass.

    Deliberately a plain dict of primitives: it is readable in a failure
    message, diffable by category, and stable across processes — no object
    identity, no memory addresses, no clock.
    """
    return {**clinical_state_view(state), **reasoning_view(reasoning)}


def _canonical(view: Dict[str, Any]) -> str:
    """
    One byte-stable rendering of a view.

    Canonical by construction rather than by hope: keys sorted, no insignificant
    whitespace, and `default=str` only ever reached by values the view already
    reduced to primitives.
    """
    return json.dumps(view, sort_keys=True, separators=(",", ":"), default=str)


def _digest(view: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(view).encode("utf-8")).hexdigest()


def fingerprint(state: ClinicalState, reasoning: ClinicalReasoningResult) -> str:
    """
    A stable digest of the clinical conclusion.

    Same clinical picture, same digest — in this process, the next one, and
    after a replay. Nothing time-, order- or identity-dependent reaches it.
    """
    return _digest(semantic_view(state, reasoning))


def state_fingerprint(state: ClinicalState) -> str:
    """
    A stable digest of the engine's input alone.

    Unchanged means reasoning would reach the same conclusion, so the pass can
    be skipped. See `clinical_state_view` for why that holds.
    """
    return _digest(clinical_state_view(state))


def _has_content(value: Any) -> bool:
    """
    Whether a category actually says anything.

    A dict of all-None (`{"age": None, "gender": None}`) is truthy in Python
    but clinically empty, so the first result would otherwise claim that
    demographics "changed" when nothing had been learned about the patient.
    """
    if isinstance(value, dict):
        return any(_has_content(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    return value is not None and value != ""


def changed_categories(previous: Dict[str, Any] | None, current: Dict[str, Any]) -> List[str]:
    """
    Which parts of the clinical picture moved.

    Carried on the emitted result so a consumer can render "new red flag"
    rather than re-diffing the whole payload, and so a test can assert on the
    reason for an emission instead of only on its existence.
    """
    if previous is None:
        return [c for c in SEMANTIC_CATEGORIES if _has_content(current.get(c))]
    return [
        category
        for category in SEMANTIC_CATEGORIES
        if previous.get(category) != current.get(category)
    ]
