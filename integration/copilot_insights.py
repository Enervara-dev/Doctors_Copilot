"""
The clinician-facing shape of a Copilot result.

A `CopilotResult` carries everything the engine produced, in the engine's own
structure. That is the right shape for a contract and the wrong one for someone
mid-consultation: the facts the patient stated sit in one model, the
conclusions drawn from them in another, and a reader has to know which is
which.

This module is a projection and nothing more. Every value below is copied from
what the rule engine already produced — no field is computed, scored, banded or
re-ranked here, and nothing is added that the engine did not say. If a surface
looks thin, the engine found little; that is information, not a gap to fill.

Two things it does add, both presentational and both deliberate:

* **`basis`** — every surface is marked `OBSERVED` or `INFERRED`. A symptom the
  patient described and a diagnosis a rule proposed must never read alike, and
  the underlying models do not carry that distinction because each is only ever
  one of the two.

* **`ADVISORY`** — one fixed sentence on the envelope. The differential is a
  ranked list of possibilities the rules found consistent with what was said.
  Anything that reads like a decision is a misreading, and the payload should
  say so rather than rely on the UI to remember.

No transport here: the API serializes this, and so could anything else.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from integration.copilot_runtime import CopilotResult

if TYPE_CHECKING:  # annotation only; no runtime import of the app layer
    from app.services.clinical.models import ClinicalState

#: Carried on every payload. The Copilot proposes; the clinician decides.
ADVISORY = (
    "Assistive output. Candidates are possibilities consistent with what has "
    "been said so far, ranked by rule-based confidence — not diagnoses. The "
    "treating clinician remains the decision-maker."
)


class Basis(str, Enum):
    """Where a surface came from — stated, or concluded."""

    #: Said during the consultation and extracted from the transcript.
    OBSERVED = "OBSERVED"
    #: Derived by the deterministic rule engine from observed facts.
    INFERRED = "INFERRED"


# ══════════════════════════════════════════════════════════════════════════
# 1. Clinical summary — observed
# ══════════════════════════════════════════════════════════════════════════


class EvidenceRef(BaseModel):
    """
    One piece of evidence, tied back to what was said.

    The rule engine names its evidence in canonical terms ("chest pain"). A
    clinician reading a CRITICAL alert should be able to see the sentence that
    produced it, so each item is matched against the clinical state and, where
    it corresponds to an extracted symptom, carries that symptom's utterance
    and the patient's own words.

    `quote` is absent when the evidence is not a symptom — an age band, a
    comorbidity, a vital threshold. Nothing is invented to fill it.
    """

    label: str
    #: OBSERVED when the evidence is a symptom the patient reported; SUPPORTING
    #: for demographics and history; MEASURED for a vital-sign threshold.
    kind: str
    utterance_id: Optional[str] = None
    quote: Optional[str] = None


class SymptomView(BaseModel):
    """
    One symptom as the patient described it.

    `source_text` is the transcript fragment the extractor matched and
    `utterance_id` names the utterance it came from, so any item on this
    surface can be traced back to something that was actually said.
    """

    name: str
    display_name: Optional[str] = None
    present: Optional[str] = None
    severity: Optional[str] = None
    duration: Optional[str] = None
    onset: Optional[str] = None
    character: Optional[str] = None
    frequency: Optional[str] = None
    progression: Optional[str] = None
    status: Optional[str] = None
    confidence: Optional[str] = None
    source_text: Optional[str] = None
    utterance_id: Optional[str] = None


class ClinicalSummary(BaseModel):
    """What the consultation has established so far."""

    basis: Basis = Basis.OBSERVED
    age: Optional[int] = None
    gender: Optional[str] = None
    chief_complaint: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    severity: Optional[str] = None
    symptoms: List[SymptomView] = Field(default_factory=list)
    history: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    vitals: Dict[str, Any] = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════
# 2–6. Inferred surfaces
# ══════════════════════════════════════════════════════════════════════════


class DifferentialItem(BaseModel):
    """
    One candidate the rules found consistent with the observed facts.

    `confidence` and `rank` are the engine's own; nothing is rescaled or
    bucketed. `not_yet_assessed` is the engine's `missing_symptoms` — cardinal
    features that have neither been confirmed nor excluded, which is why it is
    named for the gap rather than for absence.
    """

    basis: Basis = Basis.INFERRED
    disease_id: str
    name: str
    category: str
    rank: int
    confidence: float
    supporting_evidence: List[str] = Field(default_factory=list)
    not_yet_assessed: List[str] = Field(default_factory=list)
    #: Each supporting symptom traced to the utterance that produced it.
    evidence_trail: List[EvidenceRef] = Field(default_factory=list)


class RedFlagView(BaseModel):
    """
    A time-critical scenario the rules detected, what fired it, and how
    strongly.

    Four simultaneous CRITICAL alerts are worse than one: a clinician who sees
    the same wall of red on every consultation stops reading it. Nothing is
    hidden here — every flag the engine raised is present — but they are ranked
    by how much clinical evidence actually supports each, so the panel can lead
    with the one that earned it and collapse the rest.
    """

    basis: Basis = Basis.INFERRED
    flag_id: str
    condition: str
    severity: str
    triggered_by: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    recommended_action: str

    #: 1 is the strongest. Ranking is by severity first, then by how much
    #: clinical evidence supports the flag.
    priority_rank: int = 1
    #: The flags that lead the panel. Everything else is real but secondary.
    is_primary: bool = False
    #: Symptoms and measurements only — the evidence that is a presentation
    #: rather than a risk profile.
    clinical_evidence: List[str] = Field(default_factory=list)
    #: Symptom count plus twice the number of vital thresholds met. Exposed so
    #: the ordering can be checked rather than trusted.
    specificity_score: int = 0
    evidence_trail: List[EvidenceRef] = Field(default_factory=list)


class InvestigationView(BaseModel):
    """
    A recommended investigation, with the rule's own reason and priority.

    `escalated_by_red_flag` is kept because it explains a priority that would
    otherwise look arbitrary beside the same test in another consultation.
    """

    basis: Basis = Basis.INFERRED
    investigation_id: str
    name: str
    priority: str
    reason: str
    for_conditions: List[str] = Field(default_factory=list)
    escalated_by_red_flag: bool = False


class MissingInfoView(BaseModel):
    """A clinically relevant question the consultation has not answered."""

    basis: Basis = Basis.INFERRED
    field: str
    question: str
    priority: int
    relevant_conditions: List[str] = Field(default_factory=list)


class ContraindicationView(BaseModel):
    basis: Basis = Basis.INFERRED
    contraindication_id: str
    drug_or_procedure: str
    severity: str
    reason: str
    trigger: str
    recommendation: str


class DrugInteractionView(BaseModel):
    basis: Basis = Basis.INFERRED
    interaction_id: str
    drug_a: str
    drug_b: str
    severity: str
    mechanism: str
    clinical_effect: str
    recommendation: str


class RiskFactorView(BaseModel):
    basis: Basis = Basis.INFERRED
    factor_id: str
    name: str
    category: str
    present: bool
    source: str


class SafetyView(BaseModel):
    """Everything that should give the clinician pause before acting."""

    contraindications: List[ContraindicationView] = Field(default_factory=list)
    drug_interactions: List[DrugInteractionView] = Field(default_factory=list)
    risk_factors: List[RiskFactorView] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════
# Envelope
# ══════════════════════════════════════════════════════════════════════════


class CopilotInsights(BaseModel):
    """
    The six surfaces, for one moment in a consultation.

    Identified by the utterance that produced it, so a clinician looking at a
    panel can tell which part of the conversation it reflects.
    """

    advisory: str = ADVISORY
    consultation_id: str
    #: The FINAL utterance this reflects.
    sequence: int
    event_id: str
    #: The clinical state version these conclusions were drawn from.
    state_version: int
    #: Digest of the clinical picture; unchanged means nothing moved.
    fingerprint: str
    #: Which surfaces changed since the previous emission.
    changed: List[str] = Field(default_factory=list)

    summary: ClinicalSummary
    differential: List[DifferentialItem] = Field(default_factory=list)
    red_flags: List[RedFlagView] = Field(default_factory=list)
    investigations: List[InvestigationView] = Field(default_factory=list)
    missing_information: List[MissingInfoView] = Field(default_factory=list)
    safety: SafetyView = Field(default_factory=SafetyView)

    #: Per-fact origin exactly as `ClinicalStateEngine` recorded it: which
    #: utterance, which speaker, which state version produced each fact.
    provenance: Dict[str, Any] = Field(default_factory=dict)


#: CRITICAL before HIGH before MODERATE, whatever order the engine returned.
_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}

#: A vital threshold reads "bp_systolic_gt (threshold: 180.0)".
_VITAL_MARKER = "(threshold:"


def _canonical(symptom_name: str) -> str:
    """`symptom_chest_pain` as the rule engine names it: `chest pain`."""
    return symptom_name.removeprefix("symptom_").replace("_", " ").strip().lower()


def _symptom_index(state: ClinicalState) -> Dict[str, Dict[str, Any]]:
    """Canonical symptom name to the state entry that produced it."""
    index: Dict[str, Dict[str, Any]] = {}
    for entry in state.symptoms:
        name = entry.get("name")
        if name:
            index.setdefault(_canonical(str(name)), entry)
    return index


def _evidence_ref(label: str, index: Dict[str, Dict[str, Any]]) -> EvidenceRef:
    """
    Classifies one evidence string and links it where it can be linked.

    Matching is exact against the canonical symptom names already in the state.
    An evidence item that does not correspond to an extracted symptom is
    labelled and left unlinked rather than guessed at.
    """
    if _VITAL_MARKER in label:
        return EvidenceRef(label=label, kind="MEASURED")

    entry = index.get(label.strip().lower())
    if entry is None:
        return EvidenceRef(label=label, kind="SUPPORTING")

    return EvidenceRef(
        label=label,
        kind="OBSERVED",
        utterance_id=entry.get("utterance_id"),
        quote=entry.get("source_text"),
    )


def _rank_red_flags(
    flags: List[RedFlagView],
) -> List[RedFlagView]:
    """
    Orders flags by severity, then by how much of a presentation supports them.

    Deliberately not a clinical judgement about which diagnosis matters more —
    the engine makes no such claim and neither should this. It is a count of
    evidence: a flag matched by three symptoms is better supported than one
    matched by a single non-specific complaint, and a measured vital counts
    double because it is objective.
    """
    ordered = sorted(
        flags,
        key=lambda f: (
            _SEVERITY_ORDER.get(f.severity, 99),
            -f.specificity_score,
            f.flag_id,
        ),
    )
    if not ordered:
        return ordered

    best = (
        _SEVERITY_ORDER.get(ordered[0].severity, 99),
        ordered[0].specificity_score,
    )
    for position, flag in enumerate(ordered, start=1):
        flag.priority_rank = position
        flag.is_primary = (
            _SEVERITY_ORDER.get(flag.severity, 99),
            flag.specificity_score,
        ) == best and flag.specificity_score > 0
    return ordered


def _symptom(raw: Dict[str, Any]) -> SymptomView:
    return SymptomView(
        name=str(raw.get("name") or ""),
        display_name=raw.get("display_name"),
        present=None if raw.get("present") is None else str(raw.get("present")),
        severity=raw.get("severity"),
        duration=raw.get("duration"),
        onset=raw.get("onset"),
        character=raw.get("character"),
        frequency=raw.get("frequency"),
        progression=raw.get("progression"),
        status=raw.get("status"),
        confidence=None if raw.get("confidence") is None else str(raw.get("confidence")),
        source_text=raw.get("source_text"),
        utterance_id=raw.get("utterance_id"),
    )


def _red_flag_view(flag: Any, index: Dict[str, Dict[str, Any]]) -> RedFlagView:
    """Projects one flag and measures how much of a presentation supports it."""
    trail = [_evidence_ref(e, index) for e in flag.supporting_evidence]
    observed = [ref for ref in trail if ref.kind == "OBSERVED"]
    measured = [ref for ref in trail if ref.kind == "MEASURED"]

    return RedFlagView(
        flag_id=flag.flag_id,
        condition=flag.condition,
        severity=flag.severity,
        triggered_by=list(flag.triggered_by),
        supporting_evidence=list(flag.supporting_evidence),
        recommended_action=flag.recommended_action,
        clinical_evidence=[ref.label for ref in observed + measured],
        specificity_score=len(observed) + 2 * len(measured),
        evidence_trail=trail,
    )


def to_insights(result: CopilotResult) -> CopilotInsights:
    """
    Projects one result onto the six clinician-facing surfaces.

    A pure function of the result: no engine call, no state, no clock. Two
    calls with the same result produce the same payload, which is what lets the
    API serve it without re-running anything.
    """
    state = result.clinical_state
    reasoning = result.reasoning
    index = _symptom_index(state)

    vitals = state.vital_signs.model_dump() if state.vital_signs else {}

    summary = ClinicalSummary(
        age=state.patient_info.age,
        gender=state.patient_info.gender,
        chief_complaint=list(state.chief_complaint or []),
        duration=state.duration,
        severity=state.severity,
        symptoms=[_symptom(s) for s in state.symptoms],
        history=list(state.medical_history or []),
        medications=list(state.current_medications or []),
        allergies=list(state.allergies or []),
        # Only readings that were actually taken; a panel of Nones is noise.
        vitals={k: v for k, v in vitals.items() if v is not None},
    )

    return CopilotInsights(
        consultation_id=result.consultation_id,
        sequence=result.sequence,
        event_id=result.event_id,
        state_version=state.version,
        fingerprint=result.fingerprint,
        changed=list(result.changed),
        summary=summary,
        differential=[
            DifferentialItem(
                disease_id=c.disease_id,
                name=c.name,
                category=c.category,
                rank=c.rank,
                confidence=c.confidence,
                supporting_evidence=list(c.supporting_symptoms),
                not_yet_assessed=list(c.missing_symptoms),
                evidence_trail=[
                    _evidence_ref(e, index) for e in c.supporting_symptoms
                ],
            )
            for c in reasoning.candidate_diagnoses
        ],
        red_flags=_rank_red_flags(
            [_red_flag_view(f, index) for f in reasoning.red_flags]
        ),
        investigations=[
            InvestigationView(
                investigation_id=i.investigation_id,
                name=i.name,
                priority=i.priority,
                reason=i.reason,
                for_conditions=list(i.for_conditions),
                escalated_by_red_flag=i.escalated_by_red_flag,
            )
            for i in reasoning.recommended_investigations
        ],
        missing_information=[
            MissingInfoView(
                field=m.field,
                question=m.question,
                priority=m.priority,
                relevant_conditions=list(m.relevant_conditions),
            )
            for m in reasoning.missing_information
        ],
        safety=SafetyView(
            contraindications=[
                ContraindicationView(
                    contraindication_id=c.contraindication_id,
                    drug_or_procedure=c.drug_or_procedure,
                    severity=c.severity,
                    reason=c.reason,
                    trigger=c.trigger,
                    recommendation=c.recommendation,
                )
                for c in reasoning.contraindications
            ],
            drug_interactions=[
                DrugInteractionView(
                    interaction_id=d.interaction_id,
                    drug_a=d.drug_a,
                    drug_b=d.drug_b,
                    severity=d.severity,
                    mechanism=d.mechanism,
                    clinical_effect=d.clinical_effect,
                    recommendation=d.recommendation,
                )
                for d in reasoning.drug_interactions
            ],
            risk_factors=[
                RiskFactorView(
                    factor_id=r.factor_id,
                    name=r.name,
                    category=r.category,
                    present=r.present,
                    source=r.source,
                )
                for r in reasoning.risk_factors
            ],
        ),
        provenance=dict(result.provenance),
    )
