"""
What the intelligence layer is allowed to see.

`CopilotContext` is a snapshot, not a handle. It is built from a
`CopilotResult` by copying, so a provider holding one cannot reach the clinical
state, the session, the runtime or the rule engine. Handing a provider the live
objects would make "never modify" a matter of trust; handing it a copy makes it
a matter of fact.

The snapshot keeps four things apart, because conflating them is how a
suggestion comes to look like a finding:

* **observed** — what the patient said, each item quoted and attributed to the
  utterance it came from.
* **deterministic** — what the rule engine concluded, with its own identifiers
  and confidences untouched.
* **evidence** — the links between the two.
* **references** — a flat index of stable ids (`symptom:chest_pain`,
  `inv:ecg_12lead`) that a proposal must cite. Without these, "grounded" would
  be a claim rather than something checkable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from integration.copilot_insights import CopilotInsights, to_insights
from integration.copilot_runtime import CopilotResult

#: Bumped when the shape below changes in a way a provider could notice.
#: A provider that cannot handle the version it is given should decline rather
#: than guess.
CONTEXT_VERSION = "copilot-context/1"


class ContextFact(BaseModel):
    """
    One addressable thing in the consultation.

    `ref` is what a proposal cites. It is stable within a consultation and
    derived from the clinical identifier, never from list position, so a
    citation does not silently come to mean something else when the differential
    re-ranks.
    """

    ref: str
    kind: str
    label: str
    #: Present for observed facts: the patient's own words.
    quote: Optional[str] = None
    #: Present for observed facts: which utterance carried it.
    utterance_id: Optional[str] = None
    #: Whatever the underlying model already carried — confidence, priority,
    #: severity. Copied, never recomputed.
    detail: Dict[str, Any] = Field(default_factory=dict)


class ObservedFacts(BaseModel):
    """What was said. Nothing here is inferred."""

    age: Optional[int] = None
    gender: Optional[str] = None
    chief_complaint: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    severity: Optional[str] = None
    symptoms: List[ContextFact] = Field(default_factory=list)
    history: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    vitals: Dict[str, Any] = Field(default_factory=dict)


class DeterministicConclusions(BaseModel):
    """
    What the rules concluded.

    A provider may reason about these and may propose alongside them. It may not
    change them, and the boundary gives it no means to: this is a copy, and a
    proposal has no field that could carry a replacement priority or score.
    """

    differential: List[ContextFact] = Field(default_factory=list)
    red_flags: List[ContextFact] = Field(default_factory=list)
    investigations: List[ContextFact] = Field(default_factory=list)
    missing_information: List[ContextFact] = Field(default_factory=list)
    contraindications: List[ContextFact] = Field(default_factory=list)
    risk_factors: List[ContextFact] = Field(default_factory=list)


class CopilotContext(BaseModel):
    """
    One immutable view of a consultation, versioned for the provider contract.

    Frozen so a provider cannot mutate what it was handed even in-process —
    a mistake that would otherwise be invisible until it corrupted a later
    comparison.
    """

    model_config = {"frozen": True}

    context_version: str = CONTEXT_VERSION
    consultation_id: str
    #: The utterance this view reflects.
    sequence: int
    event_id: str
    state_version: int
    #: Digest of the clinical picture. A provider may use it to cache.
    fingerprint: str

    observed: ObservedFacts
    deterministic: DeterministicConclusions

    #: Every citable ref in this context, for validation and for a provider
    #: that wants to check a reference before citing it.
    references: List[str] = Field(default_factory=list)

    def has_reference(self, ref: str) -> bool:
        return ref in set(self.references)


def _fact(ref: str, kind: str, label: str, **detail: Any) -> ContextFact:
    return ContextFact(ref=ref, kind=kind, label=label, detail=detail)


def build_context(result: CopilotResult) -> CopilotContext:
    """
    Projects one result into the provider-facing snapshot.

    Pure: no engine call, no clock, no state. The same result always produces
    the same context, which is what lets a provider's output be cached against
    the fingerprint and replayed in a test.
    """
    insights: CopilotInsights = to_insights(result)
    summary = insights.summary

    symptoms = [
        ContextFact(
            ref=f"symptom:{s.name}",
            kind="symptom",
            label=s.display_name or s.name,
            quote=s.source_text,
            utterance_id=s.utterance_id,
            detail={
                "present": s.present,
                "severity": s.severity,
                "duration": s.duration,
                "onset": s.onset,
            },
        )
        for s in summary.symptoms
    ]

    observed = ObservedFacts(
        age=summary.age,
        gender=summary.gender,
        chief_complaint=list(summary.chief_complaint),
        duration=summary.duration,
        severity=summary.severity,
        symptoms=symptoms,
        history=list(summary.history),
        medications=list(summary.medications),
        allergies=list(summary.allergies),
        vitals=dict(summary.vitals),
    )

    deterministic = DeterministicConclusions(
        differential=[
            _fact(
                f"dx:{d.disease_id}",
                "diagnosis",
                d.name,
                rank=d.rank,
                confidence=d.confidence,
                supporting_evidence=list(d.supporting_evidence),
                not_yet_assessed=list(d.not_yet_assessed),
            )
            for d in insights.differential
        ],
        red_flags=[
            _fact(
                f"flag:{f.flag_id}",
                "red_flag",
                f.condition,
                severity=f.severity,
                is_primary=f.is_primary,
                priority_rank=f.priority_rank,
                clinical_evidence=list(f.clinical_evidence),
                recommended_action=f.recommended_action,
            )
            for f in insights.red_flags
        ],
        investigations=[
            _fact(
                f"inv:{i.investigation_id}",
                "investigation",
                i.name,
                priority=i.priority,
                reason=i.reason,
                for_conditions=list(i.for_conditions),
                escalated_by_red_flag=i.escalated_by_red_flag,
            )
            for i in insights.investigations
        ],
        missing_information=[
            _fact(
                f"missing:{m.field}",
                "missing_information",
                m.question,
                priority=m.priority,
                relevant_conditions=list(m.relevant_conditions),
            )
            for m in insights.missing_information
        ],
        contraindications=[
            _fact(
                f"ci:{c.contraindication_id}",
                "contraindication",
                c.drug_or_procedure,
                severity=c.severity,
                reason=c.reason,
            )
            for c in insights.safety.contraindications
        ],
        risk_factors=[
            _fact(f"risk:{r.factor_id}", "risk_factor", r.name, source=r.source)
            for r in insights.safety.risk_factors
        ],
    )

    references = [f.ref for f in symptoms]
    for group in (
        deterministic.differential,
        deterministic.red_flags,
        deterministic.investigations,
        deterministic.missing_information,
        deterministic.contraindications,
        deterministic.risk_factors,
    ):
        references.extend(f.ref for f in group)

    return CopilotContext(
        consultation_id=result.consultation_id,
        sequence=result.sequence,
        event_id=result.event_id,
        state_version=result.clinical_state.version,
        fingerprint=result.fingerprint,
        observed=observed,
        deterministic=deterministic,
        references=references,
    )
