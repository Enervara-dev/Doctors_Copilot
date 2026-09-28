"""
What the intelligence layer may return, and how it is asked.

A proposal is the only thing that crosses back over the boundary. Its shape is
deliberately poor in a specific way: there is no field for a priority, a
severity, a confidence score on a deterministic conclusion, or an
identifier that would let it stand in for rule output. A provider cannot
overrule the rule engine because the return type gives it nothing to overrule
with.

Four kinds are allowed, ordered by how much trust each requires:

* `QUESTION` — something worth asking. Additive and low risk: the worst case is
  a clinician reads a question they had already thought of.
* `INVESTIGATION_NOTE` — commentary on an investigation the rules already
  recommended. It must name one. It cannot add a test, remove one, or change a
  priority.
* `DIAGNOSIS_CONSIDERATION` — a possibility the rules did not raise. Carried in
  its own channel, never merged into the differential, and never ranked
  alongside scored candidates.
* `SUMMARY` — narrative for the clinician to read. No clinical claim is derived
  from it downstream.

Every proposal must cite context references. That is what makes "grounded"
checkable rather than asserted, and it is enforced in `validation.py`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from integration.intelligence.context import CopilotContext

#: Bumped when the proposal shape changes. Recorded on every proposal so a
#: stored one can be read back by a later version of this code.
PROPOSAL_VERSION = "copilot-proposal/1"


class ProposalKind(str, Enum):
    QUESTION = "QUESTION"
    INVESTIGATION_NOTE = "INVESTIGATION_NOTE"
    DIAGNOSIS_CONSIDERATION = "DIAGNOSIS_CONSIDERATION"
    SUMMARY = "SUMMARY"


class IntelligenceStatus(str, Enum):
    """
    Mirrors the vocabulary the rule engine already uses for its LLM fields, so
    a consumer does not have to learn a second one.
    """

    OK = "ok"
    #: The layer is switched off. Not a failure.
    DISABLED = "disabled"
    #: Configured but nothing answered — no key, no route, a timeout.
    UNAVAILABLE = "unavailable"
    #: The provider raised. Distinct from unavailable: something is wrong.
    FAILED = "failed"


class Proposal(BaseModel):
    """
    One suggestion, with its citations.

    `grounded_in` holds context refs. A proposal that cites nothing, or cites
    something absent from the context it was given, is rejected — the provider
    is claiming knowledge the consultation does not contain.
    """

    proposal_version: str = PROPOSAL_VERSION
    proposal_id: str
    kind: ProposalKind
    #: The suggestion itself, in plain language for a clinician.
    content: str
    #: Context references this rests on, e.g. ["symptom:symptom_chest_pain"].
    grounded_in: List[str] = Field(default_factory=list)
    #: Required for INVESTIGATION_NOTE: which recommendation is being annotated.
    about_ref: Optional[str] = None
    #: The provider's own confidence, if it reports one. Never used to rank
    #: anything deterministic, and never compared against a rule confidence.
    provider_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    #: Anything the provider wants to carry for debugging. Never interpreted.
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """
    One provider's answer.

    A response with `status != OK` and no proposals is a normal outcome, not an
    error: the clinical output stands on its own and the absence of model
    commentary must never look like a clinical finding.
    """

    status: IntelligenceStatus
    provider: str
    provider_version: str
    proposals: List[Proposal] = Field(default_factory=list)
    #: Why, when the status is not OK. Shown to an operator, not a clinician.
    message: Optional[str] = None
    #: Latency, token counts, graph node timings — whatever the provider tracks.
    telemetry: Dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class IntelligenceProvider(Protocol):
    """
    The seam a LangGraph layer will implement.

    Synchronous, matching the clinical pipeline it sits beside: a provider that
    runs a graph or calls a model blocks inside `propose`, and the gateway is
    called from a threadpool like every other synchronous step. Making this
    async would force the whole runtime to change shape for one participant.

    A provider must not raise for an expected condition — no key configured, a
    model that declined — and should return `UNAVAILABLE` instead. The gateway
    catches anything that escapes, but a caught exception is reported as
    `FAILED`, which is a different thing and should mean something is wrong.
    """

    name: str
    version: str

    def propose(self, context: CopilotContext) -> ProviderResponse:
        """Read the context; return proposals. Never mutate anything."""
        ...
