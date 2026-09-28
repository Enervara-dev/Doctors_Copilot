"""
Two providers that need no model.

Both exist to prove the seam works and to give the system a sane default before
a LangGraph layer is written. Neither generates anything: the deterministic one
restates what the rule engine already concluded, and says so in its provider
name, so nothing it emits can be mistaken for inference.

A LangGraph provider implements the same `propose(context) -> ProviderResponse`
and nothing else in this package changes.
"""

from __future__ import annotations

import hashlib
from typing import List

from integration.intelligence.context import CopilotContext
from integration.intelligence.contract import (
    IntelligenceStatus,
    Proposal,
    ProposalKind,
    ProviderResponse,
)


def _stable_id(context: CopilotContext, *parts: str) -> str:
    """
    A proposal id that is the same for the same context and content.

    Not a uuid: replaying a consultation must produce byte-identical proposals,
    or the replay determinism the runtime relies on stops at this boundary.
    """
    digest = hashlib.sha256("|".join((context.fingerprint, *parts)).encode()).hexdigest()
    return f"prop_{digest[:12]}"


class NullIntelligenceProvider:
    """
    The default: answers nothing, cleanly.

    `DISABLED` rather than `UNAVAILABLE` — nothing is broken, the layer is
    simply off. A consumer can tell "we are not running intelligence" from "we
    tried and got nothing", and neither reads as a clinical finding.
    """

    name = "null"
    version = "1.0.0"

    def propose(self, context: CopilotContext) -> ProviderResponse:
        return ProviderResponse(
            status=IntelligenceStatus.DISABLED,
            provider=self.name,
            provider_version=self.version,
            message="No intelligence provider is configured.",
        )


class DeterministicIntelligenceProvider:
    """
    Restates deterministic output as proposals, for contract testing.

    Every proposal it makes is traceable to a rule the engine already fired, so
    it cannot say anything the deterministic core did not already say. That
    makes it useful for exercising the boundary end to end — validation,
    grounding, replay — with no model, no network and no nondeterminism.

    It is not an intelligence layer and does not pretend to be one. Its value is
    that if a LangGraph provider ever behaves worse than this on the contract
    tests, the contract is being broken rather than the model being weak.
    """

    name = "deterministic"
    version = "1.0.0"

    #: Questions are the safest proposal kind, so the unanswered ones the rule
    #: engine already identified are the natural thing to surface.
    MAX_QUESTIONS = 5

    def propose(self, context: CopilotContext) -> ProviderResponse:
        proposals: List[Proposal] = []

        for item in context.deterministic.missing_information[: self.MAX_QUESTIONS]:
            proposals.append(
                Proposal(
                    proposal_id=_stable_id(context, "question", item.ref),
                    kind=ProposalKind.QUESTION,
                    content=item.label,
                    grounded_in=[item.ref],
                    metadata={"origin": "missing_information"},
                )
            )

        # One note against the highest-priority investigation, citing the
        # conditions the rule itself named. Nothing here is generated: the
        # reason text comes straight from the rule.
        urgent = [
            i
            for i in context.deterministic.investigations
            if i.detail.get("priority") == "URGENT"
        ]
        if urgent:
            first = urgent[0]
            proposals.append(
                Proposal(
                    proposal_id=_stable_id(context, "note", first.ref),
                    kind=ProposalKind.INVESTIGATION_NOTE,
                    content=str(first.detail.get("reason") or first.label),
                    grounded_in=[first.ref],
                    about_ref=first.ref,
                    metadata={"origin": "rule_rationale"},
                )
            )

        return ProviderResponse(
            status=IntelligenceStatus.OK,
            provider=self.name,
            provider_version=self.version,
            proposals=proposals,
            telemetry={"proposal_count": len(proposals), "model": None},
        )
