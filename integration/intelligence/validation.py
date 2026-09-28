"""
Nothing a provider says reaches a clinician unchecked.

The rule engine already has a `HallucinationGuard` that discards LLM diagnoses
the rules did not support. This is the same posture applied at the boundary
rather than inside one feature: a proposal is admitted only if it cites the
consultation it claims to be about, and an annotation is admitted only if the
thing it annotates exists.

Rejections are kept, not dropped. "The model produced something unusable" is
information — for an operator debugging a provider, and for anyone deciding
whether to trust one — and silently discarding it would hide exactly the
behaviour worth watching.
"""

from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field

from integration.intelligence.context import CopilotContext
from integration.intelligence.contract import Proposal, ProposalKind

#: A proposal longer than this is not a suggestion, it is a document. The cap
#: exists so one runaway generation cannot dominate a clinician's screen.
MAX_CONTENT_CHARS = 2000

#: Citing everything is the same as citing nothing.
MAX_REFERENCES = 12


class RejectedProposal(BaseModel):
    """A proposal that did not pass, and why."""

    proposal: Proposal
    reason: str
    code: str


class ValidationReport(BaseModel):
    accepted: List[Proposal] = Field(default_factory=list)
    rejected: List[RejectedProposal] = Field(default_factory=list)

    @property
    def rejection_codes(self) -> List[str]:
        return [r.code for r in self.rejected]


def _check(proposal: Proposal, context: CopilotContext) -> Tuple[str, str] | None:
    """Returns (code, reason) when the proposal must be refused."""
    if not proposal.content or not proposal.content.strip():
        return ("EMPTY_CONTENT", "A proposal with no content says nothing.")

    if len(proposal.content) > MAX_CONTENT_CHARS:
        return (
            "CONTENT_TOO_LONG",
            f"Content is {len(proposal.content)} characters; the limit is {MAX_CONTENT_CHARS}.",
        )

    if not proposal.grounded_in:
        return (
            "UNGROUNDED",
            "A proposal must cite the consultation facts it rests on.",
        )

    if len(proposal.grounded_in) > MAX_REFERENCES:
        return (
            "TOO_MANY_REFERENCES",
            f"Cites {len(proposal.grounded_in)} references; the limit is {MAX_REFERENCES}.",
        )

    known = set(context.references)
    unknown = [ref for ref in proposal.grounded_in if ref not in known]
    if unknown:
        return (
            "UNKNOWN_REFERENCE",
            f"Cites {unknown!r}, which is not in this consultation.",
        )

    if proposal.kind is ProposalKind.INVESTIGATION_NOTE:
        if not proposal.about_ref:
            return (
                "MISSING_SUBJECT",
                "An investigation note must name the investigation it annotates.",
            )
        if not proposal.about_ref.startswith("inv:"):
            return (
                "WRONG_SUBJECT_KIND",
                f"{proposal.about_ref!r} is not an investigation reference.",
            )
        if proposal.about_ref not in known:
            # The decisive rule from the reconciliation design: a model may
            # annotate what the rules recommended. It may not introduce a test.
            return (
                "UNKNOWN_INVESTIGATION",
                f"{proposal.about_ref!r} was not recommended by the rule engine, "
                "so there is nothing to annotate.",
            )

    if proposal.about_ref and proposal.about_ref not in known:
        return (
            "UNKNOWN_SUBJECT",
            f"{proposal.about_ref!r} is not in this consultation.",
        )

    return None


def validate_proposals(
    proposals: List[Proposal], context: CopilotContext
) -> ValidationReport:
    """
    Splits proposals into those a clinician may see and those they may not.

    Order is preserved so a provider's own prioritisation survives; this decides
    admissibility, not importance.
    """
    report = ValidationReport()
    seen_ids: set[str] = set()

    for proposal in proposals:
        if proposal.proposal_id in seen_ids:
            report.rejected.append(
                RejectedProposal(
                    proposal=proposal,
                    code="DUPLICATE_ID",
                    reason=f"{proposal.proposal_id!r} was already proposed in this response.",
                )
            )
            continue
        seen_ids.add(proposal.proposal_id)

        failure = _check(proposal, context)
        if failure is None:
            report.accepted.append(proposal)
        else:
            code, reason = failure
            report.rejected.append(
                RejectedProposal(proposal=proposal, code=code, reason=reason)
            )

    return report
