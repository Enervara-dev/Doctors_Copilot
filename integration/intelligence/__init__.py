"""
The intelligence boundary.

Everything in this package exists to answer one question precisely: what may a
reasoning layer that is not deterministic be allowed to do to a clinical
consultation?

The answer this boundary encodes is narrow on purpose.

* It may **read** a versioned, immutable snapshot of the consultation —
  `CopilotContext`. Observed facts, deterministic conclusions and the evidence
  linking them, each item carrying a stable reference.
* It may **propose**. A proposal is a suggestion with a citation, in one of a
  small set of shapes, and it arrives in its own channel clearly marked as
  model output.
* It may **never modify** anything. There is no path from a provider back into
  the clinical state, the rule output, an investigation priority or a red flag.
  Not by convention — the context it receives is a copy, and the only thing it
  can return is a proposal that is then validated.

Every proposal is checked before it is surfaced: it must cite references that
actually exist in the context it was given, and an annotation must name an
investigation the rules actually recommended. A proposal that fails is kept
with its rejection reason rather than dropped, because "the model said
something unusable" is information a reviewer needs.

Nothing here calls a model. `providers.py` ships a null provider and a
deterministic one; a LangGraph provider implements the same `IntelligenceProvider`
protocol and nothing else in the system changes.
"""

from integration.intelligence.context import (
    CONTEXT_VERSION,
    ContextFact,
    CopilotContext,
    DeterministicConclusions,
    ObservedFacts,
    build_context,
)
from integration.intelligence.contract import (
    PROPOSAL_VERSION,
    IntelligenceProvider,
    IntelligenceStatus,
    Proposal,
    ProposalKind,
    ProviderResponse,
)
from integration.intelligence.gateway import IntelligenceGateway, IntelligenceOutcome
from integration.intelligence.langgraph_provider import LangGraphIntelligenceProvider
from integration.intelligence.model_client import (
    ModelClient,
    ModelUnavailable,
    ScriptedModelClient,
    UnavailableModelClient,
)
from integration.intelligence.providers import (
    DeterministicIntelligenceProvider,
    NullIntelligenceProvider,
)
from integration.intelligence.validation import (
    RejectedProposal,
    ValidationReport,
    validate_proposals,
)

__all__ = [
    "CONTEXT_VERSION",
    "PROPOSAL_VERSION",
    "ContextFact",
    "CopilotContext",
    "DeterministicConclusions",
    "ObservedFacts",
    "build_context",
    "IntelligenceProvider",
    "IntelligenceStatus",
    "Proposal",
    "ProposalKind",
    "ProviderResponse",
    "IntelligenceGateway",
    "IntelligenceOutcome",
    "LangGraphIntelligenceProvider",
    "ModelClient",
    "ModelUnavailable",
    "ScriptedModelClient",
    "UnavailableModelClient",
    "DeterministicIntelligenceProvider",
    "NullIntelligenceProvider",
    "RejectedProposal",
    "ValidationReport",
    "validate_proposals",
]
