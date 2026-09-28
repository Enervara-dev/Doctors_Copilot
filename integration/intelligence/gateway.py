"""
The one place a provider is called.

The gateway builds the context, invokes the provider, validates what comes
back, and returns an outcome. It is deliberately not wired into
`CopilotRuntime`: the clinical path must keep working, at full speed and with
identical output, whether or not an intelligence layer exists. Composing the
two is a decision for the caller, made once a provider is worth calling.

Three guarantees, each with a test:

* A provider that raises cannot break a consultation. The exception is caught,
  reported as `FAILED`, and the clinical result is returned untouched.
* A provider that runs slowly cannot hold a consultation open indefinitely. The
  gateway records elapsed time and a caller may set a budget; exceeding it is
  reported rather than hidden.
* Nothing a provider returns changes the clinical result. The context is a
  copy, the outcome is a separate object, and the `CopilotResult` that went in
  comes back out identical.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from integration.copilot_runtime import CopilotResult
from integration.intelligence.context import build_context
from integration.intelligence.contract import (
    IntelligenceProvider,
    IntelligenceStatus,
    Proposal,
    ProviderResponse,
)
from integration.intelligence.validation import (
    RejectedProposal,
    ValidationReport,
    validate_proposals,
)

logger = logging.getLogger(__name__)


class IntelligenceOutcome(BaseModel):
    """
    Everything that happened when the intelligence layer was consulted.

    Separate from `CopilotResult` on purpose. A consumer that renders only the
    deterministic panel never has to know this exists, and one that renders
    proposals can see exactly which provider produced them and what was refused.
    """

    consultation_id: str
    sequence: int
    #: The context digest the proposals were made against. If the clinical
    #: picture has moved on, these proposals describe an earlier moment.
    fingerprint: str

    status: IntelligenceStatus
    provider: str
    provider_version: str
    message: Optional[str] = None

    proposals: List[Proposal] = Field(default_factory=list)
    rejected: List[RejectedProposal] = Field(default_factory=list)

    elapsed_ms: float = 0.0
    #: True when the provider took longer than the caller's budget. The result
    #: is still returned — it is late, not wrong.
    over_budget: bool = False
    telemetry: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_proposals(self) -> bool:
        return bool(self.proposals)


class IntelligenceGateway:
    """
    Calls one provider safely.

    `budget_ms` is advisory: the gateway cannot interrupt a synchronous
    provider mid-call, and pretending otherwise with a thread would leave the
    work running anyway. What it does is measure and report, so a slow provider
    is visible rather than merely felt.
    """

    def __init__(
        self,
        provider: IntelligenceProvider,
        *,
        budget_ms: Optional[float] = None,
        clock: Any = time.perf_counter,
    ) -> None:
        self._provider = provider
        self._budget_ms = budget_ms
        self._clock = clock

    @property
    def provider_name(self) -> str:
        return getattr(self._provider, "name", "unknown")

    def consult(self, result: CopilotResult) -> IntelligenceOutcome:
        """
        Builds the context, asks the provider, and validates the answer.

        Returns an outcome in every case, including failure. A caller never has
        to handle an exception from here to keep a consultation running.
        """
        context = build_context(result)
        started = self._clock()

        try:
            response = self._provider.propose(context)
        except Exception as exc:  # noqa: BLE001 - the whole point is to catch anything
            elapsed = (self._clock() - started) * 1000
            logger.exception(
                "Intelligence provider %s failed for consultation %s",
                self.provider_name,
                result.consultation_id,
            )
            return IntelligenceOutcome(
                consultation_id=result.consultation_id,
                sequence=result.sequence,
                fingerprint=result.fingerprint,
                status=IntelligenceStatus.FAILED,
                provider=self.provider_name,
                provider_version=getattr(self._provider, "version", "unknown"),
                message=f"{type(exc).__name__}: {exc}",
                elapsed_ms=elapsed,
                over_budget=self._over(elapsed),
            )

        elapsed = (self._clock() - started) * 1000

        if not isinstance(response, ProviderResponse):
            # A provider that returns the wrong type is a wiring bug, and one
            # that would otherwise surface as an obscure attribute error deep in
            # validation.
            return IntelligenceOutcome(
                consultation_id=result.consultation_id,
                sequence=result.sequence,
                fingerprint=result.fingerprint,
                status=IntelligenceStatus.FAILED,
                provider=self.provider_name,
                provider_version=getattr(self._provider, "version", "unknown"),
                message=(
                    "Provider returned "
                    f"{type(response).__name__}, not a ProviderResponse."
                ),
                elapsed_ms=elapsed,
                over_budget=self._over(elapsed),
            )

        report: ValidationReport = validate_proposals(response.proposals, context)
        if report.rejected:
            logger.warning(
                "Rejected %d proposal(s) from %s for consultation %s: %s",
                len(report.rejected),
                response.provider,
                result.consultation_id,
                report.rejection_codes,
            )

        return IntelligenceOutcome(
            consultation_id=result.consultation_id,
            sequence=result.sequence,
            fingerprint=result.fingerprint,
            status=response.status,
            provider=response.provider,
            provider_version=response.provider_version,
            message=response.message,
            proposals=report.accepted,
            rejected=report.rejected,
            elapsed_ms=elapsed,
            over_budget=self._over(elapsed),
            telemetry=dict(response.telemetry),
        )

    def _over(self, elapsed_ms: float) -> bool:
        return self._budget_ms is not None and elapsed_ms > self._budget_ms
