"""
The intelligence boundary contract.

What matters here is not that a provider produces good suggestions — none of
these providers produces any suggestion worth reading. It is that a provider
cannot do harm: cannot reach the clinical state, cannot overrule a rule, cannot
assert something the consultation does not contain, and cannot take a
consultation down by failing.

Each of those is a test. The hostile-provider cases are the point of the file.

Offline throughout; no provider here calls anything.
"""

from __future__ import annotations

import os
import unittest
from typing import List

os.environ.setdefault("CLINICAL_COPILOT_ENABLE_GRAPH", "False")
os.environ.setdefault("CLINICAL_COPILOT_ENABLE_LLM", "False")

from integration.copilot_event import (  # noqa: E402
    CopilotEvent,
    Provenance,
    SpeakerRole,
    UtterancePayload,
)
from integration.copilot_runtime import CopilotResult, CopilotRuntime  # noqa: E402
from integration.intelligence import (  # noqa: E402
    CONTEXT_VERSION,
    PROPOSAL_VERSION,
    CopilotContext,
    DeterministicIntelligenceProvider,
    IntelligenceGateway,
    IntelligenceProvider,
    IntelligenceStatus,
    NullIntelligenceProvider,
    Proposal,
    ProposalKind,
    ProviderResponse,
    build_context,
    validate_proposals,
)

CARDIAC = [
    "I am a 62 year old male. I have been having severe chest pain for two hours and I am sweating a lot.",
    "I also feel short of breath and nauseous.",
    "I take aspirin daily and I have high blood pressure and diabetes.",
]


def utterance(consultation_id: str, sequence: int, text: str) -> CopilotEvent:
    return CopilotEvent(
        event_id=f"utt-{sequence:03d}",
        consultation_id=consultation_id,
        sequence=sequence,
        provenance=Provenance.FIXTURE,
        payload=UtterancePayload(
            text=text,
            speaker_role=SpeakerRole.PATIENT,
            speaker_label="Patient",
            is_final=True,
            confidence=0.95,
            started_at="2026-09-27T09:00:00.000Z",
            finalized_at="2026-09-27T09:00:04.000Z",
        ),
    )


def cardiac_result(consultation_id: str = "intel-1") -> CopilotResult:
    runtime = CopilotRuntime()
    for sequence, text in enumerate(CARDIAC, start=1):
        runtime.ingest(utterance(consultation_id, sequence, text))
    return runtime.get_session(consultation_id).latest_result


def proposal(**kwargs) -> Proposal:
    defaults = {
        "proposal_id": "p1",
        "kind": ProposalKind.QUESTION,
        "content": "Has an ECG been performed?",
        "grounded_in": ["symptom:symptom_chest_pain"],
    }
    defaults.update(kwargs)
    return Proposal(**defaults)


class ScriptedProvider:
    """Returns exactly what a test hands it."""

    name = "scripted"
    version = "0.0.1"

    def __init__(self, response, *, raises: BaseException | None = None):
        self._response = response
        self._raises = raises
        self.contexts: List[CopilotContext] = []

    def propose(self, context: CopilotContext) -> ProviderResponse:
        self.contexts.append(context)
        if self._raises is not None:
            raise self._raises
        return self._response


# ══════════════════════════════════════════════════════════════════════════
# Context
# ══════════════════════════════════════════════════════════════════════════


class TestContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = cardiac_result("ctx-1")
        cls.context = build_context(cls.result)

    def test_the_context_is_versioned(self) -> None:
        self.assertEqual(self.context.context_version, CONTEXT_VERSION)

    def test_observed_and_deterministic_are_separate(self) -> None:
        self.assertEqual(self.context.observed.age, 62)
        self.assertTrue(self.context.observed.symptoms)
        self.assertTrue(self.context.deterministic.differential)
        self.assertTrue(self.context.deterministic.red_flags)

    def test_observed_facts_carry_quote_and_utterance(self) -> None:
        chest = next(
            s for s in self.context.observed.symptoms if "chest_pain" in s.ref
        )
        self.assertTrue(chest.utterance_id)
        self.assertIn("chest pain", chest.quote)

    def test_deterministic_values_are_copied_not_recomputed(self) -> None:
        top_context = self.context.deterministic.differential[0]
        top_engine = self.result.reasoning.candidate_diagnoses[0]
        self.assertEqual(top_context.ref, f"dx:{top_engine.disease_id}")
        self.assertEqual(top_context.detail["confidence"], top_engine.confidence)
        self.assertEqual(top_context.detail["rank"], top_engine.rank)

    def test_every_fact_is_citable(self) -> None:
        refs = set(self.context.references)
        self.assertTrue(refs)
        for group in (
            self.context.observed.symptoms,
            self.context.deterministic.differential,
            self.context.deterministic.red_flags,
            self.context.deterministic.investigations,
        ):
            for fact in group:
                self.assertIn(fact.ref, refs)

    def test_references_are_identifier_based_not_positional(self) -> None:
        """A citation must not change meaning when the differential re-ranks."""
        for fact in self.context.deterministic.differential:
            self.assertTrue(fact.ref.startswith("dx:"))
            self.assertNotIn("0", fact.ref.split(":")[1][:1])

    def test_the_context_is_frozen(self) -> None:
        with self.assertRaises(Exception):
            self.context.consultation_id = "somewhere-else"  # type: ignore[misc]

    def test_building_the_context_is_deterministic(self) -> None:
        self.assertEqual(
            build_context(self.result).model_dump(),
            build_context(self.result).model_dump(),
        )


# ══════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════


class TestValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = build_context(cardiac_result("val-1"))

    def accepted(self, *proposals: Proposal) -> List[Proposal]:
        return validate_proposals(list(proposals), self.context).accepted

    def rejected_code(self, p: Proposal) -> str:
        report = validate_proposals([p], self.context)
        self.assertEqual(len(report.rejected), 1, "expected a rejection")
        return report.rejected[0].code

    def test_a_grounded_question_is_accepted(self) -> None:
        self.assertEqual(len(self.accepted(proposal())), 1)

    def test_an_ungrounded_proposal_is_rejected(self) -> None:
        self.assertEqual(self.rejected_code(proposal(grounded_in=[])), "UNGROUNDED")

    def test_a_proposal_citing_something_absent_is_rejected(self) -> None:
        """The core anti-fabrication rule."""
        self.assertEqual(
            self.rejected_code(proposal(grounded_in=["symptom:symptom_haemoptysis"])),
            "UNKNOWN_REFERENCE",
        )

    def test_empty_content_is_rejected(self) -> None:
        self.assertEqual(self.rejected_code(proposal(content="   ")), "EMPTY_CONTENT")

    def test_a_runaway_generation_is_rejected(self) -> None:
        self.assertEqual(
            self.rejected_code(proposal(content="x" * 5000)), "CONTENT_TOO_LONG"
        )

    def test_citing_everything_is_rejected(self) -> None:
        self.assertEqual(
            self.rejected_code(proposal(grounded_in=self.context.references[:50])),
            "TOO_MANY_REFERENCES",
        )

    def test_duplicate_proposal_ids_are_rejected(self) -> None:
        report = validate_proposals([proposal(), proposal()], self.context)
        self.assertEqual(len(report.accepted), 1)
        self.assertEqual(report.rejection_codes, ["DUPLICATE_ID"])

    # ── the investigation rule ────────────────────────────────────────────
    def test_an_investigation_note_must_name_its_subject(self) -> None:
        self.assertEqual(
            self.rejected_code(proposal(kind=ProposalKind.INVESTIGATION_NOTE)),
            "MISSING_SUBJECT",
        )

    def test_a_note_annotating_a_recommended_investigation_is_accepted(self) -> None:
        ref = self.context.deterministic.investigations[0].ref
        accepted = self.accepted(
            proposal(
                kind=ProposalKind.INVESTIGATION_NOTE,
                about_ref=ref,
                grounded_in=[ref],
                content="Worth doing before the patient leaves.",
            )
        )
        self.assertEqual(len(accepted), 1)

    def test_a_model_cannot_introduce_an_investigation(self) -> None:
        """
        The reconciliation rule, enforced.

        Deterministic output defines the set; a model may annotate a member of
        it and nothing else.
        """
        self.assertEqual(
            self.rejected_code(
                proposal(
                    kind=ProposalKind.INVESTIGATION_NOTE,
                    about_ref="inv:whole_body_mri",
                    grounded_in=["symptom:symptom_chest_pain"],
                )
            ),
            "UNKNOWN_INVESTIGATION",
        )

    def test_a_note_pointing_at_the_wrong_kind_is_rejected(self) -> None:
        self.assertEqual(
            self.rejected_code(
                proposal(
                    kind=ProposalKind.INVESTIGATION_NOTE,
                    about_ref="dx:acs",
                    grounded_in=["dx:acs"],
                )
            ),
            "WRONG_SUBJECT_KIND",
        )

    def test_a_diagnosis_consideration_is_allowed_but_stays_in_its_channel(self) -> None:
        accepted = self.accepted(
            proposal(
                kind=ProposalKind.DIAGNOSIS_CONSIDERATION,
                content="Consider aortic dissection given the radiation.",
                grounded_in=["symptom:symptom_chest_pain"],
            )
        )
        self.assertEqual(len(accepted), 1)
        # It is a proposal, not a candidate: it has no rank and no confidence
        # that could place it alongside scored output.
        self.assertFalse(hasattr(accepted[0], "rank"))
        self.assertFalse(hasattr(accepted[0], "disease_id"))

    def test_a_proposal_has_no_field_that_could_overrule_a_rule(self) -> None:
        """The contract prevents override by omission, not by policing."""
        fields = set(Proposal.model_fields)
        for forbidden in ("priority", "severity", "rank", "escalated_by_red_flag"):
            self.assertNotIn(forbidden, fields)

    def test_rejections_are_kept_with_a_reason(self) -> None:
        report = validate_proposals([proposal(grounded_in=[])], self.context)
        self.assertEqual(len(report.rejected), 1)
        self.assertTrue(report.rejected[0].reason)
        self.assertEqual(report.rejected[0].proposal.proposal_id, "p1")


# ══════════════════════════════════════════════════════════════════════════
# Gateway
# ══════════════════════════════════════════════════════════════════════════


class TestGateway(unittest.TestCase):
    def setUp(self) -> None:
        self.result = cardiac_result("gw-1")

    def test_the_null_provider_is_disabled_not_failed(self) -> None:
        outcome = IntelligenceGateway(NullIntelligenceProvider()).consult(self.result)
        self.assertEqual(outcome.status, IntelligenceStatus.DISABLED)
        self.assertEqual(outcome.proposals, [])
        self.assertFalse(outcome.has_proposals)

    def test_the_deterministic_provider_produces_grounded_proposals(self) -> None:
        outcome = IntelligenceGateway(DeterministicIntelligenceProvider()).consult(
            self.result
        )
        self.assertEqual(outcome.status, IntelligenceStatus.OK)
        self.assertTrue(outcome.proposals)
        self.assertEqual(outcome.rejected, [])
        for p in outcome.proposals:
            self.assertEqual(p.proposal_version, PROPOSAL_VERSION)
            self.assertTrue(p.grounded_in)

    def test_a_provider_that_raises_does_not_break_the_consultation(self) -> None:
        provider = ScriptedProvider(None, raises=RuntimeError("model exploded"))
        outcome = IntelligenceGateway(provider).consult(self.result)

        self.assertEqual(outcome.status, IntelligenceStatus.FAILED)
        self.assertIn("model exploded", outcome.message)
        self.assertEqual(outcome.proposals, [])

    def test_a_provider_that_times_out_is_reported_not_swallowed(self) -> None:
        provider = ScriptedProvider(None, raises=TimeoutError("no answer in time"))
        outcome = IntelligenceGateway(provider).consult(self.result)
        self.assertEqual(outcome.status, IntelligenceStatus.FAILED)
        self.assertIn("TimeoutError", outcome.message)

    def test_a_provider_returning_the_wrong_type_is_a_failure_not_a_crash(self) -> None:
        outcome = IntelligenceGateway(ScriptedProvider({"proposals": []})).consult(
            self.result
        )
        self.assertEqual(outcome.status, IntelligenceStatus.FAILED)
        self.assertIn("ProviderResponse", outcome.message)

    def test_unavailable_is_distinct_from_failed(self) -> None:
        provider = ScriptedProvider(
            ProviderResponse(
                status=IntelligenceStatus.UNAVAILABLE,
                provider="scripted",
                provider_version="0.0.1",
                message="no route to model",
            )
        )
        outcome = IntelligenceGateway(provider).consult(self.result)
        self.assertEqual(outcome.status, IntelligenceStatus.UNAVAILABLE)

    def test_invalid_proposals_are_filtered_before_the_caller_sees_them(self) -> None:
        provider = ScriptedProvider(
            ProviderResponse(
                status=IntelligenceStatus.OK,
                provider="scripted",
                provider_version="0.0.1",
                proposals=[
                    proposal(proposal_id="good"),
                    proposal(proposal_id="bad", grounded_in=["symptom:invented"]),
                ],
            )
        )
        outcome = IntelligenceGateway(provider).consult(self.result)

        self.assertEqual([p.proposal_id for p in outcome.proposals], ["good"])
        self.assertEqual([r.code for r in outcome.rejected], ["UNKNOWN_REFERENCE"])

    def test_the_outcome_names_the_picture_it_was_made_against(self) -> None:
        outcome = IntelligenceGateway(DeterministicIntelligenceProvider()).consult(
            self.result
        )
        self.assertEqual(outcome.fingerprint, self.result.fingerprint)
        self.assertEqual(outcome.sequence, self.result.sequence)

    def test_a_budget_overrun_is_reported_and_the_result_still_returned(self) -> None:
        ticks = iter([0.0, 5.0])  # 5 seconds
        gateway = IntelligenceGateway(
            DeterministicIntelligenceProvider(),
            budget_ms=100,
            clock=lambda: next(ticks),
        )
        outcome = gateway.consult(self.result)

        self.assertTrue(outcome.over_budget)
        self.assertEqual(outcome.status, IntelligenceStatus.OK)
        self.assertTrue(outcome.proposals, "a late answer is still an answer")

    def test_elapsed_time_is_recorded(self) -> None:
        outcome = IntelligenceGateway(DeterministicIntelligenceProvider()).consult(
            self.result
        )
        self.assertGreaterEqual(outcome.elapsed_ms, 0.0)


# ══════════════════════════════════════════════════════════════════════════
# The layer cannot modify anything
# ══════════════════════════════════════════════════════════════════════════


class TestTheLayerCannotModifyAnything(unittest.TestCase):
    """The guarantee the whole boundary exists to provide."""

    def setUp(self) -> None:
        self.result = cardiac_result("immutable-1")

    def test_consulting_a_provider_leaves_the_clinical_result_identical(self) -> None:
        before = self.result.model_dump_json()
        IntelligenceGateway(DeterministicIntelligenceProvider()).consult(self.result)
        self.assertEqual(self.result.model_dump_json(), before)

    def test_a_hostile_provider_cannot_reach_the_clinical_state(self) -> None:
        class Hostile:
            name = "hostile"
            version = "0.0.1"

            def propose(self, context: CopilotContext) -> ProviderResponse:
                # Try every route a provider might take to the real objects.
                assert not hasattr(context, "clinical_state")
                assert not hasattr(context, "reasoning")
                assert not hasattr(context, "_result")
                try:
                    context.deterministic.red_flags.clear()
                except Exception:  # noqa: BLE001 - a frozen model may refuse
                    pass
                return ProviderResponse(
                    status=IntelligenceStatus.OK,
                    provider=self.name,
                    provider_version=self.version,
                )

        before = self.result.model_dump_json()
        outcome = IntelligenceGateway(Hostile()).consult(self.result)

        self.assertEqual(outcome.status, IntelligenceStatus.OK)
        self.assertEqual(self.result.model_dump_json(), before)
        # The rule output is intact even though the provider tried to clear it.
        self.assertTrue(self.result.reasoning.red_flags)

    def test_the_provider_receives_a_copy_not_the_result(self) -> None:
        provider = ScriptedProvider(
            ProviderResponse(
                status=IntelligenceStatus.OK, provider="scripted", provider_version="0.0.1"
            )
        )
        IntelligenceGateway(provider).consult(self.result)

        context = provider.contexts[0]
        self.assertIsInstance(context, CopilotContext)
        self.assertIsNot(context, self.result)

    def test_a_second_consultation_is_unaffected_by_a_failure_in_the_first(self) -> None:
        other = cardiac_result("immutable-2")
        IntelligenceGateway(ScriptedProvider(None, raises=RuntimeError("boom"))).consult(
            self.result
        )
        outcome = IntelligenceGateway(DeterministicIntelligenceProvider()).consult(other)
        self.assertEqual(outcome.status, IntelligenceStatus.OK)


# ══════════════════════════════════════════════════════════════════════════
# Interchangeability and replay
# ══════════════════════════════════════════════════════════════════════════


class TestProvidersAreInterchangeable(unittest.TestCase):
    def test_both_shipped_providers_satisfy_the_protocol(self) -> None:
        for provider in (NullIntelligenceProvider(), DeterministicIntelligenceProvider()):
            self.assertIsInstance(provider, IntelligenceProvider)

    def test_swapping_the_provider_changes_nothing_structural(self) -> None:
        result = cardiac_result("swap-1")
        outcomes = [
            IntelligenceGateway(p).consult(result)
            for p in (NullIntelligenceProvider(), DeterministicIntelligenceProvider())
        ]
        for outcome in outcomes:
            self.assertEqual(outcome.consultation_id, result.consultation_id)
            self.assertEqual(outcome.fingerprint, result.fingerprint)

    def test_the_deterministic_provider_is_replay_stable(self) -> None:
        """
        Proposal ids must survive a replay.

        The runtime rebuilds state by replaying events; if this boundary minted
        a fresh uuid per call, the same consultation would produce different
        proposal ids each time and nothing downstream could deduplicate them.
        """
        first = IntelligenceGateway(DeterministicIntelligenceProvider()).consult(
            cardiac_result("replay-a")
        )
        second = IntelligenceGateway(DeterministicIntelligenceProvider()).consult(
            cardiac_result("replay-b")
        )
        self.assertEqual(
            [p.proposal_id for p in first.proposals],
            [p.proposal_id for p in second.proposals],
        )
        self.assertEqual(
            [p.content for p in first.proposals], [p.content for p in second.proposals]
        )


if __name__ == "__main__":
    unittest.main()
