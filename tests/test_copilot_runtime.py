"""
The push-mode runtime.

Everything here goes through `CopilotRuntime` — never through the clinical
engine directly — because the point is to prove the inversion works, not to
re-prove the engine. The golden consultation is driven end to end and must
reach the same conclusions the already-validated contract test reached.

Offline by construction: `run_llm` defaults to False inside the runtime and the
graph is disabled before import, so no provider, database or paid API is
touched.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("CLINICAL_COPILOT_ENABLE_GRAPH", "False")
os.environ.setdefault("CLINICAL_COPILOT_ENABLE_LLM", "False")

from integration.copilot_event import (  # noqa: E402
    CopilotEvent,
    Provenance,
    SpeakerRole,
    UtterancePayload,
)
from integration.copilot_runtime import (  # noqa: E402
    CopilotResult,
    CopilotRuntime,
    SessionClosedError,
    SessionExpiredError,
    SessionStatus,
    StaleEventError,
    UnknownSessionError,
)
from integration.doctor_adapter import (  # noqa: E402
    UnmappableSpeakerError,
    utterance_to_copilot_event,
)

FIXTURE = Path(__file__).parent / "fixtures" / "golden_consultation.json"


def load_golden() -> Dict[str, Any]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)


def golden_events(consultation_id: str | None = None) -> List[CopilotEvent]:
    golden = load_golden()
    cid = consultation_id or golden["consultationId"]
    return [
        utterance_to_copilot_event(u, consultation_id=cid)
        for u in sorted(golden["utterances"], key=lambda u: u["sequence"])
    ]


#: One clinically distinct utterance per sequence. The ordering tests assert on
#: which results arrive and when, so each event must genuinely move the clinical
#: picture — otherwise throttling would suppress it and the test would be
#: measuring change detection instead of ordering.
DISTINCT_UTTERANCES = {
    1: "I have severe chest pain.",
    2: "I also feel short of breath.",
    3: "I feel nauseous as well.",
    4: "I take aspirin daily.",
    5: "I have high blood pressure.",
    6: "I also have diabetes.",
}


def event(
    consultation_id: str,
    sequence: int,
    text: str | None = None,
    *,
    event_id: str | None = None,
    is_final: bool = True,
    role: SpeakerRole = SpeakerRole.PATIENT,
) -> CopilotEvent:
    """A minimal hand-built event, for the ordering and lifecycle cases."""
    return CopilotEvent(
        event_id=event_id or f"utt-{sequence:03d}",
        consultation_id=consultation_id,
        sequence=sequence,
        provenance=Provenance.FIXTURE,
        payload=UtterancePayload(
            text=text if text is not None else DISTINCT_UTTERANCES.get(sequence, f"Utterance {sequence}."),
            speaker_role=role,
            speaker_label=role.value.title(),
            is_final=is_final,
            confidence=0.95,
            started_at="2026-09-27T09:00:00.000Z",
            finalized_at="2026-09-27T09:00:04.000Z" if is_final else None,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
# Lifecycle
# ══════════════════════════════════════════════════════════════════════════


class TestSessionEviction(unittest.TestCase):
    """
    Bounding what the runtime holds.

    Eviction is swept lazily on ingest rather than by a background worker, and
    the clock is injected so none of this depends on a real sleep.
    """

    def setUp(self) -> None:
        self.now = 1000.0
        self.results: List[CopilotResult] = []
        self.runtime = CopilotRuntime(
            self.results.append,
            idle_timeout_seconds=60.0,
            clock=lambda: self.now,
        )

    def advance(self, seconds: float) -> None:
        self.now += seconds

    # ── explicit close ────────────────────────────────────────────────────
    def test_close_removes_the_session(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        self.assertEqual(self.runtime.session_count, 1)

        self.runtime.close_session("c-1")
        self.assertEqual(self.runtime.session_count, 0)

    def test_an_event_after_close_is_refused_not_re_opened(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        self.runtime.close_session("c-1")

        with self.assertRaises(SessionClosedError):
            self.runtime.ingest(event("c-1", 2))
        # Crucially, no second consultation was started under the same id.
        self.assertEqual(self.runtime.session_count, 0)

    # ── idle expiry ───────────────────────────────────────────────────────
    def test_an_idle_session_expires(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        self.advance(61)

        self.assertEqual(self.runtime.evict_idle(), ["c-1"])
        self.assertEqual(self.runtime.session_count, 0)
        self.assertEqual(self.runtime.expired_count, 1)

    def test_a_session_inside_the_window_survives(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        self.advance(59)

        self.assertEqual(self.runtime.evict_idle(), [])
        self.assertEqual(self.runtime.session_count, 1)

    def test_activity_resets_the_timeout(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        self.advance(50)
        self.runtime.ingest(event("c-1", 2))  # activity
        self.advance(50)  # 100s since the first event, 50s since the last

        self.assertEqual(self.runtime.evict_idle(), [])
        self.assertEqual(self.runtime.session_count, 1)

    def test_expiry_is_swept_on_the_next_ingest(self) -> None:
        """No worker: the sweep happens on the path that is already running."""
        self.runtime.ingest(event("c-1", 1))
        self.advance(61)

        # An event for a different consultation drives the sweep.
        self.runtime.ingest(event("c-2", 1))
        self.assertEqual(self.runtime.expired_count, 1)
        self.assertEqual(self.runtime.active_sessions, ["c-2"])

    def test_an_event_for_an_expired_session_is_refused(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        self.advance(61)

        with self.assertRaises(SessionExpiredError):
            self.runtime.ingest(event("c-1", 2))
        # It is not silently resurrected as a fresh consultation.
        self.assertEqual(self.runtime.session_count, 0)

    def test_an_expired_session_cannot_be_reopened(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        self.advance(61)
        self.runtime.evict_idle()

        with self.assertRaises(SessionExpiredError):
            self.runtime.open_session("c-1")

    def test_expiry_is_distinguishable_from_a_close(self) -> None:
        """
        Two different situations, two different errors.

        Nobody decided an expired consultation was over, so a caller can tell
        "this is finished" from "the runtime gave up and you should replay".
        """
        self.runtime.ingest(event("c-1", 1))
        self.runtime.ingest(event("c-2", 1))
        self.runtime.close_session("c-1")
        self.advance(61)
        self.runtime.evict_idle()

        with self.assertRaises(SessionClosedError):
            self.runtime.get_session("c-1")
        with self.assertRaises(SessionExpiredError):
            self.runtime.get_session("c-2")

    # ── isolation ─────────────────────────────────────────────────────────
    def test_one_session_expiring_leaves_the_other_working(self) -> None:
        self.runtime.ingest(event("A", 1, "I have severe chest pain."))
        self.advance(40)
        self.runtime.ingest(event("B", 1, "I have a headache."))
        self.advance(40)  # A idle 80s, B idle 40s

        self.assertEqual(self.runtime.evict_idle(), ["A"])
        self.assertEqual(self.runtime.active_sessions, ["B"])

        # B still advances, with its own state intact.
        self.runtime.ingest(event("B", 2, "I also feel nauseous."))
        self.assertEqual(self.runtime.get_session("B").final_count, 2)

    def test_an_expired_session_does_not_take_anothers_state_with_it(self) -> None:
        self.runtime.ingest(event("A", 1, "I have severe chest pain."))
        self.advance(40)
        self.runtime.ingest(event("B", 1, "I have severe chest pain."))
        self.advance(40)
        self.runtime.evict_idle()

        names = {s["name"] for s in self.runtime.get_session("B").current_state().symptoms}
        self.assertIn("symptom_chest_pain", names)

    # ── tombstones stay bounded too ───────────────────────────────────────
    def test_tombstones_are_capped(self) -> None:
        runtime = CopilotRuntime(idle_timeout_seconds=60.0, retired_capacity=3)
        for i in range(5):
            runtime.ingest(event(f"c-{i}", 1))
            runtime.close_session(f"c-{i}")

        self.assertEqual(runtime.retired_ids, ["c-2", "c-3", "c-4"])
        # The oldest tombstone has been dropped, so that id is merely unknown.
        with self.assertRaises(UnknownSessionError):
            runtime.get_session("c-0")

    def test_eviction_can_be_switched_off(self) -> None:
        runtime = CopilotRuntime(idle_timeout_seconds=0, clock=lambda: self.now)
        runtime.ingest(event("c-1", 1))
        self.advance(10_000)

        self.assertEqual(runtime.evict_idle(), [])
        self.assertEqual(runtime.session_count, 1)

    def test_the_default_timeout_does_not_evict_a_live_consultation(self) -> None:
        """Thirty minutes is longer than any gap inside a real consultation."""
        runtime = CopilotRuntime(clock=lambda: self.now)
        runtime.ingest(event("c-1", 1))
        self.advance(29 * 60)

        self.assertEqual(runtime.evict_idle(), [])
        self.advance(2 * 60)
        self.assertEqual(runtime.evict_idle(), ["c-1"])


class TestSessionLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.results: List[CopilotResult] = []
        self.runtime = CopilotRuntime(self.results.append)

    def test_a_new_session_opens_active(self) -> None:
        session = self.runtime.open_session("c-1")
        self.assertEqual(session.status, SessionStatus.ACTIVE)
        self.assertEqual(session.next_expected_sequence, 1)
        self.assertIn("c-1", self.runtime.active_sessions)

    def test_opening_twice_returns_the_same_session(self) -> None:
        first = self.runtime.open_session("c-1")
        self.assertIs(self.runtime.open_session("c-1"), first)

    def test_an_event_opens_a_session_when_none_exists(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        self.assertEqual(self.runtime.get_session("c-1").status, SessionStatus.ACTIVE)

    def test_auto_open_can_be_refused(self) -> None:
        with self.assertRaises(UnknownSessionError):
            self.runtime.ingest(event("c-1", 1), auto_open=False)

    def test_closing_releases_the_session(self) -> None:
        session = self.runtime.open_session("c-1")
        self.runtime.close_session("c-1")

        # The object is marked closed, and the runtime no longer holds it.
        self.assertEqual(session.status, SessionStatus.CLOSED)
        self.assertNotIn("c-1", self.runtime.active_sessions)
        self.assertEqual(self.runtime.session_count, 0)
        # A tombstone remains, so asking for it says "closed", not "unknown".
        with self.assertRaises(SessionClosedError):
            self.runtime.get_session("c-1")

    def test_events_after_close_are_refused(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        self.runtime.close_session("c-1")
        with self.assertRaises(SessionClosedError):
            self.runtime.ingest(event("c-1", 2))

    def test_a_closed_session_cannot_be_reopened(self) -> None:
        self.runtime.open_session("c-1")
        self.runtime.close_session("c-1")
        with self.assertRaises(SessionClosedError):
            self.runtime.open_session("c-1")

    def test_closing_an_unknown_session_is_an_error(self) -> None:
        with self.assertRaises(UnknownSessionError):
            self.runtime.close_session("never-opened")

    def test_closing_drops_events_still_waiting_on_a_gap(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        self.runtime.ingest(event("c-1", 3))  # held
        session = self.runtime.get_session("c-1")
        self.assertEqual(session.pending_sequences, [3])

        self.runtime.close_session("c-1")
        # Applying it now would build a state the consultation never passed through.
        self.assertEqual(session.pending_sequences, [])


# ══════════════════════════════════════════════════════════════════════════
# Ordering
# ══════════════════════════════════════════════════════════════════════════


class TestSequenceHandling(unittest.TestCase):
    """
    Sequence is the ordering authority; arrival order is not.

    The clinical state accumulates, so a gap is buffered rather than skipped.
    A *different* event claiming an already-consumed sequence is an error,
    because silently dropping it would hide an inconsistent stream.
    """

    def setUp(self) -> None:
        self.results: List[CopilotResult] = []
        self.runtime = CopilotRuntime(self.results.append)

    def test_ordered_delivery_1_2_3(self) -> None:
        for seq in (1, 2, 3):
            self.runtime.ingest(event("c-1", seq))
        self.assertEqual([r.sequence for r in self.results], [1, 2, 3])
        self.assertEqual(self.runtime.get_session("c-1").next_expected_sequence, 4)

    def test_out_of_order_delivery_1_3_2_is_reordered_not_dropped(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        self.assertEqual([r.sequence for r in self.results], [1])

        self.runtime.ingest(event("c-1", 3))
        # Held, not processed: 2 has not arrived.
        self.assertEqual([r.sequence for r in self.results], [1])
        self.assertEqual(self.runtime.get_session("c-1").pending_sequences, [3])

        released = self.runtime.ingest(event("c-1", 2))
        # 2 closes the gap and releases 3 with it, in order.
        self.assertEqual([r.sequence for r in released], [2, 3])
        self.assertEqual([r.sequence for r in self.results], [1, 2, 3])
        self.assertEqual(self.runtime.get_session("c-1").pending_sequences, [])

    def test_duplicate_delivery_1_2_2_3(self) -> None:
        for seq in (1, 2, 2, 3):
            self.runtime.ingest(event("c-1", seq))
        self.assertEqual([r.sequence for r in self.results], [1, 2, 3])

    def test_a_repeated_event_produces_no_second_result(self) -> None:
        self.runtime.ingest(event("c-1", 1))
        again = self.runtime.ingest(event("c-1", 1))
        self.assertEqual(again, [])

    def test_a_different_event_reusing_a_consumed_sequence_is_refused(self) -> None:
        self.runtime.ingest(event("c-1", 1, event_id="utt-001"))
        self.runtime.ingest(event("c-1", 2, event_id="utt-002"))
        with self.assertRaises(StaleEventError):
            self.runtime.ingest(event("c-1", 1, event_id="utt-999"))

    def test_a_long_gap_releases_everything_at_once(self) -> None:
        for seq in (1, 5, 4, 3):
            self.runtime.ingest(event("c-1", seq))
        self.assertEqual([r.sequence for r in self.results], [1])

        released = self.runtime.ingest(event("c-1", 2))
        self.assertEqual([r.sequence for r in released], [2, 3, 4, 5])

    def test_a_misrouted_event_is_refused(self) -> None:
        session = self.runtime.open_session("c-1")
        with self.assertRaises(Exception):
            session.ingest(event("c-other", 1))


# ══════════════════════════════════════════════════════════════════════════
# Partial / final
# ══════════════════════════════════════════════════════════════════════════


class TestPartialAndFinal(unittest.TestCase):
    """
    The Doctor platform reuses one id across PARTIAL and FINAL.

    Only FINAL reaches the clinical state. `update_state_from_event` has no
    such guard of its own — verified — so the runtime applies it. Without that,
    a partial's text would be extracted and never retracted when the final
    turned out to say something different.
    """

    def setUp(self) -> None:
        self.results: List[CopilotResult] = []
        self.runtime = CopilotRuntime(self.results.append)

    def test_a_partial_produces_no_result(self) -> None:
        self.runtime.ingest(event("c-1", 1, "I have chest", is_final=False))
        self.assertEqual(self.results, [])

    def test_a_partial_is_visible_as_live_text(self) -> None:
        self.runtime.ingest(event("c-1", 1, "I have chest", is_final=False))
        self.assertEqual(
            self.runtime.get_session("c-1").live_partials, {"utt-001": "I have chest"}
        )

    def test_partial_then_final_is_one_utterance_not_two(self) -> None:
        self.runtime.ingest(
            event("c-1", 1, "I have chest", event_id="abc", is_final=False)
        )
        self.runtime.ingest(
            event("c-1", 1, "I have severe chest pain.", event_id="abc", is_final=True)
        )

        self.assertEqual(len(self.results), 1)
        session = self.runtime.get_session("c-1")
        self.assertEqual(session.final_count, 1)
        # The partial is cleared by its own final.
        self.assertEqual(session.live_partials, {})

    def test_the_partial_text_never_reaches_the_clinical_state(self) -> None:
        """
        The regression that matters.

        The partial mentions nausea; the final does not. If the partial were
        applied, nausea would be in the state permanently with nothing to
        remove it.
        """
        self.runtime.ingest(
            event("c-1", 1, "I feel nauseous and", event_id="abc", is_final=False)
        )
        self.runtime.ingest(
            event("c-1", 1, "I have severe chest pain.", event_id="abc", is_final=True)
        )

        names = {s["name"] for s in self.results[0].clinical_state.symptoms}
        self.assertIn("symptom_chest_pain", names)
        self.assertNotIn("symptom_nausea", names)

    def test_successive_partials_update_the_live_text(self) -> None:
        """
        The platform emits several partials per utterance as speech settles.

        Deduplicating them would freeze the display on the first few words, so
        a partial overwrites rather than being refused.
        """
        for text in ("I have", "I have severe", "I have severe chest"):
            self.runtime.ingest(event("c-1", 1, text, event_id="abc", is_final=False))

        session = self.runtime.get_session("c-1")
        self.assertEqual(session.live_partials, {"abc": "I have severe chest"})
        self.assertEqual(self.results, [])
        # The utterance has not settled, so the sequence has not advanced.
        self.assertEqual(session.next_expected_sequence, 1)

    def test_partials_then_final_then_the_next_utterance(self) -> None:
        self.runtime.ingest(event("c-1", 1, "I have", event_id="a", is_final=False))
        self.runtime.ingest(
            event("c-1", 1, "I have severe chest pain.", event_id="a", is_final=True)
        )
        self.runtime.ingest(event("c-1", 2, "I also feel nauseous.", event_id="b"))

        self.assertEqual([r.sequence for r in self.results], [1, 2])
        self.assertEqual(self.runtime.get_session("c-1").next_expected_sequence, 3)

    def test_a_late_partial_for_a_settled_utterance_is_refused(self) -> None:
        self.runtime.ingest(event("c-1", 1, event_id="a", is_final=True))
        with self.assertRaises(StaleEventError):
            self.runtime.ingest(event("c-1", 1, "stale", event_id="a", is_final=False))

    def test_a_repeated_final_is_ignored(self) -> None:
        final = event("c-1", 1, event_id="abc", is_final=True)
        self.runtime.ingest(final)
        self.runtime.ingest(final)
        self.assertEqual(len(self.results), 1)
        self.assertEqual(self.runtime.get_session("c-1").final_count, 1)

    def test_state_is_not_accumulated_twice_by_a_repeated_final(self) -> None:
        final = event("c-1", 1, "I have severe chest pain.", event_id="abc")
        self.runtime.ingest(final)
        version_after_first = self.results[0].clinical_state.version
        self.runtime.ingest(final)
        self.assertEqual(
            self.runtime.get_session("c-1").current_state().version,
            version_after_first,
        )


# ══════════════════════════════════════════════════════════════════════════
# Isolation and validation
# ══════════════════════════════════════════════════════════════════════════


class TestIsolationAndValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.results: List[CopilotResult] = []
        self.runtime = CopilotRuntime(self.results.append)

    def test_two_consultations_keep_separate_state(self) -> None:
        self.runtime.ingest(event("A", 1, "I have severe chest pain."))
        self.runtime.ingest(event("B", 1, "I have a headache."))

        a = {s["name"] for s in self.runtime.get_session("A").current_state().symptoms}
        b = {s["name"] for s in self.runtime.get_session("B").current_state().symptoms}
        self.assertIn("symptom_chest_pain", a)
        self.assertNotIn("symptom_chest_pain", b)

    def test_a_failure_in_one_consultation_leaves_the_other_working(self) -> None:
        self.runtime.ingest(event("A", 1, "I have severe chest pain."))
        self.runtime.ingest(event("B", 1, "I have a headache."))

        # A is closed, then pushed to — A raises.
        self.runtime.close_session("A")
        with self.assertRaises(SessionClosedError):
            self.runtime.ingest(event("A", 2))

        # B is untouched and still advancing.
        self.runtime.ingest(event("B", 2, "I also feel nauseous."))
        self.assertEqual(self.runtime.get_session("B").next_expected_sequence, 3)
        self.assertEqual([r.consultation_id for r in self.results][-1], "B")

    def test_a_stale_event_in_one_consultation_does_not_disturb_another(self) -> None:
        self.runtime.ingest(event("A", 1))
        self.runtime.ingest(event("A", 2))
        self.runtime.ingest(event("B", 1))

        with self.assertRaises(StaleEventError):
            self.runtime.ingest(event("A", 1, event_id="different"))

        self.runtime.ingest(event("B", 2))
        self.assertEqual(self.runtime.get_session("B").final_count, 2)

    def test_an_invalid_speaker_is_refused_at_the_boundary(self) -> None:
        golden = load_golden()
        raw = dict(golden["utterances"][1], speaker="SPEAKER_00")
        with self.assertRaises(UnmappableSpeakerError):
            utterance_to_copilot_event(raw, consultation_id="c-1")

    def test_an_invalid_status_is_refused_at_the_boundary(self) -> None:
        golden = load_golden()
        broken = dict(golden["utterances"][1], status="SETTLING")
        with self.assertRaises(ValueError):
            utterance_to_copilot_event(broken, consultation_id="c-1")

    def test_a_missing_required_field_is_refused(self) -> None:
        golden = load_golden()
        broken = dict(golden["utterances"][1])
        del broken["sequence"]
        with self.assertRaises(KeyError):
            utterance_to_copilot_event(broken, consultation_id="c-1")

    def test_a_consumer_that_raises_does_not_break_ingestion(self) -> None:
        def explode(_result: CopilotResult) -> None:
            raise RuntimeError("the consumer blew up")

        runtime = CopilotRuntime(explode)
        runtime.ingest(event("c-1", 1, "I have severe chest pain."))
        # Clinical work survived the consumer's failure.
        self.assertEqual(runtime.get_session("c-1").final_count, 1)
        runtime.ingest(event("c-1", 2, "I also feel nauseous."))
        self.assertEqual(runtime.get_session("c-1").next_expected_sequence, 3)

    def test_the_runtime_works_without_any_consumer(self) -> None:
        runtime = CopilotRuntime()
        results = runtime.ingest(event("c-1", 1, "I have severe chest pain."))
        self.assertEqual(len(results), 1)


# ══════════════════════════════════════════════════════════════════════════
# Golden consultation, through push mode
# ══════════════════════════════════════════════════════════════════════════


class TestGoldenConsultationThroughPushMode(unittest.TestCase):
    """
    The same consultation, the same conclusions — but pushed.

    Nothing here calls the clinical engine directly. If these assertions hold,
    the inversion changed the direction without changing the medicine.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.results: List[CopilotResult] = []
        runtime = CopilotRuntime(cls.results.append)
        for e in golden_events():
            runtime.ingest(e)
        runtime.close_session(load_golden()["consultationId"])
        cls.final = cls.results[-1]

    def test_only_utterances_that_change_the_picture_emit(self) -> None:
        """
        Six FINAL utterances, five emissions.

        Sequence 4 is the doctor asking "do you take any regular medication,
        and do you have any medical conditions?" — a question that adds no
        clinical fact, so the conclusion is unchanged and nothing is sent.
        """
        self.assertEqual([r.sequence for r in self.results], [1, 2, 3, 5, 6])

    def test_every_result_names_the_utterance_that_produced_it(self) -> None:
        for result in self.results:
            self.assertTrue(result.event_id.startswith("utt-"))
            self.assertEqual(result.consultation_id, "golden-acs-001")

    # ── the validated golden expectations ─────────────────────────────────
    def test_demographics(self) -> None:
        info = self.final.clinical_state.patient_info
        self.assertEqual(info.age, 62)
        self.assertEqual(info.gender, "Male")

    def test_chief_complaint(self) -> None:
        self.assertIn("symptom_chest_pain", self.final.clinical_state.chief_complaint)

    def test_symptoms(self) -> None:
        names = {s["name"] for s in self.final.clinical_state.symptoms}
        self.assertIn("symptom_chest_pain", names)
        self.assertIn("symptom_shortness_of_breath", names)
        self.assertEqual(self.final.clinical_state.severity, "Severe")

    def test_negation(self) -> None:
        present = {
            s["name"]
            for s in self.final.clinical_state.symptoms
            if str(s.get("present")) == "True"
        }
        self.assertNotIn("symptom_fever", present)

    def test_history_and_medication(self) -> None:
        self.assertIn("Hypertension", self.final.clinical_state.medical_history)
        self.assertIn("Diabetes", self.final.clinical_state.medical_history)
        self.assertIn("Aspirin", self.final.clinical_state.current_medications)

    def test_clinical_state_is_versioned(self) -> None:
        self.assertGreater(self.final.clinical_state.version, 1)

    def test_provenance_traces_back_to_doctor_utterances(self) -> None:
        self.assertTrue(self.final.provenance)
        self.assertTrue(
            any(e["event_id"].startswith("utt-") for e in self.final.provenance.values())
        )

    def test_diagnosis_candidates(self) -> None:
        candidates = self.final.reasoning.candidate_diagnoses
        self.assertGreater(len(candidates), 0)
        ids = {c.disease_id for c in candidates}
        self.assertTrue({"unstable_angina", "acs"} & ids, f"got {ids}")
        self.assertEqual(
            [c.rank for c in candidates], list(range(1, len(candidates) + 1))
        )

    def test_red_flags(self) -> None:
        self.assertIn("CRITICAL", {f.severity for f in self.final.reasoning.red_flags})

    def test_risk_factors(self) -> None:
        ids = {r.factor_id for r in self.final.reasoning.risk_factors}
        self.assertIn("hypertension", ids)
        self.assertIn("diabetes", ids)

    def test_investigation_recommendations(self) -> None:
        investigations = self.final.reasoning.recommended_investigations
        ecg = next((i for i in investigations if i.investigation_id == "ecg_12lead"), None)
        self.assertIsNotNone(ecg)
        self.assertEqual(ecg.priority, "URGENT")
        ids = [i.investigation_id for i in investigations]
        self.assertEqual(len(ids), len(set(ids)))

    def test_missing_information(self) -> None:
        fields = {m.field for m in self.final.reasoning.missing_information}
        self.assertIn("blood_pressure", fields)

    def test_contraindication(self) -> None:
        self.assertTrue(self.final.reasoning.contraindications)

    def test_the_deterministic_set_is_not_touched_by_any_model(self) -> None:
        """
        Investigation authority stays with the rule engine.

        The runtime never enables enrichment, so this asserts the current
        guarantee: what the clinician sees came from a rule, not a model.
        """
        self.assertEqual(self.final.reasoning.llm_recommended_investigations, [])


# ══════════════════════════════════════════════════════════════════════════
# Replay
# ══════════════════════════════════════════════════════════════════════════


class TestDeterministicReplay(unittest.TestCase):
    """
    Restart safety without persistence.

    Every piece of runtime state is a projection of the ordered events, so
    replaying them rebuilds it exactly. That is what makes a database a later
    decision rather than a prerequisite.
    """

    @staticmethod
    def _run(consultation_id: str) -> CopilotResult:
        results: List[CopilotResult] = []
        runtime = CopilotRuntime(results.append)
        for e in golden_events(consultation_id):
            runtime.ingest(e)
        return results[-1]

    def test_replaying_the_same_events_reconstructs_the_same_state(self) -> None:
        first = self._run("replay-a")
        second = self._run("replay-b")

        self.assertEqual(
            first.clinical_state.model_dump(exclude={"session_id"}),
            second.clinical_state.model_dump(exclude={"session_id"}),
        )

    def test_replay_reconstructs_the_same_reasoning(self) -> None:
        first = self._run("replay-c")
        second = self._run("replay-d")

        self.assertEqual(
            [(c.disease_id, round(c.confidence, 6)) for c in first.reasoning.candidate_diagnoses],
            [(c.disease_id, round(c.confidence, 6)) for c in second.reasoning.candidate_diagnoses],
        )
        self.assertEqual(
            sorted(i.investigation_id for i in first.reasoning.recommended_investigations),
            sorted(i.investigation_id for i in second.reasoning.recommended_investigations),
        )
        self.assertEqual(
            sorted(f.flag_id for f in first.reasoning.red_flags),
            sorted(f.flag_id for f in second.reasoning.red_flags),
        )

    def test_shuffled_arrival_reconstructs_the_ordered_result(self) -> None:
        """Arrival order must not change the consultation."""
        ordered = self._run("replay-ordered")

        results: List[CopilotResult] = []
        runtime = CopilotRuntime(results.append)
        events = golden_events("replay-shuffled")
        for e in [events[0], events[2], events[1], events[4], events[3], events[5]]:
            runtime.ingest(e)

        self.assertEqual([r.sequence for r in results], [1, 2, 3, 5, 6])
        self.assertEqual(
            results[-1].clinical_state.model_dump(exclude={"session_id"}),
            ordered.clinical_state.model_dump(exclude={"session_id"}),
        )


if __name__ == "__main__":
    unittest.main()
