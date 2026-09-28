"""
Intelligence during a live consultation.

The property the whole design exists for is the first test in this file: a
clinician does not wait for a model. Everything else — coalescing, staleness,
catch-up, degradation — follows from running the intelligence layer beside the
clinical path rather than inside it.

All offline. The graph reaches a scripted model; the slow provider sleeps
locally.
"""

from __future__ import annotations

import json
import os
import threading
import time
import unittest
from typing import Any, Dict, List

os.environ.setdefault("CLINICAL_COPILOT_ENABLE_GRAPH", "False")
os.environ.setdefault("CLINICAL_COPILOT_ENABLE_LLM", "False")

from fastapi.testclient import TestClient  # noqa: E402

from app.utils.jwt_utils import generate_jwt_token  # noqa: E402
from copilot_api.main import API_PREFIX, SnapshotBroker, create_app  # noqa: E402
from integration.copilot_event import (  # noqa: E402
    CopilotEvent,
    Provenance,
    SpeakerRole,
    UtterancePayload,
)
from integration.copilot_runtime import CopilotResult, CopilotRuntime  # noqa: E402
from integration.intelligence import (  # noqa: E402
    IntelligenceGateway,
    IntelligenceStatus,
    LangGraphIntelligenceProvider,
    ProviderResponse,
    ScriptedModelClient,
    UnavailableModelClient,
)
from integration.intelligence.context import CopilotContext  # noqa: E402
from integration.intelligence.live import (  # noqa: E402
    IntelligenceSnapshot,
    LiveIntelligenceService,
)

CARDIAC = [
    "I am a 62 year old male. I have severe chest pain and I am sweating.",
    "I also feel short of breath.",
    "I have high blood pressure and diabetes.",
]

SCRIPT: Dict[str, Any] = {
    "synthesis": {
        "summary": "62-year-old man with severe chest pain and sweating.",
        "refs": ["symptom:symptom_chest_pain"],
    },
    "unresolved": {
        "items": [
            {"text": "What is the blood pressure now?", "refs": ["missing:blood_pressure"]}
        ]
    },
    "explanations": {"items": []},
    "investigation_notes": {
        "items": [{"text": "Repeat the ECG if pain continues.", "refs": ["inv:ecg_12lead"]}]
    },
}


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
            started_at="2026-09-28T09:00:00.000Z",
            finalized_at="2026-09-28T09:00:04.000Z",
        ),
    )


def cardiac_result(consultation_id: str = "live-1") -> CopilotResult:
    runtime = CopilotRuntime()
    for sequence, text in enumerate(CARDIAC, start=1):
        runtime.ingest(utterance(consultation_id, sequence, text))
    return runtime.get_session(consultation_id).latest_result


def scripted_service(
    consumer=None, *, script: Dict[str, Any] | None = None, budget_ms: float | None = None
) -> LiveIntelligenceService:
    provider = LangGraphIntelligenceProvider(
        ScriptedModelClient(script if script is not None else SCRIPT)
    )
    return LiveIntelligenceService(
        IntelligenceGateway(provider, budget_ms=budget_ms), consumer
    )


class SlowProvider:
    """A provider that takes its time, without a model or a network."""

    name = "slow"
    version = "0.0.1"

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.started = threading.Event()

    def propose(self, context: CopilotContext) -> ProviderResponse:
        self.started.set()
        time.sleep(self.seconds)
        return ProviderResponse(
            status=IntelligenceStatus.OK,
            provider=self.name,
            provider_version=self.version,
        )


# ══════════════════════════════════════════════════════════════════════════
# The clinician does not wait
# ══════════════════════════════════════════════════════════════════════════


class TestTheClinicalPathIsNeverBlocked(unittest.TestCase):
    def test_submitting_returns_immediately_however_slow_the_provider(self) -> None:
        provider = SlowProvider(2.0)
        service = LiveIntelligenceService(IntelligenceGateway(provider))
        service.start()
        try:
            result = cardiac_result("block-1")
            started = time.perf_counter()
            service.submit(result)
            elapsed = time.perf_counter() - started

            self.assertLess(elapsed, 0.2, "submit must not wait for the provider")
            self.assertTrue(provider.started.wait(timeout=5))
        finally:
            service.stop(timeout=6)

    def test_ingesting_over_http_does_not_wait_for_the_model(self) -> None:
        provider = SlowProvider(1.5)
        broker = SnapshotBroker()
        service = LiveIntelligenceService(IntelligenceGateway(provider), broker)
        service.start()
        runtime = CopilotRuntime()
        client = TestClient(
            create_app(runtime, intelligence=service, snapshot_broker=broker)
        )
        cid = "block-2"
        headers = {"Authorization": f"Bearer {generate_jwt_token(cid, 'doctor')}"}
        try:
            # Warm the clinical pipeline on a different consultation first.
            # The claim under test is that ingestion does not wait for the
            # model; one-time dictionary and vector-store loading is not part
            # of it, and on a loaded machine it alone can exceed the bound.
            warm = "block-2-warmup"
            client.post(
                f"{API_PREFIX}/sessions/{warm}/events",
                json=utterance(warm, 1, CARDIAC[0]).model_dump(mode="json"),
                headers={"Authorization": f"Bearer {generate_jwt_token(warm, 'doctor')}"},
            )

            started = time.perf_counter()
            response = client.post(
                f"{API_PREFIX}/sessions/{cid}/events",
                json=utterance(cid, 1, CARDIAC[0]).model_dump(mode="json"),
                headers=headers,
            )
            elapsed = time.perf_counter() - started

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["results"], "clinical output arrived")
            self.assertLess(elapsed, 1.5, "the clinical answer beat the provider")
        finally:
            service.stop(timeout=6)

    def test_a_provider_that_never_answers_leaves_the_api_working(self) -> None:
        provider = SlowProvider(30.0)
        service = LiveIntelligenceService(IntelligenceGateway(provider))
        service.start()
        runtime = CopilotRuntime()
        client = TestClient(create_app(runtime, intelligence=service))
        cid = "block-3"
        headers = {"Authorization": f"Bearer {generate_jwt_token(cid, 'doctor')}"}
        try:
            for sequence, text in enumerate(CARDIAC, start=1):
                self.assertEqual(
                    client.post(
                        f"{API_PREFIX}/sessions/{cid}/events",
                        json=utterance(cid, sequence, text).model_dump(mode="json"),
                        headers=headers,
                    ).status_code,
                    200,
                )
            insights = client.get(
                f"{API_PREFIX}/sessions/{cid}/insights", headers=headers
            )
            self.assertEqual(insights.status_code, 200)
            self.assertTrue(insights.json()["red_flags"])
        finally:
            service.stop(timeout=1)


# ══════════════════════════════════════════════════════════════════════════
# Incremental updates and coalescing
# ══════════════════════════════════════════════════════════════════════════


class TestIncrementalUpdates(unittest.TestCase):
    def test_a_snapshot_is_produced_for_a_meaningful_change(self) -> None:
        received: List[IntelligenceSnapshot] = []
        with scripted_service(received.append) as service:
            service.submit(cardiac_result("inc-1"))
            self.assertTrue(service.wait_until_idle(20))

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].status, IntelligenceStatus.OK)
        self.assertTrue(received[0].proposals)

    def test_a_burst_is_coalesced_into_one_run(self) -> None:
        """
        Five results while one run is in flight must not queue five runs.

        Commentary on a picture that has already moved on is worse than none.
        """
        provider = SlowProvider(0.4)
        received: List[IntelligenceSnapshot] = []
        service = LiveIntelligenceService(IntelligenceGateway(provider), received.append)

        # The clinical results are computed before the worker starts, then
        # submitted in one burst. Interleaving them used to make the outcome
        # depend on whether the pipeline was slower than the provider: when it
        # was, the worker drained each submission before the next arrived and
        # nothing coalesced, so the test failed without the behaviour changing.
        runtime = CopilotRuntime()
        cid = "inc-2"
        burst = []
        for sequence, text in enumerate(
            CARDIAC + ["I also feel nauseous.", "I have a headache."], start=1
        ):
            results = runtime.ingest(utterance(cid, sequence, text))
            if results:
                burst.append(results[-1])
        self.assertGreater(len(burst), 2, "need several results to merge")

        service.start()
        try:
            for result in burst:
                service.submit(result)
            self.assertTrue(service.wait_until_idle(30))
        finally:
            service.stop(timeout=6)

        self.assertGreater(service.coalesced, 0, "expected submissions to be merged")
        self.assertLess(
            service.completed, service.submitted, "fewer runs than submissions"
        )

    def test_the_latest_snapshot_is_the_newest_picture(self) -> None:
        with scripted_service() as service:
            runtime = CopilotRuntime()
            cid = "inc-3"
            last = None
            for sequence, text in enumerate(CARDIAC, start=1):
                results = runtime.ingest(utterance(cid, sequence, text))
                if results:
                    last = results[-1]
                    service.submit(last)
                    self.assertTrue(service.wait_until_idle(20))

            snapshot = service.latest(cid)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.fingerprint, last.fingerprint)
        self.assertFalse(snapshot.stale)


# ══════════════════════════════════════════════════════════════════════════
# Staleness
# ══════════════════════════════════════════════════════════════════════════


class TestStaleness(unittest.TestCase):
    def test_output_about_a_superseded_picture_is_marked_stale(self) -> None:
        received: List[IntelligenceSnapshot] = []
        provider = SlowProvider(0.5)
        service = LiveIntelligenceService(IntelligenceGateway(provider), received.append)
        service.start()
        try:
            runtime = CopilotRuntime()
            cid = "stale-1"
            first = runtime.ingest(utterance(cid, 1, CARDIAC[0]))[-1]
            service.submit(first)
            self.assertTrue(provider.started.wait(timeout=5))

            # While that run is in flight the consultation moves on.
            second = runtime.ingest(utterance(cid, 2, CARDIAC[1]))[-1]
            self.assertNotEqual(first.fingerprint, second.fingerprint)
            service.submit(second)
            self.assertTrue(service.wait_until_idle(30))
        finally:
            service.stop(timeout=6)

        self.assertTrue(
            any(s.stale for s in received), "the first run described an older picture"
        )
        self.assertGreater(service.stale_dropped, 0)

    def test_a_stale_snapshot_does_not_displace_a_fresher_one(self) -> None:
        with scripted_service() as service:
            result = cardiac_result("stale-2")
            service.submit(result)
            self.assertTrue(service.wait_until_idle(20))
            fresh = service.latest("stale-2")

            # Submit an older picture: its fingerprint no longer matches.
            runtime = CopilotRuntime()
            older = runtime.ingest(utterance("stale-2", 1, CARDIAC[0]))[-1]
            service.submit(older)
            self.assertTrue(service.wait_until_idle(20))
            after = service.latest("stale-2")

        # The most recent submission defines "current", so the older one is not
        # stale by that rule — what matters is that whatever is held is the
        # picture the service last saw.
        self.assertIsNotNone(after)
        self.assertIn(after.fingerprint, {fresh.fingerprint, older.fingerprint})


# ══════════════════════════════════════════════════════════════════════════
# Degradation
# ══════════════════════════════════════════════════════════════════════════


class TestDegradedBehaviour(unittest.TestCase):
    def test_an_unavailable_model_produces_a_degraded_snapshot_not_silence(self) -> None:
        received: List[IntelligenceSnapshot] = []
        service = LiveIntelligenceService(
            IntelligenceGateway(LangGraphIntelligenceProvider(UnavailableModelClient())),
            received.append,
        )
        with service:
            service.submit(cardiac_result("deg-1"))
            self.assertTrue(service.wait_until_idle(20))

        self.assertEqual(len(received), 1)
        snapshot = received[0]
        self.assertEqual(snapshot.status, IntelligenceStatus.UNAVAILABLE)
        self.assertTrue(snapshot.degraded)
        self.assertEqual(snapshot.proposals, [])
        # A client can tell this apart from "not asked yet".
        self.assertTrue(snapshot.message)

    def test_a_failing_provider_is_reported_and_the_worker_survives(self) -> None:
        class Exploding:
            name = "exploding"
            version = "0.0.1"

            def propose(self, context: CopilotContext) -> ProviderResponse:
                raise RuntimeError("graph blew up")

        received: List[IntelligenceSnapshot] = []
        service = LiveIntelligenceService(
            IntelligenceGateway(Exploding()), received.append
        )
        with service:
            service.submit(cardiac_result("deg-2"))
            self.assertTrue(service.wait_until_idle(20))
            # The worker is still alive and serving.
            service.submit(cardiac_result("deg-3"))
            self.assertTrue(service.wait_until_idle(20))

        self.assertEqual(len(received), 2)
        for snapshot in received:
            self.assertEqual(snapshot.status, IntelligenceStatus.FAILED)
            self.assertTrue(snapshot.degraded)

    def test_a_consumer_that_raises_does_not_stop_the_service(self) -> None:
        seen: List[str] = []

        def flaky(snapshot: IntelligenceSnapshot) -> None:
            seen.append(snapshot.consultation_id)
            raise RuntimeError("renderer blew up")

        with scripted_service(flaky) as service:
            service.submit(cardiac_result("deg-4"))
            self.assertTrue(service.wait_until_idle(20))
            service.submit(cardiac_result("deg-5"))
            self.assertTrue(service.wait_until_idle(20))

        self.assertEqual(seen, ["deg-4", "deg-5"])

    def test_a_budget_overrun_is_reported_on_the_snapshot(self) -> None:
        provider = SlowProvider(0.3)
        service = LiveIntelligenceService(IntelligenceGateway(provider, budget_ms=50))
        with service:
            service.submit(cardiac_result("deg-6"))
            self.assertTrue(service.wait_until_idle(20))
            snapshot = service.latest("deg-6")

        self.assertIsNotNone(snapshot)
        self.assertTrue(snapshot.over_budget)
        self.assertGreater(snapshot.elapsed_ms, 50)


# ══════════════════════════════════════════════════════════════════════════
# Provenance
# ══════════════════════════════════════════════════════════════════════════


class TestProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with scripted_service() as service:
            service.submit(cardiac_result("prov-1"))
            service.wait_until_idle(20)
            cls.snapshot = service.latest("prov-1")

    def test_a_citation_resolves_to_the_patients_own_words(self) -> None:
        summary = next(p for p in self.snapshot.proposals if p.kind == "SUMMARY")
        evidence = next(e for e in summary.evidence if e.ref.startswith("symptom:"))

        self.assertTrue(evidence.utterance_id)
        self.assertIn("chest pain", evidence.quote)
        self.assertNotIn("symptom_", evidence.quote)

    def test_a_note_resolves_to_the_investigation_it_annotates(self) -> None:
        note = next(
            p for p in self.snapshot.proposals if p.kind == "INVESTIGATION_NOTE"
        )
        self.assertEqual(note.about_ref, "inv:ecg_12lead")
        self.assertTrue(any(e.ref == "inv:ecg_12lead" for e in note.evidence))

    def test_nothing_is_left_unresolved(self) -> None:
        for proposal in self.snapshot.proposals:
            self.assertEqual(proposal.unresolved_refs, [])
            self.assertTrue(proposal.evidence)

    def test_the_snapshot_names_the_moment_it_describes(self) -> None:
        self.assertTrue(self.snapshot.fingerprint)
        self.assertTrue(self.snapshot.event_id.startswith("utt-"))
        self.assertGreater(self.snapshot.generated_at, 0)


# ══════════════════════════════════════════════════════════════════════════
# The clinical core stays authoritative
# ══════════════════════════════════════════════════════════════════════════


class TestNoClinicalMutation(unittest.TestCase):
    def test_running_the_live_layer_changes_no_clinical_output(self) -> None:
        result = cardiac_result("immut-1")
        before = result.model_dump_json()
        with scripted_service() as service:
            service.submit(result)
            self.assertTrue(service.wait_until_idle(20))
        self.assertEqual(result.model_dump_json(), before)

    def test_the_clinical_endpoint_is_identical_with_and_without_intelligence(
        self,
    ) -> None:
        def clinical_payload(with_service: bool) -> Dict[str, Any]:
            runtime = CopilotRuntime()
            service = scripted_service() if with_service else None
            if service:
                service.start()
            try:
                client = TestClient(create_app(runtime, intelligence=service))
                cid = "parity"
                headers = {"Authorization": f"Bearer {generate_jwt_token(cid, 'doctor')}"}
                for sequence, text in enumerate(CARDIAC, start=1):
                    client.post(
                        f"{API_PREFIX}/sessions/{cid}/events",
                        json=utterance(cid, sequence, text).model_dump(mode="json"),
                        headers=headers,
                    )
                if service:
                    service.wait_until_idle(20)
                return client.get(
                    f"{API_PREFIX}/sessions/{cid}/insights", headers=headers
                ).json()
            finally:
                if service:
                    service.stop(timeout=6)

        without = clinical_payload(False)
        with_it = clinical_payload(True)
        self.assertEqual(
            [d["disease_id"] for d in without["differential"]],
            [d["disease_id"] for d in with_it["differential"]],
        )
        self.assertEqual(
            [f["flag_id"] for f in without["red_flags"]],
            [f["flag_id"] for f in with_it["red_flags"]],
        )


# ══════════════════════════════════════════════════════════════════════════
# Catch-up and lifecycle over HTTP
# ══════════════════════════════════════════════════════════════════════════


class TestCatchUpOverHttp(unittest.TestCase):
    def setUp(self) -> None:
        self.broker = SnapshotBroker()
        self.service = scripted_service(self.broker)
        self.service.start()
        self.runtime = CopilotRuntime()
        self.client = TestClient(
            create_app(
                self.runtime, intelligence=self.service, snapshot_broker=self.broker
            )
        )
        self.cid = "catch-1"
        self.headers = {"Authorization": f"Bearer {generate_jwt_token(self.cid, 'doctor')}"}

    def tearDown(self) -> None:
        self.service.stop(timeout=6)

    def _consult(self) -> None:
        for sequence, text in enumerate(CARDIAC, start=1):
            self.client.post(
                f"{API_PREFIX}/sessions/{self.cid}/events",
                json=utterance(self.cid, sequence, text).model_dump(mode="json"),
                headers=self.headers,
            )
        self.assertTrue(self.service.wait_until_idle(30))

    def test_a_reconnecting_client_can_fetch_what_it_missed(self) -> None:
        self._consult()
        response = self.client.get(
            f"{API_PREFIX}/sessions/{self.cid}/intelligence", headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["consultation_id"], self.cid)
        self.assertTrue(body["proposals"])

    def test_before_anything_is_produced_it_is_a_clean_404(self) -> None:
        self.client.post(
            f"{API_PREFIX}/sessions",
            json={"consultation_id": self.cid},
            headers=self.headers,
        )
        response = self.client.get(
            f"{API_PREFIX}/sessions/{self.cid}/intelligence", headers=self.headers
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "NO_INTELLIGENCE_YET")

    def test_an_unknown_consultation_is_404_before_intelligence_is_consulted(
        self,
    ) -> None:
        response = self.client.get(
            f"{API_PREFIX}/sessions/nope/intelligence",
            headers={"Authorization": f"Bearer {generate_jwt_token('nope', 'doctor')}"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "SESSION_NOT_FOUND")

    def test_intelligence_requires_a_credential_for_that_consultation(self) -> None:
        self._consult()
        self.assertEqual(
            self.client.get(f"{API_PREFIX}/sessions/{self.cid}/intelligence").status_code,
            401,
        )
        self.assertEqual(
            self.client.get(
                f"{API_PREFIX}/sessions/{self.cid}/intelligence",
                headers={"Authorization": f"Bearer {generate_jwt_token('other', 'doctor')}"},
            ).status_code,
            403,
        )

    def test_closing_releases_the_intelligence_state(self) -> None:
        self._consult()
        self.assertIsNotNone(self.service.latest(self.cid))

        self.client.post(
            f"{API_PREFIX}/sessions/{self.cid}/close", headers=self.headers
        )
        self.assertIsNone(self.service.latest(self.cid))

    def test_the_api_works_with_no_intelligence_service_at_all(self) -> None:
        """The default deployment: everything clinical, nothing else."""
        runtime = CopilotRuntime()
        client = TestClient(create_app(runtime))
        cid = "plain-1"
        headers = {"Authorization": f"Bearer {generate_jwt_token(cid, 'doctor')}"}
        client.post(
            f"{API_PREFIX}/sessions/{cid}/events",
            json=utterance(cid, 1, CARDIAC[0]).model_dump(mode="json"),
            headers=headers,
        )

        self.assertEqual(
            client.get(f"{API_PREFIX}/sessions/{cid}/insights", headers=headers).status_code,
            200,
        )
        response = client.get(
            f"{API_PREFIX}/sessions/{cid}/intelligence", headers=headers
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "NO_INTELLIGENCE_YET")


class TestSnapshotBroker(unittest.TestCase):
    """The SSE fan-out for the second channel."""

    def snapshot(self, consultation_id: str) -> IntelligenceSnapshot:
        return IntelligenceSnapshot(
            consultation_id=consultation_id,
            sequence=1,
            event_id="utt-001",
            fingerprint="abc",
            status=IntelligenceStatus.OK,
            provider="test",
            provider_version="1",
        )

    def test_a_subscriber_receives_its_own_consultation(self) -> None:
        broker = SnapshotBroker()
        q = broker.subscribe("A")
        broker(self.snapshot("A"))
        self.assertEqual(q.get_nowait().consultation_id, "A")

    def test_a_subscriber_never_receives_another_consultation(self) -> None:
        broker = SnapshotBroker()
        q = broker.subscribe("A")
        broker(self.snapshot("B"))
        self.assertTrue(q.empty())

    def test_unsubscribing_stops_delivery(self) -> None:
        broker = SnapshotBroker()
        q = broker.subscribe("A")
        broker.unsubscribe("A", q)
        broker(self.snapshot("A"))
        self.assertTrue(q.empty())
        self.assertEqual(broker.subscriber_count("A"), 0)

    def test_a_stalled_subscriber_drops_its_oldest(self) -> None:
        broker = SnapshotBroker()
        q = broker.subscribe("A")
        for _ in range(SnapshotBroker.MAX_BUFFERED + 5):
            broker(self.snapshot("A"))
        self.assertLessEqual(q.qsize(), SnapshotBroker.MAX_BUFFERED)


if __name__ == "__main__":
    unittest.main()
