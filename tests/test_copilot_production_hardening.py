"""
Coverage for the production-readiness additions: the ENVIRONMENT-gated auth
guard, optional SQLite session persistence, and structured logging setup.

Everything here defaults off (no ENVIRONMENT, no COPILOT_STATE_DB_PATH), so
none of it changes what every other test in this repository already exercises
— these tests turn each feature on deliberately, one at a time.
"""

from __future__ import annotations

import logging
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CLINICAL_COPILOT_ENABLE_GRAPH", "False")
os.environ.setdefault("CLINICAL_COPILOT_ENABLE_LLM", "False")

from copilot_api.bootstrap import PLACEHOLDER_JWT_SECRET, check_auth_secret  # noqa: E402
from copilot_api.logging_config import configure_structured_logging  # noqa: E402
from copilot_api.persistence import SnapshotStore  # noqa: E402
from copilot_api.settings import is_production_environment  # noqa: E402
from integration.copilot_event import (  # noqa: E402
    CopilotEvent,
    EventType,
    Provenance,
    SpeakerRole,
    UtterancePayload,
)
from integration.copilot_runtime import CopilotRuntime, UnknownSessionError  # noqa: E402


def _event(seq: int, role: str, text: str, cid: str, event_id: str) -> CopilotEvent:
    return CopilotEvent(
        event_id=event_id,
        consultation_id=cid,
        sequence=seq,
        event_type=EventType.UTTERANCE,
        provenance=Provenance.SERVICE,
        payload=UtterancePayload(
            text=text, speaker_role=SpeakerRole(role), speaker_label=role, is_final=True
        ),
    )


class TestProductionEnvironmentGuard(unittest.TestCase):
    def test_unset_is_not_production(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENVIRONMENT", None)
            self.assertFalse(is_production_environment())

    def test_only_the_exact_value_counts(self) -> None:
        with mock.patch.dict(os.environ, {"ENVIRONMENT": "Production"}):
            self.assertTrue(is_production_environment(), "case-insensitive match")
        with mock.patch.dict(os.environ, {"ENVIRONMENT": "prod"}):
            self.assertFalse(is_production_environment(), "not the exact word")

    def test_the_insecure_override_is_rejected_in_production_regardless_of_the_flag(self) -> None:
        with mock.patch("app.core.config.settings.JWT_SECRET_KEY", PLACEHOLDER_JWT_SECRET):
            with mock.patch.dict(os.environ, {"ENVIRONMENT": "production"}):
                with self.assertRaises(RuntimeError) as raised:
                    check_auth_secret(allow_insecure=True)
        self.assertIn("ENVIRONMENT=production", str(raised.exception))

    def test_the_override_still_works_outside_production(self) -> None:
        """Unchanged behaviour for local dev and every other existing test."""
        with mock.patch("app.core.config.settings.JWT_SECRET_KEY", PLACEHOLDER_JWT_SECRET):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ENVIRONMENT", None)
                check_auth_secret(allow_insecure=True)  # no raise

    def test_a_real_secret_passes_even_in_production(self) -> None:
        with mock.patch("app.core.config.settings.JWT_SECRET_KEY", "a-real-secret"):
            with mock.patch.dict(os.environ, {"ENVIRONMENT": "production"}):
                check_auth_secret(allow_insecure=False)  # no raise


class TestSnapshotStore(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.remove(self.path)  # SnapshotStore creates it; start from nothing.
        self.store = SnapshotStore(self.path)

    def tearDown(self) -> None:
        self.store.close()
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_missing_consultation_loads_as_none(self) -> None:
        self.assertIsNone(self.store.load("never-seen"))

    def test_save_then_load_round_trips(self) -> None:
        self.store.save("c1", "ACTIVE", {"next_sequence": 3, "final_count": 2})
        loaded = self.store.load("c1")
        self.assertEqual(loaded["next_sequence"], 3)
        self.assertEqual(loaded["final_count"], 2)

    def test_save_overwrites_the_previous_snapshot_for_the_same_id(self) -> None:
        self.store.save("c1", "ACTIVE", {"final_count": 1})
        self.store.save("c1", "ACTIVE", {"final_count": 5})
        self.assertEqual(self.store.load("c1")["final_count"], 5)

    def test_delete_removes_it(self) -> None:
        self.store.save("c1", "ACTIVE", {"final_count": 1})
        self.store.delete("c1")
        self.assertIsNone(self.store.load("c1"))

    def test_two_consultations_do_not_overwrite_each_other(self) -> None:
        self.store.save("c1", "ACTIVE", {"final_count": 1})
        self.store.save("c2", "ACTIVE", {"final_count": 9})
        self.assertEqual(self.store.load("c1")["final_count"], 1)
        self.assertEqual(self.store.load("c2")["final_count"], 9)

    def test_the_resolved_path_is_absolute(self) -> None:
        self.assertTrue(os.path.isabs(self.store._path))


class TestRuntimeRestartRecovery(unittest.TestCase):
    """
    The scenario persistence exists for: a runtime is discarded (as happens on
    every container restart, since state lives in the process) and a new one,
    backed by the same store, must recover both the clinical facts and the
    ordering guarantees — not just for the caller who triggers a fresh event,
    but for a plain read arriving before the next event does.
    """

    def setUp(self) -> None:
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.remove(self.path)
        self._stores: list = []

    def _open_store(self) -> SnapshotStore:
        store = SnapshotStore(self.path)
        self._stores.append(store)
        return store

    def tearDown(self) -> None:
        for store in self._stores:
            store.close()
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except PermissionError:
                pass  # Windows file-lock release can lag a process exit slightly.

    def test_clinical_state_and_sequence_survive_a_simulated_restart(self) -> None:
        cid = "restart-test-1"
        store_a = self._open_store()
        runtime_a = CopilotRuntime(snapshot_store=store_a)
        runtime_a.ingest(_event(1, "patient", "Severe chest pain since this morning.", cid, "e1"))
        runtime_a.ingest(_event(2, "patient", "I take amlodipine for my blood pressure.", cid, "e2"))

        # A brand-new runtime and store over the same file -- nothing shared
        # with runtime_a in memory, exactly like a fresh container.
        store_b = self._open_store()
        runtime_b = CopilotRuntime(snapshot_store=store_b)

        # The read path (GET, not an event) must recover it too.
        session = runtime_b.get_session(cid)
        self.assertEqual(session.final_count, 2)
        self.assertEqual(session._next_sequence, 3)
        state = session.current_state()
        self.assertIn("Amlodipine", state.current_medications)

        # And the sequence contract still holds: the runtime built no memory
        # of sequences 1-2 other than what it recovered, so 3 continues cleanly.
        runtime_b.ingest(_event(3, "doctor", "Your blood pressure is 160 over 95.", cid, "e3"))
        self.assertEqual(runtime_b.get_session(cid)._next_sequence, 4)

    def test_a_closed_session_does_not_resurrect_after_restart(self) -> None:
        cid = "restart-test-2"
        store_a = self._open_store()
        runtime_a = CopilotRuntime(snapshot_store=store_a)
        runtime_a.ingest(_event(1, "doctor", "Hello.", cid, "e1"))
        runtime_a.close_session(cid)  # deletes its own snapshot

        store_b = self._open_store()
        runtime_b = CopilotRuntime(snapshot_store=store_b)
        with self.assertRaises(UnknownSessionError):
            runtime_b.get_session(cid)

    def test_persistence_off_is_unaffected_by_this_module_existing(self) -> None:
        """The default configuration: no store, behaviour unchanged."""
        cid = "no-persistence"
        runtime = CopilotRuntime()  # snapshot_store=None, the default
        runtime.ingest(_event(1, "doctor", "Hello.", cid, "e1"))
        self.assertEqual(runtime.get_session(cid).final_count, 1)


class TestStructuredLogging(unittest.TestCase):
    def tearDown(self) -> None:
        # Leave the root logger as this test found it, so other test modules'
        # own logging assumptions aren't affected by running after this one.
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)

    def test_attaches_a_json_formatter_to_root(self) -> None:
        from app.utils.logging import JSONFormatter

        logging.getLogger().handlers.clear()
        configure_structured_logging()
        formatters = [h.formatter for h in logging.getLogger().handlers]
        self.assertTrue(any(isinstance(f, JSONFormatter) for f in formatters))

    def test_calling_it_twice_does_not_duplicate_handlers(self) -> None:
        logging.getLogger().handlers.clear()
        configure_structured_logging()
        configure_structured_logging()
        self.assertEqual(len(logging.getLogger().handlers), 1)


if __name__ == "__main__":
    unittest.main()
