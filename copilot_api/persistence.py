"""
Optional consultation-state persistence, for safe restart and recovery.

Off by default. With no `COPILOT_STATE_DB_PATH` configured, nothing in this
module is imported by the running service and behaviour is identical to
before persistence existed — every existing test runs in that configuration.

## What this does and does not cover

On restart, a deployment with persistence enabled recovers:
- the clinical state itself (symptoms, history, medications, vitals, ... —
  the safety-critical data a clinician is relying on)
- fact provenance (which utterance established which fact)
- enough sequencing state (`next_sequence`, settled event ids, live partial
  text) that the session's ordering guarantees hold across the restart: a
  replayed or resumed event stream is accepted or rejected exactly as it
  would have been if the process had never stopped

It deliberately does NOT persist result-emission throttling state (the last
emitted fingerprint, reasoning-skip counters, the cached `CopilotResult`).
Losing that on restart costs at most one redundant reasoning pass and one
result a reconnecting client may already have — a UX nicety, not a safety
property. Recomputing it from the restored clinical state is strictly
sufficient and far simpler than replicating every counter exactly.

## Why SQLite, not a new service

A consultation's session state is a handful of small JSON blobs, written
after every final event and read once per session per process lifetime. That
does not justify standing up Redis or Postgres — an external dependency this
service would then have to be built, deployed, and monitored around, for a
volume of writes a single SQLite file handles without noticing. `sqlite3` is
in the Python standard library: no new dependency, no new failure mode beyond
"the disk this file lives on."

## The one thing this does NOT solve by itself

On ECS/Fargate (or any container platform without a persistent local disk), a
SQLite file on the container's own filesystem is lost exactly like in-memory
state is — the container's disk is as ephemeral as its memory. For this to
survive a task replacement, the path this module writes to must be a mounted
persistent volume (EFS on AWS; see `copilot_api/DEPLOYMENT.md`). Enabling
`COPILOT_STATE_DB_PATH` without such a mount changes nothing about restart
safety — it only changes where the ephemeral copy happens to sit.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consultation_snapshots (
    consultation_id TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    payload         TEXT NOT NULL,
    updated_at      REAL NOT NULL
);
"""


class SnapshotStore:
    """
    A tiny key-value store over SQLite: one row per consultation, the whole
    snapshot as one JSON blob. No migrations, no ORM — the schema is one
    table because the access pattern is "read one row at startup, write one
    row after every final event."

    One connection, guarded by a lock. `CopilotRuntime` calls this from
    FastAPI's threadpool (ingestion routes are sync, by design — see
    `copilot_api/main.py`), so concurrent writes from different consultations
    are expected and must not corrupt each other; SQLite's own locking would
    serialize them anyway, but the explicit lock keeps the failure mode a
    predictable wait rather than a `sqlite3.OperationalError` under load.
    """

    def __init__(self, db_path: str) -> None:
        # Resolved and logged, not used as given: a relative path silently
        # means "relative to whatever the process's CWD happens to be", which
        # is invisible in a log line that only echoes the configured string
        # back. Logging the resolved path is what makes "the volume isn't
        # mounted where I think it is" a boot-time log line instead of an
        # afternoon of "the data just isn't there."
        resolved = os.path.abspath(db_path)
        directory = os.path.dirname(resolved)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._path = resolved
        self._lock = threading.Lock()
        # check_same_thread=False: FastAPI's threadpool means different
        # requests may use this connection from different threads. The lock
        # above is what actually makes that safe, not this flag alone.
        self._conn = sqlite3.connect(resolved, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        # A round-trip, not just a commit: proves the file is genuinely
        # writable and readable from this path before the service claims
        # persistence is enabled, rather than discovering a permissions or
        # mount problem on the first real consultation.
        self._conn.execute(
            "INSERT OR REPLACE INTO consultation_snapshots VALUES (?, ?, ?, ?)",
            ("__startup_probe__", "PROBE", "{}", _now()),
        )
        self._conn.execute(
            "DELETE FROM consultation_snapshots WHERE consultation_id = ?",
            ("__startup_probe__",),
        )
        self._conn.commit()
        logger.info("SnapshotStore: opened %s (resolved from %r)", resolved, db_path)

    def save(self, consultation_id: str, status: str, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO consultation_snapshots (consultation_id, status, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(consultation_id) DO UPDATE SET
                    status = excluded.status,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (consultation_id, status, body, _now()),
            )
            self._conn.commit()

    def load(self, consultation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM consultation_snapshots WHERE consultation_id = ?",
                (consultation_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.error(
                "SnapshotStore: corrupt snapshot for %s, ignoring it", consultation_id
            )
            return None

    def delete(self, consultation_id: str) -> None:
        """Called when a consultation closes — a closed session is a tombstone in
        memory (see `CopilotRuntime.close_session`); it should not resurrect
        itself as an open one from a stale snapshot after the next restart."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM consultation_snapshots WHERE consultation_id = ?",
                (consultation_id,),
            )
            self._conn.commit()

    def all_consultation_ids(self) -> Iterator[str]:
        """For a future warm-restore-all-on-boot path. Not called today —
        sessions restore lazily, on first access, which is simpler and pays
        the recovery cost only for consultations actually resumed."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT consultation_id FROM consultation_snapshots"
            ).fetchall()
        return (r[0] for r in rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _now() -> float:
    import time

    return time.time()
