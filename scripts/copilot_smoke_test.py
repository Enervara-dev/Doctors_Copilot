#!/usr/bin/env python
"""
Production smoke test for a running Copilot API instance.

Unlike `tests/test_copilot_api.py` (which exercises the app in-process via
FastAPI's TestClient, and is what CI runs), this script speaks real HTTP to a
real base URL — local, staging, or the deployed Railway/AWS instance — proving
the actual deployed process answers correctly, not just the code in isolation.

Usage:
    python scripts/copilot_smoke_test.py --base-url http://127.0.0.1:8000 \\
        --jwt-secret <the CLINICAL_COPILOT_JWT_SECRET_KEY the target is running with>

Exits non-zero on any failed check, so it's usable as a deployment gate:
    python scripts/copilot_smoke_test.py --base-url $COPILOT_URL --jwt-secret $JWT_SECRET || exit 1

What it does NOT do: open an SSE stream (a CI runner closing a long-lived
connection is more trouble than the coverage is worth here; SSE has its own
dedicated, thorough coverage in tests/test_copilot_api.py::TestStreaming) or
require a live model key (LLM checks are skipped, not failed, when
--intelligence is not passed — a deployment with COPILOT_INTELLIGENCE_ENABLED
unset is a valid, fully-supported configuration).
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import httpx

try:
    import jwt as pyjwt
except ImportError:
    print("This script needs PyJWT: pip install PyJWT", file=sys.stderr)
    sys.exit(2)


@dataclass
class Report:
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed.append(name)
            print(f"  PASS  {name}")
        else:
            self.failed.append(name)
            print(f"  FAIL  {name}  {detail}")

    def summarize(self) -> int:
        total = len(self.passed) + len(self.failed)
        print(f"\n{len(self.passed)}/{total} checks passed.")
        if self.failed:
            print("Failed:", ", ".join(self.failed))
        return 1 if self.failed else 0


def make_token(secret: str, consultation_id: str, role: str = "doctor") -> str:
    now = int(time.time())
    return pyjwt.encode(
        {"session_id": consultation_id, "role": role, "iat": now, "exp": now + 3600},
        secret,
        algorithm="HS256",
    )


def event(cid: str, seq: int, text: str, role: str, event_id: Optional[str] = None) -> dict:
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "consultation_id": cid,
        "sequence": seq,
        "event_type": "utterance",
        "source": "smoke-test",
        "provenance": "SERVICE",
        "payload": {
            "text": text,
            "speaker_role": role,
            "speaker_label": role,
            "is_final": True,
            "confidence": 0.9,
        },
    }


def run(base_url: str, jwt_secret: str, check_intelligence: bool) -> int:
    r = Report()
    client = httpx.Client(base_url=base_url, timeout=20.0)
    prefix = "/copilot/v1"

    # ── liveness ─────────────────────────────────────────────────────────
    resp = client.get("/health")
    r.check("health is 200", resp.status_code == 200, f"got {resp.status_code}")
    r.check("health reports clinical status", resp.json().get("status") == "healthy")

    # ── auth: negative cases first, before anything is created ─────────────
    cid = f"smoke-{uuid.uuid4().hex[:8]}"
    token = make_token(jwt_secret, cid)

    resp = client.get(f"{prefix}/sessions/{cid}")
    r.check("no credential -> 401", resp.status_code == 401, f"got {resp.status_code}")

    resp = client.get(f"{prefix}/sessions/{cid}", headers={"Authorization": "Bearer garbage"})
    r.check("garbage token -> 403", resp.status_code == 403, f"got {resp.status_code}")

    other_cid = f"smoke-other-{uuid.uuid4().hex[:8]}"
    resp = client.get(f"{prefix}/sessions/{other_cid}", headers={"Authorization": f"Bearer {token}"})
    r.check(
        "token scoped to a different consultation -> 403",
        resp.status_code == 403,
        f"got {resp.status_code}",
    )

    resp = client.get(f"{prefix}/sessions/{cid}", headers={"Authorization": f"Bearer {token}"})
    r.check(
        "unopened consultation -> 404, not a silent empty result",
        resp.status_code == 404,
        f"got {resp.status_code}",
    )

    # ── session lifecycle ────────────────────────────────────────────────
    H = {"Authorization": f"Bearer {token}"}
    resp = client.post(f"{prefix}/sessions", json={"consultation_id": cid}, headers=H)
    r.check("create session -> 201", resp.status_code == 201, f"got {resp.status_code}: {resp.text[:200]}")

    resp = client.post(
        f"{prefix}/sessions/{cid}/events",
        json=event(cid, 1, "Doctor, I have severe chest pain since this morning.", "patient"),
        headers=H,
    )
    r.check("ingest final event -> 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    r.check("event was accepted", resp.status_code == 200 and resp.json().get("accepted") is True)

    resp = client.get(f"{prefix}/sessions/{cid}/insights", headers=H)
    r.check("insights -> 200", resp.status_code == 200, f"got {resp.status_code}")
    if resp.status_code == 200:
        symptoms = [s.get("name") for s in resp.json().get("summary", {}).get("symptoms", [])]
        r.check(
            "insights reflect the ingested text",
            "symptom_chest_pain" in symptoms,
            f"symptoms were {symptoms}",
        )

    resp = client.post(
        f"{prefix}/sessions/{cid}/events",
        json=event(cid, 1, "duplicate of the same sequence", "patient", event_id="different-id"),
        headers=H,
    )
    r.check(
        "replaying an already-consumed sequence -> 409 STALE_EVENT",
        resp.status_code == 409,
        f"got {resp.status_code}: {resp.text[:200]}",
    )

    if check_intelligence:
        # Intelligence is optional and asynchronous: a deployment with no
        # model key configured correctly answers 404 forever, and one that
        # has a key needs real network time before a snapshot exists. Both
        # 404 (not yet / not configured) and 200 (produced) are the contract
        # working correctly; only a 5xx or a malformed body is a failure.
        resp = client.get(f"{prefix}/sessions/{cid}/intelligence", headers=H)
        r.check(
            "intelligence endpoint answers its documented contract",
            resp.status_code in (200, 404),
            f"got {resp.status_code}: {resp.text[:200]}",
        )
        if resp.status_code == 200:
            body = resp.json()
            r.check("intelligence response has a status field", "status" in body)
            r.check(
                "a failed model call degrades rather than 500s",
                body.get("status") in ("ok", "unavailable"),
                f"status was {body.get('status')!r}",
            )

    resp = client.post(f"{prefix}/sessions/{cid}/close", headers=H)
    r.check("close session -> 200", resp.status_code == 200, f"got {resp.status_code}")

    resp = client.get(f"{prefix}/sessions/{cid}", headers=H)
    r.check("read after close -> 409 SESSION_CLOSED", resp.status_code == 409, f"got {resp.status_code}")

    resp = client.post(
        f"{prefix}/sessions/{cid}/events",
        json=event(cid, 2, "should be refused", "doctor"),
        headers=H,
    )
    r.check("ingest after close -> 409", resp.status_code == 409, f"got {resp.status_code}")

    return r.summarize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000 or the Railway/AWS URL")
    parser.add_argument(
        "--jwt-secret", required=True,
        help="Must match CLINICAL_COPILOT_JWT_SECRET_KEY on the target instance, or every check fails at the auth stage.",
    )
    parser.add_argument(
        "--intelligence", action="store_true",
        help="Reserved for a future pass; currently just flags the check as not implemented rather than skipping silently.",
    )
    args = parser.parse_args()

    print(f"Smoke-testing {args.base_url}\n")
    sys.exit(run(args.base_url, args.jwt_secret, args.intelligence))


if __name__ == "__main__":
    main()
