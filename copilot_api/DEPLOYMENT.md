# Copilot API — deployment guide

This document covers the Copilot service **alone**: `copilot_api/` +
`integration/` + the shared `clinical_intelligence/` reasoning core it depends
on. It does not cover the audio/STT pipeline (`app.main:app`), the frontend
console, or the appointment system — those are separate services with their
own deployment concerns.

---

## 1. What this service is, in one paragraph

A stateless-per-request, **not stateless-per-process** HTTP API. It accepts
transcript events for a consultation (`POST /copilot/v1/sessions/{id}/events`),
runs them through a deterministic clinical rule engine, optionally asks a
LangGraph-orchestrated LLM for grounded commentary, and serves both back
(`GET .../insights` for the deterministic result, `GET .../intelligence` for
the advisory one, plus an SSE `.../stream`). Session state — the accumulating
clinical picture for one consultation — lives in an in-memory dict unless you
enable the optional SQLite persistence described in §7. **This is the single
architectural fact every other decision below follows from: run exactly one
instance of this service, or its session state will not be consistent.**

---

## 2. Required and optional environment variables

### Required for any real deployment

| Variable | Purpose |
|---|---|
| `CLINICAL_COPILOT_JWT_SECRET_KEY` | Signs/verifies the bearer JWTs the Doctor platform mints per consultation. **The service refuses to start** if this is still `production-jwt-secret-key-change-me` (the value shipped in `app/core/config.py`), unless `COPILOT_ALLOW_INSECURE_AUTH=true` — which itself is refused when `ENVIRONMENT=production`. There is no path to running production traffic on the placeholder. |
| `ENVIRONMENT` | Set to exactly `production`. This is what makes the insecure-auth override unconditionally refused, regardless of `COPILOT_ALLOW_INSECURE_AUTH`. Logged at boot (`describe()`), never gates anything else. |

### Required if the intelligence layer is on

| Variable | Purpose |
|---|---|
| `COPILOT_INTELLIGENCE_ENABLED` | `true` to turn on the LangGraph layer. Off (default) means a fully functional clinical-only deployment — this is a legitimate, complete production configuration, not a degraded one. |
| `GEMINI_API_KEY` (or `OPENAI_API_KEY` + `COPILOT_MODEL_PROVIDER=openai`) | The model credential. A missing/invalid key does **not** stop the service from starting or from serving deterministic clinical output — `/intelligence` reports `status: "unavailable"` instead. Verified in this pass with a deliberately broken key: ingestion, insights, and `/health` were unaffected. |

### Optional — safe to leave unset

| Variable | Default | Effect |
|---|---|---|
| `COPILOT_INTELLIGENCE_PROVIDER` | `langgraph` | `deterministic` restates rule output with no model call (useful for a staging environment with no key); `null` disables output entirely without disabling the worker. |
| `COPILOT_MODEL_NAME` | provider default | e.g. a specific Gemini model version. |
| `COPILOT_MODEL_TEMPERATURE` | `0.2` | |
| `COPILOT_INTELLIGENCE_BUDGET_MS` | `8000` | Advisory only — marks a slow snapshot as late, never cancels the call. |
| `COPILOT_INTELLIGENCE_CAPACITY` | `256` | How many consultations keep a cached intelligence snapshot. |
| `COPILOT_SESSION_IDLE_TIMEOUT_SECONDS` | `1800` | A consultation with no event for this long is evicted (410 on the next access). |
| `COPILOT_ALLOW_INSECURE_AUTH` | `false` | Local development only. No effect when `ENVIRONMENT=production`. |
| `COPILOT_CORS_ALLOWED_ORIGINS` | *(empty)* | Comma-separated origins. Empty means no CORS headers are ever sent — correct for a service called server-to-server (Railway → this service), never from a browser. Only set this if a browser client calls the API directly. |
| `COPILOT_TRUSTED_HOSTS` | *(empty)* | Comma-separated `Host` header values this instance accepts (`TrustedHostMiddleware`). Empty disables the check. Set to your ALB DNS name / custom domain once known — see §5. |
| `COPILOT_STATE_DB_PATH` | *(empty)* | Enables SQLite session persistence. See §7 for what it does and does not solve. |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | *(empty)* | Optional knowledge-graph enrichment. Degrades cleanly (`graph_status: DOWN`) when absent — verified throughout this pass with no Neo4j reachable. |

### Explicitly not needed

`CLINICAL_COPILOT_ELEVENLABS_API_KEY`, `CLINICAL_COPILOT_HUGGINGFACE_TOKEN`,
`CLINICAL_COPILOT_STT_PROVIDER_MODE`, `CLINICAL_COPILOT_DIARIZATION_MODE` — all
belong to `app.main`, the audio pipeline, which this image does not contain
and this service never imports (`requirements-copilot.txt`).

---

## 3. Building the image

```bash
docker build -f Dockerfile.copilot -t copilot-api:<tag> .
```

**Not validated end-to-end in this pass.** Docker's CLI is present in this
environment (`29.6.2`) but no daemon was reachable (`Docker Desktop` not
running, and its executable was not at the expected install path) — `docker
build` could not be executed here. Reported explicitly rather than assumed.

What **was** validated as the strongest available substitute:
`requirements-copilot.txt` installed cleanly into an isolated virtual
environment containing nothing else, and `copilot_api.bootstrap:create_production_app`
booted successfully — full route table, structured logging, LangGraph
provider construction — using **only** those packages. This is the same
dependency set and the same entry point the image's `CMD` invokes; the
remaining risk in an actual `docker build` is packaging-layer only (base image
availability, `pip install` inside the container), not application-layer.

Run `docker build` yourself before the first real deployment and treat it as
the one validation step this pass could not complete.

### Running it locally

```bash
docker run --rm -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e CLINICAL_COPILOT_JWT_SECRET_KEY=<a real secret> \
  -e COPILOT_INTELLIGENCE_ENABLED=true \
  -e GEMINI_API_KEY=<your key> \
  copilot-api:<tag>
curl http://127.0.0.1:8000/health
```

---

## 4. What was actually validated in this pass (and how)

Every scenario below was run against a real `uvicorn
copilot_api.bootstrap:create_production_app --factory` process — real HTTP,
real JWTs, real Gemini calls where noted — not just the in-process test suite.

| Scenario | Result |
|---|---|
| Boot with `ENVIRONMENT=production` + placeholder secret + `COPILOT_ALLOW_INSECURE_AUTH=true` | **Correctly refuses to start** — the production-environment hardening added in this pass |
| Boot with a real secret | Starts; `/health` → `200` |
| No credential | `401 UNAUTHENTICATED` |
| Garbage bearer token | `403 FORBIDDEN` |
| Token valid for a different consultation id | `403 FORBIDDEN` |
| Create session → ingest → insights | `201` → `200` (accepted) → `200`, correct symptom extraction (`symptom_chest_pain`) |
| Real Gemini + LangGraph intelligence | 8 grounded proposals, `degraded: false`, produced in ~15–20s |
| SSE stream | `: connected` frame, then a live `event: result` frame with the ingested clinical data, delivered to a second concurrent connection |
| Session close, then event/read after close | `200` close → `409 SESSION_CLOSED` on both |
| Idle-timeout expiry | Lazily swept **on the next `ingest()`**, not on a read — a `GET` right after the timeout still returns the live state; the following event correctly gets `410 SESSION_EXPIRED`. This is by design (see `evict_idle`'s docstring), not a bug — documented here because it is easy to mistake for one. |
| Broken/invalid Gemini key | `/intelligence` → `unavailable`, `degraded: true`, with the real HTTP error in the (now-structured) log. Deterministic ingestion and insights were completely unaffected. `/health` stayed `"healthy"`. |
| SQLite persistence across a process restart | A session opened, ingested into, then the process killed and a **new** process started against the same file: `GET /sessions/{id}` (a plain read, not just the next event) correctly recovered the clinical state and continued accepting the next sequence number. Verified after fixing a real gap found during this check — see §7. |
| Full test suite | `708 passed, 1 pre-existing unrelated flaky, 2 skipped` (`test_hybrid_intelligence.py::test_pipeline_integration` — shares process-global provider state with unrelated tests; passes in isolation; documented before this pass, not introduced by it) |

### Not validated here

- **`docker build` itself** — no daemon reachable (§3).
- **Lint/typecheck** — no tool is configured anywhere in this repository
  (`ruff.toml`, `mypy.ini`, `pyproject.toml`, `.flake8`, `setup.cfg` — none
  exist). `python -m py_compile` / `ast.parse` was run over every changed
  file as the closest available static check. Installing a linter was judged
  out of scope for this pass — it would be a new standing project decision,
  not a deployment-readiness fix, and wasn't requested.
- **Actual AWS/Railway connectivity** — §5/§6 are exact, tested-shape
  instructions; they were not run against a live AWS account or Railway
  project from this environment.

---

## 5. AWS deployment (ECS Fargate)

This assumes an existing VPC with at least two subnets and a default security
group setup; adjust names to your account's conventions.

```bash
# 1. Build and push
aws ecr create-repository --repository-name copilot-api
aws ecr get-login-password --region <region> | docker login --username AWS \
    --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker build -f Dockerfile.copilot -t copilot-api .
docker tag copilot-api:latest <account-id>.dkr.ecr.<region>.amazonaws.com/copilot-api:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/copilot-api:latest

# 2. Store secrets (never put these in the task definition's plain `environment` block)
aws secretsmanager create-secret --name copilot/jwt-secret \
    --secret-string '<a long random value — not the placeholder>'
aws secretsmanager create-secret --name copilot/gemini-api-key \
    --secret-string '<your Gemini key>'
```

**3. Task definition** — the parts that matter beyond the standard boilerplate:

```json
{
  "family": "copilot-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [{
    "name": "copilot-api",
    "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/copilot-api:latest",
    "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
    "environment": [
      {"name": "ENVIRONMENT", "value": "production"},
      {"name": "COPILOT_INTELLIGENCE_ENABLED", "value": "true"},
      {"name": "COPILOT_TRUSTED_HOSTS", "value": "<your-alb-dns-name>"}
    ],
    "secrets": [
      {"name": "CLINICAL_COPILOT_JWT_SECRET_KEY", "valueFrom": "arn:aws:secretsmanager:<region>:<account-id>:secret:copilot/jwt-secret"},
      {"name": "GEMINI_API_KEY", "valueFrom": "arn:aws:secretsmanager:<region>:<account-id>:secret:copilot/gemini-api-key"}
    ],
    "healthCheck": {
      "command": ["CMD-SHELL", "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)\""],
      "interval": 30, "timeout": 5, "retries": 3, "startPeriod": 20
    },
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/copilot-api",
        "awslogs-region": "<region>",
        "awslogs-stream-prefix": "copilot"
      }
    }
  }]
}
```

The container's own `HEALTHCHECK` is already in `Dockerfile.copilot`; this
repeats it at the task-definition level because ECS does not read a Docker
`HEALTHCHECK` for its own service-level health decisions — the ALB target
group health check (below) is what actually drives replacement.

```bash
aws ecs register-task-definition --cli-input-json file://task-def.json

# 4. Service — desired count 1. Not 2, not "min 1 max 3 autoscaling".
#    See §1: session state is in-memory per process. A second task would have
#    its own empty dict; a request landing on it for a consultation opened on
#    the first task would 404. Scale this service vertically (more CPU/memory
#    per task), not horizontally, unless persistence (§7) is combined with a
#    router that pins each consultation to one task — which is out of scope
#    for this pass and not requested.
aws ecs create-service \
  --cluster <your-cluster> \
  --service-name copilot-api \
  --task-definition copilot-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-ids>],securityGroups=[<sg-id>],assignPublicIp=DISABLED}" \
  --load-balancers "targetGroupArn=<target-group-arn>,containerName=copilot-api,containerPort=8000"
```

**5. ALB target group** — health check path `/health`, protocol HTTP, port
`8000` (the ALB terminates TLS; the container itself speaks plain HTTP behind
it — this is what `--proxy-headers` in the Dockerfile's `CMD` is for, so the
app still sees `https` as the effective scheme). Idle timeout: leave at the
ALB default (60s) or lower — the container's own `--timeout-keep-alive 75`
was chosen to outlive it, not the other way around.

**6. Security group** — restrict the container's security group to accept
inbound `8000` **only** from the ALB's security group. This is not optional:
`--proxy-headers --forwarded-allow-ips '*'` in the container `CMD` means the
container trusts `X-Forwarded-For`/`X-Forwarded-Proto` from whoever can reach
it directly. That trust is only safe because the security group — not the
application — is what limits "whoever" to the ALB alone.

**7. HTTPS**: terminate TLS at the ALB (ACM certificate, standard listener
setup) or at Railway's own edge if this ends up hosted there instead — the
container never needs to hold a TLS certificate itself.

---

## 6. Railway configuration (the Doctor backend's side)

The Doctor backend does not need this service's source, only its URL and a
way to mint valid bearer tokens. Set these in the Railway project running the
Doctor backend:

| Railway variable | Value |
|---|---|
| `COPILOT_BASE_URL` | The Copilot service's URL — the ALB's DNS name/custom domain (AWS) or this service's own Railway URL if co-hosted there instead. Include the scheme: `https://copilot.yourdomain.com`. |
| `COPILOT_JWT_SIGNING_KEY` | **Must be byte-for-byte the same value** as `CLINICAL_COPILOT_JWT_SECRET_KEY` on the Copilot service. This is what lets the Doctor backend mint a token the Copilot service will accept — there is no shared database or handshake, only a shared secret. |
| `COPILOT_JWT_ALGORITHM` | `HS256` |

The Doctor backend mints one JWT per consultation, the same way
`app/utils/jwt_utils.generate_jwt_token` does on the Python side:

```json
{ "session_id": "<consultation_id>", "role": "doctor", "iat": <now>, "exp": <now + 3600> }
```

signed with `COPILOT_JWT_SIGNING_KEY` / HS256, sent as
`Authorization: Bearer <token>` on every call to `{COPILOT_BASE_URL}/copilot/v1/...`.
A token minted for one `consultation_id` is refused (`403`) against any other
— this was verified directly in this pass, not assumed from reading the code.

**The HTTP contract itself** (already implemented, verified against a real
running instance in this pass, not new in this milestone):

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/copilot/v1/sessions` | Body `{"consultation_id": "..."}` → `201` |
| `POST` | `/copilot/v1/sessions/{id}/events` | Body is a `CopilotEvent` (see `integration/copilot_event.py`) → `200` with `{"accepted", "buffered", "results", "insights"}` |
| `GET` | `/copilot/v1/sessions/{id}` | Session/sequence state |
| `GET` | `/copilot/v1/sessions/{id}/insights` | Deterministic clinical picture — never varies with model availability |
| `GET` | `/copilot/v1/sessions/{id}/intelligence` | Latest advisory LangGraph snapshot, or `404` if none yet / layer disabled |
| `GET` | `/copilot/v1/sessions/{id}/stream` | SSE: `result` and `insight` frames as they're produced |
| `POST` | `/copilot/v1/sessions/{id}/close` | Ends the consultation |

Error shape is uniform: `{"error": {"code": "...", "message": "..."}}` with
one of `SESSION_NOT_FOUND` (404), `SESSION_CLOSED` (409), `SESSION_EXPIRED`
(410), `STALE_EVENT` (409), `INVALID_SPEAKER` (422), `VALIDATION_ERROR` (422),
or `RUNTIME_REFUSED` (409, catch-all). Auth failures use FastAPI's own
`{"detail": {"code": ..., "message": ...}}` shape instead — a pre-existing,
harmless inconsistency between the auth layer and the domain-error layer, not
introduced by this pass. Worth normalizing later; not blocking.

---

## 7. Session-state persistence (optional)

**The zero-code option, and the one this codebase was originally designed
around** (see `integration/copilot_runtime.py`'s own module docstring): the
Doctor platform is the source of truth for the transcript. If it can re-POST
every event for a consultation from sequence 1 after detecting the Copilot
service lost its session (a `404` where it expected a live one), no
persistence is needed on this side at all — the deterministic clinical state
is, by construction, a pure projection of the event stream and rebuilds
identically from a replay.

**If replay isn't something the Doctor backend can do today**, this pass adds
an opt-in alternative: set `COPILOT_STATE_DB_PATH` to enable SQLite-backed
snapshotting (`copilot_api/persistence.py`). What it covers and what it
doesn't is documented in full in that module's docstring; the short version:

- Recovers the clinical state, fact provenance, and sequencing/dedup state —
  everything needed for correctness after a restart.
- Does **not** recover result-emission throttling counters (a UX detail, not
  a safety one — costs at most one redundant reasoning pass after a restart).
- **Requires a persistent volume**, not the container's own ephemeral disk.
  On Fargate, that means mounting an EFS access point and pointing
  `COPILOT_STATE_DB_PATH` at a path inside it:

```json
"volumes": [{
  "name": "copilot-state",
  "efsVolumeConfiguration": {"fileSystemId": "<fs-id>", "transitEncryption": "ENABLED"}
}],
"containerDefinitions": [{
  "mountPoints": [{"sourceVolume": "copilot-state", "containerPath": "/data"}],
  "environment": [{"name": "COPILOT_STATE_DB_PATH", "value": "/data/copilot-sessions.db"}]
}]
```

Without that mount, setting the variable changes nothing about restart safety
— it only changes where the equally-ephemeral copy happens to sit, which is
exactly the failure mode a manual check in this pass reproduced before the
volume question was even asked.

A real gap was found and fixed while validating this: the original
implementation restored a session from its snapshot only when the *next
event* arrived (`open_session`), not on a plain read. A client polling
`GET /insights` immediately after a restart — before any new event —
would have seen a false `404` for a consultation that genuinely existed.
`get_session` now attempts the same restore. Verified with a real two-process
test: ingest → kill the process → a **new** process, same file → `GET`
(no event) correctly recovers `final_count`, `next_expected_sequence`, and
every extracted clinical fact.

---

## 8. Service URL format

Whatever the deployment target, the Copilot service is reached as:

```
https://<host>/copilot/v1/...
```

- AWS: `https://<your-alb-dns-name-or-custom-domain>/copilot/v1/...`
- Railway (if co-hosted there instead of AWS): `https://<service>.up.railway.app/copilot/v1/...`

`/health` (no prefix, no auth) is the liveness/readiness check for whichever
platform's load balancer or health-check system sits in front of it.
