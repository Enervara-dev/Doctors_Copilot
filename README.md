# Clinical Consultation Copilot

A live clinical-consultation assistant. It transcribes a doctor/patient conversation,
attributes each sentence to a speaker, extracts structured clinical facts, and runs a
deterministic rule engine (with optional LLM enrichment) to surface differentials, red
flags, and recommendations.

**This repository contains two backend services, now wired together by a small bridge**,
plus a static demo console for the first one. See
[Two services, bridged](#two-services-bridged) below before deploying anything.

For a from-the-code deep dive (execution paths, response shapes, what's real vs. mocked),
see [`docs/`](docs/README.md) and in particular [`docs/status.md`](docs/status.md).

---

## Two services, bridged

| | `app.main:app` | `copilot_api.bootstrap:create_production_app` |
|---|---|---|
| **What it does** | Ingests raw microphone audio over a WebSocket, runs STT (ElevenLabs) + diarization (pyannote) + speaker ID (SpeechBrain ECAPA) + clinical extraction, and serves the resulting state/reasoning over HTTP | Ingests transcript **events**, runs them through a session runtime, the deterministic rule engine, and an optional LangGraph intelligence layer, and serves insights back |
| **Input** | Live PCM audio | JSON events via `POST /copilot/v1/sessions/{id}/events` |
| **Consumers today** | The static console in `frontend/`, and — when bridged — `copilot_api` | The audio pipeline, via `app/services/copilot_bridge.py` |
| **Auth on its routes** | WebSocket + all HTTP GET routes require a bearer credential scoped to the session | Every route requires a bearer token scoped to the consultation (`copilot_api/auth.py`) |
| **Container** | None shipped | `Dockerfile.copilot` |
| **Dependencies** | `requirements.txt` — includes torch/pyannote/speechbrain (~GB-scale) | `requirements-copilot.txt` — lean, no ML audio stack |
| **State** | In-memory Python dicts, per process | In-memory Python dicts, per process |

They share the rule engine (`clinical_intelligence/`) and the auth primitive
(`app/core/security.py`). `app/services/copilot_bridge.py` is what actually connects
them: every **finalized** transcript event `app.main` produces is forwarded to
`copilot_api` as it happens, so a consultation recorded through the console also
accumulates clinical state and (if enabled) LangGraph intelligence on the Copilot side.

The bridge exists specifically because the two services disagree on event ordering:
`app.main` numbers transcript events **per speaker role**, while `copilot_api` requires
one single, gap-free sequence per consultation. The bridge assigns that sequence as it
forwards, and only forwards *final* transcripts (never partials — see the module
docstring for why). It is off by default (`COPILOT_BRIDGE_ENABLED=false`); with it
disabled, `app.main` behaves exactly as if the bridge did not exist.

There is still no UI that renders `copilot_api`'s insights/intelligence — that side is
consumed via its HTTP API only. "Doctor UI" in this repo means the `frontend/` console,
which shows the audio pipeline's own view (transcript, extracted facts, rule-engine
reasoning), not the Copilot service's.

---

## Quick start — audio pipeline + demo console

```powershell
.\run.ps1
```

Starts `app.main:app`, serves `frontend/` as a static console, waits for the diarization
models to load (~40 s cold, loading from the local Hugging Face cache), and opens the
browser. `Ctrl+C` stops both. Options: `-ApiPort`, `-UiPort`, `-NoBrowser`.

### Manual

```powershell
# Always the venv Python: torch, pyannote and speechbrain live there only.
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Separate terminal — the mic needs a secure context, so it must be served, not opened as a file.
python -m http.server 5500 --directory frontend
```

Open <http://127.0.0.1:5500>. Interactive API docs at <http://127.0.0.1:8000/docs>.

**This service intentionally has no offline/mock mode.** `STT_PROVIDER_MODE` and
`DIARIZATION_MODE` must both be `production`, real credentials must be set, and the real
model weights must be reachable — otherwise startup fails with a explicit error. (An
earlier development-mode mock silently fabricated transcripts and speaker labels, which
made clinical facts vanish with no error; it was deleted rather than left as a footgun.)

### First-time model setup

The diarization and voice-embedding models are not vendored. Once, with
`HF_HUB_OFFLINE=0` and a Hugging Face token that has accepted the
`pyannote/speaker-diarization-3.1` license:

```powershell
$env:HF_HUB_OFFLINE = "0"
.venv\Scripts\python.exe -c "from app.services.diarization_worker import _lazy_init_pyannote; _lazy_init_pyannote()"
.venv\Scripts\python.exe -c "from app.services.speaker_verification.backend import SpeechBrainBackend; SpeechBrainBackend().load_model()"
```

After that, the app runs offline from the local cache.

### Replaying a recording instead of speaking

```powershell
$env:DEMO_SESSION_ID = "demo-1"
.venv\Scripts\python.exe stream_audio.py path\to\consultation.wav doctor
```

Then open `http://127.0.0.1:5500/?session=demo-1`.

---

## Quick start — Copilot API (event/insights service)

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-copilot.txt
.venv\Scripts\python.exe -m uvicorn copilot_api.bootstrap:create_production_app --factory --port 8010
```

With no environment variables set, it serves the full deterministic API and reports no
intelligence — `GET /copilot/v1/sessions/{id}/intelligence` returns a clean 404. See
`.env.example` for every `COPILOT_*` switch (intelligence provider, model, budget,
capacity, idle timeout).

**It refuses to start** if `CLINICAL_COPILOT_JWT_SECRET_KEY` is still the placeholder
shipped in `app/core/config.py` — that secret is public, so leaving it in place would let
anyone mint a valid token for any consultation. Set the real variable (note the
`CLINICAL_COPILOT_` prefix) or, for local development only, `COPILOT_ALLOW_INSECURE_AUTH=true`.

### Container

```bash
docker build -f Dockerfile.copilot -t copilot-api .
docker run -p 8000:8000 -e CLINICAL_COPILOT_JWT_SECRET_KEY=... copilot-api
```

Runs as a non-root user, ~150 MB image (no torch/pyannote). `/health` reflects the
deterministic core only — a degraded intelligence layer is visible in the response body
but is deliberately never allowed to make the container unhealthy.

---

## Running both services, bridged

Start `copilot_api` first (it's what the bridge forwards to), then `app.main` with the
bridge switched on:

```powershell
# Terminal 1
.venv\Scripts\python.exe -m uvicorn copilot_api.bootstrap:create_production_app --factory --port 8010

# Terminal 2
$env:COPILOT_BRIDGE_ENABLED = "true"
$env:COPILOT_BRIDGE_BASE_URL = "http://127.0.0.1:8010"
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000

# Terminal 3 — the console
python -m http.server 5500 --directory frontend
```

Record a consultation through the console as usual. Every finalized transcript now also
lands in `copilot_api` — confirm with:

```powershell
$token = "production-secure-token-change-me"
curl -H "Authorization: Bearer $token" http://127.0.0.1:8010/copilot/v1/sessions/<your-session-id>/insights
```

If nothing shows up, check `app.main`'s logs for `CopilotBridge:` lines — a bridge that
can't reach `copilot_api` logs and drops the event rather than breaking audio ingestion.

---

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ clinical_intelligence/tests/ -q
```

~695 tests across both services. One test
(`test_hybrid_intelligence.py::test_pipeline_integration`) is order-dependent — it passes
alone and can fail in the full run because it shares a process-global provider instance
(circuit breaker / latency counters) with other tests that run before it. Not a
production defect; a test-isolation gap worth fixing before relying on suite-green as a
release gate.

---

## Can this go on AWS?

**Not as-is, without deciding what "deployed" means first.** Neither service is a
production blocker on code quality grounds — the rule engine is well-tested, the newer
`copilot_api` service is genuinely hardened (per-consultation auth, a real startup guard
against the placeholder secret, graceful shutdown, a lean image). The blockers are
architectural and operational, not "buggy code":

1. **No persistence in either service.** All session state is an in-memory Python `dict`.
   A restart, a deploy, or an ECS task recycling loses every in-progress consultation.
   This alone rules out anything requiring multiple instances, rolling deploys, or
   auto-scaling — you'd be limited to one long-lived task per service with no failover.
2. **No infrastructure-as-code.** No Dockerfile for the audio pipeline, no
   docker-compose, no Terraform/CDK/CloudFormation, no ECS task definitions, no CI. Only
   `Dockerfile.copilot` exists. Everything else would need to be written before this is
   "an AWS deployment" rather than "code that could run on AWS."
3. ~~The audio pipeline's HTTP endpoints are unauthenticated~~ — **fixed.** `GET
   /v1/transcripts/{id}`, `/v1/clinical-state/{id}`, its `/provenance` sibling, and
   `/v1/clinical-reasoning/{id}` now require the same bearer credential a participant used
   to open the WebSocket for that session (`app/core/security.py`
   `require_session_auth`). A guessable session ID alone no longer reads anything.
4. ~~The two services aren't integrated~~ — **partially fixed.** `app/services/copilot_bridge.py`
   now forwards every finalized transcript from `app.main` into `copilot_api` in real
   time (off by default; see [Copilot bridge](#running-both-services-bridged)). What's
   still missing: nothing in this repo *reads* `copilot_api`'s insights/intelligence back
   — there is no "Doctor UI" rendering that side, only its HTTP API.
5. **The audio pipeline needs a large, GPU-friendly base image or a beefy CPU instance**
   for pyannote/ECAPA inference, plus the model license acceptance step above baked into
   the image build — none of this is packaged.
6. **Secret hygiene issue, separate from this pass:** a live ElevenLabs key was committed
   to `.env` in this repo's history and reached `origin/main` before `.gitignore` covered
   it. If that key is still in use anywhere, rotate it before this goes anywhere near a
   public cloud account. The JWT signing key is also still the shipped placeholder in
   every environment file in this repo — set a real one before anything but local testing.

**What AWS deployment would concretely look like, once the above is addressed:**
- `copilot_api`: straightforward on ECS Fargate or App Runner — it's already
  stateless-per-request except for its in-memory session map, so persistence (Redis /
  DynamoDB for session + result state) is the main gap before it can run as >1 task.
- `app.main`: needs an EC2 or Fargate task sized for pyannote inference (CPU works, GPU is
  faster), the model cache baked into the image or mounted from EFS/S3 at boot, and a
  decision on whether WebSocket audio ingestion sits behind an ALB (supported) or needs
  sticky sessions (it does, since state is in-process).
- Neither service needs a database for its *current* feature set beyond session
  state — nothing else persists today.

If "deploy to AWS" means "put the demo console + audio pipeline behind a URL for internal
testing with one instance and no auto-scaling," that's realistic this week. If it means
"production, patient data, multiple instances, zero-downtime deploys," items 1–3 above are
non-negotiable first.

---

## Configuration

All settings load from environment variables; `.env` is read automatically
(`python-dotenv`) and is git-ignored. See `.env.example` for the full list with comments.
The two services share the `CLINICAL_COPILOT_*` prefixed settings (auth, STT, diarization)
and `copilot_api` adds its own `COPILOT_*` switches, all optional.

| Variable | Required for | Notes |
|---|---|---|
| `CLINICAL_COPILOT_JWT_SECRET_KEY` | Both, in production | `copilot_api` refuses to start on the placeholder value |
| `CLINICAL_COPILOT_ELEVENLABS_API_KEY` | `app.main` | No offline fallback |
| `CLINICAL_COPILOT_HUGGINGFACE_TOKEN` | `app.main` | Must have accepted the pyannote license |
| `CLINICAL_COPILOT_STT_PROVIDER_MODE` | `app.main` | Must be `production` |
| `CLINICAL_COPILOT_DIARIZATION_MODE` | `app.main` | Must be `production` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | Optional | LLM enrichment. Without one, `llm_status` reports `"unavailable"` — no fabricated fallback text |
| `COPILOT_INTELLIGENCE_ENABLED` | Optional | Master switch for `copilot_api`'s model layer; off by default |
| `CLINICAL_COPILOT_DIARIZATION_DEBUG` | Optional | **Turn this off outside local debugging.** When `True`, every diarized turn writes a full word-by-word transcript dump to `logs/diarization_debug_{session_id}.json` on disk. Defaults to `False`; `logs/` is git-ignored specifically because this has, in the past, been left on and committed |
| `COPILOT_BRIDGE_ENABLED` | Optional | Forwards finalized `app.main` transcripts into `copilot_api`. Off by default |
| `COPILOT_BRIDGE_BASE_URL` | With the bridge on | Where `app.main` reaches `copilot_api`. Default `http://127.0.0.1:8010` |

Full reference: [`docs/configuration.md`](docs/configuration.md).

---

## Repository layout

```
app/                       Audio pipeline service (app.main:app)
  api/                     WebSocket ingress + read-only transcript egress
  services/                Buffering, STT, diarization, speaker ID, clinical extraction
  core/                    Settings, auth primitives
copilot_api/               Copilot event/insights service (separate ASGI app)
integration/               Session runtime, LangGraph intelligence provider, model clients
clinical_intelligence/     Shared deterministic rule engine + optional LLM enrichment
frontend/                  Static demo console for app.main:app (mic capture, live view)
tests/, clinical_intelligence/tests/   ~695 tests across both services
docs/                      Verified, code-referenced documentation — start at docs/README.md
scratch/                   Ad-hoc dev scripts; not part of any shipped service
```

---

## Further reading

- [`docs/README.md`](docs/README.md) — documentation index
- [`docs/architecture.md`](docs/architecture.md) — component map, execution paths
- [`docs/status.md`](docs/status.md) — implemented / mocked / removed, per subsystem
- [`docs/api/README.md`](docs/api/README.md) — every HTTP/WebSocket endpoint on `app.main`
