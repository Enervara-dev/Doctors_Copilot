# Configuration

Two configuration systems coexist, and they read variables differently.

| System | File | Mechanism | Prefix |
|---|---|---|---|
| Application core | `app/core/config.py` | `pydantic-settings` `BaseSettings`, reads `.env` | `CLINICAL_COPILOT_` (enforced by `env_prefix`) |
| Clinical intelligence | `clinical_intelligence/config/*.py` | plain `os.getenv` after `load_dotenv()` | `CLINICAL_COPILOT_` written manually, **except** the keys below |

`load_dotenv()` runs at `clinical_intelligence/config/__init__.py:4`, which is what makes
un-prefixed variables such as `GEMINI_API_KEY` visible to `os.getenv`.

Variables read **without** the prefix: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `NEO4J_URI`,
`NEO4J_USERNAME`, `NEO4J_PASSWORD`.

`extra = "ignore"` on the `Settings` config means unknown `.env` keys are silently accepted.

---

## Settings that change behaviour

### Modes

| Variable | Default | Effect |
|---|---|---|
| `CLINICAL_COPILOT_STT_PROVIDER_MODE` | `production` | Must be `production`. Any other value raises at startup — the development mock was removed |
| `CLINICAL_COPILOT_DIARIZATION_MODE` | `production` | Must be `production`. Any other value raises at startup — the development mock was removed |

A validator (`config.py:118-128`) raises at startup if a `production` mode is set without
its key: `ELEVENLABS_API_KEY` for STT, `HUGGINGFACE_TOKEN` for diarization.

### Credentials

| Variable | Used by |
|---|---|
| `CLINICAL_COPILOT_ELEVENLABS_API_KEY` | `providers/scribe_v2.py` |
| `CLINICAL_COPILOT_HUGGINGFACE_TOKEN` | `diarization_worker.py` |
| `CLINICAL_COPILOT_AUTH_SECRET_TOKEN` | `core/security.py` — default `production-secure-token-change-me` |
| `CLINICAL_COPILOT_JWT_SECRET_KEY` | `utils/jwt_utils.py` — default `production-jwt-secret-key-change-me` |
| `GEMINI_API_KEY` | `llm/providers.py` (no prefix) |
| `OPENAI_API_KEY` | `llm/providers.py` (no prefix) |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | `knowledge_graph/graph_manager.py` (no prefix) |

### Audio

| Variable | Default | Notes |
|---|---|---|
| `AUDIO_SAMPLE_RATE` | 16000 | Drives frame-size limits |
| `AUDIO_CHANNELS` | 1 | |
| `AUDIO_BITS_PER_SAMPLE` | 16 | |
| `MIN_FRAME_DURATION_MS` / `MAX_FRAME_DURATION_MS` | 20 / 50 | → 640 / 1600 byte frame bounds |
| `BUFFER_CHUNK_DURATION_SEC` | 1.0 | → 32 000-byte chunks |
| `MAX_BUFFER_MEM_LIMIT_BYTES` | 10 MB | Per-stream ceiling; exceeding it clears the buffer and raises |

### Queue and buffers

| Variable | Default | Notes |
|---|---|---|
| `QUEUE_MAX_SIZE` | 100 | Ready-chunk queue depth |
| `QUEUE_OVERFLOW_STRATEGY` | `drop_oldest` | `drop_oldest` \| `block` \| `ignore_and_warn`. `.env` sets `block` |
| `TRANSCRIPT_BUFFER_SIZE` | 500 | Transcript events retained per session |

### Diarization

| Variable | Default | `.env` | Notes |
|---|---|---|---|
| `DIARIZATION_WINDOW_SECONDS` | 5 | 6 | Audio per inference |
| `DIARIZATION_INTERVAL_SECONDS` | 2 | 4 | Inference cadence |
| `DIARIZATION_MIN_SPEAKERS` | 2 | — | |
| `DIARIZATION_MAX_SPEAKERS` | 3 | — | Hard cap on global labels |
| `DIARIZATION_DEBUG` | False | True | Writes `logs/diarization_debug_{session}.json` — **contains transcript text** |
| `DIARIZATION_BUFFER_SECONDS` | 7 | 8 | **Never read by any code** — see below |

### Feature flags — `clinical_intelligence/config/runtime.py`

All default to `True`.

| Variable | Effect |
|---|---|
| `CLINICAL_COPILOT_ENABLE_LLM` | Enables the LLM layer |
| `CLINICAL_COPILOT_ENABLE_RAG` | Enables guideline retrieval |
| `CLINICAL_COPILOT_ENABLE_GRAPH` | Enables knowledge-graph enrichment |
| `CLINICAL_COPILOT_ENABLE_SOAP` | Enables the `SOAPGeneration` task |
| `CLINICAL_COPILOT_ENABLE_PATIENT_EXPLANATION` | Enables the `PatientExplanation` task |
| `CLINICAL_COPILOT_ENABLE_ADAPTIVE_QUESTIONS` | Enables the `AdaptiveQuestions` task |
| `CLINICAL_COPILOT_ENABLE_EMERGENCY_WORKFLOW` | Enables the red-flag escalation bypass |
| `CLINICAL_COPILOT_ENABLE_RED_FLAG_ALERTS` | **Presentation only** — red flags are always detected; when false they are stripped from the response by `presentation/serializer.py:13-17` |

### LLM tuning — `clinical_intelligence/config/llm.py`

| Variable | Default | Notes |
|---|---|---|
| `CLINICAL_COPILOT_MAX_RESPONSE_TOKENS` | 4096 | Gemini 2.5 spends thinking tokens from the same budget |
| `CLINICAL_COPILOT_THINKING_BUDGET` | unset | When set, sends `thinkingConfig.thinkingBudget`; `0` disables thinking |
| (constant) `DEFAULT_TEMPERATURE` | 0.2 | |
| (constant) `CACHE_TTL` | 300 s | |

### Speaker verification — `app/services/speaker_verification/config.py`

| Variable | Default |
|---|---|
| `CLINICAL_COPILOT_ENABLE_SPEAKER_VERIFICATION` | True |
| `CLINICAL_COPILOT_SPEAKER_MATCH_THRESHOLD` | 0.85 |
| `CLINICAL_COPILOT_UNKNOWN_THRESHOLD` | 0.70 |
| `CLINICAL_COPILOT_MIN_SEGMENT_DURATION` | 0.8 s |
| `CLINICAL_COPILOT_MINIMUM_CONFIRMATIONS` | 3 |
| `CLINICAL_COPILOT_MAX_UNKNOWN_SPEAKERS` | 5 |
| `CLINICAL_COPILOT_UNKNOWN_EXPIRY_SECONDS` | 600 |
| `CLINICAL_COPILOT_MIN_RMS_THRESHOLD` | 0.005 |
| `CLINICAL_COPILOT_CLIPPING_THRESHOLD` | 0.98 |

These read `getattr(settings, NAME, default)` first, but none of these names exist on
`Settings`, so the literal defaults apply unless the environment variable is set.

### Process environment

| Variable | Set where | Effect |
|---|---|---|
| `HF_HUB_OFFLINE` | `app/main.py:24` via `setdefault` | Defaults to `1`; models load from cache only. Set `0` for a first download |

---

## Settings defined but never read

Verified by searching for `settings.<NAME>` across `app/` and `clinical_intelligence/`:

| Setting | Status | Consequence |
|---|---|---|
| `SESSION_TIMEOUT_SEC` (300) | **Not implemented** | No inactivity timeout exists. Sessions end only when streams and workers are gone |
| `MAX_ACTIVE_SESSIONS` (1000) | **Not implemented** | No cap is enforced |
| `DIARIZATION_BUFFER_SECONDS` | **Not implemented** | Appears only in `config.py` and a comment at `diarization_worker.py:141`. The audio deque is never trimmed |
| `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | **Not implemented** | `pinecone` is never imported |
| `AURA_INSTANCEID`, `AURA_INSTANCENAME` | **Not implemented** | No code reads them; the Neo4j client targets `NEO4J_URI`, default localhost |

Also note `MAX_RESPONSE_TOKENS` is defined **twice** — `config/llm.py` (4096, used by the
orchestrator) and `config/runtime.py:32` (1024, not used by the Gemini provider). The
duplicate is a trap for anyone tuning it.

---

## `.env` and secrets

`.env` is git-ignored (`.gitignore:13`) but **was committed** in `27a2845`, which is on
`origin/main`. That revision contained a live ElevenLabs API key and the JWT secret.
Ignoring the file now does not remove it from history.

`.env.example` documents only a subset — it omits `GEMINI_API_KEY`, the `PINECONE_*` and
`AURA_*` keys, and every `ENABLE_*` flag.

## Minimum to run

With models already cached:

```
CLINICAL_COPILOT_DIARIZATION_MODE=production
CLINICAL_COPILOT_HUGGINGFACE_TOKEN=<token with the pyannote licence accepted>
CLINICAL_COPILOT_ELEVENLABS_API_KEY=<key>
CLINICAL_COPILOT_STT_PROVIDER_MODE=production
```

Optional: `GEMINI_API_KEY` (LLM enrichment; without it `llm_status` reports
`unavailable` and the `llm_*` fields stay empty), `NEO4J_*` (graph enrichment; degrades
cleanly without it).
