# External services and integrations

Five external dependencies are referenced. Four are used by code; one is configured but
never imported.

| Service | Purpose | Required to run | Status |
|---|---|---|---|
| ElevenLabs Scribe v2 | Speech-to-text | Yes, for real transcription | **Implemented** |
| Hugging Face + pyannote | Diarization models | Yes in production mode | **Implemented** |
| SpeechBrain ECAPA | Voice embeddings | No — degrades | **Implemented** |
| Google Gemini | LLM enrichment | No — reports unavailable | **Implemented** |
| Neo4j | Knowledge graph | No — degrades | **Implemented**, unpopulated |
| Pinecone | — | No | **Not implemented** |
| OpenAI | Alternative LLM | No | **Partially implemented** |

---

## ElevenLabs Scribe v2

`app/services/providers/scribe_v2.py`

| Property | Value |
|---|---|
| Endpoint | `wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime&include_timestamps=true` |
| Auth | `xi-api-key` header |
| Config | `CLINICAL_COPILOT_ELEVENLABS_API_KEY`, `CLINICAL_COPILOT_STT_PROVIDER_MODE` |

Protocol as implemented:

1. Connect, expect `{"message_type": "session_started"}`, else raise.
2. Send `{"message_type": "input_audio_chunk", "audio_base_64": …, "commit": false}`.
3. Receive `partial_transcript` (cumulative text, no timings) and, per commit, **both**
   `committed_transcript` and `committed_transcript_with_timestamps`.
4. Commit with `"commit": true` and an empty payload.

**Observed behaviour:** every commit arrives twice — once without word timings, once with.
The client requests `include_timestamps=true`, so only the timestamped variant is acted on
(`scribe_v2.py:203-215`).

Retry: 5 attempts, exponential backoff from 1 s (`scribe_v2.py:63-110`). Exhausting them
raises `ConnectionError`.

**No offline mode.** The development mock that returned `"Mock transcription text for
chunk N"` was removed. `STT_PROVIDER_MODE` must be `production` and a key must be present,
or startup fails — there is no offline transcription fallback.

---

## Hugging Face / pyannote

`app/services/diarization_worker.py`

| Property | Value |
|---|---|
| Model | `pyannote/speaker-diarization-3.1` (gated) |
| Auth | `CLINICAL_COPILOT_HUGGINGFACE_TOKEN`, passed as `token=` |
| Offline | `HF_HUB_OFFLINE` defaults to `1` (`app/main.py:24`) |

Loaded once at startup by `preload_pipeline()` when `DIARIZATION_MODE=production`. Runs on
CUDA if available, else CPU (`diarization_worker.py:218`).

**Failure is fatal to startup.** `_lazy_init_pyannote` raises `ImportError` if
torch/pyannote are missing, and the exception propagates out of the startup event, so the
application exits rather than degrading.

The token must have accepted the gated repository's licence. Because `HF_HUB_OFFLINE=1` is
the default, the model must already be in the local cache — a first run requires
`HF_HUB_OFFLINE=0`.

Requires **pyannote.audio 4.x**: the code passes `token=`, which 3.x spells
`use_auth_token=`.

---

## SpeechBrain ECAPA

`app/services/speaker_verification/backend.py`

| Property | Value |
|---|---|
| Model | `speechbrain/spkrec-ecapa-voxceleb` (not gated) |
| Local path | `pretrained_models/spkrec-ecapa-voxceleb/` |
| Output | 192-dimension embedding |
| Enabled by | `CLINICAL_COPILOT_ENABLE_SPEAKER_VERIFICATION` (default true) |

Loaded at startup and warmed with one second of silence. **Failure is caught** and disables
verification without stopping the server (`speaker_verification.py:57-62`).

`LocalStrategy.COPY` is passed to `from_hparams` (`backend.py:36`). This matters on Windows:
SpeechBrain's default `SYMLINK` strategy requires privileges the service usually lacks, and
its fallback writes **small text files containing a cache path** in place of the weights —
which loads without error and produces meaningless embeddings. A real
`embedding_model.ckpt` is ~83 MB.

`pretrained_models/` is git-ignored; weights are fetched on first run.

---

## Google Gemini

`clinical_intelligence/llm/providers.py`

| Property | Value |
|---|---|
| Endpoint | `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` |
| Model | `gemini-2.5-flash` (`config/llm.py` `DEFAULT_GEMINI_MODEL`) |
| Auth | `GEMINI_API_KEY` as a URL query parameter |
| Client | `httpx.AsyncClient`, 30 s timeout |

`generationConfig` sends `temperature`, `maxOutputTokens`, and when a schema is supplied
`responseMimeType: application/json` plus `responseSchema` derived from the prompt
registry's `expected_json_schema`. Keywords Gemini rejects (`$schema`,
`additionalProperties`) are stripped by `_to_response_schema`.

TLS uses the **operating system trust store** via `_build_ssl_context`, not certifi. This is
deliberate: on a network with a TLS-inspecting proxy the intercepting CA is in the OS store
but not in certifi's bundle, and every call would otherwise fail with
`CERTIFICATE_VERIFY_FAILED`.

Resilience: circuit breaker opens after 3 failures for 30 s (`config/runtime.py:35`);
`RETRY_LIMIT = 3`.

### Failure is explicit

There is no fallback provider. `MockLLMProvider` and the automatic
fallback-on-failure were removed: they returned templated strings into the same
`llm_*` response fields real output uses, with nothing marking the source, so a
consumer could not tell a model answer from a canned one.

When no provider answers:

- the `llm_*` fields stay empty (`null` / `[]` / `{}`)
- `llm_status` is `"unavailable"` (or `"disabled"` when `ENABLE_LLM` is off)
- the per-prompt telemetry carries `status: "unavailable"`
- `execute_prompt` returns `None` rather than a substituted payload

The rule-engine output — diagnoses, red flags, investigations, scores — is unaffected;
it is deterministic and does not depend on the LLM.

**At the time of writing** the configured key returned HTTP 429 (quota exhausted), so
`llm_status` reads `"unavailable"` on every run here.

---

## Neo4j

`clinical_intelligence/knowledge_graph/graph_manager.py`

| Setting | Env var | Default |
|---|---|---|
| URI | `NEO4J_URI` | `bolt://localhost:7687` |
| User | `NEO4J_USERNAME` | `neo4j` |
| Password | `NEO4J_PASSWORD` | `password` |

> These use **bare** names, not the `CLINICAL_COPILOT_` prefix, so they are read from the
> process environment via `os.getenv` rather than through `Settings`.

Degradation is graceful: a connection failure is logged, graph enrichment is disabled, and
reasoning continues with `graph_status.status = "DOWN"` (`graph_manager.py:44-62`).

`.env` contains `AURA_INSTANCEID` and `AURA_INSTANCENAME`, which suggest a Neo4j Aura
instance. **No code reads either variable**, and no `NEO4J_URI` is set, so the client
targets localhost. Whether an Aura instance exists is **Unknown / not determinable from the
current code**.

The graph builder, importer, schema and reasoner modules exist but **never execute** on any
live path; `importer.py` is a CLI utility. During verification the graph reported
`node_count: 0`, `last_import: "N/A"`.

---

## Pinecone — not implemented

`.env` sets `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` (`enervara-specialists`).

**No module imports `pinecone`.** Verified by searching the whole codebase for
`import pinecone` / `from pinecone` — zero matches outside the installed package.

Retrieval is instead a **local TF-IDF index** built in-process over the 5 guidelines in
`clinical_intelligence/vector_store/guidelines.yaml` (`vector_store/store.py:82`). There is
no embedding model and no external vector database.

Status: **Not implemented**. The configuration is inert.

---

## OpenAI

`OpenAIProvider` (`llm/providers.py`) targets `https://api.openai.com/v1/chat/completions`
with `gpt-4o-mini`, and is constructed by the orchestrator. `OPENAI_API_KEY` is **not set**
in `.env`, so `generate()` raises immediately and the channel is never exercised.

Unlike the Gemini path, it sends `response_format: {"type": "json_object"}` but **does not
send the schema**, so it is subject to the envelope-shape problem the Gemini path now
avoids.

Status: **Partially implemented**.

---

## Network requirements

| Destination | When |
|---|---|
| `api.elevenlabs.io:443` | Every consultation with a real key |
| `generativelanguage.googleapis.com:443` | Every reasoning call with LLM enabled |
| `huggingface.co:443` | First model download only (`HF_HUB_OFFLINE=0`) |
| `localhost:7687` | Every reasoning call; failure is tolerated |

With `HF_HUB_OFFLINE=1` and models cached, only ElevenLabs and Gemini need egress. Behind a
TLS-inspecting proxy, `pip` and the Hugging Face downloader need an OS-trust-store CA bundle
(see the repository README); the application itself does not.
