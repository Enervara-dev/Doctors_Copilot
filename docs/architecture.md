# Architecture

## What the system is

A single FastAPI process that ingests live consultation audio over WebSockets, transcribes
it, attributes each utterance to a speaker, extracts clinical facts into a per-session
state object, and runs that state through a deterministic rule engine with optional LLM
enrichment. A static HTML/CSS/JS console reads the results over HTTP.

There is no separate worker process, no message broker, no database. Everything runs in
one Python process; all coordination is `asyncio` tasks and in-memory dictionaries.

## Processes at runtime

| Process | Command | Required |
|---|---|---|
| API + all processing | `uvicorn app.main:app` | Yes |
| Static file server for the console | `python -m http.server 5500 --directory frontend` | Only to use the browser console |

`run.ps1` starts both. There is no containerisation, no CI configuration, and no process
manager in the repository.

## Component map

```
Browser (frontend/)                         [static files, no build step]
   │  WebSocket: PCM audio frames
   │  HTTP GET: transcript / state / reasoning
   ▼
FastAPI app  (app/main.py)
   │
   ├── app/api/websocket.py          WebSocket ingress, auth handshake
   ├── app/services/session.py       session + participant stream registry (in-memory)
   ├── app/services/audio_buffer.py  frame validation, 1 s chunking, backpressure
   │        │
   │        ├──> app/services/stt_worker.py ──> providers/scribe_v2.py ──> ElevenLabs
   │        └──> app/services/audio_tap.py  ──> diarization_worker.py   ──> pyannote
   │                                                    │
   │                                              speaker_timeline.py
   │                                                    │
   │                                          speaker_alignment.py  (who said what)
   │                                                    │
   │                                     speaker_verification/  ──> SpeechBrain ECAPA
   │                                                    │
   │                                          transcript_bus.py  (pub/sub, in-memory)
   │                                                    │
   │                                   clinical/state_engine.py  (subscriber)
   │                                          │  pipeline → normalizer → extractors
   │                                          ▼
   │                                     ClinicalState  (in-memory dict per session)
   │
   └── GET /v1/clinical-reasoning/{id}
            └──> clinical_intelligence/integration.py
                     └──> rule_engine/engine.py
                              ├── rule engine components (deterministic)
                              ├── knowledge_graph/  ──> Neo4j        [degrades if absent]
                              ├── rag/ + vector_store/ (local TF-IDF)
                              └── llm/ ──> Gemini    [falls back to canned text]
```

## The two execution paths

Two distinct paths run at different times. Knowing which is which explains most of the
system's behaviour.

### Path A — ingestion (continuous, while audio streams)

Triggered by audio arriving on the WebSocket. Verified by tracing a transcript-event
sequence; these are the modules in which a function actually executed:

```
app/services/transcript_bus.py
app/services/clinical/state_engine.py
app/services/clinical/pipeline.py
app/services/clinical/normalizer.py
app/services/clinical/extractor.py
app/services/clinical/validator.py
app/services/clinical/dictionary/loader.py
app/services/clinical/extractors/{confidence, context, diagnosis_extractor,
    history_extractor, investigation_extractor, medication_extractor,
    name_extractor, negation, section_detector, symptom_extractor,
    vitals_extractor}.py
app/services/session.py
app/services/audio_buffer.py
```

### Path B — reasoning (on demand, per HTTP request)

Triggered **only** by `GET /v1/clinical-reasoning/{session_id}`. Nothing schedules it.
Modules in which a function actually executed during a reasoning call:

```
clinical_intelligence/integration.py
clinical_intelligence/rule_engine/{engine, loader, matcher, scorer, red_flags,
    risk_evaluator, investigations, missing_info, contraindications, interactions,
    explainability, aggregator, validator}.py
clinical_intelligence/rule_engine/scores/{wells, heart, timi, news2, qsofa,
    curb65, chads2vasc}.py
clinical_intelligence/knowledge_graph/{graph_manager, graph_queries, cache, service}.py
clinical_intelligence/evidence/{disease_profile_registry, evidence_models,
    profile_compiler}.py          # the only three that survived the cleanup
clinical_intelligence/rag/retriever.py
clinical_intelligence/vector_store/store.py
clinical_intelligence/prompts/registry.py
clinical_intelligence/llm/{orchestrator, providers, task_manager, runner,
    context_builder, response_formatter, session_manager}.py
clinical_intelligence/validators/{schema_validator, response_validator,
    hallucination_guard}.py
```

`clinical_intelligence/presentation/serializer.py` executes in the endpoint
(`app/main.py:149`), one level above the traced call.

**There is no automatic re-analysis.** The state engine computes change signals
(`has_significant_state_change`, `clinical_state_complete`, `ready_for_llm` —
`app/services/clinical/state_engine.py:156-175`) but **no production code calls them**;
their only callers are in `tests/test_clinical_state_engine.py`. Status: **Placeholder**.

## Background tasks

| Task | Started by | Interval / trigger |
|---|---|---|
| STT worker reconciliation | `stt_manager.start_orchestrator()` at startup (`app/main.py:68`) | every 100 ms (`stt_manager.py:120`) |
| Per-stream STT worker | reconciliation, one per `(session_id, role)` | on stream registration |
| Diarization loop | `diarization_worker.feed_audio()` | every `DIARIZATION_INTERVAL_SECONDS` |
| Audio tap consumer | first `feed_audio` call | drains a queue |
| Speaker verification worker | `speaker_verification_service.start_worker()` | drains a queue |

## Startup sequence

`app/main.py:54-84`:

1. Log configuration.
2. Start the STT reconciliation loop.
3. Import `app.services.clinical`, which subscribes `ClinicalStateEngine` to the transcript
   bus (`state_engine.py`, logged as "Subscribed to TranscriptEventBus").
4. If `DIARIZATION_MODE == "production"`: `await diarization_worker_manager.preload_pipeline()`.
   **This raises and aborts startup if torch/pyannote are unavailable** — the server does
   not start in degraded mode.
5. If speaker verification is enabled: load the ECAPA backend. A failure here is caught and
   logged; the server continues with verification disabled
   (`speaker_verification.py:57-62`).

`HF_HUB_OFFLINE` is set to `1` via `os.environ.setdefault` at `app/main.py:24`, so model
loads read from the local Hugging Face cache unless the variable is already set.

## Shutdown

`app/main.py:86-90` stops the STT orchestrator and cancels its workers. No state is
persisted. Status: **Implemented** (process-local cleanup only).

## Technology dependencies

| Dependency | Used for | Where |
|---|---|---|
| FastAPI / Uvicorn | HTTP + WebSocket server | `app/main.py` |
| Pydantic v2 + pydantic-settings | models, `.env` loading | `app/core/config.py` |
| websockets | ElevenLabs client | `providers/scribe_v2.py` |
| torch, pyannote.audio 4.x | diarization | `diarization_worker.py` |
| speechbrain | ECAPA speaker embeddings | `speaker_verification/backend.py` |
| numpy, scipy, soundfile | audio handling | various |
| httpx | Gemini/OpenAI HTTP | `llm/providers.py` |
| nest-asyncio | re-entering the running loop for LLM tasks | `llm/runner.py:57` |
| neo4j | knowledge graph | `knowledge_graph/graph_manager.py` |
| PyJWT | token decoding | `app/utils/jwt_utils.py` |
| PyYAML | rule files | `rule_engine/loader.py` |

`pyannote.audio` must be 4.x: the code calls `Pipeline.from_pretrained(..., token=...)`
(`diarization_worker.py:222`), which 3.x spells `use_auth_token`.

## Known architectural constraints

- **Single process, in-memory state.** Two API instances would not share sessions; a
  restart loses every consultation. No horizontal scaling path exists in the code.
- **Clinical extraction is gated on speaker role.** `state_engine.py:221` only extracts
  symptoms, history, medications, allergies and demographics when
  `speaker_id in ["patient", "attender"]`. Extraction accuracy is therefore bounded by
  diarization accuracy; with `DIARIZATION_MODE=development` the labels are synthetic and
  facts are silently dropped.
- **Unbounded per-session audio buffer.** `diarization_worker.py:142` declares
  `chunks_deque` with a comment stating older bytes are evicted in `feed_audio()`; no
  eviction exists (no `popleft`, no `maxlen`). Every byte of a session is retained until
  session cleanup: roughly 32 kB/s, about 57 MB for a 30-minute consultation.
