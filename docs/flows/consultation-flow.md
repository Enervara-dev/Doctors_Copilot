# Consultation flow — end to end

Audio to clinical intelligence, in execution order, with the file that does each step.

## Overview

```
microphone ──► WebSocket ──► buffer ──┬──► STT (ElevenLabs) ──┐
                                      │                       ├──► alignment ──► transcript bus
                                      └──► diarization ───────┘                        │
                                                                                       ▼
                                                                          clinical state engine
                                                                                       │
                                                          GET /v1/clinical-reasoning ──►│
                                                                                       ▼
                                                                            rule engine + LLM
```

Two independent consumers read every 1-second audio chunk: the STT worker and the
diarization worker. Their outputs are joined by timestamp in the alignment layer.

---

## 1. Connection

`app/api/websocket.py:31-65`

1. Client opens `ws://…/v1/streaming/audio?session_id=…&role=…&token=…`.
2. Token verified, role verified. Failure closes with code `1008`.
3. `session_manager.get_or_create_session(session_id)` — creating a session **clears any
   existing transcript-bus buffer for that ID** (`session.py:88-98`).
4. `consultation.register_stream(role, websocket)` creates an `AudioSessionBuffer`.

**Edge case (implemented):** re-registering an existing role replaces the stream object but
carries the old buffer over, so a reconnect does not lose partially accumulated audio
(`session.py:49-67`).

## 2. Frame ingestion

`app/api/websocket.py:67-111` → `app/services/audio_buffer.py`

Each frame is validated (`audio_buffer.py:56-72`):

| Check | Failure |
|---|---|
| 640 ≤ size ≤ 1600 bytes | `InvalidAudioFrameError` |
| size even (16-bit alignment) | `InvalidAudioFrameError` |

An invalid frame is reported back on the socket as `{"type":"error",…}` and the connection
stays open. A per-session memory ceiling (`MAX_BUFFER_MEM_LIMIT_BYTES`, 10 MB) clears the
buffer and raises if exceeded (`audio_buffer.py:116-122`).

## 3. Chunking and fan-out

`audio_buffer.py:128-190`

Frames accumulate until `buffer_chunk_bytes` (32 000 = 1.0 s). Each completed chunk is:

1. Wrapped as `AudioChunk` with sequence number and timestamp.
2. Put on `ready_chunks_queue` under the configured backpressure strategy
   (`QUEUE_MAX_SIZE = 100`):
   - `drop_oldest` — discard the oldest chunk, enqueue the new one
   - `block` — await space, applying backpressure to ingestion
   - `ignore_and_warn` — discard the new chunk
3. Passed to `audio_tap.feed_audio(session_id, chunk)` (`audio_buffer.py:190`) — a
   fire-and-forget queue feeding diarization without adding STT latency.

## 4a. Speech-to-text

`app/services/stt_worker.py`, `app/services/providers/scribe_v2.py`

The `STTManager` reconciliation loop (every 100 ms) starts one worker per
`(session_id, role)`. Each worker:

1. Connects to ElevenLabs Scribe v2 (`scribe_v2.py:63-110`), up to 5 attempts with
   exponential backoff, expecting a `session_started` handshake.
2. Streams chunks; drains the queue until the `None` sentinel.
3. Receives `partial_transcript` and `committed_transcript*` messages.

**ElevenLabs emits every commit twice** — once as `committed_transcript` (no word timings)
and once as `committed_transcript_with_timestamps`. Because the client requests
`include_timestamps=true`, only the timestamped variant is acted on; the plain copy is
dropped (`scribe_v2.py:203-215`).

On the final committed transcript the worker triggers and awaits a final global
diarization pass before alignment (`stt_worker.py:86-107`), with a 1200 s safety timeout.

## 4b. Diarization

`app/services/diarization_worker.py`, `app/services/speaker_timeline.py`

Runs on a timer per session:

| Setting | Default | `.env` |
|---|---|---|
| `DIARIZATION_BUFFER_SECONDS` | 7 | 8 |
| `DIARIZATION_WINDOW_SECONDS` | 5 | 6 |
| `DIARIZATION_INTERVAL_SECONDS` | 2 | 4 |
| `DIARIZATION_MIN_SPEAKERS` | 2 | — |
| `DIARIZATION_MAX_SPEAKERS` | 3 | — |

Each cycle converts PCM to a normalised float32 tensor with peak normalisation
(`diarization_worker.py:69-86`), runs pyannote, then maps the window's local labels
(`SPEAKER_00`…) to global labels (`Speaker_0`…) by greedy overlap against the existing
timeline (`diarization_worker.py:548-575`).

A **final global pass** re-runs diarization over the whole session and **overwrites the
timeline**, relabelling by chronological first appearance (`diarization_worker.py:473-500`).

When `DIARIZATION_MODE=development`, `_get_mock_segments` assigns speakers by
`timestamp % 30` (`diarization_worker.py:92-105`). Status: **Mocked**.

## 5. Alignment — who said what

`app/services/speaker_alignment.py`. Detailed in
[speaker-identification.md](speaker-identification.md).

Committed transcripts are split into sentences using the word timestamps, each sentence is
queried against the speaker timeline, contiguous same-speaker sentences are grouped into
turns, and a `TranscriptEvent` is emitted per turn.

Partial transcripts take a simpler path: one event mapped to whoever was speaking in the
last 3 seconds, estimated from bytes received (`speaker_alignment.py:110-146`).

## 6. Transcript bus

`app/services/transcript_bus.py`

In-memory pub/sub with a per-session sliding buffer (`TRANSCRIPT_BUFFER_SIZE`, default 500).

**Requires a running event loop**: `publish()` dispatches to subscribers via
`asyncio.create_task` (`transcript_bus.py:128`). Calling it from synchronous code with no
loop logs `"no running event loop"` and the event reaches no subscriber — the clinical
state silently stays empty.

## 7. Clinical state extraction

`app/services/clinical/state_engine.py` — subscribed at import of `app.services.clinical`.

Per event:

1. `pipeline.process()` — normalise colloquial terms, then run every extractor.
2. Name extraction (all speakers).
3. **Everything else is gated on `speaker_id in ["patient", "attender"]`**
   (`state_engine.py:221`): demographics, chief complaint, symptoms, duration, severity,
   medical history, family history, risk factors, medications, allergies.
4. A separate branch handles doctor speech (`state_engine.py:536`): vitals, examination
   findings, treatment plan, follow-up.
5. `ClinicalStateValidator.validate()` (`clinical/validator.py`).
6. Provenance recorded per fact; `state.version` incremented.

Notable implemented behaviours:

| Behaviour | Where |
|---|---|
| A condition next to a family relation is recorded as family history, not the patient's own | `state_engine.py` §6 |
| Past-history phrasing (`"I was diagnosed with…"`) is not treated as a doctor's working assessment | `state_engine.py` `is_past_history` |
| Negated symptoms are stored with `status="Negated"`, `present="False"` — not discarded | `state_engine.py` §4 |
| Spoken BP (`"160 over 95"`) is parsed as well as `"160/95"` | `extractors/vitals_extractor.py:31` |

**Consequence of the role gate:** with mock diarization, patient utterances labelled
`doctor` are dropped entirely — history and medications go missing with no error.

## 8. Reasoning

Triggered **only** by `GET /v1/clinical-reasoning/{session_id}`. Nothing schedules it.

`clinical_intelligence/integration.py:52` → `rule_engine/engine.py:reason()`:

1. Parse state — vitals, age, gender, history, medications, allergies, family history;
   social history flattened from `risk_factors`.
2. Risk-factor evaluation (33 rules).
3. Symptom matching against 31 diseases with alias resolution.
4. Disease scoring — weights, age/gender modifiers, vitals criteria, negation penalties.
5. Red-flag detection (7 rules).
6. Contraindications (17) and drug interactions (18).
7. Investigation recommendation (39), escalated by active red flags.
8. Missing-information detection.
9. Clinical scores — Wells, HEART, TIMI, NEWS2, qSOFA, CURB-65, CHA₂DS₂-VASc.
10. Knowledge-graph enrichment if Neo4j is reachable; skipped with `graph_status: DOWN` otherwise.
11. Guideline retrieval — local TF-IDF over 5 guidelines.
12. LLM enrichment if `ENABLE_LLM` and a key is present.
13. Explainability trace assembled.

`run_llm` is `ENABLE_LLM and bool(GEMINI_API_KEY or OPENAI_API_KEY)`
(`integration.py:68`).

## 9. LLM enrichment

`clinical_intelligence/llm/task_manager.py`

A DAG of tasks with dependencies and per-task timeouts (`config/runtime.py:61+`):

```
EmergencyEscalation (priority 1, only if red flags and ENABLE_EMERGENCY_WORKFLOW)
AdaptiveQuestions (2)
  ├── InvestigationRecommendations (3)
  └── DifferentialExplanation (4)
        └── ClinicalSummary (5) ─── SOAPGeneration, PatientExplanation
```

Results are cached per session **and context digest** for 300 s
(`task_manager.py:_cache_scope`), so the cache invalidates when the clinical picture moves.

`runner.py:57` imports `nest_asyncio` to re-enter FastAPI's running loop. This import only
happens when a loop is already running — so a missing install fails under the server while
passing from a script.

Every response passes `JSONSchemaValidator` and `HallucinationGuard`, which discards
diagnoses not present in the rule-engine output. On failure the orchestrator falls back to
`MockLLMProvider` (**Mocked** — templated text in the same response fields).

## Failure modes actually implemented

| Failure | Behaviour |
|---|---|
| Invalid audio frame | Error message to client, socket stays open |
| Audio queue full | Per `QUEUE_OVERFLOW_STRATEGY` |
| Session buffer > 10 MB | Buffer cleared, `InvalidAudioFrameError` raised |
| ElevenLabs connect fails | 5 retries with backoff, then `ConnectionError` |
| Neo4j unreachable | Logged, graph enrichment disabled, reasoning continues |
| Gemini fails / rate-limited | Circuit breaker after 3 failures, 30 s cool-down, fall back to `MockLLMProvider` |
| LLM returns invalid JSON | Schema validation fails, fallback used |
| Unsupported diagnosis from LLM | Discarded by `HallucinationGuard` |
| torch/pyannote missing in production mode | **Startup aborts** — the server does not run |
| ECAPA model missing | Logged; verification disabled; the rest continues |
| Client disconnect | Trailing audio flushed, sentinel enqueued, stream unregistered |
