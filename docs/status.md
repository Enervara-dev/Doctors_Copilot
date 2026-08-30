# Implementation status

Every subsystem, with its status and where to verify it.

Status labels are defined in [README.md](README.md#how-to-read-this).

---

## Audio ingestion

| Feature | Status | Evidence |
|---|---|---|
| WebSocket audio ingress | **Implemented** | `app/api/websocket.py:31` |
| Binary PCM frames | **Implemented** | `websocket.py:76-88` |
| Base64 JSON frames | **Implemented** | `websocket.py:90-111` |
| Frame validation (size, 16-bit alignment) | **Implemented** | `audio_buffer.py:60-72` |
| 1-second chunking | **Implemented** | `audio_buffer.py:129-143` |
| Backpressure: `drop_oldest` / `block` / `ignore_and_warn` | **Implemented** | `audio_buffer.py:146-175` |
| Reconnect preserving the buffer | **Implemented** | `session.py:49-67` — a re-registering role inherits the previous `AudioSessionBuffer` |
| Per-frame error reported to the client | **Implemented** | `websocket.py:88` sends `{"type":"error","message":...}` |
| Trailing-audio flush on disconnect | **Implemented** | `websocket.py:127-150` |
| Multi-participant sessions (doctor/patient/attender) | **Implemented** | one WebSocket per role, keyed in `session.streams` |
| Rolling-buffer eviction | **Not implemented** | `diarization_worker.py:141` comments that eviction happens in `feed_audio()`; no eviction code exists |

## Speech-to-text

| Feature | Status | Evidence |
|---|---|---|
| ElevenLabs Scribe v2 streaming | **Implemented** | `providers/scribe_v2.py`, `ws_url` line 27 |
| Connection retry with backoff (5 attempts) | **Implemented** | `scribe_v2.py:63-110` |
| Word-level timestamps | **Implemented** | `include_timestamps=true` in the URL |
| Duplicate-commit suppression | **Implemented** | `scribe_v2.py:203-215` — ElevenLabs emits each commit twice; the untimestamped copy is dropped |
| Partial transcripts | **Implemented** | `scribe_v2.py:196-201`; surfaced only when `include_partials=true` |
| Worker lifecycle reconciliation | **Implemented** | `stt_manager.py:68-108` |
| `STT_PROVIDER_MODE=development` | **Mocked** | `scribe_v2.py:119,144,160` — emits `"Mock transcription text for chunk N"`. Only active when the API key is **also** empty |

## Diarization and speaker identification

See [flows/speaker-identification.md](flows/speaker-identification.md).

| Feature | Status | Evidence |
|---|---|---|
| pyannote `speaker-diarization-3.1` | **Implemented** | `diarization_worker.py:220-225` |
| Sliding-window inference | **Implemented** | `diarization_worker.py`, `DIARIZATION_*` settings |
| Window→global label matching (greedy overlap) | **Implemented** | `diarization_worker.py:548-575` |
| Final global re-label pass | **Implemented** | `diarization_worker.py:473-500` — overwrites the timeline, relabelling by chronological first appearance |
| Speaker timeline with overlap/proximity/continuity fallback | **Implemented** | `speaker_timeline.py` |
| Sentence-level alignment to speakers | **Implemented** | `speaker_alignment.py:86+` |
| Turn-gap splitting (1.5 s) | **Implemented** | `speaker_alignment.py`, `TURN_GAP_SECONDS` |
| Commit-boundary word rejoining | **Implemented** | `speaker_alignment.py`, `_stitch_commit_boundaries` |
| Cluster-level role assignment | **Implemented** | `speaker_alignment.py`, `assign_cluster_roles` — keyword evidence pooled per speaker cluster |
| ECAPA voice embeddings | **Implemented** | `speaker_verification/backend.py` |
| Voice profile matching, confirmation counters, locking | **Implemented** | `speaker_verification/speaker_verification.py`, `voice_registry.py` |
| `DIARIZATION_MODE=development` | **Mocked** | `diarization_worker.py:92-105` — speakers assigned by `timestamp % 30`. **Will mis-attribute speech**, and because extraction is role-gated, facts are silently dropped |
| Speaker verification correcting a wrong label | **Partially implemented** | The mechanism exists (`_process_verified_mapping`) but requires ≥3 confirmations and a voice match; not observed correcting a real mislabel in this codebase's test material |

## Clinical extraction

| Feature | Status | Evidence |
|---|---|---|
| Terminology normalisation | **Implemented** | `clinical/normalizer.py` |
| Symptom / medication / diagnosis / procedure / investigation extraction | **Implemented** | `clinical/extractors/*.py` |
| Negation detection | **Implemented** | `extractors/negation.py` |
| Vitals extraction (BP, pulse, temp, SpO₂, weight, height) | **Implemented** | `extractors/vitals_extractor.py` |
| Family history attribution | **Implemented** | `state_engine.py` §6 — conditions next to a family relation are recorded as family history, not the patient's own |
| Social risk factors (smoking, alcohol, …) | **Implemented** | `extractors/history_extractor.py` + persisted in `state_engine.py` §6b |
| Confidence scoring | **Implemented** | `extractors/confidence.py` |
| Provenance tracking | **Implemented** | `state_engine._record_provenance`, exposed at `/v1/clinical-state/{id}/provenance` |
| Section detection | **Implemented** | `extractors/section_detector.py` |
| State validation | **Implemented** | `clinical/validator.py`, called from `state_engine.py:193` |
| Change-significance signals | **Placeholder** | `state_engine.py:156-175` — computed, never read by production code; only `tests/` call them |
| Dictionary coverage | **Partially implemented** | 17 symptoms, 17 diseases, 17 drugs, 12 procedures, 10 anatomy, 7 family, 2 negations — small enough to be a demo vocabulary |

## Clinical reasoning (rule engine)

| Feature | Status | Evidence |
|---|---|---|
| Deterministic disease scoring | **Implemented** | `rule_engine/scorer.py`, 31 diseases in `rules/diseases.yaml` |
| Symptom matching with aliases | **Implemented** | `rule_engine/matcher.py`, 72 symptom groups |
| Risk-factor evaluation | **Implemented** | `rule_engine/risk_evaluator.py`, 33 rules |
| Red-flag detection | **Implemented** | `rule_engine/red_flags.py`, 7 rules |
| Contraindication checks | **Implemented** | `rule_engine/contraindications.py`, 17 rules |
| Drug-interaction checks | **Implemented** | `rule_engine/interactions.py`, 18 rules |
| Investigation recommendation | **Implemented** | `rule_engine/investigations.py`, 39 rules |
| Missing-information detection | **Implemented** | `rule_engine/missing_info.py` |
| Explainability trace | **Implemented** | `rule_engine/explainability.py` |
| Clinical scores (Wells, HEART, TIMI, NEWS2, qSOFA, CURB-65, CHA₂DS₂-VASc) | **Implemented** | `rule_engine/scores/` — all seven executed in tracing |
| Red flags in the API response | **Partially implemented** | Detected always; **stripped from the response** when `ENABLE_RED_FLAG_ALERTS=False` (`presentation/serializer.py:13-17`). The deployed `.env` sets it to `False` |
| Red-flag specificity | **Partially implemented** | A chest-pain + sweating case produced 6 CRITICAL flags including stroke and sepsis. Rules fire broadly; tuning is a known gap |

## Knowledge graph

| Feature | Status | Evidence |
|---|---|---|
| Neo4j connection management | **Implemented** | `knowledge_graph/graph_manager.py` |
| Query execution + TTL cache | **Implemented** | `graph_queries.py`, `cache.py` |
| Graceful degradation when Neo4j is absent | **Implemented** | `graph_manager.py:44-62`; reasoning continues, `graph_status.status = "DOWN"` |
| Graph builder / importer / schema / reasoner | **Not executed** | `graph_builder.py`, `importer.py`, `schema.py`, `graph_reasoner.py` — no function runs on any live path. `importer.py` is a CLI tool |
| A populated graph | **Unknown** | No Neo4j instance was reachable during verification; `node_count: 0`, `last_import: "N/A"` observed. Whether a populated instance exists elsewhere is not determinable from the code |

## Retrieval (RAG)

| Feature | Status | Evidence |
|---|---|---|
| Guideline retrieval | **Implemented** | `rag/retriever.py` + `vector_store/store.py` |
| Vector index | **Partially implemented** | Local TF-IDF built in-process (`store.py:82`), over **5 guidelines** in `vector_store/guidelines.yaml`. No embedding model, no external vector database |
| Pinecone | **Not implemented** | `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` are set in `.env`, but `pinecone` is **never imported anywhere in the codebase** |

## LLM enrichment

| Feature | Status | Evidence |
|---|---|---|
| Prompt registry with JSON schemas | **Implemented** | `prompts/registry.py`, 7 prompts |
| DAG task scheduler with dependencies | **Implemented** | `llm/task_manager.py`, `config/runtime.py:61+` |
| Gemini client | **Implemented** | `llm/providers.py`, REST via httpx |
| Structured output (`responseSchema`) | **Implemented** | `llm/providers.py` — registry schema forwarded, unsupported keywords stripped |
| OS trust-store TLS | **Implemented** | `llm/providers.py:_build_ssl_context` |
| Circuit breaker + retry | **Implemented** | `llm/providers.py`, `config/runtime.py:35` |
| Result caching keyed on session + context digest | **Implemented** | `llm/task_manager.py:_cache_scope`, 300 s TTL |
| Hallucination guard | **Implemented** | `validators/hallucination_guard.py` — discards diagnoses not present in the rule output |
| Response schema validation | **Implemented** | `validators/schema_validator.py` |
| **Fallback text presented as model output** | **Mocked** | `llm/providers.py` `MockLLMProvider` — on provider failure, returns templated strings (`"Patient is presenting with symptoms suggestive of {diagnosis}."`). These populate the same `llm_*` response fields as real output, **with no field marking them as fallback** |
| OpenAI channel | **Partially implemented** | `OpenAIProvider` exists and is constructed; `OPENAI_API_KEY` is not set in `.env`, so it is never exercised |
| Live Gemini output | **Unknown at time of writing** | The configured key returned HTTP 429 (quota exhausted) throughout verification. The request/response wiring is unit-tested; end-to-end model output was not observable |

## API

| Endpoint | Status | Evidence |
|---|---|---|
| `GET /health` | **Implemented** | `app/main.py:93` |
| `WS /v1/streaming/audio` | **Implemented** | `app/api/websocket.py:31` |
| `GET /v1/transcripts/{id}` | **Implemented** | `app/main.py:107` |
| `GET /v1/clinical-state/{id}` | **Implemented** | `app/main.py:119` |
| `GET /v1/clinical-state/{id}/provenance` | **Implemented** | `app/main.py:128` |
| `GET /v1/clinical-reasoning/{id}` | **Implemented** | `app/main.py:136` |
| Session end / finalise endpoint | **Not implemented** | No route creates, closes or finalises a consultation |
| Report/PDF endpoint | **Not implemented** | `generate_pdf_report.py` and `generate_summary_report.py` are standalone scripts, not wired to any route |
| Voice enrolment endpoint | **Not implemented** | The voice registry is populated only from live diarized segments |

## Authentication and authorization

See [auth/README.md](auth/README.md).

| Feature | Status | Evidence |
|---|---|---|
| Static shared token | **Implemented** | `core/security.py:38` |
| JWT with session + role claims | **Implemented** | `core/security.py:42-58`, `utils/jwt_utils.py` |
| Role allow-list | **Implemented** | `core/security.py:62-68` |
| Auth on HTTP endpoints | **Not implemented** | Only the WebSocket authenticates. All four `GET` endpoints are unauthenticated |
| User accounts / identity | **Not implemented** | No user model, no login, no per-user data |
| Per-session authorization | **Partially implemented** | JWT claims are checked against the requested session and role — but the static token bypasses this entirely for any session and any role |

## Frontend

See [flows/frontend-flow.md](flows/frontend-flow.md).

| Feature | Status | Evidence |
|---|---|---|
| Microphone capture → 16 kHz PCM → 20 ms frames | **Implemented** | `frontend/app.js` `startRecording`, `resample`, `toInt16`, `flushFrames` |
| Input level meter from real RMS | **Implemented** | `app.js` `paintLevel` |
| Live transcript rendering | **Implemented** | `app.js` `refreshTranscript`, 2 s polling |
| Clinical state rendering | **Implemented** | `app.js` `refreshState` |
| Reasoning rendering | **Implemented** | `app.js` `runAnalysis` |
| Vitals status thresholds | **Implemented** | `app.js` `vitalState` — **client-side display thresholds only**, not from the backend |
| Settings drawer | **Implemented** | `app.js`, drawer handlers |
| Theme toggle + persistence | **Implemented** | `app.js`, `localStorage["copilot-theme"]` |
| Deep links (`?session=`, `?api=`, `?role=`, `?token=`, `?theme=`) | **Implemented** | `app.js` `applyQuery` |
| Auto re-analyse checkbox (30 s) | **Implemented** | `app.js` `startPolling` |
| Rail "Consultation" nav button | **Placeholder** | `index.html:22` has `data-view="consult"`; **no event listener is bound to it** anywhere in `app.js` |
| Routing / multiple screens | **Not implemented** | Single page, no router, no history manipulation |
| Login / logout UI | **Not implemented** | The token is a plain text field in the settings drawer |
| Patient record persistence | **Not implemented** | The patient header is rendered from the live session's `ClinicalState` only |

## Dormant code

**Removed.** The packages previously listed here — `interview/`, `workflow/`, most of
`evidence/`, `events/`, `session/`, `storage/`, `timeline/`, `core/`,
`knowledge_graph/graph_reasoner.py`, `config/evidence.py`, `config/workflow.py` and
`app/services/conversation/` — were deleted in the dead-code cleanup, together with the
five test modules that existed only to exercise them.

Two files on the original list were **kept**, because the trace-based audit understated
their reachability:

| Kept | Why |
|---|---|
| `knowledge_graph/schema.py` | `graph_manager.create_constraints_and_indexes()` imports `CONSTRAINTS` and `INDEXES` from it. That method returns early when Neo4j is inactive, which is why the trace (run with Neo4j down) never reached the import. Removing it would break constraint creation whenever a graph is actually connected |
| `knowledge_graph/graph_builder.py` | Required by `importer.py`, retained as the standalone CLI that populates the graph |

`knowledge_graph/importer.py` is retained as a CLI utility; `test_knowledge_graph.py` also
imports `validate_graph_data` from it.

One dead block was removed from a live file: `rule_engine/engine.py` imported
`core.container` and `workflow.protocols` to attach workflow telemetry, inside a
`try/except Exception: pass`. Nothing ever registered those interfaces in the container
outside the deleted tests, so `resolve()` always raised and the block never had an effect.

## Operational gaps

| Gap | Status | Evidence |
|---|---|---|
| Persistence of any kind | **Not implemented** | See [data/storage.md](data/storage.md) |
| Test coverage of the reasoning engine and extraction | **Implemented** | 344 tests pass (`tests/` + `clinical_intelligence/tests/`) |
| Deployment artifacts (Dockerfile, CI) | **Not implemented** | None in the repository |
| Structured audit log persistence | **Partially implemented** | Audit lines are logged to stdout (`speaker_verification.py`); nothing writes them to durable storage |
| Secret hygiene | **Known issue** | `.env` was committed in history (`27a2845`) and is reachable on `origin/main`; it contained a live ElevenLabs key. `.gitignore:13` covers `.env` only from that commit onward |
