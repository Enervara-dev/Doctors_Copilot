# API reference

Complete inventory of the HTTP and WebSocket surface. There are **six** endpoints; there
are no others.

Source: `app/main.py`, `app/api/websocket.py`.

Interactive docs are served by FastAPI at `/docs` and `/redoc`.

## CORS

`app/main.py:38-44` — `allow_origins=["*"]`, `allow_credentials=True`, all methods, all
headers. Status: **Implemented**, and permissive: any origin may call every HTTP endpoint.

## Authentication summary

| Surface | Authenticated |
|---|---|
| `WS /v1/streaming/audio` | Yes — `token` query parameter |
| All four `GET` endpoints | **No** — no token, header or cookie is checked |

Knowing a `session_id` is sufficient to read that consultation's transcript, clinical
state and reasoning. See [../auth/README.md](../auth/README.md).

---

## `WS /v1/streaming/audio`

Real-time audio ingress. `app/api/websocket.py:31`.

### Query parameters — all required

| Parameter | Type | Validation |
|---|---|---|
| `session_id` | string | None beyond presence. Any string creates or joins a session |
| `role` | string | Must be one of `doctor`, `patient`, `attender` (`config.py` `ALLOWED_ROLES`) |
| `token` | string | Static token or a valid JWT — see below |

### Handshake

1. `verify_session_token(token, session_id, role)` (`core/security.py:24`).
   Failure → close `1008` with reason `"Invalid authentication token"`.
2. `verify_participant_role(role)` (`core/security.py:62`).
   Failure → close `1008` with reason `"Invalid participant role"`.
3. `websocket.accept()`, then the session and participant stream are registered.

Re-connecting with a `(session_id, role)` pair that is already registered **replaces** the
stream and **carries over the existing audio buffer** (`session.py:49-67`).

### Audio contract

| Property | Value | Source |
|---|---|---|
| Encoding | Linear PCM, 16-bit signed, little-endian | `config.py` |
| Sample rate | 16 000 Hz | `AUDIO_SAMPLE_RATE` |
| Channels | 1 (mono) | `AUDIO_CHANNELS` |
| Frame size | 640–1600 bytes inclusive (20–50 ms) | `min_frame_bytes` / `max_frame_bytes` |
| Frame alignment | Length must be even | `audio_buffer.py:68` |

### Message formats

**Binary** — the raw PCM frame as the WebSocket payload.

**Text (JSON)**:

```json
{ "seq": 42, "audio": "<base64-encoded PCM frame>" }
```

`seq` is optional and used only for drop monitoring (`audio_buffer.append_frame(..., seq=)`).
A payload missing `audio` is rejected.

### Server → client messages

The server sends only error notifications; **transcripts are not pushed over this socket**.

```json
{ "type": "error", "message": "Audio frame size of 320 bytes is outside the allowed range of [640, 1600] bytes..." }
```

The connection stays open after a frame error (`websocket.py:81-88`).

### Disconnect handling

`websocket.py:124-174` — flushes trailing buffered audio into the ready queue, enqueues a
`None` sentinel for the STT worker, signals end-of-stream to the diarization worker via
`audio_tap.feed_sentinel`, then unregisters the stream.

---

## `GET /health`

`app/main.py:93`.

```json
{
  "status": "healthy",
  "timestamp": 1788009723.71,
  "active_sessions_count": 0,
  "active_sessions": []
}
```

`active_sessions` is the list of session IDs currently in `session_manager`. Status is the
literal string `"healthy"` whenever the process is serving — it does **not** reflect the
health of ElevenLabs, Neo4j, Gemini or the models.

---

## `GET /v1/transcripts/{session_id}`

`app/main.py:107`. Returns buffered transcript events.

| Query parameter | Default | Effect |
|---|---|---|
| `include_partials` | `false` | When false, only events with `is_final == true` are returned |

Returns a **JSON array** (not an object). An unknown `session_id` returns `[]`, not a 404.

Buffer size is capped by `TRANSCRIPT_BUFFER_SIZE` (default 500) per session
(`transcript_bus.py`), and the buffer is **cleared** when a session is created
(`session.py:90`) and during speaker re-alignment (`speaker_alignment.realign_session`).

### Element shape — `TranscriptEvent` (`app/services/providers/models.py`)

| Field | Type | Notes |
|---|---|---|
| `session_id` | string | |
| `role` | string | The role of the **WebSocket connection** the audio arrived on |
| `sequence_number` | int | Per-role counter |
| `timestamp` | float | Wall-clock, `time.time()` |
| `transcript` | string | The text |
| `is_partial` | bool | |
| `is_final` | bool | |
| `confidence` | float \| null | Provider transcription confidence |
| `provider` | string | `"scribe_v2"` |
| `event_id` | string | UUID |
| `speaker_id` | string \| null | **The attributed speaker** — `doctor`/`patient`/`attender`/`UNKNOWN`. Derived from diarization, *not* from `role` |
| `speaker_label` | string \| null | Same value as `speaker_id` in current code |
| `start_time` | float \| null | Seconds from session audio start |
| `end_time` | float \| null | |
| `text` | string \| null | Duplicate of `transcript` |
| `version` | int | Incremented on retroactive enrichment |
| `updated_at` | float \| null | |
| `speaker_confidence` | float \| null | |
| `speaker_method` | string \| null | `"timestamp"`, `"diarization"`, `"overlap"` |

> `role` and `speaker_id` are different things. `role` is which socket sent the audio;
> `speaker_id` is who the pipeline decided was talking.

Serialized with the Pydantic v1 `.dict()` shim (`main.py:117`).

---

## `GET /v1/clinical-state/{session_id}`

`app/main.py:119`. Returns `ClinicalState.model_dump()`.

An unknown session returns a **fresh empty state**, not a 404 —
`clinical_state_engine.get_state()` creates one on access.

See [../data/models.md](../data/models.md#clinicalstate) for the full shape.

---

## `GET /v1/clinical-state/{session_id}/provenance`

`app/main.py:128`. Returns the provenance dictionary: which transcript event and state
version established each fact.

Keys are dotted fact paths — `"medical_history.hypertension"`,
`"family_history.father.coronary artery disease"`, `"patient_info.age"`, `"severity"` —
written by `state_engine._record_provenance`. Status: **Implemented**.

The exact value structure is not documented here because it is written at a single call
site; read `_record_provenance` in `app/services/clinical/state_engine.py` for the shape.

---

## `GET /v1/clinical-reasoning/{session_id}`

`app/main.py:136`. Runs the full reasoning pass **synchronously on the request**.

When no state exists:

```json
{ "message": "No reasoning results available yet." }
```

Otherwise returns `serialize_reasoning_result(result, show_red_flags=ENABLE_RED_FLAG_ALERTS)`.

### Response keys

Observed from a live response and matching `ClinicalReasoningResult`:

| Key | Type | Notes |
|---|---|---|
| `session_id` | string | |
| `reasoning_timestamp` | string | |
| `candidate_diagnoses` | array | `disease_id`, `name`, `category`, `confidence` (0–1), `rank`, `supporting_symptoms`, `missing_symptoms`, `negated_symptoms`, … |
| `red_flags` | array | **Emptied** when `ENABLE_RED_FLAG_ALERTS=False` |
| `risk_factors` | array | `factor_id`, `name`, `category`, `present`, `source`, `confidence_modifier` |
| `recommended_investigations` | array | `investigation_id`, `name`, `priority`, `reason`, `for_conditions` |
| `missing_information` | array | `field`, `question`, `priority`, `relevant_conditions` |
| `contraindications` | array | `contraindication_id`, `drug_or_procedure`, `reason`, `trigger`, `severity` (`ABSOLUTE`/`RELATIVE`), `recommendation` |
| `drug_interactions` | array | `interaction_id`, `drug_a`, `drug_b`, `severity` (`MAJOR`/`MODERATE`/`MINOR`), `mechanism`, `clinical_effect`, `recommendation` |
| `clinical_alerts` | array | Red-flag alerts filtered out with the same flag |
| `explainability` | object | `reasoning_steps` |
| `engine_version`, `rule_set_version` | string / object | Rule counts per file |
| `clinical_scores` | array | `name`, `score`, `interpretation`, `evidence` |
| `debug_diagnostics` | object \| null | |
| `graph_metadata`, `graph_status` | object \| null | `graph_status.status` is `"DOWN"` when Neo4j is unreachable |
| `llm_adaptive_questions` | array | |
| `llm_differential_explanations` | object | |
| `llm_recommended_investigations` | array | |
| `llm_clinical_summary` | object \| null | |
| `llm_soap_notes` | object \| null | `subjective`, `objective`, `assessment`, `plan` |
| `llm_patient_explanation` | object \| null | |
| `llm_emergency_escalation` | object \| null | |
| `llm_explainability` | object | `generated_from_prompt_id`, `prompt_version`, `citations` |
| `needs_human_review` | bool | |
| `decision_support` | object \| null | |
| `pipeline_version`, `retriever_version`, `prompt_version`, `provider_version` | string | |

> **`llm_*` fields do not indicate their source.** When the provider fails, the same fields
> are populated by `MockLLMProvider` with templated text. Nothing in the response
> distinguishes model output from fallback text. See
> [../integrations/README.md](../integrations/README.md#google-gemini).

### Performance

Latency varies with LLM behaviour: sub-second when results are cached or the circuit
breaker is open, up to ~30 s observed when provider retries and timeouts are in play
(`PROVIDER_TIMEOUT_SECONDS = 30.0`, `RETRY_LIMIT = 3`). There is no request timeout on the
endpoint itself.

---

## Not present

Verified absent from the codebase:

- Session create / end / finalise
- Report or PDF generation over HTTP
- Voice enrolment
- Any `POST`, `PUT`, `PATCH` or `DELETE` route
- Pagination, filtering or sorting parameters (beyond `include_partials`)
- Rate limiting
- API versioning beyond the literal `/v1` prefix
