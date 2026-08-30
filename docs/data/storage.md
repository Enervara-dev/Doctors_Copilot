# Storage and session lifecycle

## There is no database

Verified by searching the codebase for `sqlite`, `postgres`, `psycopg`, `sqlalchemy`,
`redis`, `motor` and `pymongo`: **no database driver is imported by any application
module.** (`sqlalchemy` and `alembic` are present in `.venv` as transitive dependencies of
other packages; no project code imports them.)

Status: **Not implemented** — no database, no ORM, no migrations, no connection string.

## What holds state

Everything is a Python dictionary in the API process, lost on restart.

| Store | Location | Keyed by | Contents | Bounded |
|---|---|---|---|---|
| Sessions | `session.py:81` `SessionManager._sessions` | `session_id` | `ConsultationSession` with per-role streams and sequence counters | No |
| Audio buffer | `audio_buffer.py` `AudioSessionBuffer.raw_buffer` | per stream | Frames not yet formed into a 1 s chunk | 10 MB (`MAX_BUFFER_MEM_LIMIT_BYTES`) |
| Ready chunks | `audio_buffer.py` `ready_chunks_queue` | per stream | 1 s chunks awaiting STT | 100 (`QUEUE_MAX_SIZE`) |
| Diarization audio | `diarization_worker.py:142` `chunks_deque` | `session_id` | **Every audio byte of the session** | **No — see below** |
| Speaker timeline | `speaker_timeline.py` | `session_id` | Speaker segments | No |
| Committed STT cache | `speaker_alignment.py` `_committed_cache` | `session_id` | Raw provider responses for replay | No |
| Cluster→role mapping | `speaker_alignment.py` `_session_mappings` | `session_id` | Label→role | No |
| Voice profiles | `voice_registry.py` `_registries` | `session_id` | ECAPA embeddings + running statistics | 5 unknown slots; 600 s expiry for unknowns |
| Transcript events | `transcript_bus.py` | `session_id` | Sliding window of events | 500 (`TRANSCRIPT_BUFFER_SIZE`) |
| Clinical state | `state_engine.py:33` `_states` | `session_id` | `ClinicalState` | No |
| Provenance | `state_engine.py:34` `_provenance` | `session_id` | Fact → source event | No |
| State metadata | `state_engine.py:35` `_metadata` | `session_id` | Change signals (**never read** — see [status](../status.md)) | No |
| LLM results | `llm/orchestrator.py:35` `LLMResultCache` | `session_id` + context digest | Parsed LLM output | 300 s TTL |
| Graph query cache | `knowledge_graph/cache.py` | query | Neo4j results | TTL-based |

### Unbounded audio retention

`diarization_worker.py:141` carries the comment:

> `# Older bytes are evicted in feed_audio() to respect DIARIZATION_BUFFER_SECONDS.`

**No eviction exists.** There is no `popleft`, no `maxlen`, no slicing of `chunks_deque`
anywhere in the file. Every byte is retained for the session's lifetime: ~32 kB/s, roughly
57 MB for a 30-minute consultation, per session.

This is load-bearing, not merely a leak: `get_session_audio_slice`
(`diarization_worker.py:296`) indexes the concatenated deque by **absolute** byte offset,
which is only correct because nothing is ever evicted. Trimming the buffer without changing
that indexing would silently return the wrong audio to speaker verification.

Status: **Not implemented** (eviction), and the code depends on its absence.

## Files written to disk

Exactly one write path exists in the application (`speaker_alignment.py:345-357`):

```
logs/diarization_debug_{session_id}.json
```

Written only when `CLINICAL_COPILOT_DIARIZATION_DEBUG=True`. Contains pyannote segments,
raw words with timestamps, the word→speaker mapping with reasons, and the final events.
Overwritten on each commit, so it holds only the **most recent** commit for that session.

**It contains consultation transcript text**, so it is patient data on disk with no
retention policy, no rotation and no access control. The repository currently tracks 26
such files.

No other application code writes files. `generate_pdf_report.py` and
`generate_summary_report.py` write PDFs but are standalone scripts, not reachable from the
API.

## Session lifecycle

### Creation

Implicit. `get_or_create_session` (`session.py:84`) creates a session on the first
WebSocket connection with that ID. There is **no create endpoint** and no validation of the
identifier. Creating a session **clears any existing transcript-bus buffer** for that ID.

The `GET` endpoints also create state implicitly: `clinical_state_engine.get_state()`
returns a fresh empty `ClinicalState` for an unknown ID rather than 404.

### Cleanup

`clean_empty_sessions` (`session.py:108`) is called from the STT reconciliation loop every
100 ms. A session is deleted when it has **no streams and no running workers**, which also
clears the diarization state, speaker timeline, alignment caches and voice registry.

`SESSION_TIMEOUT_SEC` (300) is defined in `app/core/config.py` but **no code reads it** —
there is no inactivity timeout. Status: **Not implemented**.

### End of consultation

There is **no end/finalise action**. A consultation ends when participants disconnect and
cleanup removes it. Nothing is archived, exported or summarised on the way out.

Status: **Not implemented**.

### Restart

All state is lost. Transcripts, clinical state, reasoning results and voice profiles do not
survive a restart, and cannot be recovered.

## Consequences

| Consequence | Detail |
|---|---|
| Single instance only | Two API processes would not share sessions; there is no shared store or sticky-session mechanism |
| No consultation history | Nothing can be retrieved after cleanup or restart |
| No audit trail at rest | Audit lines (`speaker_verification.py`) go to stdout only |
| Memory grows with session count and duration | Only the audio queue and transcript buffer are bounded; state, provenance and diarization audio are not |
| No data retention or deletion controls | No expiry, no purge, no per-patient deletion path |
