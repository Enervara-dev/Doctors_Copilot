# Speaker identification

How the system decides who said each sentence. This is the most intricate flow in the
codebase and the one that most affects clinical output, because extraction is gated on
speaker role.

## Why it matters

`app/services/clinical/state_engine.py:221` extracts symptoms, history, medications,
allergies and demographics **only** when `speaker_id in ["patient", "attender"]`. A patient
utterance mislabelled as `doctor` is discarded silently — no error, no warning, the facts
simply never appear in the clinical state.

## The four layers

```
1. pyannote diarization      → acoustic clusters:  Speaker_0, Speaker_1, Speaker_2
2. speaker timeline          → time ranges per cluster
3. alignment                 → sentences mapped to clusters, grouped into turns
4. cluster role assignment   → cluster → doctor / patient / attender
   (+ voice verification, which anchors identity acoustically)
```

Clusters are acoustic. Roles are clinical. The mapping between them is decided once per
session, not per sentence.

## Layer 1 — diarization

`app/services/diarization_worker.py`

**Production** (`DIARIZATION_MODE=production`): pyannote `speaker-diarization-3.1`,
constrained to `min_speakers=2`, `max_speakers=3`.

Sliding windows are stitched into a global timeline by **greedy overlap matching**
(`diarization_worker.py:548-575`): local labels from the new window are matched to existing
global labels by total temporal overlap, highest first, each global label claimed once.
Unmatched locals take a free slot; if all three are taken,
`_map_unmapped_to_existing` falls back to overlap → proximity → continuity.

A **final global pass** (`diarization_worker.py:473-500`) re-diarizes the whole session and
**completely overwrites** the timeline, assigning `Speaker_0/1/2` by chronological first
appearance. This discards the incrementally matched labels.

There is no development mode. The mock generator that cycled speakers by `timestamp % 30`
was removed: its clusters did not correspond to real people, and because extraction is
role-gated it silently discarded clinical facts. `DIARIZATION_MODE` must be `production`.

## Layer 2 — timeline

`app/services/speaker_timeline.py`

Holds `{start, end, label}` segments. `update_timeline(new_segments, window_start)` keeps
segments ending before `window_start`, truncates those straddling it, appends the new ones,
then smooths and filters tiny segments. Labels are hard-capped to
`DIARIZATION_MAX_SPEAKERS` (`SPEAKER_LABELS`).

`get_speaker_for_range(start, end)` resolves a time range to a label, falling back through
overlap → proximity → continuity, and can return details for debugging.

## Layer 3 — alignment

`app/services/speaker_alignment.py:align_and_segment`

For a committed transcript **with** word timestamps:

1. Group words into sentences on `.`, `?`, `!`.
2. Compute each sentence's true span from valid word timestamps.
3. `timeline.get_speaker_for_range(start, end)` → raw cluster label.
4. Map through the session's cluster→role mapping.
5. Group contiguous sentences into turns, splitting when either the speaker changes **or**
   the silence between sentences exceeds `TURN_GAP_SECONDS` (1.5 s). Without the gap rule a
   single 35-second commit collapses into one turn and a short reply arrives carrying the
   preceding conversation.
6. Emit one `TranscriptEvent` per turn and enqueue a voice-verification task per turn.

**Without** word timestamps there is a fallback path that estimates timings at roughly
15 characters per second (`speaker_alignment.py:176`). With duplicate-commit
suppression in place this path is not reached for ElevenLabs commits; it remains for
providers that supply no timings. Status: **Partially implemented** — present, low accuracy
by construction.

### Commit-boundary rejoining

ElevenLabs closes commits on byte boundaries, so a word can arrive split
(`"…I take aml"` / `"odipine daily."`). `_stitch_commit_boundaries` rejoins two consecutive
same-speaker events when the first does not end in sentence punctuation, the second starts
lowercase, and the gap is within `TURN_GAP_SECONDS`. Without it the broken token defeats
entity extraction and the medication is lost.

Applied in two places: across a full replay in `realign_session`, and against the last
published event in `stitch_against_published`, called from `stt_worker.py:113` — realignment
runs before later commits arrive, so the live path needs its own application.

## Layer 4 — cluster role assignment

`speaker_alignment.assign_cluster_roles`

Runs once per session during `realign_session`:

1. `_cluster_sentences` buckets every cached committed sentence by its raw cluster label.
   Sentences without word timings are skipped — they cannot be tied to a cluster.
2. Each cluster is scored over **all** its utterances against `DOCTOR_PATTERNS` and
   `PATIENT_PATTERNS` (regex phrase lists on the class).
3. Roles are assigned best-first: the strongest `doctor − patient` margin becomes `doctor`;
   of the rest, the strongest `patient − doctor` margin becomes `patient`; any remainder
   becomes `attender`.
4. Clusters absent from the cache keep their positional default from
   `SPEAKER_ROLE_MAPPING` (`Speaker_0→doctor`, `Speaker_1→patient`, `Speaker_2→attender`).

Scoring at cluster level rather than per sentence is deliberate: a cluster is a person, so
evidence pooled across everything they said is far more stable than keyword-matching a
single utterance.

## Voice verification

`app/services/speaker_verification/`

| Component | Role |
|---|---|
| `backend.py` | SpeechBrain ECAPA-TDNN, 192-dimension embeddings, CPU or CUDA |
| `voice_registry.py` | Per-session voice profiles with Welford running mean/variance |
| `speaker_verification.py` | Quality gating, matching, confirmation counting, locking |
| `embedding_store.py` | Cosine similarity |

Flow: alignment enqueues `(session_id, start, end, proposed_role, raw_speaker)` per turn →
a background worker slices the exact PCM via
`diarization_worker.get_session_audio_slice` → audio quality gate (duration ≥
`MIN_SEGMENT_DURATION` 0.8 s, RMS, clipping) → ECAPA embedding → cosine match against
registered profiles.

Thresholds (`speaker_verification/config.py`):

| Setting | Default |
|---|---|
| `SPEAKER_MATCH_THRESHOLD` | 0.85 |
| `UNKNOWN_THRESHOLD` | 0.70 |
| `MIN_SEGMENT_DURATION` | 0.8 s |
| `MINIMUM_CONFIRMATIONS` | 3 |
| `MAX_UNKNOWN_SPEAKERS` | 5 |
| `UNKNOWN_EXPIRY_SECONDS` | 600 |

Profiles are keyed by **diarization cluster**, not by role. Roles are applied on top by the
alignment layer. After `MINIMUM_CONFIRMATIONS` a cluster→role mapping locks; a later
attempt to map that cluster elsewhere is refused and logged as an audit line.

`load_backend()` failure is caught and disables verification without stopping the server
(`speaker_verification.py:57-62`).

**Model files:** `pretrained_models/spkrec-ecapa-voxceleb/` must contain real weights
(`embedding_model.ckpt` ≈ 83 MB). SpeechBrain's default `SYMLINK` strategy fails on Windows
without elevated privileges and leaves behind small text files containing a cache path;
`backend.py:36` passes `LocalStrategy.COPY` to avoid this.

## Verified accuracy

Measured against a two-speaker recording with a machine-generated ground truth
(`scratch/ground_truth.json`), scored by segment overlap, with
`DIARIZATION_MODE=production`:

**7 / 7 turns correctly attributed.**

Cluster scores from that run: `Speaker_0 {doctor: 5, patient: 0}` → doctor;
`Speaker_1 {doctor: 0, patient: 4}` → patient.

This is a single controlled recording with two clearly distinct synthetic voices and no
crosstalk. It is not a measure of clinical accuracy on real consultations, which is
**Unknown / not determinable from the current code**.

## Known limitations

| Limitation | Status |
|---|---|
| Diarization has no offline mode; pyannote and its model cache are mandatory | **Implemented** |
| Role assignment relies on English keyword lists; a consultation using different phrasing scores 0 on both sides and falls back to positional defaults | **Partially implemented** |
| The final global pass discards incremental label continuity | **Implemented** as written; a deliberate re-label |
| Overlapping speech is not separated — pyannote assigns one label per range | **Not implemented** |
| Voice profiles do not persist across sessions; there is no enrolment | **Not implemented** |
| Real-world accuracy with more than two speakers, accents, or crosstalk | **Unknown** |
