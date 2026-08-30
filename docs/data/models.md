# Data models

All models are Pydantic. There are no database schemas — see [storage.md](storage.md).

| Model group | File |
|---|---|
| Transcript events | `app/services/providers/models.py` |
| Clinical extraction + state | `app/services/clinical/models.py` |
| Reasoning results | `clinical_intelligence/rule_engine/models.py` |
| Speaker verification | `app/services/speaker_verification/models.py` |

---

## TranscriptEvent

`app/services/providers/models.py`. The unit published on the transcript bus and returned
by `GET /v1/transcripts/{id}`. Full field table in
[../api/README.md](../api/README.md#element-shape--transcriptevent).

Two fields are easy to confuse:

- **`role`** — the role declared on the WebSocket connection that carried the audio.
- **`speaker_id`** — who the pipeline decided was talking, from diarization plus cluster
  role assignment. This is the clinically meaningful one, and the one
  `ClinicalStateEngine` gates extraction on.

---

## Extraction entities

`app/services/clinical/models.py`

### SymptomEntity

`name` (canonical, e.g. `symptom_cough`), `display_name`, `severity`, `duration`,
`present`, `confidence`, `onset`, `progression`, `frequency`, `character`, `source_text`,
`utterance_id`, `speaker`.

`source_text` and `utterance_id` carry provenance back to the originating utterance.

### MedicationEntity

`name` (generic), `display_name` (as reported, e.g. "Tylenol"), `canonical_name`,
`drug_id`, `present`, `confidence`, `dosage`, `frequency`, `duration`, `route`,
`instructions`, `prn`, `source_text`, `utterance_id`, `speaker`.

### DiagnosisEntity / ProcedureEntity / RiskFactorEntity

`name`, `present`, `confidence`.

### FamilyHistoryEntity

`relationship`, `condition`, `present`, `confidence`.

### VitalObservation / VitalSigns

`VitalSigns` has six optional slots: `bp`, `pulse`, `temperature`, `spo2`, `weight`,
`height`. Each is a `VitalObservation` with `value`, `unit`, `qualitative`, `source`,
`confidence`, `timestamp`, `source_text`, `utterance_id`.

`qualitative` carries non-numeric readings — `"hypertensive"`, `"hypotensive"`, `"normal"`.

### ClinicalExtractionResult

Per-utterance output of the pipeline: `session_id`, `speaker_id`, `timestamp`, plus
`symptoms`, `medications`, `diagnoses`, `procedures`, `risk_factors`, `family_histories`.

---

## ClinicalState

`app/services/clinical/models.py:119`. Returned by `GET /v1/clinical-state/{id}`.

| Field | Type | Populated from |
|---|---|---|
| `session_id` | str | |
| `patient_info` | `PatientInfo` — `patient_name`, `age`, `gender` | Patient/attender speech |
| `chief_complaint` | `List[str]` | Utterances matching complaint phrasing |
| `symptoms` | `List[Dict]` | Symptom extractor. **Dicts, not `SymptomEntity`** |
| `duration` | str \| null | Overall duration |
| `severity` | str \| null | Overall severity |
| `medical_history` | `List[str]` | Patient's **own** past conditions |
| `family_history` | `List[Dict]` | `relationship`, `condition`, `present`, `confidence` |
| `risk_factors` | `List[Dict]` | `name`, `present`, `confidence` — smoking, alcohol, … |
| `current_medications` | `List[str]` | |
| `allergies` | `List[str]` | Regex on "allergic to X" |
| `vital_signs` | `VitalSigns` | Doctor speech |
| `diagnosis_tentative` | `List[str]` | Doctor's working assessment |
| `treatment_plan` | `TreatmentPlan` — `medicines`, `investigations`, `advice` | Doctor speech |
| `follow_up` | `List[str]` | |
| `examination` | `List[ExaminationFinding]` | |
| `audit_trail` | `List[Dict]` | |
| `version` | int | Incremented per applied event |

Notes on shapes that differ from what the names suggest:

- `symptoms` and `family_history` are **lists of plain dicts**, not the entity models. Their
  keys mirror the entities but are written by hand in `state_engine.py`.
- Inside `symptoms`, `present` is the **string** `"True"` / `"False"`, not a bool
  (`state_engine.py` §4). Consumers must compare strings.
- `risk_factors[].present` is likewise a string.
- `medical_history` holds the **patient's own** conditions. A relative's condition goes to
  `family_history` — see [../flows/consultation-flow.md](../flows/consultation-flow.md#7-clinical-state-extraction).

Negated symptoms are **retained**, with `status="Negated"` and `present="False"` — a denial
is clinical information, not an absence of information.

---

## Reasoning models

`clinical_intelligence/rule_engine/models.py`. `ClinicalReasoningResult` is dumped directly
as the API response; the full key list is in
[../api/README.md](../api/README.md#response-keys).

### CandidateDiagnosis

`disease_id`, `name`, `category`, `confidence` (0.0–1.0), `rank`, `supporting_symptoms`,
`missing_symptoms`, `negated_symptoms`, plus match metadata.

> The field is `confidence`, not `score`.

### Contraindication

`contraindication_id`, `drug_or_procedure`, `reason`, `trigger`, `severity`
(`ABSOLUTE` | `RELATIVE`), `recommendation`, optional `guideline`, `rule_metadata`.

### DrugInteraction

`interaction_id`, `drug_a`, `drug_b`, `severity` (`MAJOR` | `MODERATE` | `MINOR`),
`mechanism`, `clinical_effect`, `recommendation`.

> Two named drugs — `drug_a` / `drug_b` — not a `drugs` list.

### Others

`IdentifiedRiskFactor` (`factor_id`, `name`, `category`, `present`, `source`,
`confidence_modifier`), `RedFlag` (`flag_id`, `condition`, `severity`,
`supporting_evidence`, `recommended_action`), `RecommendedInvestigation`
(`investigation_id`, `name`, `priority`, `reason`, `for_conditions`), `MissingInformation`
(`field`, `question`, `priority`, `relevant_conditions`), `ClinicalScore` (`name`, `score`,
`interpretation`, `evidence`).

---

## Speaker verification models

`app/services/speaker_verification/models.py`

- **`AudioSegment`** — dataclass: `samples` (numpy), `sample_rate`, `start_time`, `end_time`
- **`AudioQuality`** — `passed`, `rms`, `speech_ratio`, `clipping_ratio`, `reason`
- **`SpeakerEmbedding`** — `embedding` (list of float), `timestamp`, `duration`, `session_id`
- **`SpeakerVerificationTelemetry`** — model name/version, load/embed/verify timings,
  `similarity_score`, `speaker_changed`, `verification_enabled`

`RegisteredVoice` (`voice_registry.py`) is a plain class, not Pydantic: `speaker_id`,
`role`, `average_embedding`, Welford accumulators, `embedding_count`, `last_seen`, `locked`.

---

## Rule definitions

YAML in `clinical_intelligence/rules/`, hot-reloadable via `rule_engine/loader.py`.

| File | Entries |
|---|---|
| `symptoms.yaml` | 72 |
| `diseases.yaml` | 31 |
| `investigations.yaml` | 39 |
| `risk_factors.yaml` | 33 |
| `drug_interactions.yaml` | 18 |
| `contraindications.yaml` | 17 |
| `red_flags.yaml` | 7 |

## Extraction dictionaries

JSON in `app/services/clinical/dictionary/`, loaded by `loader.py`.

| File | Entries |
|---|---|
| `symptoms.json` | 17 |
| `diseases.json` | 17 |
| `drug_dictionary.json` | 17 |
| `procedures.json` | 12 |
| `anatomy.json` | 10 |
| `family.json` | 7 |
| `negations.json` | 2 |

These are demo-scale vocabularies. Terminology outside them is not extracted.

## Guidelines corpus

`clinical_intelligence/vector_store/guidelines.yaml` — **5 guidelines** (AHA chest pain,
AHA/ASA stroke, and three others), each with `id`, `title`, `organization`, `version`,
`evidence_level`, `specialty`, `publication_date`, `text_content`.
