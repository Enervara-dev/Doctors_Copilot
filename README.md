# Clinical Consultation Copilot

An enterprise-grade, low-latency, and deterministic clinical assistant system designed for live consultation session recording, real-time speech processing, clinical fact extraction, and safety-focused rule-based clinical reasoning.

---

## 0. Running It

### Quick start

```powershell
.
un.ps1
```

Starts the API, serves the console, waits for the models to load and opens the browser.
`Ctrl+C` stops both. Options: `-ApiPort`, `-UiPort`, `-NoBrowser`.

### Manual

Two processes. **Always use `.venv\Scripts\python.exe`** — torch, pyannote and
speechbrain are installed there and nowhere else, so a system Python starts and then
fails when it loads the diarizer.

```powershell
# 1. API — first start takes ~40 s while pyannote and ECAPA load
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 2. Console (separate terminal) — the mic needs a secure context, so serve it
python -m http.server 5500 --directory frontend
```

Open <http://127.0.0.1:5500> and press **Start consultation**.
Interactive API docs are at <http://127.0.0.1:8000/docs>.

### Replaying a recording instead of speaking

```powershell
$env:DEMO_SESSION_ID = "demo-1"
.venv\Scripts\python.exe stream_audio.py path	o\consultation.wav doctor
```

Any sample rate or channel count is accepted; the script converts. Then open
`http://127.0.0.1:5500/?session=demo-1` to view that consultation.

### Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ clinical_intelligence/tests/ -q
```

### Setup from scratch

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then fetch the models once, with `HF_HUB_OFFLINE=0` and a Hugging Face token that has
accepted the `pyannote/speaker-diarization-3.1` licence:

```powershell
$env:HF_HUB_OFFLINE = "0"
.venv\Scripts\python.exe -c "from app.services.diarization_worker import _lazy_init_pyannote; _lazy_init_pyannote()"
.venv\Scripts\python.exe -c "from app.services.speaker_verification.backend import SpeechBrainBackend; SpeechBrainBackend().load_model()"
```

Afterwards the app runs offline from the cache (`HF_HUB_OFFLINE` defaults to `1`).

**Behind a TLS-inspecting proxy** — corporate networks, some endpoint-security agents —
`pip` and the Hugging Face downloader verify against certifi and will fail with
`CERTIFICATE_VERIFY_FAILED`. Export the OS trust store once and point them at it:

```powershell
.venv\Scripts\python.exe -c "import ssl,base64;certs=[c for s in ('ROOT','CA') for c,e,_ in ssl.enum_certificates(s) if e=='x509_asn'];open('scratch/win-ca-bundle.pem','w').write(''.join('-----BEGIN CERTIFICATE-----
'+'
'.join(base64.b64encode(c).decode()[i:i+64] for i in range(0,len(base64.b64encode(c).decode()),64))+'
-----END CERTIFICATE-----
' for c in certs))"
$env:REQUESTS_CA_BUNDLE = "$PWD\scratch\win-ca-bundle.pem"
$env:SSL_CERT_FILE      = "$PWD\scratch\win-ca-bundle.pem"
# pip additionally needs it passed explicitly:
.venv\Scripts\python.exe -m pip install --cert $env:REQUESTS_CA_BUNDLE -r requirements.txt
```

The application itself does **not** need these at runtime — it loads the OS trust store
directly for its outbound API calls.

### Configuration

Settings live in `.env` (prefix `CLINICAL_COPILOT_`). The ones that change behaviour most:

| Setting | Effect |
|---|---|
| `DIARIZATION_MODE` | Must be `production`. The development mock was removed; any other value fails at startup |
| `STT_PROVIDER_MODE` | Must be `production`. The development mock was removed; any other value fails at startup |
| `ENABLE_RED_FLAG_ALERTS` | when `False`, red flags are detected but stripped from the API response |
| `ENABLE_LLM` | turns the LLM enrichment layer on/off |

---

## 1. Executive Summary & Core Capabilities

The **Clinical Consultation Copilot** acts as an ambient assistant during doctor-patient consultations. It consumes raw audio streams from multiple participants (Doctor, Patient, and Attender), processes them through speech-to-text and speaker-diarization layers, extracts structured clinical facts, and evaluates the clinical state against a safety-first clinical reasoning engine.

### What the System Can Do:
*   **Multi-Role Audio Streaming:** Ingests live PCM audio from three separate client microphones (Doctor, Patient, Attender) simultaneously with sub-100ms latency.
*   **Speaker Diarization & Alignment:** Uses PyAnnote to segment speakers and matches the spoken transcripts with their roles, forming a structured conversational timeline.
*   **Speech-to-Text Processing:** Integrates with ElevenLabs Scribe V2 to transcribe audio segments on-the-fly.
*   **Clinical Entity Extraction:** Normalizes colloquial layman terminology and extracts key medical entities (Symptoms, Medications, Tentative Diagnoses, Procedures, Risk Factors, and Family History).
*   **Safety-First Clinical Reasoning:** Runs a fully deterministic, explainable, and LLM-free reasoning engine to:
    *   *Score Candidate Diagnoses:* Ranks potential conditions based on clinical rule weights, vitals, age, gender, and negation penalties.
    *   *Detect Red Flags:* Flags life-threatening conditions (e.g., Acute Coronary Syndrome, Anaphylaxis) requiring immediate attention.
    *   *Perform Safety Checks:* Checks for drug-drug interactions and patient-specific contraindications (e.g., prescribing a drug to an allergic patient).
    *   *Suggest Next Steps:* Recommends targeted clinical investigations and identifies missing critical information to help the doctor during the visit.

---

## 2. End-to-End Ingestion & Processing Pipeline

The diagram below outlines the full flow of information from physical microphone vibrations to structured clinical intelligence outputs.

```
+--------------------+
| Doctor / Patient / |
| Attender Microphones|
+---------+----------+
          |
          | WebRTC (UDP/RTP)
          v
+--------------------+
| WebRTC Media GW    | (Terminates WebRTC UDP packets, converts to raw LPCM)
+---------+----------+
          |
          | WebSocket Connection (ws:// or wss://)
          v
+--------------------+
| FastAPI Ingest GW  | [app/api/websocket.py]
| (Auth, Sessions)   | - Handshake & active session verification
+---------+----------+
          |
          | Micro-frames (20-50ms)
          v
+--------------------+
| Audio Buffer Mgr   | [app/services/audio_buffer.py]
| (Validation & Buff)| - Validates LPCM parameters, aggregates into 1.0s chunks
+---------+----------+
          |
          | 1.0s PCM Chunks
          v
+--------------------+
| Speech-to-Text /   | [app/services/stt_worker.py]
| Diarization Worker | - ElevenLabs Scribe V2 transcription
+---------+----------+ - PyAnnote diarization speaker alignment
          |
          | Aligned Conversation Utterances (Role + Transcript)
          v
+--------------------+
| Term Normalizer &  | [app/services/clinical/pipeline.py]
| Entity Extractor   | - Maps colloquial terms to medical terminologies
+---------+----------+ - Filters negated items and updates ClinicalState
          |
          | ClinicalState (Vitals, Symptoms, Meds, History)
          v
+--------------------+
| Clinical Reasoning | [clinical_intelligence/rule_engine/engine.py]
| Engine (Module 1)  | - Runs deterministic checks against YAML rulesets
+---------+----------+
          |
          v
+-------------------------------------------------------------------------------+
|                        CLINICAL REASONING RESULTS                             |
|  - Candidate Diagnoses (ranked)   - Red Flag Alerts      - Risk Factors       |
|  - Investigation Recommendations  - Missing Info Questions                    |
|  - Contraindications & Drug-Drug Interactions - Complete Explainability Trace |
+-------------------------------------------------------------------------------+
```

---

## 3. Milestone 1: Audio Processing & Clinical Extraction Pipeline

### Step 1: Audio Ingestion Layer
*   **Protocol:** WebSocket (`ws://` or `wss://`) endpoint at `/v1/streaming/audio`
*   **PCM Contract:** Raw LPCM 16-bit Mono, 16 kHz sample rate, little-endian.
*   **Buffering Strategy:** Accumulates client micro-frames (20-50ms) into **1.0 second chunks** (32,000 bytes) to minimize downstream API latency and network overhead.
*   **Drop & Reconnect Policy:** If a client drops due to network handoff, the session manager keeps the buffer state active and resumes appending once the client reconnects.

#### Connection Handshake Query Parameters
*   `session_id` (string, Required): Unique session UUID.
*   `role` (string, Required): One of `doctor`, `patient`, or `attender`.
*   `token` (string, Required): Handshake authentication token.

### Step 2: Speech-to-Text (STT) Layer
Transcribes PCM chunks:
*   **ElevenLabs Scribe V2:** High-accuracy, low-latency streaming Speech-to-Text.

### Step 3: Diarization & Speaker Alignment Layer
*   Uses **PyAnnote speaker-diarization-3.1** to segment audio.
*   Determines which participant was speaking at any given time.
*   Merges speaker timeline segments and aligns them with transcripts, outputting role-annotated conversational turns.

### Step 4: Clinical Extraction & State Engine
*   **Terminology Normalization:** Translates layman descriptions into formal medical terms (e.g., "throbbing right sided head pain" -> "unilateral throbbing headache").
*   **Entity Extraction:** Extracts clinical entities (symptoms, diagnoses, medications, procedures, risk factors, family histories) from the normalized turns.
*   **Negation Classifier:** Identifies whether a symptom is present or explicitly negated (e.g., "No, I don't have a fever" -> `present: False`).
*   **Clinical State Engine:** Aggregates facts across all dialogue turns and maintains a thread-safe, session-level `ClinicalState` object.

---

## 4. Milestone 2: Rule-Based Clinical Reasoning Engine

The clinical reasoning engine is a deterministic, rule-based medical intelligence pipeline. It uses **no LLMs** during reasoning, ensuring absolute predictability, speed, and auditability.

### Engine Components (`clinical_intelligence/rule_engine/`)

1.  **`ClinicalReasoningEngine` ([engine.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/rule_engine/engine.py)):**
    Orchestrates the entire reasoning pass, parses incoming `ClinicalState`, loads rules, coordinates sub-components, and structures the final response.
2.  **`SymptomMatcher` ([matcher.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/rule_engine/matcher.py)):**
    Matches extracted patient symptoms against candidate conditions. Uses alias mappings to resolve colloquial synonyms.
3.  **`RiskFactorEvaluator` ([risk_evaluator.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/rule_engine/risk_evaluator.py)):**
    Searches the patient's medical history, current medications, allergies, and family history to identify matches for defined clinical risk factors.
4.  **`DiseaseScorer` ([scorer.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/rule_engine/scorer.py)):**
    Calculates disease scores ($0.0$ to $1.0$) by summing weights of cardinal/supportive symptoms, applying modifiers for age/gender, satisfying vital signs criteria, and applying negation penalties.
5.  **`RedFlagDetector` ([red_flags.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/rule_engine/red_flags.py)):**
    Identifies life-threatening scenarios (e.g., cardiac arrest, sepsis, stroke indicators) that demand immediate clinical action.
6.  **`ContraindicationChecker` ([contraindications.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/rule_engine/contraindications.py)):**
    Cross-references current or proposed procedures and medications against the patient's allergies and medical history to alert on absolute or relative contraindications.
7.  **`DrugInteractionDetector` ([interactions.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/rule_engine/interactions.py)):**
    Scans the patient's current medications list to detect drug-drug interactions (classified as Major, Moderate, or Minor).
8.  **`InvestigationRecommender` ([investigations.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/rule_engine/investigations.py)):**
    Proposes clinical tests, labs, or imaging based on suspected diagnoses and elevates their urgency if active red flags are present.
9.  **`MissingInfoDetector` ([missing_info.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/rule_engine/missing_info.py)):**
    Evaluates what crucial clinical information (e.g., pain radiation, symptom duration) is missing for candidate conditions and generates targeted questions for the doctor to ask.
10. **`ExplainabilityTracer` ([explainability.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/rule_engine/explainability.py)):**
    Records every evaluation step, rule match, and score modification, compiling a comprehensive audit trail for transparency.

### Configurable Rule Definition Files (`clinical_intelligence/rules/`)

All rules are defined in readable, hot-reloadable YAML files:
*   [`symptoms.yaml`](file:///c:/Intern/transcriptmodel/clinical_intelligence/rules/symptoms.yaml): Mappings of canonical symptoms to lists of layperson aliases.
*   [`diseases.yaml`](file:///c:/Intern/transcriptmodel/clinical_intelligence/rules/diseases.yaml): Primary disease models with symptom weights, vitals criteria, age/gender modifiers, and negation penalties.
*   [`risk_factors.yaml`](file:///c:/Intern/transcriptmodel/clinical_intelligence/rules/risk_factors.yaml): Mappings from patient history, allergies, and family history to clinical risk factor IDs.
*   [`red_flags.yaml`](file:///c:/Intern/transcriptmodel/clinical_intelligence/rules/red_flags.yaml): Definition of critical scenarios, severities, and immediate actions.
*   [`contraindications.yaml`](file:///c:/Intern/transcriptmodel/clinical_intelligence/rules/contraindications.yaml): Rules specifying under what conditions specific drugs or procedures are contra-indicated.
*   [`drug_interactions.yaml`](file:///c:/Intern/transcriptmodel/clinical_intelligence/rules/drug_interactions.yaml): Structured lists of drug-drug interactions, clinical mechanisms, and recommendations.
*   [`investigations.yaml`](file:///c:/Intern/transcriptmodel/clinical_intelligence/rules/investigations.yaml): Definitions of diagnostic tests, target diseases, baseline priorities, and red-flag escalations.

### Clinical Knowledge Graph Engine (`clinical_intelligence/knowledge_graph/`)

The Knowledge Graph acts as an evidence enrichment database mapped to standard clinical schemas. It resolves relationships (such as `Disease -> Symptoms`, `Medication -> Interactions`) using Cypher projections, and enriches candidates without altering deterministic rankings.

1.  **`Neo4jConnectionManager` ([graph_manager.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/knowledge_graph/graph_manager.py)):**
    Coordinates active connections to the Neo4j database using unique constraints and indexes. Implements connection-failure degradation to run Rule Engine normally without crashing.
2.  **`GraphQueryEngine` ([graph_queries.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/knowledge_graph/graph_queries.py)):**
    Executes parameterized traversals (e.g. connected guidelines, treatment recommendations, structured symptom-disease paths) and profiles execution times.
3.  **`GraphQueryCache` ([cache.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/knowledge_graph/cache.py)):**
    Provides a TTL-based query cache layer to satisfy the <20 ms traversal time requirement and tracks hit/miss performance ratios.
4.  **`Graph Builder & Importer` ([importer.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/knowledge_graph/importer.py)):**
    Pre-compiles YAML definitions into graph elements. Enforces pre-import constraints (aborts on duplicate nodes, dangling edges, invalid labels, or orphans) to ensure graph consistency.
5.  **`EvidenceAggregator` ([aggregator.py](file:///c:/Intern/transcriptmodel/clinical_intelligence/rule_engine/aggregator.py)):**
    A lightweight broker layer that queries the graph for candidate diagnoses and attaches a structured `GraphEvidence` model payload to candidates.

---

## 5. Directory Structure

```
c:/Intern/transcriptmodel/
├── requirements.txt                  # Python dependencies
├── test_client.py                    # Multi-role simulation client
├── generate_pdf_report.py            # PDF report compiler
├── generate_summary_report.py        # Cohesive clinical summary compiler
├── verify_step_4.py                  # Step 4 verification runner
│
├── app/                              # Milestone 1 Application Core
│   ├── main.py                       # FastAPI entrypoint
│   ├── api/
│   │   └── websocket.py              # WebSocket ingress endpoint
│   ├── core/
│   │   ├── config.py                 # System parameters (PCM size, limits)
│   │   └── security.py               # Token verification and role auth
│   ├── services/
│   │   ├── audio_buffer.py           # Audio frame validation and chunking
│   │   ├── session.py                # Consultation session registry
│   │   ├── stt_worker.py             # Speech-to-Text orchestrator
│   │   ├── diarization_worker.py     # Speaker diarization task coordinator
│   │   ├── speaker_alignment.py      # Speaker turn & text alignment logic
│   │   └── clinical/
│   │       ├── pipeline.py           # Fact extraction & normalization pipeline
│   │       ├── extractor.py          # Entity extractor from segments
│   │       ├── normalizer.py         # Normalizes colloquial terms
│   │       └── state_engine.py       # Manages session ClinicalState
│   └── utils/
│       └── logging.py                # JSON structured logging utility
│
├── clinical_intelligence/            # Milestone 2 Clinical Intelligence
│   ├── integration.py                # Milestone 1 to Milestone 2 shim
│   ├── rule_engine/                  # Logic components
│   │   ├── engine.py                 # Core reasoning orchestrator
│   │   ├── matcher.py                # Symptom mapping & normalization
│   │   ├── risk_evaluator.py         # Patient risk factors checker
│   │   ├── scorer.py                 # Disease score builder
│   │   ├── red_flags.py              # Red flags detector
│   │   ├── contraindications.py      # Contraindications analyzer
│   │   ├── interactions.py           # Drug-drug interaction checker
│   │   ├── investigations.py         # Diagnostic recommender
│   │   ├── missing_info.py           # Missing info detector
│   │   ├── explainability.py         # Explainability trace builder
│   │   ├── models.py                 # Pydantic models for results
│   │   ├── loader.py                 # Hot-reloading YAML loader
│   │   ├── aggregator.py             # Broker merging graph evidence
│   │   └── exceptions.py             # Rule engine exceptions
│   ├── knowledge_graph/              # Knowledge Graph components
│   │   ├── graph_manager.py          # Neo4j connection lifecycle & indexes
│   │   ├── graph_builder.py          # rule elements mapper
│   │   ├── graph_queries.py          # parameterized query executions
│   │   ├── graph_models.py           # Neo4j schema definitions
│   │   ├── cache.py                  # TTL-based traversal cache
│   │   └── importer.py               # Pre-import validations CLI tool
│   ├── rules/                        # YAML rule definition files
│   │   ├── symptoms.yaml
│   │   ├── diseases.yaml
│   │   ├── risk_factors.yaml
│   │   ├── red_flags.yaml
│   │   ├── contraindications.yaml
│   │   ├── drug_interactions.yaml
│   │   └── investigations.yaml
│   └── tests/                        # Engine unit and integration tests
│       ├── fixtures/
│       │   └── sample_clinical_state.json
│       ├── test_engine.py
│       ├── test_matcher.py
│       ├── test_scorer.py
│       ├── test_red_flags.py
│       ├── test_contraindications.py
│       └── test_knowledge_graph.py
```

---

## 6. Integration and Usage API

### Python API Integration
You can use the integration shim to pass a `session_id` directly or run reasoning from an arbitrary `ClinicalState` / dictionary representation.

```python
from clinical_intelligence.integration import get_reasoning_result, reason_from_state

# Option A: Get reasoning results using session_id from ClinicalStateEngine
result = get_reasoning_result("session-uuid-1234")
if result:
    print(f"Suspected Condition: {result.candidate_diagnoses[0].name}")
    print(f"Active Red Flags: {len(result.red_flags)}")

# Option B: Run reasoning directly from a dictionary clinical state
dummy_state = {
    "session_id": "manual-state-1",
    "patient_info": {"age": 55, "gender": "male"},
    "medical_history": ["hypertension", "diabetes"],
    "current_medications": ["amlodipine"],
    "symptoms": [
        {"name": "chest pain", "severity": "severe", "present": True}
    ]
}
direct_result = reason_from_state(dummy_state)
```

### Running Tests
To verify all rule engine logic, schema models, matching rules, and trace explainability, execute:
```powershell
# Run all unit and integration tests under clinical_intelligence/tests
python -m unittest discover -s clinical_intelligence/tests -p "test_*.py"
```
