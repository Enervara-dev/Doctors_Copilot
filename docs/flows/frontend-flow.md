# Frontend flow

Documents `frontend/` as it currently exists. Source: `index.html` (173 lines),
`styles.css` (563), `app.js` (612). No build step, no framework, no package manifest.

## Structure

| Concern | Approach |
|---|---|
| Framework | None — vanilla DOM APIs |
| Build | None — files served as-is |
| Routing | **Not implemented** — one page, no router, no `history` calls |
| State management | A single module-level `state` object (`app.js:24`) plus DOM as the render target |
| Styling | One stylesheet, CSS custom properties, light/dark via `prefers-color-scheme` + `[data-theme]` |
| Data fetching | `fetch` + `setInterval` polling. No client cache, no retry |

## Screens

**One screen.** A rail, a header, and a two-column stage. The settings drawer is an overlay
on the same page, not a route.

```
┌──────┬──────────────── header: patient · timer · levels · status · record ─────────┐
│      ├───────────────────────────────┬───────────────────────────────────────────┤
│ rail │ Live consultation             │ Clinical intelligence                     │
│      │  (transcript turns)           │  alerts → hero assessment → vitals →      │
│      │                               │  differential → tabs → captured facts     │
└──────┴───────────────────────────────┴───────────────────────────────────────────┘
```

### Rail

| Element | Status |
|---|---|
| Brand mark | Static SVG |
| "Consultation" button (`data-view="consult"`) | **Placeholder** — `index.html:22`; **no event listener is bound to it** in `app.js`. It renders in an active state permanently and does nothing when clicked |
| Settings button (`btnSettings`) | **Implemented** — opens the drawer. Disabled while recording |
| Theme button (`btnTheme`) | **Implemented** — toggles and persists to `localStorage` |

## User actions

Every interactive control, and what it actually does. Nine listeners are bound in total.

| Control | Handler | Behaviour |
|---|---|---|
| **Start / End consultation** (`btnRecord`) | `startRecording` / `stopRecording` | Requests the mic, opens the WebSocket, starts capture, timer and polling. Ending tears everything down and schedules one final refresh + analysis after 1.2 s |
| **Analyse** (`btnAnalyse`) | `runAnalysis` | `GET /v1/clinical-reasoning/{session}`. Guarded by `state.analysing`; the button disables during the call |
| **Settings** (`btnSettings`) | inline | Shows drawer + scrim |
| **Close drawer** (`btnCloseDrawer`), scrim click, `Escape` | `closeDrawer` | Hides both |
| **Auto re-analyse** (`autoAnalyse`) | inline | When checked *and* polling is active, adds a 30 s `runAnalysis` interval |
| **Theme** (`btnTheme`) | inline | Toggles `data-theme`, writes `localStorage["copilot-theme"]` in a `try/catch` |
| **Tabs** (`.tab` × 4) | delegated | Switches `is-active` between Ask next / Investigations / Safety / Note |

There are **no other interactive controls**. The transcript, vitals, differential and
captured-fact lists are display-only — no click handlers, no selection, no editing.

## Form flows and validation

The only form is the settings drawer: **Backend**, **Session ID**, **My role** (select),
**Token**, and the auto-analyse checkbox.

**Client-side validation: none.** No `required`, no `pattern`, no length limits, no format
checks, no submit handler. Values are read directly at call time:

- `base()` — `apiBase` with trailing slashes stripped (`app.js:47`)
- `session()` — `sessionId` trimmed (`app.js:48`)

An empty or malformed session ID is sent to the backend as-is. Invalid values surface as a
failed request, not as field validation. Status: **Not implemented**.

Values are **not persisted** — only the theme is. A reload restores the defaults in the
markup unless query parameters are supplied.

## Deep links

`applyQuery` (`app.js:36-45`) reads on load and writes into the form fields:

| Parameter | Target |
|---|---|
| `api` | Backend URL |
| `session` | Session ID |
| `role` | Role select |
| `token` | Token |
| `theme` | `light` or `dark` only; anything else ignored |

## State management

`app.js:24-31`:

```js
{ ws, audioCtx, micStream, processor, source,   // audio plumbing
  pending: Int16Array,                          // samples not yet framed
  recording, startedAt,                         // session status
  timerId, pollId, autoId,                      // interval handles
  seenEvents: Set,                              // event_ids already rendered (for animation)
  analysing }                                   // re-entrancy guard
```

No store, no reactivity, no derived state. Each refresh **replaces `innerHTML`** of its
container and re-renders from the API payload. `seenEvents` exists only so a newly arrived
turn animates once.

## Data flow

### Microphone → backend

`startRecording` (`app.js:140`):

1. `getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })`
2. Open `ws://…/v1/streaming/audio?session_id=…&role=…&token=…`
3. `new AudioContext({ sampleRate: 16000 })` → `createMediaStreamSource` →
   `createScriptProcessor(2048, 1, 1)`
4. Per audio callback:
   - RMS computed and painted to the level meter (`paintLevel`)
   - `resample()` to 16 kHz (linear interpolation) if the context differs
   - `toInt16()` to 16-bit PCM
   - appended to `state.pending`; `flushFrames()` emits exact 320-sample (640-byte) frames

`ScriptProcessorNode` is deprecated in the Web Audio spec; it is what the code uses.

### Backend → UI

| Source | Interval | Renders |
|---|---|---|
| `GET /v1/transcripts/{s}?include_partials=true` | 2 s | Turn list, turn counter |
| `GET /v1/clinical-state/{s}` | 2 s | Patient header, flags, vitals tiles, captured facts |
| `GET /v1/clinical-reasoning/{s}` | manual, or 30 s when auto-analyse is on | Alerts, hero assessment, differential, tabs |
| `GET /health` | 10 s | Connection pill |

Polling starts on load (`app.js` tail) and restarts when recording begins.

## Loading, empty and error states

| State | Behaviour |
|---|---|
| **Loading** | **Not implemented** — no spinners or skeletons. The Analyse button disables during its request; nothing else indicates work in progress. A 30 s reasoning call shows no feedback |
| **Empty — transcript** | "Ready to listen" placeholder with an animated bar motif |
| **Empty — assessment** | "No assessment yet" |
| **Empty — questions / tests / safety** | Per-list messages, e.g. "No contraindications or drug interactions detected." |
| **Empty — SOAP** | "The consultation note is generated once an assessment exists." |
| **Empty — vitals / captured facts** | Sections hidden entirely via `hidden` |
| **Backend unreachable** | Connection pill turns red, "Backend offline". `refreshTranscript` and `refreshState` **swallow errors and return silently** — stale data stays on screen with no indication it is stale |
| **Reasoning failure** | Toast: "Could not reach the reasoning engine." |
| **Microphone denied** | Toast: "Microphone blocked. Serve this page from http://localhost and allow access." |
| **WebSocket error** | Toast: "Could not open the audio stream." |
| **Frame rejected by server** | The server's error message is shown as a toast |
| **WebSocket closes while recording** | `stopRecording()` runs, resetting the UI |

Toasts auto-dismiss after 3.2 s. There is no error log, no retry button, no reconnect logic.

## Display logic that is client-side only

`vitalState()` (`app.js:~300`) applies thresholds **in the browser**. These are display
cues, not backend clinical assessments:

| Vital | Warning | Critical |
|---|---|---|
| BP | sys ≥ 140 or dia ≥ 90 → "Elevated" | sys ≥ 180 or dia ≥ 120 → "Crisis range" |
| Pulse | > 100 → "Tachycardic" | ≥ 120 or < 50 → "Out of range" |
| SpO₂ | < 95 → "Low" | < 92 → "Hypoxic" |
| Temperature | ≥ 100.4 °F / ≥ 38 °C → "Febrile" | ≥ 103 °F / ≥ 39.5 °C → "High fever" |

The temperature branch infers the unit from magnitude (`n < 50` treated as Celsius) rather
than reading `obs.unit`.

The patient header derives **allergy** chips from `state.allergies` and **risk** chips from
`state.risk_factors` where `present` is not `"false"`.

## Security notes

- The **token is a plain text input** in the drawer and is placed in the WebSocket **URL
  query string** — it will appear in browser history and any proxy or server access log.
- There is **no login, no session cookie, and no logout**.
- The four `GET` endpoints require no credential, so the console reads any session ID
  entered into the field.

## Not implemented in the frontend

Verified absent:

- Routing or multiple screens
- Authentication UI
- Patient search, selection, or history across consultations
- Editing, correcting, or annotating the transcript
- Accepting or dismissing a suggestion
- Export, print, or save of the note
- Consultation list or history
- Offline support or a service worker
- Automated frontend tests in `frontend/` (a jsdom harness exists at
  `scratch/test_frontend.js`, outside the package)
