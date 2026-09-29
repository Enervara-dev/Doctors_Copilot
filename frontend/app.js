/*
 * Enervera Copilot — consultation console.
 *
 * Captures the microphone, converts it to the ingestion contract the backend
 * expects (16 kHz mono 16-bit LPCM, little-endian, 20 ms frames) and streams it
 * over /v1/streaming/audio. Transcript, clinical state and reasoning are read
 * back from the REST endpoints. Everything rendered here comes from the API —
 * nothing on this page is simulated.
 */

const SAMPLE_RATE = 16000;
const FRAME_SAMPLES = 320;             // 20 ms @ 16 kHz
const FRAME_BYTES = FRAME_SAMPLES * 2; // 640 bytes — inside the server's [640, 1600] window
const LEVEL_BARS = 14;

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const state = {
  ws: null, audioCtx: null, micStream: null, processor: null, source: null,
  pending: new Int16Array(0),
  recording: false, startedAt: 0,
  timerId: null, pollId: null, autoId: null,
  seenEvents: new Set(),
  analysing: false,
};

/* ── config ─────────────────────────────────────────────────────── */
// A consultation is addressable: ?session=<id>&api=<url> so a view can be shared
// or reopened without retyping the session settings.
(function applyQuery() {
  const q = new URLSearchParams(location.search);
  const set = (id, key) => { const v = q.get(key); if (v) $(id).value = v; };
  set("apiBase", "api");
  set("sessionId", "session");
  set("role", "role");
  set("token", "token");
  const theme = q.get("theme");
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
})();

const base = () => $("apiBase").value.replace(/\/+$/, "");
const session = () => $("sessionId").value.trim();

async function getJSON(path) {
  // The transcript/state/reasoning GET endpoints now require the same bearer
  // credential used to open the WebSocket for this session (a session id
  // alone used to be enough to read it). Reuse the token already entered in
  // the settings drawer — no new UI, just the header these endpoints now check.
  const token = $("token")?.value?.trim();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const res = await fetch(base() + path, { headers });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function toast(message) {
  const t = $("toast");
  t.textContent = message;
  t.hidden = false;
  clearTimeout(toast._id);
  toast._id = setTimeout(() => { t.hidden = true; }, 3200);
}

/* ── theme ──────────────────────────────────────────────────────── */
(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("copilot-theme"); } catch (_) { /* private mode */ }
  if (saved) document.documentElement.dataset.theme = saved;
})();

$("btnTheme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme
    ? root.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("copilot-theme", root.dataset.theme); } catch (_) { /* ignore */ }
});

/* ── health ─────────────────────────────────────────────────────── */
async function checkHealth() {
  const box = $("conn");
  try {
    const h = await getJSON("/health");
    box.className = "conn is-up";
    $("connText").textContent = `Connected · ${h.active_sessions_count} live`;
  } catch (_) {
    box.className = "conn is-down";
    $("connText").textContent = "Backend offline";
  }
}

/* ── microphone ─────────────────────────────────────────────────── */
function resample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const pos = i * ratio, idx = Math.floor(pos), frac = pos - idx;
    const a = input[idx] ?? 0, b = input[idx + 1] ?? a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

function toInt16(floats) {
  const out = new Int16Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    const s = Math.max(-1, Math.min(1, floats[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function buildLevelMeter() {
  const box = $("levels");
  box.innerHTML = "";
  for (let i = 0; i < LEVEL_BARS; i++) box.appendChild(el("i"));
}

/** Drives the meter from real RMS so it reflects what the mic actually hears. */
function paintLevel(rms) {
  const bars = $("levels").children;
  const active = Math.min(LEVEL_BARS, Math.round(Math.sqrt(rms) * 3.2 * LEVEL_BARS));
  for (let i = 0; i < bars.length; i++) {
    const lit = i < active;
    bars[i].style.height = (lit ? 4 + (i / LEVEL_BARS) * 16 : 3) + "px";
    bars[i].style.opacity = lit ? 0.9 : 0.22;
  }
}

function flushFrames() {
  while (state.pending.length >= FRAME_SAMPLES) {
    const frame = state.pending.slice(0, FRAME_SAMPLES);
    state.pending = state.pending.slice(FRAME_SAMPLES);
    if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(frame.buffer);
  }
}

async function startRecording() {
  try {
    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (_) {
    toast("Microphone blocked. Serve this page from http://localhost and allow access.");
    return;
  }

  const url = `${base().replace(/^http/, "ws")}/v1/streaming/audio`
    + `?session_id=${encodeURIComponent(session())}`
    + `&role=${encodeURIComponent($("role").value)}`
    + `&token=${encodeURIComponent($("token").value)}`;

  state.ws = new WebSocket(url);
  state.ws.binaryType = "arraybuffer";
  state.ws.onerror = () => toast("Could not open the audio stream.");
  state.ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "error") toast(msg.message);
    } catch (_) { /* not JSON */ }
  };
  state.ws.onclose = () => { if (state.recording) stopRecording(); };

  state.audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  state.source = state.audioCtx.createMediaStreamSource(state.micStream);
  state.processor = state.audioCtx.createScriptProcessor(2048, 1, 1);

  state.processor.onaudioprocess = (e) => {
    const raw = e.inputBuffer.getChannelData(0);

    let sum = 0;
    for (let i = 0; i < raw.length; i++) sum += raw[i] * raw[i];
    paintLevel(Math.sqrt(sum / raw.length));

    const ints = toInt16(resample(raw, state.audioCtx.sampleRate, SAMPLE_RATE));
    const merged = new Int16Array(state.pending.length + ints.length);
    merged.set(state.pending, 0);
    merged.set(ints, state.pending.length);
    state.pending = merged;
    flushFrames();
  };

  state.source.connect(state.processor);
  state.processor.connect(state.audioCtx.destination);

  state.recording = true;
  state.startedAt = Date.now();
  buildLevelMeter();
  $("levels").hidden = false;
  $("timer").hidden = false;
  $("btnRecord").classList.add("is-live");
  $("recordLabel").textContent = "End consultation";
  $("btnSettings").disabled = true;

  state.timerId = setInterval(tickTimer, 500);
  tickTimer();
  startPolling();
}

function stopRecording() {
  if (state.processor) { state.processor.disconnect(); state.processor.onaudioprocess = null; }
  if (state.source) state.source.disconnect();
  if (state.audioCtx) state.audioCtx.close();
  if (state.micStream) state.micStream.getTracks().forEach((t) => t.stop());
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.close(1000, "consultation ended");

  Object.assign(state, { ws: null, audioCtx: null, micStream: null, processor: null, source: null,
                         pending: new Int16Array(0), recording: false });

  clearInterval(state.timerId);
  $("levels").hidden = true;
  $("btnRecord").classList.remove("is-live");
  $("recordLabel").textContent = "Start consultation";
  $("btnSettings").disabled = false;

  // One last sweep so the closing turns and their analysis land.
  setTimeout(() => { refreshTranscript(); refreshState(); runAnalysis(); }, 1200);
}

function tickTimer() {
  const s = Math.floor((Date.now() - state.startedAt) / 1000);
  $("timerValue").textContent =
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

/* ── transcript ─────────────────────────────────────────────────── */
const ROLES = { doctor: "Doctor", patient: "Patient", attender: "Attender" };

async function refreshTranscript() {
  let events;
  try {
    events = await getJSON(`/v1/transcripts/${encodeURIComponent(session())}?include_partials=true`);
  } catch (_) { return; }

  $("turnCount").textContent = `${events.length} turn${events.length === 1 ? "" : "s"}`;
  const box = $("transcript");

  if (!events.length) {
    if (!$("transcriptEmpty")) {
      box.innerHTML = "";
      box.appendChild(emptyTranscript());
    }
    return;
  }

  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 90;
  box.innerHTML = "";

  events.forEach((e) => {
    const who = String(e.speaker_id || e.role || "unknown").toLowerCase();
    const known = who in ROLES;
    const row = el("div", `turn ${known ? who : "unknown"}${e.is_final ? "" : " partial"}`);
    if (!state.seenEvents.has(e.event_id)) {
      row.classList.add("is-new");
      state.seenEvents.add(e.event_id);
    }

    const whoBox = el("div", "turn-who");
    whoBox.appendChild(el("span", "turn-role", known ? ROLES[who] : "Unknown"));
    if (e.start_time != null) {
      whoBox.appendChild(el("span", "turn-time", fmtClock(e.start_time)));
    }

    row.append(whoBox, el("div", "turn-body", e.transcript || e.text || ""));
    box.appendChild(row);
  });

  if (nearBottom) box.scrollTop = box.scrollHeight;
}

function emptyTranscript() {
  const wrap = el("div", "placeholder");
  wrap.id = "transcriptEmpty";
  const art = el("div", "placeholder-art");
  for (let i = 0; i < 5; i++) art.appendChild(el("span"));
  wrap.append(art, el("h3", null, "Ready to listen"),
    el("p", null, "Start the consultation and the conversation will appear here, attributed to each speaker as they talk."));
  return wrap;
}

const fmtClock = (sec) =>
  `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;

/* ── clinical state ─────────────────────────────────────────────── */
const VITAL_META = {
  bp:          { label: "Blood pressure", unit: "mmHg" },
  pulse:       { label: "Pulse",          unit: "bpm" },
  temperature: { label: "Temperature",    unit: "" },
  spo2:        { label: "SpO₂",           unit: "%" },
  weight:      { label: "Weight",         unit: "kg" },
  height:      { label: "Height",         unit: "cm" },
};

/** Ranges flag a reading for attention; they never replace clinical judgement. */
function vitalState(key, value) {
  const n = parseFloat(value);
  if (key === "bp") {
    const [sys, dia] = String(value).split("/").map(Number);
    if (sys >= 180 || dia >= 120) return ["critical", "Crisis range"];
    if (sys >= 140 || dia >= 90) return ["warning", "Elevated"];
    return ["", ""];
  }
  if (key === "pulse") {
    if (n >= 120 || n < 50) return ["critical", "Out of range"];
    if (n > 100) return ["warning", "Tachycardic"];
    return ["", ""];
  }
  if (key === "spo2") {
    if (n < 92) return ["critical", "Hypoxic"];
    if (n < 95) return ["warning", "Low"];
    return ["", ""];
  }
  if (key === "temperature") {
    if (n >= 103 || (n < 50 && n >= 39.5)) return ["critical", "High fever"];
    if (n >= 100.4 || (n < 50 && n >= 38)) return ["warning", "Febrile"];
    return ["", ""];
  }
  return ["", ""];
}

async function refreshState() {
  let s;
  try { s = await getJSON(`/v1/clinical-state/${encodeURIComponent(session())}`); }
  catch (_) { return; }

  // Patient identity
  const info = s.patient_info || {};
  const name = info.patient_name;
  $("patientName").textContent = name || "No patient identified";
  $("patientName").classList.toggle("is-placeholder", !name);
  $("patientAvatar").textContent = name ? name.trim()[0].toUpperCase() : "—";
  $("patientAvatar").classList.toggle("is-known", Boolean(name));

  const bits = [];
  if (info.age != null) bits.push(`${info.age} yrs`);
  if (info.gender) bits.push(info.gender);
  if (s.medical_history?.length) bits.push(s.medical_history.join(", "));
  $("patientMeta").textContent = bits.length ? bits.join(" · ") : "Waiting for consultation to begin";

  // Flags that change what is safe to prescribe
  const flags = $("patientFlags");
  flags.innerHTML = "";
  (s.allergies || []).forEach((a) => flags.appendChild(chip(`Allergy: ${a}`, "is-critical")));
  (s.risk_factors || []).filter((r) => String(r.present).toLowerCase() !== "false")
    .forEach((r) => flags.appendChild(chip(cap(r.name), "is-warning")));

  // Vitals
  const vitals = Object.entries(s.vital_signs || {})
    .filter(([, v]) => v && (v.value || v.qualitative));
  $("vitals").hidden = vitals.length === 0;
  const tiles = $("vitalTiles");
  tiles.innerHTML = "";
  vitals.forEach(([key, obs]) => {
    const meta = VITAL_META[key] || { label: cap(key), unit: obs.unit || "" };
    const reading = obs.value || obs.qualitative;
    const [level, note] = obs.value ? vitalState(key, obs.value) : ["", ""];
    const tile = el("div", `tile${level ? " is-" + level : ""}`);
    tile.appendChild(el("div", "tile-label", meta.label));
    const val = el("div", "tile-value", String(reading));
    if (meta.unit) val.appendChild(el("span", "tile-unit", meta.unit));
    tile.appendChild(val);
    if (note) tile.appendChild(el("div", "tile-state", `▲ ${note}`));
    tiles.appendChild(tile);
  });

  renderCapture(s);
}

function chip(text, cls) { return el("span", `flag-chip ${cls || ""}`, text); }
const cap = (s) => String(s).charAt(0).toUpperCase() + String(s).slice(1);

function renderCapture(s) {
  const rows = [];

  const symptoms = (s.symptoms || []).map((sym) => {
    const negated = String(sym.present).toLowerCase() === "false";
    const label = (sym.display_name || sym.name || "").replace(/_/g, " ");
    const extra = [sym.severity, sym.duration, sym.onset].filter(Boolean).join(", ");
    return { text: extra ? `${label} (${extra})` : label,
             cls: negated ? "is-negated" : (sym.severity === "severe" ? "is-severe" : "") };
  });
  if (symptoms.length) rows.push(["Symptoms", symptoms]);

  const plain = (arr) => arr.map((x) => ({ text: x, cls: "" }));
  if (s.medical_history?.length) rows.push(["History", plain(s.medical_history)]);
  if (s.current_medications?.length) rows.push(["Medications", plain(s.current_medications)]);
  if (s.allergies?.length) rows.push(["Allergies", plain(s.allergies)]);
  if (s.family_history?.length) {
    rows.push(["Family", s.family_history.map((f) => ({ text: `${cap(f.relationship)}: ${f.condition}`, cls: "" }))]);
  }

  $("capture").hidden = rows.length === 0;
  const grid = $("captureGrid");
  grid.innerHTML = "";
  rows.forEach(([key, values]) => {
    const row = el("div", "cap-row");
    row.appendChild(el("div", "cap-key", key));
    const vals = el("div", "cap-vals");
    values.forEach((v) => vals.appendChild(el("span", `pill ${v.cls}`, v.text)));
    row.appendChild(vals);
    grid.appendChild(row);
  });
}

/* ── reasoning ──────────────────────────────────────────────────── */
async function runAnalysis() {
  if (state.analysing) return;
  state.analysing = true;
  $("btnAnalyse").disabled = true;

  let r;
  try { r = await getJSON(`/v1/clinical-reasoning/${encodeURIComponent(session())}`); }
  catch (_) { toast("Could not reach the reasoning engine."); return; }
  finally { state.analysing = false; $("btnAnalyse").disabled = false; }

  if (!r || r.message) return;   // nothing to reason over yet

  renderAlerts(r.red_flags || []);
  renderLead(r.candidate_diagnoses || []);
  renderDifferentials(r.candidate_diagnoses || []);
  renderAsk(r.missing_information || [], r.llm_adaptive_questions || []);
  renderTests(r.recommended_investigations || []);
  renderSafety(r.contraindications || [], r.drug_interactions || []);
  renderSoap(r.llm_soap_notes);
  $("tabsWrap").hidden = false;
}

function renderAlerts(flags) {
  const box = $("alerts");
  box.innerHTML = "";
  flags.forEach((f) => {
    const a = el("div", "alert");
    const head = el("div", "alert-head");
    const ico = document.createElement("div");
    ico.className = "ico";
    ico.innerHTML = '<svg viewBox="0 0 24 24"><path d="M12 9v5m0 3h.01M10.3 3.9 2.4 17.5A2 2 0 0 0 4.1 20.5h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>';
    head.append(ico, el("span", null, f.condition || f.name || "Red flag"));
    head.appendChild(el("span", "alert-sev", f.severity || "ALERT"));
    a.appendChild(head);
    if (f.recommended_action) a.appendChild(el("div", "alert-action", f.recommended_action));
    if (f.supporting_evidence?.length) {
      a.appendChild(el("div", "alert-why", `Because: ${f.supporting_evidence.join(", ")}`));
    }
    box.appendChild(a);
  });
}

function renderLead(dx) {
  const box = $("assessment");
  box.innerHTML = "";
  if (!dx.length) {
    const p = el("div", "placeholder small");
    p.append(el("h3", null, "No assessment yet"),
             el("p", null, "Findings appear once the consultation has enough detail to reason over."));
    box.appendChild(p);
    return;
  }
  const top = dx[0];
  const pct = Math.round((top.confidence ?? 0) * 100);

  const lead = el("div", "lead");
  const fig = el("div", "lead-figure");
  const val = el("div", "lead-value", String(pct));
  val.appendChild(Object.assign(document.createElement("sup"), { textContent: "%" }));
  fig.append(val, el("div", "lead-cap", "Confidence"));

  const text = el("div", "lead-text");
  text.append(el("div", "lead-name", top.name),
              el("div", "lead-cat", cap(top.category || "Leading differential")));
  if (top.supporting_symptoms?.length) {
    const sup = el("div", "lead-support");
    sup.append(document.createTextNode("Supported by "), el("b", null, top.supporting_symptoms.join(", ")));
    text.appendChild(sup);
  }
  lead.append(fig, text);
  box.appendChild(lead);
}

function renderDifferentials(dx) {
  $("differentialsWrap").hidden = dx.length < 2;
  const list = $("differentials");
  list.innerHTML = "";
  dx.slice(1, 6).forEach((d) => {
    const pct = Math.round((d.confidence ?? 0) * 100);
    const li = document.createElement("li");
    const row = el("div", "dx-row");
    row.append(el("span", "dx-rank", `${d.rank}`), el("span", "dx-name", d.name), el("span", "dx-pct", `${pct}%`));
    const track = el("div", "dx-track");
    const fill = el("span", "dx-fill");
    fill.style.width = `${pct}%`;
    track.appendChild(fill);
    li.append(row, track);
    if (d.missing_symptoms?.length) {
      li.appendChild(el("div", "dx-why", `Not yet established: ${d.missing_symptoms.join(", ")}`));
    }
    list.appendChild(li);
  });
}

function renderAsk(missing, llmQuestions) {
  const list = $("askList");
  list.innerHTML = "";
  const items = [
    ...llmQuestions.map((q) => ({ q: q.question, why: q.purpose })),
    ...missing.map((m) => ({ q: m.question, why: (m.relevant_conditions || []).join(", ") })),
  ];
  if (!items.length) return emptyInto(list, "Nothing outstanding — the picture is complete enough to act on.");
  items.slice(0, 8).forEach(({ q, why }) => {
    const li = el("li", "qitem", q);
    if (why) li.appendChild(el("span", "qitem-why", `To establish: ${why}`));
    list.appendChild(li);
  });
}

function renderTests(tests) {
  const list = $("testList");
  list.innerHTML = "";
  if (!tests.length) return emptyInto(list, "No investigations recommended yet.");
  tests.forEach((t) => {
    const li = el("li", "titem");
    li.appendChild(el("span", `pri is-${String(t.priority || "").toLowerCase()}`, t.priority || "—"));
    const body = el("div");
    body.append(el("span", null, t.name));
    if (t.reason) body.appendChild(el("span", "titem-why", t.reason));
    li.appendChild(body);
    list.appendChild(li);
  });
}

function renderSafety(contras, interactions) {
  const list = $("safetyList");
  list.innerHTML = "";
  const rows = [
    ...contras.map((c) => ({ tag: c.severity, title: `${c.drug_or_procedure} — ${c.reason}`,
                             why: `Triggered by ${c.trigger}. ${c.recommendation || ""}`.trim() })),
    ...interactions.map((d) => ({ tag: d.severity, title: `${d.drug_a} + ${d.drug_b} — ${d.clinical_effect}`,
                                  why: `${d.mechanism || ""} ${d.recommendation || ""}`.trim() })),
  ];
  if (!rows.length) return emptyInto(list, "No contraindications or drug interactions detected.");
  rows.forEach((r) => {
    const li = el("li", "titem");
    li.appendChild(el("span", `pri is-${String(r.tag || "").toLowerCase()}`, r.tag || "—"));
    const body = el("div");
    body.append(el("span", null, r.title));
    if (r.why) body.appendChild(el("span", "titem-why", r.why));
    li.appendChild(body);
    list.appendChild(li);
  });
}

function renderSoap(soap) {
  const box = $("soap");
  box.innerHTML = "";
  if (!soap) {
    box.appendChild(el("p", "placeholder small", "The consultation note is generated once an assessment exists."));
    return;
  }
  [["Subjective", "subjective"], ["Objective", "objective"],
   ["Assessment", "assessment"], ["Plan", "plan"]].forEach(([label, key]) => {
    const sec = document.createElement("section");
    sec.append(el("h4", null, label), el("p", null, soap[key] || "—"));
    box.appendChild(sec);
  });
}

function emptyInto(list, message) {
  const li = document.createElement("li");
  li.appendChild(el("div", "placeholder small", message));
  list.appendChild(li);
}

/* ── polling ────────────────────────────────────────────────────── */
function startPolling() {
  stopPolling();
  state.pollId = setInterval(() => { refreshTranscript(); refreshState(); }, 2000);
  if ($("autoAnalyse").checked) state.autoId = setInterval(runAnalysis, 30000);
}
function stopPolling() {
  clearInterval(state.pollId); state.pollId = null;
  clearInterval(state.autoId); state.autoId = null;
}

/* ── wiring ─────────────────────────────────────────────────────── */
$("btnRecord").addEventListener("click", () => state.recording ? stopRecording() : startRecording());
$("btnAnalyse").addEventListener("click", runAnalysis);

$("btnSettings").addEventListener("click", () => { $("drawer").hidden = false; $("scrim").hidden = false; });
const closeDrawer = () => { $("drawer").hidden = true; $("scrim").hidden = true; };
$("btnCloseDrawer").addEventListener("click", closeDrawer);
$("scrim").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

$("autoAnalyse").addEventListener("change", () => {
  if (state.pollId) startPolling();
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t === tab));
    document.querySelectorAll(".tabpane").forEach((p) =>
      p.classList.toggle("is-active", p.dataset.pane === tab.dataset.tab));
  });
});

checkHealth();
setInterval(checkHealth, 10000);
refreshTranscript();
refreshState();
runAnalysis();
startPolling();
