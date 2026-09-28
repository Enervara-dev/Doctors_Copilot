/*
 * Renders frontend/index.html in jsdom against the live backend and asserts the
 * console reflects what the API returned. Assertions compare the DOM to the
 * fetched JSON rather than to hardcoded clinical content, so this measures the
 * frontend and not the pipeline's variance.
 *
 *   node scratch/test_frontend.js <session-id> [port]
 */
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const SESSION = process.argv[2] || "final-2";
const PORT = process.argv[3] || "8012";
const API = `http://127.0.0.1:${PORT}`;
const root = path.join(__dirname, "..", "frontend");

const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const js = fs.readFileSync(path.join(root, "app.js"), "utf8");

const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true, url: "http://localhost/" });
const { window } = dom;

window.fetch = (...a) => fetch(...a);
window.performance = performance;
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
window.navigator.mediaDevices = { getUserMedia: async () => { throw new Error("no mic in test"); } };
window.AudioContext = function () { throw new Error("no audio in test"); };
window.WebSocket = function () { throw new Error("no ws in test"); };

window.document.getElementById("sessionId").value = SESSION;
window.document.getElementById("apiBase").value = API;

window.eval(js);

const $ = (id) => window.document.getElementById(id);
const txt = (id) => ($(id) ? $(id).textContent.replace(/\s+/g, " ").trim() : "<missing element>");
const get = (p) => fetch(API + p).then((r) => r.json());

let failed = 0;
function check(name, ok, detail) {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(28)} ${String(detail).slice(0, 74)}`);
}
function checkAll(name, id, items, labelOf) {
  const rendered = txt(id).toLowerCase();
  if (!items || !items.length) return check(name + " (none)", true, "nothing to render");
  const missing = items.filter((i) => !rendered.includes(String(labelOf(i)).toLowerCase()));
  check(name, missing.length === 0, missing.length ? `missing: ${missing.map(labelOf)}` : `${items.length} rendered`);
}

(async () => {
  await new Promise((r) => setTimeout(r, 2500));
  $("btnAnalyse").dispatchEvent(new window.Event("click"));
  await new Promise((r) => setTimeout(r, 6000));

  const [events, state, reasoning] = await Promise.all([
    get(`/v1/transcripts/${SESSION}?include_partials=true`),
    get(`/v1/clinical-state/${SESSION}`),
    get(`/v1/clinical-reasoning/${SESSION}`),
  ]);

  console.log("── shell ──");
  check("backend status", txt("connText").includes("Connected"), txt("connText"));
  check("no unresolved template", !html.includes("{{"), "clean markup");

  console.log("── patient header ──");
  const info = state.patient_info || {};
  check("patient name", !info.patient_name || txt("patientName").includes(info.patient_name), txt("patientName"));
  check("patient meta", info.age == null || txt("patientMeta").includes(String(info.age)), txt("patientMeta"));
  const riskCount = (state.risk_factors || []).length + (state.allergies || []).length;
  check("risk/allergy chips", riskCount === 0 || txt("patientFlags").length > 1, txt("patientFlags") || "(none)");

  console.log("── transcript ──");
  const turns = window.document.querySelectorAll("#transcript .turn");
  check("turn count", turns.length === events.length, `${turns.length} of ${events.length}`);
  check("turn counter label", txt("turnCount").startsWith(String(events.length)), txt("turnCount"));
  const speakerClassed = [...turns].every((t) => /doctor|patient|attender|unknown/.test(t.className));
  check("speaker styling", speakerClassed, "every turn carries a speaker class");
  check("transcript text", events.every((e) => txt("transcript").includes((e.transcript || "").slice(0, 25))), "all turns present");

  console.log("── vitals ──");
  const vitals = Object.entries(state.vital_signs || {}).filter(([, v]) => v && v.value);
  const tiles = window.document.querySelectorAll("#vitalTiles .tile");
  check("vital tiles", tiles.length === vitals.length, `${tiles.length} of ${vitals.length}`);
  const bp = (state.vital_signs || {}).bp;
  if (bp && bp.value) {
    check("bp value shown", txt("vitalTiles").includes(bp.value), bp.value);
    const sys = Number(String(bp.value).split("/")[0]);
    if (sys >= 140) {
      check("elevated bp flagged", /is-(warning|critical)/.test($("vitalTiles").innerHTML), "status ring applied");
    }
  }

  console.log("── assessment ──");
  const dx = reasoning.candidate_diagnoses || [];
  if (dx.length) {
    const pct = Math.round((dx[0].confidence ?? 0) * 100);
    check("hero confidence", txt("assessment").includes(String(pct)), `${pct}%`);
    check("hero diagnosis", txt("assessment").includes(dx[0].name), dx[0].name);
    const heroes = window.document.querySelectorAll(".lead-value");
    check("exactly one hero figure", heroes.length === 1, `${heroes.length} found`);
  }
  checkAll("differential list", "differentials", dx.slice(1, 6), (d) => d.name);

  console.log("── panels ──");
  checkAll("red flags", "alerts", reasoning.red_flags || [], (f) => f.condition);
  checkAll("investigations", "testList", (reasoning.recommended_investigations || []).slice(0, 5), (t) => t.name);
  checkAll("ask-next", "askList", (reasoning.missing_information || []).slice(0, 3), (m) => m.question.slice(0, 28));
  const soap = reasoning.llm_soap_notes;
  check("SOAP note", !soap || txt("soap").includes((soap.objective || "").slice(0, 22)), txt("soap").slice(0, 60));

  console.log("── captured facts ──");
  checkAll("medications", "captureGrid", state.current_medications || [], (m) => m);
  checkAll("history", "captureGrid", state.medical_history || [], (h) => h);
  checkAll("family history", "captureGrid", state.family_history || [], (f) => f.condition);

  console.log(failed === 0 ? "\nAll frontend checks passed." : `\n${failed} check(s) failed.`);
  process.exit(failed === 0 ? 0 : 1);
})();
