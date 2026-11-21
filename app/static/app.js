"use strict";

// Discovery Clock page. Recorder view: film or pick a video, send it with the steps, poll for
// the first read and then the full read. Watch view (?watch=1): show a QR code and follow the
// newest run on the server.

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const WATCH = new URLSearchParams(location.search).get("watch") === "1";
// ?case=NAME opens one use-case tab: packing, handover, evidence, lockout or own.
const CASE_PARAM = (new URLSearchParams(location.search).get("case") || "").toLowerCase();
const POLL_MS = 500;
const GIVE_UP_MS = 95000;     // the server calls a run an error at 75 s; the page waits a little longer
const AUDIT_POLL_MS = 1000;   // the audit comes 45 to 70 s after the full read, so it is polled slowly
const AUDIT_GIVE_UP_MS = 175000;
const AUDIT_WORDS = { confirm: "confirms", dispute: "DISPUTES" };
const LEAD_IN_S = 3;
const VIDEO_BITS = 1200000;
const MIMES = ["video/mp4", "video/webm;codecs=vp9", "video/webm"];
const WORDS = { done: "Done", not_found: "Not found", unsure: "Unsure" };
const WITNESS_NAMES = { ladder: "Decisions", hydrogen: "Hydrogen", hydrogen4: "Hydrogen 4 fps", hydrogen2: "Hydrogen 2 fps", neon: "Neon", neon4: "Neon 4 fps", carbon: "Carbon" };
const WITNESS_ORDER = ["ladder", "hydrogen4", "hydrogen2", "neon4", "hydrogen", "neon"];
const narrow = window.matchMedia("(max-width: 900px)");

const video = $("video");
const state = { stream: null, camPending: null, facing: "environment", starting: false, discard: false,
  recorder: null, timer: null, blob: null, name: "", fullUrl: "", busy: false };

const mainView = { box: $("results"), summary: $("summary"), alarm: $("alarm"), fast: $("line-fast"),
  full: $("line-full"), footer: $("footer"), legend: $("legend"), status: $("status"), rows: [], run: "",
  audit: $("line-audit"), sheet: $("sheet"), sheetText: $("sheet-text"), signoff: $("signoff") };
const askView = { box: $("answer"), summary: $("answer-line"), alarm: null, fast: $("ask-fast"),
  full: $("ask-full"), footer: null, legend: null, status: $("ask-status"), rows: [], run: "", one: true,
  audit: $("ask-audit"), sheet: null, sheetText: null, signoff: null };

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function svg(tag, attrs) {
  const n = document.createElementNS(SVG_NS, tag);
  Object.keys(attrs).forEach((k) => n.setAttribute(k, attrs[k]));
  return n;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const secs = (t) => t.toFixed(1) + " s";
const pct = (t, dur) => Math.min(100, Math.max(0, (t / dur) * 100));

function say(node, text, isError) {
  if (!node) return;
  node.textContent = text;
  node.classList.toggle("err", Boolean(isError));
}

function caption(text) {
  $("caption").textContent = text;
  $("screen-note").hidden = true;
}

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch (e) { /* private mode: nothing is remembered */ }
  return null;
}

function forget(key) {
  try { localStorage.removeItem(key); } catch (e) { /* private mode: nothing was kept */ }
}

// ---------- theme: dark by default, light on request ----------

function setTheme(name, keep) {
  const theme = name === "light" ? "light" : "dark";
  const other = theme === "dark" ? "Light" : "Dark";
  document.documentElement.setAttribute("data-theme", theme);
  const b = $("theme");
  if (b) {
    b.textContent = other;
    b.title = "Switch to the " + other.toLowerCase() + " theme";
  }
  if (keep) store("clock.theme", theme);
}

// ---------- checklist ----------

function readSteps() {
  return $("steps").value.split("\n").map((s) => s.trim()).filter(Boolean);
}

function countSteps() {
  const n = readSteps().length;
  $("count").textContent = n ? n + (n === 1 ? " step." : " steps.") : "";
}

// ---------- use cases: one tab per kind of job, each with its own remembered checklist ----------

// Only the packing check and the handover were measured. The other lists are examples, and say so.
const CASES = {
  packing: { rec: "Film the packing", run: "Check the packing",
    role: "For a seller: which item did not go in?", note: "Measured on nine phone takes.",
    steps: ["put the AirPods case in the bag", "put the Hello Panda box in the bag", "put the watch in the bag",
      "zip the bag closed", "knock on the table twice"] },
  handover: { rec: "Film the handover", run: "Check the handover",
    role: "For a receiver: was it handed to the right person?", note: "Measured on two short takes.",
    steps: ["shake hands with the person wearing a white sweater", "shake hands with the person wearing an orange sweater"] },
  evidence: { rec: "Film the sealing", run: "Check the seal",
    role: "For a custodian: was it sealed before it left the table?", note: "Example list. Not measured.",
    steps: ["put the item in the evidence bag", "seal the bag", "write on the bag label", "hand the bag to the second person"] },
  lockout: { rec: "Film the lockout", run: "Check the lockout",
    role: "For a supervisor: lock, tag, then try the switch.", note: "Example list. Not measured.",
    steps: ["switch the machine off", "put the lock on the switch", "hang the tag on the lock", "try the switch"] },
  own: { rec: "Film the job", run: "Check the list",
    role: "Any job you can type, one step per line.", note: "", steps: [] },
};
const caseKey = (name) => "clock.steps." + name;
const caseText = {};      // each tab's text, also held in memory, so a switch loses nothing in private mode
let activeCase = "packing";
// By default the checklist is one free-text box. The use-case tabs with their starting lists show only with ?cases=1.
const SHOW_CASES = new URLSearchParams(location.search).get("cases") === "1";
let caseReady = false;    // false until the first tab is on screen, so the empty box is not saved over a list

const recWord = () => CASES[activeCase].rec;
const runWord = () => CASES[activeCase].run;

// The text of the box stays with the tab it was typed in.
function keepSteps() {
  caseText[activeCase] = $("steps").value;
  store(caseKey(activeCase), $("steps").value);
}

function stepsChanged() {
  countSteps();
  keepSteps();
}

// A tab's own text if it has any, else its starting list.
function caseSteps(name) {
  const held = typeof caseText[name] === "string" ? caseText[name] : store(caseKey(name));
  return held && held.trim() ? held : CASES[name].steps.join("\n");
}

function setCase(name, keep) {
  const next = CASES[name] ? name : "packing";
  if (caseReady) keepSteps();
  activeCase = next;
  caseReady = true;
  const c = CASES[next];
  $("steps").value = caseSteps(next);
  document.querySelectorAll("#cases button").forEach((b) => {
    b.setAttribute("aria-pressed", b.dataset.case === next ? "true" : "false");
  });
  if ($("case-role")) $("case-role").textContent = c.role;   // absent in a cached older page
  if ($("case-note")) $("case-note").textContent = c.note;
  if (!state.recorder && !state.timer) $("rec").textContent = c.rec;   // while recording the button keeps its stop word
  $("run").textContent = c.run;
  if (keep) store("clock.case", next);
  countSteps();
}

// The single list of the earlier page moves to the "Your own list" tab, once.
function moveOldSteps() {
  const old = store("clock.steps");
  if (!old || !old.trim() || (store(caseKey("own")) || "").trim()) return;
  store(caseKey("own"), old);
  forget("clock.steps");
}

function clipLength() {
  const v = Math.min(45, Math.max(15, parseInt($("len").value, 10) || 25));
  $("len-out").textContent = v + " s";
  return v;
}

// ---------- camera, recording, file ----------

// A second press while the camera is still opening joins the first request, so only one stream is opened.
function cameraOn() {
  if (!state.camPending) state.camPending = openCamera().finally(() => { state.camPending = null; });
  return state.camPending;
}

function cannotRecord(text) {
  say($("status"), text, true);
  $("filebox").classList.add("point");
}

async function openCamera() {
  if (state.stream) return true;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
    cannotRecord("This browser cannot record here. Use the file button, marked in red.");
    return false;
  }
  const want = { facingMode: state.facing, width: { ideal: 1280 }, height: { ideal: 720 } };
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: want, audio: true });
  } catch (e1) {
    try {   // the microphone may be blocked while the camera is allowed
      state.stream = await navigator.mediaDevices.getUserMedia({ video: want, audio: false });
    } catch (e2) {
      cannotRecord("The camera was blocked or is missing. Allow it, or use the file button, marked in red.");
      return false;
    }
  }
  say($("status"), "");
  showLive();
  return true;
}

function cameraOff() {
  if (!state.stream) return;
  state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
}

async function flipCamera() {
  if (state.recorder || state.timer || state.starting) return;
  state.facing = state.facing === "environment" ? "user" : "environment";
  $("flip").textContent = state.facing === "environment" ? "Use front camera" : "Use back camera";
  if (state.stream) {
    cameraOff();
    await cameraOn();
  }
}

function unfloat() {
  $("player").classList.remove("float");
  $("player").style.top = "";
  if ($("player-slot")) $("player-slot").style.minHeight = "";   // absent in a cached older page
  $("full").hidden = true;
  markRow(null);
}

function showLive() {
  video.onerror = null;
  video.removeAttribute("src");
  video.srcObject = state.stream;
  video.muted = true;          // no echo from the speakers while the microphone is live
  video.controls = false;
  video.play().catch(() => {});
  unfloat();
  caption(state.facing === "environment" ? "Camera preview, back camera." : "Camera preview, front camera.");
}

function showFull() {
  unfloat();
  video.onerror = null;
  video.srcObject = null;
  if (!state.fullUrl) {
    video.removeAttribute("src");
    video.load();
    video.controls = false;
    $("caption").textContent = String.fromCharCode(160);   // keeps the line height
    $("screen-note").hidden = false;
    return;
  }
  video.src = state.fullUrl;
  video.muted = false;
  video.controls = true;
  caption("Full video: " + state.name);
}

function setVideo(blob, name) {
  if (state.fullUrl) URL.revokeObjectURL(state.fullUrl);
  state.blob = blob;
  state.name = name;
  state.fullUrl = URL.createObjectURL(blob);
  showFull();
}

function endCountdown() {
  clearInterval(state.timer);
  state.timer = null;
  $("countdown").hidden = true;
  $("rec").textContent = recWord();
  $("rec").classList.remove("live");
  $("flip").disabled = false;
}

function stopRecording() {
  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();   // onstop tidies up
  else endCountdown();
}

function countdown(word, num, live) {
  $("cd-word").textContent = word;
  $("cd-num").textContent = String(num);
  $("countdown").className = live ? "countdown rec" : "countdown";
  $("countdown").hidden = false;
}

async function record() {
  if (state.starting || state.busy) return;   // a double press while the camera opens must not start two recordings
  if (state.recorder || state.timer) { stopRecording(); return; }
  state.starting = true;
  const ok = await cameraOn();
  state.starting = false;
  if (!ok) return;
  showLive();
  const length = clipLength();
  $("rec").textContent = "Stop";
  $("flip").disabled = true;
  let lead = LEAD_IN_S;
  countdown("Starts in", lead, false);
  state.timer = setInterval(() => {
    lead -= 1;
    if (lead > 0) { countdown("Starts in", lead, false); return; }
    clearInterval(state.timer);
    state.timer = null;
    startRecorder(length);
  }, 1000);
}

function startRecorder(length) {
  const mime = MIMES.find((m) => MediaRecorder.isTypeSupported(m)) || "";
  const opts = { videoBitsPerSecond: VIDEO_BITS };
  if (mime) opts.mimeType = mime;
  let rec;
  try {
    rec = new MediaRecorder(state.stream, opts);
  } catch (e) {
    endCountdown();
    cannotRecord("Recording could not start in this browser. Use the file button, marked in red.");
    return;
  }
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  rec.onerror = () => { cannotRecord("The recording stopped with an error. Use the file button, marked in red."); };
  rec.onstop = () => {
    state.recorder = null;
    endCountdown();
    cameraOff();
    const keep = !state.discard;   // a file chosen during the recording replaces it
    state.discard = false;
    const type = (rec.mimeType || mime || "video/webm").split(";")[0];
    if (keep && chunks.length) {
      setVideo(new Blob(chunks, { type: type }), "recording." + (type.includes("mp4") ? "mp4" : "webm"));
      say($("status"), "Recorded. Now press \"" + runWord() + "\".");
    } else if (keep) showFull();
  };
  state.recorder = rec;
  rec.start(1000);
  $("rec").classList.add("live");
  const end = Date.now() + length * 1000;
  const tick = () => {
    const left = Math.max(0, Math.ceil((end - Date.now()) / 1000));
    countdown("REC", left, true);
    if (left <= 0 && rec.state !== "inactive") rec.stop();
  };
  tick();
  state.timer = setInterval(tick, 250);
}

function pickFile() {
  const f = $("file").files[0];
  if (!f) return;
  if (!(f.type || "").startsWith("video/") && !/\.(mp4|mov|webm|m4v|mkv)$/i.test(f.name)) {
    say($("status"), "Choose a video file (.mp4, .mov or .webm).", true);
    return;
  }
  if (state.recorder && state.recorder.state !== "inactive") state.discard = true;
  stopRecording();
  cameraOff();
  $("filebox").classList.remove("point");
  setVideo(f, f.name || "video.mp4");
  say($("status"), "Video chosen. Now press \"" + runWord() + "\".");
}

// ---------- network: every call has a timeout ----------

async function fetchJson(url, opts, ms, retries) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    const res = await fetch(url, Object.assign({ signal: ctl.signal }, opts));
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(typeof body.detail === "string" ? body.detail : "The server answered with error " + res.status + ".");
      err.status = res.status;
      throw err;
    }
    return body;
  } catch (e) {
    if (e.name === "AbortError") throw new Error("The server did not answer in " + Math.round(ms / 1000) + " seconds.");
    if (e instanceof TypeError) {   // the request did not reach the server
      if (retries > 0) return fetchJson(url, opts, ms, retries - 1);
      throw new Error("The server could not be reached. Check the connection.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

function setBusy(busy) {
  state.busy = busy;
  ["run", "ask-btn", "rec", "flip"].forEach((id) => { $(id).disabled = busy; });
}

// ---------- drawing ----------

function tickTimes(dur) {
  const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((s) => dur / s <= 8) || 600;
  const out = [];
  for (let t = 0; t <= dur + 1e-6; t += step) out.push(t);
  return out;
}

function orderFlags(res) {
  const flags = {};
  const add = (i, text) => { (flags[i] = flags[i] || []).push(text); };
  (res.order_issues || []).forEach((o) => {
    add(o.earlier, "Out of order: seen before step " + (o.later + 1));
    add(o.later, "Out of order: seen after step " + (o.earlier + 1));
  });
  return flags;
}

function rangeText(s) {
  const lo = s.lo === null ? s.time : s.lo;
  return lo.toFixed(1) + " to " + (s.hi === null ? "end" : s.hi.toFixed(1) + " s");
}

// Posterior over the start time as a filled curve. The peak is 1, so every row uses the full height.
function densityCurve(s, dur) {
  const g = svg("svg", { class: "dens", viewBox: "0 0 100 40", preserveAspectRatio: "none", role: "img" });
  g.setAttribute("aria-label", "Step " + (s.index + 1) + ": probability of each start time. Most likely " +
    secs(s.time) + ", range " + rangeText(s) + ".");
  const pts = s.taus.map((t, j) => pct(t, dur).toFixed(2) + " " + (38 - 35 * Math.max(0, Math.min(1, s.density[j]))).toFixed(2));
  const d = "M" + pct(s.taus[0], dur).toFixed(2) + " 38L" + pts.join("L") + "L" + pct(s.taus[s.taus.length - 1], dur).toFixed(2) + " 38Z";
  g.append(svg("path", { d: d }));
  return g;
}

function ladderVote(p, res) {
  const th = res.thresholds || { done_at: 0.5, not_found_at: 0.5 };
  return p >= th.done_at ? "yes" : (p < th.not_found_at ? "no" : "");
}

// What each source said about this step. A source whose yes or no goes against the verdict is boxed.
function witnessMarks(s, res) {
  const w = s.witnesses;
  if (!w) return null;
  const box = el("span", "wit");
  const names = WITNESS_ORDER.filter((n) => n in w).concat(Object.keys(w).filter((n) => !WITNESS_ORDER.includes(n)));
  const marks = [];
  names.forEach((name) => {
    const o = w[name];
    const label = WITNESS_NAMES[name] || name;
    let text;
    let vote;
    if (name === "ladder") {
      vote = ladderVote(o.p, res);
      text = label + " " + o.p.toFixed(2) + (o.time === null || o.time === undefined ? "" : " at " + secs(o.time));
    } else {
      vote = o.happened ? "yes" : "no";
      text = label + " " + vote + (o.happened && typeof o.start === "number" ? " " + o.start.toFixed(1) : "");
    }
    const against = (s.status === "done" && vote === "no") || (s.status === "not_found" && vote === "yes");
    const m = el("span", against ? "w against" : "w", text);
    if (against) m.append(el("span", "tag", "disagrees"));
    marks.push(m);
  });
  Object.keys(res.witness_calls || {}).forEach((name) => {
    if (!(name in w)) marks.push(el("span", "w silent", (WITNESS_NAMES[name] || name) + " no answer"));
  });
  marks.forEach((m, i) => {
    if (i) box.append(el("span", "sep", "/"));
    box.append(m);
  });
  return box;
}

// ---------- audit: an optional second check that arrives after the full read ----------

// The audit's verdicts by step index, or null. An older server sends no audit fields at all.
function auditMap(p) {
  const a = p && p.fused && p.audit_state === "done" ? p.audit : null;
  if (!a || !Array.isArray(a.steps) || !a.steps.length) return null;
  const map = {};
  a.steps.forEach((r) => { if (r && typeof r.index === "number") map[r.index] = r; });
  return map;
}

const disputedDone = (s, a) => Boolean(a && a.audit === "dispute" && s.status === "done");

// The third status line. Empty and hidden until there is a full read and the server speaks of an audit.
function auditLine(view, p) {
  const node = view.audit;
  if (!node) return;
  const st = p && p.fused ? p.audit_state : undefined;
  const map = auditMap(p);
  let text = "";
  let quiet = false;
  if (st === "pending") {
    text = "Audit: a Gemini Managed Agent is checking start and end frames in a sandbox ..." +
      (typeof p.audit_elapsed === "number" ? " " + Math.round(p.audit_elapsed) + " s" : "");
  } else if (st === "done" && map) {
    const rows = Object.keys(map).map((k) => map[k].audit);
    const yes = rows.filter((v) => v === "confirm").length;
    const no = rows.filter((v) => v === "dispute").length;
    text = "Audit: " + (typeof p.audit.seconds === "number" ? secs(p.audit.seconds) + ", " : "") + "Gemini Managed Agent, " +
      (p.audit.agent_ran_code ? "rules run in a Google sandbox" : "sandbox run not confirmed") +
      ". Confirmed " + yes + ", disputed " + no + ", could not tell " + (rows.length - yes - no) + ".";
  } else if (typeof st === "string") {   // off, failed, or an audit with nothing usable in it
    text = "Audit: not available for this run.";
    quiet = true;
  }
  node.textContent = text;
  node.hidden = !text;
  node.classList.toggle("quiet", quiet);
}

function auditSheet(view, p) {
  if (!view.sheet || !view.sheetText) return;
  const text = auditMap(p) && typeof p.audit.sheet === "string" ? p.audit.sheet.trim() : "";
  if (view.sheetText.textContent !== text) view.sheetText.textContent = text;
  // The agent's sheet counts steps from 0 and the table counts from 1. The sample sheet counts from 1.
  const head = view.sheet.querySelector("summary");
  const label = "Sign-off sheet" + (text && !p.audit.stub ? " (the agent numbers steps from 0)" : "");
  if (head && head.textContent !== label) head.textContent = label;
  view.sheet.hidden = !text;
}

// The audit's notes for one step. Shown under the row while the row is selected or has the focus.
function auditNotes(a) {
  const box = el("span", "audx");
  [["Audit reason", a.reason], ["At start", a.at_start], ["At end", a.at_end]].forEach((pair) => {
    const line = el("span", "audl");
    line.append(el("span", "k", pair[0] + ": "), document.createTextNode(String(pair[1] || "no note")));
    box.append(line);
  });
  return box;
}

function fillRow(b, s, res, ticks, flags, aud) {
  const dur = res.video.duration;
  const fused = res.stage === "fused" && Array.isArray(s.taus) && s.taus.length > 1;
  const on = b.classList.contains("on");
  b.className = "row " + s.status + (fused ? " fused" : "") + (disputedDone(s, aud) ? " disputed" : "") + (on ? " on" : "");
  const idx = el("span", "idx", String(s.index + 1));

  const what = el("span", "what");
  what.append(el("span", "txt", s.text), el("span", "st", WORDS[s.status] || s.status));
  if (disputedDone(s, aud)) what.append(el("span", "flag disputed", "DISPUTED BY AUDIT"));
  flags.forEach((f) => what.append(el("span", "flag", f)));

  const track = el("span", "track");
  ticks.forEach((t) => {
    const k = el("span", "tick");
    k.style.left = pct(t, dur) + "%";
    track.append(k);
  });
  const time = el("span", "time");
  if (s.time === null || s.time === undefined) {
    track.append(el("span", "none", s.status === "not_found" ? "NOT FOUND" : "no start time"));
    time.append(el("span", "big", "none"), el("span", "small", "no start time"));
    b.disabled = !aud;   // with audit notes the row can be selected to read them; there is no clip to play
    b.onclick = aud ? () => markRow(b.classList.contains("on") ? null : b) : null;
    if (aud) b.title = "Show the audit notes for this step";
    else b.removeAttribute("title");
  } else {
    const lo = s.lo === null ? s.time : s.lo;
    const hi = s.hi === null ? dur : s.hi;
    if (fused) track.append(densityCurve(s, dur));
    const r = el("span", fused ? "range thin" : "range");
    r.style.left = pct(lo, dur) + "%";
    r.style.width = Math.max(0, pct(hi, dur) - pct(lo, dur)) + "%";
    const e = el("span", "est");
    e.style.left = pct(s.time, dur) + "%";
    track.append(r, e);
    time.append(el("span", "big", s.time.toFixed(1)), el("span", "small", rangeText(s)));
    b.disabled = false;
    b.title = "Play the 4 second clip at " + secs(s.time);
    b.onclick = () => playClip(res, s, b);
  }
  const score = el("span", "score", s.p_done.toFixed(2));
  const kids = [idx, what, track, time, score];
  let marks = witnessMarks(s, res);
  if (aud) {
    if (marks) marks.append(el("span", "sep", "/"));
    else marks = el("span", "wit");
    // a step that was not found keeps its plain mark: the audit never softens that row
    const loud = aud.audit === "dispute" && s.status !== "not_found";
    marks.append(el("span", loud ? "w aud dispute" : "w aud", "Audit: " + (AUDIT_WORDS[aud.audit] || "cannot tell")));
  }
  if (marks) kids.push(marks);
  if (aud) kids.push(auditNotes(aud));
  b.replaceChildren(...kids);
}

// Draw a result into a view. The same row nodes are reused when the full read replaces the first read.
function draw(view, res, audit) {
  const dur = res.video.duration;
  const ticks = tickTimes(dur);
  const flags = orderFlags(res);
  if (view.run !== res.run || view.rows.length !== res.steps.length) {
    const head = el("div", "head");
    const axis = el("span", "track");
    ticks.forEach((t, i) => {
      const l = el("span", "ticklabel", i === ticks.length - 1 ? t + " s" : String(t));
      l.style.left = pct(t, dur) + "%";
      if (i > 0 && pct(t, dur) > 92) l.classList.add("end");   // keep the last label inside the strip
      axis.append(l);
    });
    head.append(el("span", "idx", "No."), el("span", "what", "Step and status"), axis, el("span", "time", "Starts at, s"),
      el("span", "score", "P(done)"));
    view.rows = res.steps.map(() => { const b = el("button", "row"); b.type = "button"; return b; });
    view.run = res.run;
    view.box.replaceChildren(head, ...view.rows);
  }
  res.steps.forEach((s, i) => fillRow(view.rows[i], s, res, ticks, flags[s.index] || [], audit ? audit[s.index] || null : null));
}

function summarize(res, audit) {
  const n = res.steps.length;
  const count = (st) => res.steps.filter((s) => s.status === st).length;
  const flagged = [];
  res.steps.forEach((s) => {
    if (s.status !== "done") flagged.push("step " + (s.index + 1) + " " + (WORDS[s.status] || s.status).toLowerCase());
    else if (audit && disputedDone(s, audit[s.index])) flagged.push("step " + (s.index + 1) + " done but disputed by the audit");
  });
  (res.order_issues || []).forEach((o) => flagged.push("step " + (o.earlier + 1) + " seen before step " + (o.later + 1)));
  return n + (n === 1 ? " step: " : " steps: ") + count("done") + " done, " + count("not_found") + " not found, " +
    count("unsure") + " unsure. " + (flagged.length ? "Flagged for review: " + flagged.join("; ") + "." : "No step flagged.");
}

function answerLine(res, audit) {
  const s = res.steps[0];
  const p = " P(done) " + s.p_done.toFixed(2) + "." + (audit && disputedDone(s, audit[s.index]) ? " Disputed by the audit: check it." : "");
  if (s.status === "not_found") return "Not found in this video." + p;
  if (s.time === null || s.time === undefined) return "Unsure, and no start time could be read." + p;
  return (s.status === "done" ? "It started at about " : "Unsure. Best estimate ") + secs(s.time) + ", range " + rangeText(s) + "." + p;
}

function witnessCount(res) {
  let n = res.call && typeof res.call.seconds === "number" && !res.call.error ? 1 : 0;
  Object.keys(res.witness_calls || {}).forEach((k) => {
    const c = res.witness_calls[k];
    if (c.ok && !c.left_out) n += 1;
  });
  return n;
}

function legendText(res, audit) {
  const fused = res.stage === "fused";
  return "Each strip spans the video, 0 to " + secs(res.video.duration) + ". " +
    (fused ? "Filled curve: probability of each start time after all sources are combined. Line under it: the range. "
      : "Bar: the range of the start time from the first read. ") +
    "Tick: the estimate. P(done): 0 to 1 that the step happened at all. " +
    (fused ? "Under each row: what each source said, with a box around any source that goes against the verdict. " : "") +
    "Select a row to play its 4 second clip." +
    (audit ? " Audit: a managed agent's check of still frames from the start and the end. Select a row to read its notes." : "");
}

// p is one answer from /api/result
function show(view, p) {
  const res = p.fused || p.fast;
  $("sample").hidden = !(p.stub || (res && res.stub));   // the strip follows the run on screen
  const el1 = Math.round(p.elapsed) + " s";
  if (p.fast) {
    say(view.fast, "First read: " + secs(p.fast.seconds_total) + ", one Decisions call, " + p.fast.call.n_questions + " questions");
  } else if (p.fused) {
    say(view.fast, "First read: no answer from the Decisions call");
  } else {
    say(view.fast, p.stage === "error" ? "" : "First read: running, " + el1);
  }
  if (p.fused) {
    const n = witnessCount(p.fused);
    say(view.full, "Full read: " + secs(p.fused.seconds_total) + ", " + n + (n === 1 ? " witness" : " witnesses"));
  } else if (p.stage === "error") {
    say(view.full, (p.fast ? "Full read: failed. " : "") + (p.error || "The run failed."), true);
  } else {
    say(view.full, "Full read: running, " + el1);
  }
  const audit = auditMap(p);
  auditLine(view, p);
  auditSheet(view, p);
  if (!res) return;
  if (view.signoff) view.signoff.hidden = !p.fused;   // the line to sign appears with the full read
  draw(view, res, audit);
  if (view.summary) view.summary.textContent = view.one ? answerLine(res, audit) : summarize(res, audit);
  if (view.alarm) {
    const missing = res.steps.filter((s) => s.status === "not_found");
    view.alarm.hidden = !missing.length;
    view.alarm.replaceChildren(el("strong", "", "NOT FOUND"),
      ...missing.map((s) => el("span", "", "Step " + (s.index + 1) + ": " + s.text)));
  }
  if (view.legend) {
    view.legend.textContent = legendText(res, audit);
    view.legend.hidden = false;
  }
  if (view.footer) {
    const c = res.call || {};
    view.footer.textContent = (res.stub ? "Sample data, no model call. " : "") +
      "Seconds: first read " + (p.fast ? p.fast.seconds_total.toFixed(1) : "none") +
      ", full read " + (p.fused ? p.fused.seconds_total.toFixed(1) : (p.stage === "error" ? "none" : "running")) + ". " +
      "Questions: " + (c.n_questions || 0) + ". Input tokens, Decisions call: " + (c.input_tokens || 0).toLocaleString("en-US") +
      ". Run " + res.run + ".";
  }
}

function clearView(view, summary) {
  view.rows = [];
  view.run = "";
  view.box.replaceChildren();
  say(view.fast, "");
  say(view.full, "");
  auditLine(view, null);
  auditSheet(view, null);
  if (view.summary) view.summary.textContent = summary || "";
  if (view.alarm) view.alarm.hidden = true;
  if (view.legend) view.legend.hidden = true;
  if (view.footer) view.footer.textContent = "";
  if (view.signoff) view.signoff.hidden = true;
}

function markRow(btn) {
  document.querySelectorAll(".row.on").forEach((r) => r.classList.remove("on"));
  if (btn) btn.classList.add("on");
}

function playClip(res, s, btn) {
  if (state.recorder || state.timer) return;
  cameraOff();
  const url = "/api/clip?run=" + res.run + "&t=" + s.time.toFixed(1);
  let retried = false;
  video.srcObject = null;
  video.muted = false;
  video.controls = true;
  video.onerror = () => {
    if (retried) { $("caption").textContent = "The clip for step " + (s.index + 1) + " could not be loaded."; return; }
    retried = true;
    video.src = url + "&retry=1";
    video.play().catch(() => {});
  };
  video.src = url;
  video.play().catch(() => {});
  markRow(btn);
  const player = $("player");
  // On a narrow screen the player stays on top of the table. Its place in the page is kept,
  // so the table does not jump, and it sits under the sample-data strip when that is shown.
  if (narrow.matches && $("player-slot") && !player.classList.contains("float")) {
    $("player-slot").style.minHeight = player.offsetHeight + "px";
  }
  player.style.top = $("sample").hidden ? "" : $("sample").offsetHeight + "px";
  player.classList.add("float");
  $("full").hidden = !(state.fullUrl || narrow.matches);
  $("full").textContent = state.fullUrl ? "Show full video" : "Close clip";
  caption("Clip for step " + (s.index + 1) + ": 4 seconds around " + secs(s.time) + ".");
  if (narrow.matches && btn) {   // keep the selected row in sight, just under the floating player
    const gap = btn.getBoundingClientRect().top - player.getBoundingClientRect().bottom - 6;
    if (gap < 0) window.scrollBy(0, gap);
  }
}

// ---------- recorder view: run and poll ----------

async function follow(runId, view) {
  const t0 = Date.now();
  for (;;) {
    let p = null;
    try {
      p = await fetchJson("/api/result?run=" + runId, {}, 8000, 1);
    } catch (e) {
      if (e.status === 404 || Date.now() - t0 > GIVE_UP_MS) { say(view.status, e.message, true); return; }
    }
    if (p) {
      show(view, p);
      if (p.stage === "fused") { say(view.status, ""); followAudit(runId, view, p); return; }
      if (p.stage === "error") { say(view.status, p.error || "The run failed.", true); return; }
      say(view.status, (p.stage === "fast" ? "First read is in. Full read running. " : "Running. ") + Math.round(p.elapsed) + " s");
    }
    if (Date.now() - t0 > GIVE_UP_MS) { say(view.status, "No result after 95 seconds. Run again.", true); return; }
    await sleep(POLL_MS);
  }
}

// The audit arrives after the full read. It is followed apart from the run, so the Run button is
// free again as soon as the full read is in. Only the audit line moves until the audit is over.
async function followAudit(runId, view, first) {
  if (!first || first.audit_state !== "pending") return;   // no audit for this run, or an older server
  const t0 = Date.now();
  while (Date.now() - t0 < AUDIT_GIVE_UP_MS) {
    await sleep(AUDIT_POLL_MS);
    if (view.run !== runId) return;   // another run took over this view
    let p = null;
    try {
      p = await fetchJson("/api/result?run=" + runId, {}, 8000, 0);
    } catch (e) {
      if (e.status === 404) break;
    }
    if (view.run !== runId) return;
    if (p && p.audit_state !== "pending") { show(view, p); return; }
    if (p) auditLine(view, p);
  }
  if (view.run === runId) auditLine(view, { fused: true, audit_state: "failed" });
}

// A server still running the earlier one-stage code answers /api/run with the whole first read.
// Show it as the first read and say that the full read needs a restart.
function oneStage(view, res) {
  const took = res.call && typeof res.call.seconds === "number" ? res.call.seconds : 0;
  res.stage = "fast";
  res.seconds_total = took;
  const note = "This server runs the earlier one-stage code. Restart it to get the full read.";
  show(view, { run: res.run, stage: "error", fast: res, fused: null, error: note, elapsed: took, stub: Boolean(res.stub) });
  say(view.status, note, true);
}

async function runSteps(steps, view) {
  if (state.busy) return;
  if (state.recorder || state.timer) { say(view.status, "Wait for the recording to finish.", true); return; }
  if (!state.blob) { say(view.status, "Record or choose a video first.", true); return; }
  if (!steps.length) { say(view.status, "Type at least one step.", true); return; }
  const fd = new FormData();
  fd.append("video", state.blob, state.name);
  fd.append("steps", JSON.stringify(steps));
  setBusy(true);
  clearView(view, "");
  say(view.status, "Sending the video.");
  try {
    const started = await fetchJson("/api/run", { method: "POST", body: fd }, 60000, 0);
    $("sample").hidden = !started.stub;
    if (narrow.matches && view === mainView) $("timeline-h").scrollIntoView({ block: "start" });
    if (Array.isArray(started.steps)) oneStage(view, started);
    else await follow(started.run, view);
  } catch (e) {
    say(view.status, e.message, true);
  } finally {
    setBusy(false);
  }
}

function askOne(event) {
  event.preventDefault();
  const q = $("q").value.trim();
  if (!q) { say($("ask-status"), "Type what to look for.", true); return; }
  runSteps([q], askView);
}

// ---------- watch view: QR code, then follow the newest run ----------

async function loadQr() {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 8000);
  try {
    const res = await fetch("/api/qr", { signal: ctl.signal });
    if (!res.ok) throw new Error("bad answer");
    $("qr-url").textContent = (res.headers.get("X-Public-Url") || "").replace(/\/$/, "");
    $("qr").src = URL.createObjectURL(await res.blob());
  } catch (e) {
    $("qr-url").textContent = "The QR code could not be loaded. Open this address on the phone: " + location.origin;
  } finally {
    clearTimeout(timer);
  }
}

async function watchLoop() {
  const WAITING = "Waiting for a recording. Results appear here as they arrive.";
  let run = "";
  let final = false;
  let auditSince = 0;   // when this page first saw the run's audit pending
  let lastLatest = 0;
  clearView(mainView, WAITING);
  for (;;) {
    try {
      if (Date.now() - lastLatest >= 1000) {
        lastLatest = Date.now();
        const latest = await fetchJson("/api/latest", {}, 5000, 0);
        if (typeof latest.stub === "boolean") $("sample").hidden = !latest.stub;   // the server may have restarted
        if (latest.run && latest.run !== run) {
          run = latest.run;
          final = false;
          auditSince = 0;
          showFull();
          clearView(mainView, "A new recording arrived. Reading.");
        }
      }
      if (run && !final) {
        const p = await fetchJson("/api/result?run=" + run, {}, 8000, 0);
        const waiting = p.stage === "fused" && p.audit_state === "pending";
        if (waiting && auditSince) auditLine(mainView, p);   // the rows are drawn; only the audit line moves
        else show(mainView, p);
        if (waiting && !auditSince) auditSince = Date.now();
        final = p.stage === "error" || (p.stage === "fused" && !waiting);
        if (waiting && Date.now() - auditSince > AUDIT_GIVE_UP_MS) {
          final = true;
          auditLine(mainView, { fused: true, audit_state: "failed" });
        }
      }
      say($("status"), "");
    } catch (e) {
      if (e.status === 404) { run = ""; final = false; auditSince = 0; clearView(mainView, WAITING); }
    }
    await sleep(run && !final ? (auditSince ? AUDIT_POLL_MS : POLL_MS) : 1000);
  }
}

// ---------- start ----------

setTheme(document.documentElement.getAttribute("data-theme"), false);
if ($("theme")) {   // absent in a cached older page
  $("theme").addEventListener("click", () => setTheme(document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light", true));
}
$("full").addEventListener("click", showFull);

if (WATCH) {
  document.body.classList.add("watch");
  $("screen-note").textContent = "Select a step row to play its 4 second clip here.";
  loadQr();
  watchLoop();
} else {
  moveOldSteps();
  if (!SHOW_CASES) document.body.classList.add("free");
  setCase(!SHOW_CASES ? "own" : (CASES[CASE_PARAM] ? CASE_PARAM : store("clock.case")), SHOW_CASES && Boolean(CASES[CASE_PARAM]));
  document.querySelectorAll("#cases button").forEach((b) => {
    b.addEventListener("click", () => setCase(b.dataset.case, true));
  });
  $("len").value = store("clock.len") || "25";
  clipLength();
  $("steps").addEventListener("input", stepsChanged);
  $("steps").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runSteps(readSteps(), mainView); });
  $("len").addEventListener("input", () => store("clock.len", String(clipLength())));
  $("rec").addEventListener("click", record);
  $("flip").addEventListener("click", flipCamera);
  $("file").addEventListener("change", pickFile);
  $("run").addEventListener("click", () => runSteps(readSteps(), mainView));
  $("ask").addEventListener("submit", askOne);
  window.addEventListener("pagehide", cameraOff);
}

fetchJson("/api/health", {}, 5000, 1).then((h) => { if (h.stub) $("sample").hidden = false; }).catch(() => {});
