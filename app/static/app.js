"use strict";

// Discovery Clock page: record or pick a video, send it with the steps, draw the step timeline.

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const RUN_WAIT_MS = 100000;   // the server gives up at 90 s; the page waits a little longer
const LEAD_IN_S = 3;
const MIMES = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"];
const WORDS = { done: "Done", not_found: "Not found", unsure: "Unsure" };

const video = $("video");
const state = { stream: null, camPending: null, starting: false, discard: false, recorder: null, timer: null,
  blob: null, name: "", fullUrl: "", busy: false };

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

const secs = (t) => t.toFixed(1) + " s";
const pct = (t, dur) => Math.min(100, Math.max(0, (t / dur) * 100));

function say(node, text, isError) {
  node.textContent = text;
  node.classList.toggle("err", Boolean(isError));
}

function caption(text) {
  $("caption").textContent = text;
  $("screen-note").hidden = true;
}

// ---------- checklist ----------

function readSteps() {
  return $("steps").value.split("\n").map((s) => s.trim()).filter(Boolean);
}

function drawPreview() {
  $("preview").replaceChildren(...readSteps().map((s) => el("li", "", s)));
}

// ---------- camera, recording, file ----------

// A second click while the camera is still opening joins the first request, so only one stream is opened.
function cameraOn() {
  if (!state.camPending) state.camPending = openCamera().finally(() => { state.camPending = null; });
  return state.camPending;
}

async function openCamera() {
  if (state.stream) return true;
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    say($("status"), "This browser cannot record. Choose a file instead.", true);
    return false;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: true });
  } catch (e) {
    say($("status"), "The camera or microphone was blocked. Allow both, or choose a file.", true);
    return false;
  }
  say($("status"), "");
  $("cam").textContent = "Camera off";
  showLive();
  return true;
}

function cameraOff() {
  if (!state.stream) return;
  state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
  $("cam").textContent = "Camera on";
}

function showLive() {
  video.onerror = null;
  video.removeAttribute("src");
  video.srcObject = state.stream;
  video.muted = true;          // no echo from the speakers while the microphone is live
  video.controls = false;
  video.play().catch(() => {});
  markRow(null);
  $("full").hidden = true;
  caption("Camera preview.");
}

function showFull() {
  if (!state.fullUrl) {
    $("caption").textContent = String.fromCharCode(160);   // keeps the line height
    $("screen-note").hidden = false;
    return;
  }
  video.onerror = null;
  video.srcObject = null;
  video.src = state.fullUrl;
  video.muted = false;
  video.controls = true;
  markRow(null);
  $("full").hidden = true;
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
  $("rec").textContent = "Record";
}

function stopRecording() {
  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();   // onstop tidies up
  else endCountdown();
}

async function record() {
  if (state.starting) return;   // a double click while the camera opens must not start two recordings
  if (state.recorder || state.timer) { stopRecording(); return; }
  state.starting = true;
  const ok = await cameraOn();
  state.starting = false;
  if (!ok) return;
  showLive();
  const length = Math.min(45, Math.max(15, parseInt($("len").value, 10) || 25));
  $("len").value = length;
  $("rec").textContent = "Stop";
  const cd = $("countdown");
  let lead = LEAD_IN_S;
  cd.className = "countdown";
  cd.textContent = "Starts in " + lead;
  cd.hidden = false;
  state.timer = setInterval(() => {
    lead -= 1;
    if (lead > 0) { cd.textContent = "Starts in " + lead; return; }
    clearInterval(state.timer);
    state.timer = null;
    startRecorder(length);
  }, 1000);
}

function startRecorder(length) {
  const mime = MIMES.find((m) => MediaRecorder.isTypeSupported(m)) || "";
  let rec;
  try {
    rec = new MediaRecorder(state.stream, mime ? { mimeType: mime } : undefined);
  } catch (e) {
    endCountdown();
    say($("status"), "Recording could not start in this browser. Choose a file instead.", true);
    return;
  }
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  rec.onstop = () => {
    state.recorder = null;
    endCountdown();
    cameraOff();
    const keep = !state.discard;   // a file chosen during the recording replaces it
    state.discard = false;
    const type = (rec.mimeType || mime || "video/webm").split(";")[0];
    if (keep && chunks.length) setVideo(new Blob(chunks, { type: type }), "recording." + (type.includes("mp4") ? "mp4" : "webm"));
    else if (keep) showFull();
  };
  state.recorder = rec;
  rec.start(1000);
  const end = Date.now() + length * 1000;
  const cd = $("countdown");
  cd.className = "countdown rec";
  const tick = () => {
    const left = Math.max(0, Math.ceil((end - Date.now()) / 1000));
    cd.textContent = "REC " + left + " s left";
    if (left <= 0 && rec.state !== "inactive") rec.stop();
  };
  tick();
  state.timer = setInterval(tick, 250);
}

function pickFile() {
  const f = $("file").files[0];
  if (!f) return;
  if (!/\.(mp4|webm)$/i.test(f.name)) {
    say($("status"), "Choose an .mp4 or .webm file.", true);
    return;
  }
  if (state.recorder && state.recorder.state !== "inactive") state.discard = true;
  stopRecording();
  cameraOff();
  say($("status"), "");
  setVideo(f, f.name);
}

// ---------- network: every call has a timeout and one retry ----------

async function fetchJson(url, opts, ms, retries) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    const res = await fetch(url, Object.assign({ signal: ctl.signal }, opts));
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(typeof body.detail === "string" ? body.detail : "The server answered with error " + res.status + ".");
    }
    return body;
  } catch (e) {
    if (e.name === "AbortError") throw new Error("No answer after 90 seconds. Try a shorter video or fewer steps.");
    if (e instanceof TypeError) {   // the request did not reach the server, so asking again is safe
      if (retries > 0) return fetchJson(url, opts, ms, retries - 1);
      throw new Error("The server could not be reached. Check that it is running.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

function setBusy(busy) {
  state.busy = busy;
  ["run", "ask-btn", "rec", "cam"].forEach((id) => { $(id).disabled = busy; });
}

async function runSteps(steps, statusNode) {
  if (state.busy) return null;
  if (state.recorder || state.timer) { say(statusNode, "Wait for the recording to finish.", true); return null; }
  if (!state.blob) { say(statusNode, "Record or choose a video first.", true); return null; }
  if (!steps.length) { say(statusNode, "Type at least one step.", true); return null; }
  const fd = new FormData();
  fd.append("video", state.blob, state.name);
  fd.append("steps", JSON.stringify(steps));
  if ($("clock").checked) fd.append("clock", "1");
  setBusy(true);
  const t0 = Date.now();
  say(statusNode, "Running. 0 s");
  const tick = setInterval(() => say(statusNode, "Running. " + Math.round((Date.now() - t0) / 1000) + " s"), 1000);
  try {
    const res = await fetchJson("/api/run", { method: "POST", body: fd }, RUN_WAIT_MS, 1);
    say(statusNode, "");
    if (res.stub) $("sample").hidden = false;
    return res;
  } catch (e) {
    say(statusNode, e.message, true);
    return null;
  } finally {
    clearInterval(tick);
    setBusy(false);
  }
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

function curve(s, res) {
  const dur = res.video.duration;
  const x = (j) => pct(res.grid[j], dur).toFixed(2);
  const y = (p) => (3 + (1 - p) * 34).toFixed(2);
  const points = (arr) => arr.map((p, j) => x(j) + "," + y(p)).join(" ");
  const g = svg("svg", { class: "curve", viewBox: "0 0 100 40", preserveAspectRatio: "none", role: "img" });
  g.setAttribute("aria-label", "Step " + (s.index + 1) + " answers over " + res.grid.length + " time points. Raw answers run from " +
    Math.min(...s.raw).toFixed(2) + " to " + Math.max(...s.raw).toFixed(2) + ". The fitted curve ends at " +
    s.fitted[s.fitted.length - 1].toFixed(2) + ".");
  g.append(
    svg("line", { class: "half", x1: 0, x2: 100, y1: y(0.5), y2: y(0.5) }),
    svg("polyline", { class: "raw", points: points(s.raw) }),
    svg("path", { class: "dots", d: s.raw.map((p, j) => "M" + x(j) + " " + y(p) + "h0").join("") }),
    svg("polyline", { class: "fit", points: points(s.fitted) })
  );
  return g;
}

function row(s, res, ticks, flags) {
  const dur = res.video.duration;
  const b = el("button", "row");
  b.type = "button";
  b.append(el("span", "idx", String(s.index + 1)));

  const what = el("span");
  what.append(el("span", "txt", s.text), el("span", "st " + s.status, WORDS[s.status] || s.status));
  flags.forEach((f) => what.append(el("span", "flag", f)));
  b.append(what);

  const track = el("span", "track");
  ticks.forEach((t) => {
    const k = el("span", "tick");
    k.style.left = pct(t, dur) + "%";
    track.append(k);
  });
  track.append(curve(s, res));
  const time = el("span", "time");
  if (s.time === null) {
    time.append(el("span", "big", "none"), el("span", "small", "no start time"));
    b.disabled = true;
  } else {
    const lo = s.lo === null ? s.time : s.lo;
    const hi = s.hi === null ? dur : s.hi;
    const r = el("span", "range");
    r.style.left = pct(lo, dur) + "%";
    r.style.width = Math.max(0, pct(hi, dur) - pct(lo, dur)) + "%";
    const e = el("span", "est");
    e.style.left = pct(s.time, dur) + "%";
    track.append(r, e);
    time.append(el("span", "big", s.time.toFixed(1)), el("span", "small", rangeText(s)));
    b.title = "Play the 4 second clip at " + secs(s.time);
    b.addEventListener("click", () => playClip(res, s, b));
  }
  b.append(track, time, el("span", "score", s.p_done.toFixed(2)));
  return b;
}

function render(box, res) {
  const dur = res.video.duration;
  const ticks = tickTimes(dur);
  const flags = orderFlags(res);
  const head = el("div", "head");
  const axis = el("span", "track");
  ticks.forEach((t, i) => {
    const l = el("span", "ticklabel", i === ticks.length - 1 ? t + " s" : String(t));
    l.style.left = pct(t, dur) + "%";
    if (i > 0 && pct(t, dur) > 92) l.classList.add("end");   // keep the last label inside the strip
    axis.append(l);
  });
  head.append(el("span", "", "No."), el("span", "", "Step and status"), axis, el("span", "", "Start, s"), el("span", "score", "Score"));
  box.replaceChildren(head, ...res.steps.map((s) => row(s, res, ticks, flags[s.index] || [])));
}

function summarize(res) {
  const n = res.steps.length;
  const count = (st) => res.steps.filter((s) => s.status === st).length;
  const flagged = [];
  res.steps.forEach((s) => {
    if (s.status !== "done") flagged.push("step " + (s.index + 1) + " " + (WORDS[s.status] || s.status).toLowerCase());
  });
  (res.order_issues || []).forEach((o) => flagged.push("step " + (o.earlier + 1) + " seen before step " + (o.later + 1)));
  return n + (n === 1 ? " step: " : " steps: ") + count("done") + " done, " + count("not_found") + " not found, " +
    count("unsure") + " unsure. " + (flagged.length ? "Flagged for review: " + flagged.join("; ") + "." : "No step flagged.");
}

function callLine(res) {
  const c = res.call;
  return (res.stub ? "Sample data, no model call. " : "") + "Took " + c.seconds.toFixed(1) + " s. " + c.n_questions +
    " questions. " + c.input_tokens.toLocaleString("en-US") + " input tokens. Run " + res.run + ".";
}

function legendText(res) {
  return "Each strip spans the video, 0 to " + secs(res.video.duration) + ". Dots and thin line: raw answers to \"had this step " +
    "started by now?\". Thick line: fitted curve. Dashed line: 0.5. Bar and marker: start estimate with its range, placed " +
    res.bias_s.toFixed(1) + " s earlier than the curve by the engine's fixed correction. Score: 0 to 1 that the step happened at all. " +
    "Select a row to play its 4 second clip.";
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
  $("full").hidden = !state.fullUrl;
  caption("Clip for step " + (s.index + 1) + ": 4 seconds around " + secs(s.time) + ".");
}

// ---------- actions ----------

async function runChecklist() {
  const res = await runSteps(readSteps(), $("status"));
  if (!res) return;
  $("summary").textContent = summarize(res);
  render($("results"), res);
  $("legend").textContent = legendText(res);
  $("legend").hidden = false;
  $("footer").textContent = callLine(res);
}

async function askOne(event) {
  event.preventDefault();
  const q = $("q").value.trim();
  if (!q) { say($("ask-status"), "Type what to look for.", true); return; }
  const res = await runSteps([q], $("ask-status"));
  if (!res) return;
  const s = res.steps[0];
  const score = " Score " + s.p_done.toFixed(2) + ". ";
  let line;
  if (s.status === "not_found") line = "Not found in this video.";
  else if (s.time === null) line = "Unsure, and no start time could be read.";
  else line = (s.status === "done" ? "It started at about " : "Unsure. Best estimate ") + secs(s.time) + ", range " + rangeText(s) + ".";
  $("answer-line").textContent = line + score + callLine(res);
  render($("answer"), res);
}

$("steps").addEventListener("input", drawPreview);
$("steps").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runChecklist(); });
$("cam").addEventListener("click", () => { if (state.stream) { stopRecording(); cameraOff(); showFull(); } else cameraOn(); });
$("rec").addEventListener("click", record);
$("file").addEventListener("change", pickFile);
$("full").addEventListener("click", showFull);
$("run").addEventListener("click", runChecklist);
$("ask").addEventListener("submit", askOne);
window.addEventListener("pagehide", cameraOff);

drawPreview();
fetchJson("/api/health", {}, 5000, 1).then((h) => { if (h.stub) $("sample").hidden = false; }).catch(() => {});
