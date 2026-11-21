"""Web app for Discovery Clock: record or upload a video, list the steps, read a step timeline.

Start it from the project root:
    .venv/Scripts/python.exe -m uvicorn app.server:app --port 8765
Then open http://localhost:8765/ to record, or http://localhost:8765/?watch=1 for the watch view.

A run goes in two stages. POST /api/run stores the video, starts engine.run_fused on a
background thread and answers at once. GET /api/result is polled by the page: first the fast
read arrives (one Decisions call), then the full read (ladder plus three model timelines).

Set CLOCK_STUB=1 to get sample data without any model call. The page labels it as sample data.
Set PUBLIC_URL to the public address of the app when it sits behind a tunnel (used for the QR code).

After the full read is stored, an audit starts on its own thread (clock/auditor.py, a Gemini Managed
Agent). It is optional: it never delays or changes the first read or the full read, and a failed
audit only leaves a note on the run. Set CLOCK_AUDIT=0 to switch it off.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clock import engine, ladder, media  # noqa: E402

UPLOADS = ROOT / "runs" / "uploads"
CLIPS = ROOT / "runs" / "clips"
TUNNEL_LOG = ROOT / "runs" / "tunnel.log"
STATIC = Path(__file__).resolve().parent / "static"

RUN_DEADLINE_S = 75         # a run with no full read by then is reported as an error
MAX_ACTIVE_RUNS = 2
MAX_KEPT_RUNS = 40
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_STEPS = 20              # 20 steps x 21 questions stays inside one Decisions call
MAX_STEP_CHARS = 200
CLIP_BEFORE_S = 1.5         # the clip runs from 1.5 s before the estimate to 2.5 s after it
CLIP_LENGTH_S = 4.0
EMPTY_CLIP_BYTES = 500      # ffmpeg writes a 261 byte file when the start is past the end
STUB_FAST_S = 1.0
STUB_FUSED_S = 4.0
STUB_AUDIT_S = 6.0          # the sample audit arrives this long after the sample full read
AUDIT_TIMEOUT_S = 110       # given to the auditor for its one managed-agent call (no retry: the slot must free up)
AUDIT_DEADLINE_S = 125      # an audit with no answer by then is reported as failed
AUDIT_KEYS = ("ok", "seconds", "model", "steps", "sheet", "agent_ran_code", "order_pairs", "rules_agree")
AUDIT_STEP_KEYS = ("index", "text", "at_start", "at_end", "audit", "reason")
VIDEO_EXTS = (".webm", ".mp4", ".mov", ".m4v", ".mkv")
EXT_BY_TYPE = {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
               "video/x-matroska": ".mkv", "video/x-m4v": ".m4v"}

app = FastAPI(title="Discovery Clock")
_runs: dict[str, dict] = {}     # run id -> state; insertion order is start order
_runs_lock = threading.Lock()
_clip_lock = threading.Lock()
_audit_lock = threading.Lock()  # held while an audit runs: at most one at a time across the server


def stub_on() -> bool:
    return os.environ.get("CLOCK_STUB") == "1"


def audit_on() -> bool:
    return os.environ.get("CLOCK_AUDIT") != "0"


def parse_steps(raw: str) -> list[str]:
    try:
        steps = json.loads(raw)
    except ValueError:
        steps = None
    if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
        raise HTTPException(400, "The steps field must be a JSON list of strings.")
    steps = [" ".join(s.split())[:MAX_STEP_CHARS] for s in steps if s.strip()]
    if not steps:
        raise HTTPException(400, "Type at least one step.")
    if len(steps) > MAX_STEPS:
        raise HTTPException(400, f"At most {MAX_STEPS} steps per run.")
    return steps


async def save_upload(video: UploadFile, run_id: str) -> Path:
    ext = Path(video.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        # some phone browsers send a name without an extension; the content type still says what it is
        ext = EXT_BY_TYPE.get((video.content_type or "").split(";")[0].strip().lower(), "")
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, "The video must be a .mp4, .mov, .webm, .m4v or .mkv file.")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{run_id}{ext}"
    size = 0
    with open(path, "wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                break
            f.write(chunk)
    if size == 0 or size > MAX_UPLOAD_BYTES:
        path.unlink(missing_ok=True)
        if size == 0:
            raise HTTPException(400, "The video file is empty.")
        raise HTTPException(413, "The video is larger than 200 MB.")
    return path


def plain_error(e: Exception) -> str:
    """One readable sentence for the page. Never carries the key or a local path."""
    text = f"{type(e).__name__}: {e}".replace(str(ROOT), ".")
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.environ.get(name)
        if key:
            text = text.replace(key, "[key]")
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "The model's per-minute limit was reached. Wait one minute, then run again."
    if "No GEMINI_API_KEY" in text:
        return "No API key is set on the server. Put GEMINI_API_KEY in .env and restart."
    if "ffmpeg" in text or "no duration" in text:
        return "The video could not be read. Record again or use an .mp4 file."
    if "No model answered" in text:
        return "No model answered. Check the network, then run again."
    if "timeout" in text.lower() or "timed out" in text.lower():
        return "The model did not answer in time. Run again."
    return "The run failed. " + text[:300]


def scrub(value):
    """Take the project folder and the API key out of every string in a result (model error texts)."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        for root in (str(ROOT), ROOT.as_posix()):
            value = value.replace(root, ".")
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            key = os.environ.get(name)
            if key:
                value = value.replace(key, "[key]")
    return value


def public_view(result: dict, run_id: str, name: str, stub: bool) -> dict:
    """The result as the page gets it: no absolute local path, plus what the page needs to draw."""
    out = scrub(result)
    out["video"] = dict(result.get("video") or {}, path=f"runs/uploads/{name}")
    out.update({"run": run_id, "stub": stub,
                "thresholds": {"done_at": engine.DONE_AT, "not_found_at": engine.NOT_FOUND_AT},
                "clip": {"before": CLIP_BEFORE_S, "length": CLIP_LENGTH_S}})
    return out


# ---------- sample data (CLOCK_STUB=1) ----------

def _order_issues(steps: list[dict]) -> list[dict]:
    done = [s for s in steps if s["status"] == "done"]
    return [{"earlier": b["index"], "later": a["index"]} for a in done for b in done
            if b["index"] > a["index"] and b["hi"] is not None and a["lo"] is not None and b["hi"] < a["lo"]]


def stub_results(path: Path, steps: list[str]) -> tuple[dict, dict]:
    """Sample fast and full results in the engine's shapes, made without a model call.

    The ladder answers are synthetic curves pushed through the real fit and the real fusion.
    With three or more steps two of them are swapped, with four or more the last is not found,
    with five or more the fourth is unsure, and one sample witness disagrees on the first step,
    so the page can show every state. The unsure step is set by hand in the full sample, because
    what the fusion makes of a half-sure step depends on the video length.
    """
    try:
        duration = media.duration_s(path)
    except ValueError:
        duration = 25.0     # browser recordings often carry no duration in the header
    grid = ladder.time_grid(duration, max_points=20)
    n = len(steps)
    unsure_top = (engine.DONE_AT + engine.NOT_FOUND_AT) / 2
    starts = [duration * (i + 1) / (n + 1) for i in range(n)]
    if n >= 3:
        starts[1], starts[2] = starts[2], starts[1]
    missing = n - 1 if n >= 4 else -1
    shaky = 3 if n >= 5 else -1
    fits, fast_steps = [], []
    for i, text in enumerate(steps):
        top = 0.08 if i == missing else unsure_top if i == shaky else 0.93
        raw = [min(1.0, max(0.0, top / (1 + math.exp((starts[i] - t) / 0.7)) + 0.04 * math.sin(5 * t + i)))
               for t in grid]
        f = ladder.fit(grid, raw, p_any=top)
        fits.append(f)
        pd = engine.p_done(f)
        status = ("done" if pd >= engine.DONE_AT and f.t50 is not None
                  else "not_found" if pd < engine.NOT_FOUND_AT else "unsure")
        show = status != "not_found" and f.t50 is not None
        fast_steps.append({"index": i, "text": text, "status": status,
                           "time": round(f.t50, 2) if show else None,
                           "lo": round(f.lo, 2) if show else None, "hi": round(f.hi, 2) if show else None,
                           "p_done": round(pd, 3),
                           "raw": [round(x, 3) for x in f.raw], "fitted": [round(x, 3) for x in f.fitted]})
    video = {"path": str(path), "sha256": media.sha256(path), "duration": round(duration, 2)}
    call = {"seconds": 0.0, "input_tokens": 0, "n_questions": n * (len(grid) + 1)}
    fast = {"video": video, "grid": grid, "steps": fast_steps, "order_issues": _order_issues(fast_steps),
            "call": call, "stage": "fast", "seconds_total": STUB_FAST_S}

    def says(name: str, i: int, shift: float) -> dict:
        happened = i != missing and not (i == shaky and name != "hydrogen")
        if name == "neon" and i == 0 and n >= 2:
            happened = False                # one witness disagrees with the verdict
        start = round(min(duration, max(0.0, starts[i] + shift)), 1) if happened else None
        return {"index": i, "happened": happened, "start": start}

    reports = {name: [says(name, i, shift) for i in range(n)]
               for name, shift in (("hydrogen", -0.4), ("hydrogen4", -0.2), ("neon", 0.5))}
    fused_steps = engine.fuse_steps(steps, grid, fits, duration, reports)
    if shaky >= 0 and fused_steps[shaky]["status"] != "unsure":
        fused_steps[shaky].update(status="unsure", p_done=0.35)
    fused = {"video": video, "grid": grid, "steps": fused_steps, "order_issues": _order_issues(fused_steps),
             "call": call, "stage": "fused",
             "witness_calls": {name: {"ok": True, "seconds": 0.0, "model": "sample"} for name in reports},
             "seconds_total": STUB_FUSED_S}
    return fast, fused


def stub_audit(fused: dict) -> dict:
    """A sample audit in the auditor's shape, made without a model call.

    One step read as done is disputed, one is confirmed, the rest are "cannot tell".
    """
    steps = fused["steps"]
    done = [s["index"] for s in steps if s["status"] == "done"]
    dispute = done[-1] if done else steps[0]["index"]
    confirm = done[0] if len(done) > 1 else None
    rows = []
    for s in steps:
        i = s["index"]
        if i == dispute:
            row = {"at_start": "yes: sample note, the result is already in place in the START frame",
                   "at_end": "yes: sample note, the result is in place in the END frame", "audit": "dispute",
                   "reason": "Sample reason: the lasting result was already in place at START, "
                             "so the video does not show this step being done."}
        elif i == confirm:
            row = {"at_start": "no: sample note, the result is not in place in the START frame",
                   "at_end": "yes: sample note, the result is in place in the END frame", "audit": "confirm",
                   "reason": "Sample reason: absent at START and in place at END."}
        else:
            row = {"at_start": "cannot tell: sample note, the START frame does not show it",
                   "at_end": "cannot tell: sample note, the END frame does not show it", "audit": "cannot tell",
                   "reason": "Sample reason: the frames do not show this step's lasting result."}
        rows.append(dict(row, index=i, text=s["text"]))
    count = {v: sum(1 for r in rows if r["audit"] == v) for v in ("confirm", "dispute", "cannot tell")}
    sheet = "\n".join([
        "SAMPLE SIGN-OFF SHEET, not a real audit",
        f"step {dispute + 1}: disputed" + (f"; step {confirm + 1}: confirmed" if confirm is not None else ""),
        f"confirm {count['confirm']}, dispute {count['dispute']}, cannot tell {count['cannot tell']}"])
    return {"ok": True, "seconds": STUB_AUDIT_S, "model": "sample", "steps": rows, "sheet": sheet,
            "agent_ran_code": True, "error": None, "order_pairs": []}


def audit_view(res: dict, stub: bool) -> dict:
    """The audit as the page gets it: named fields only, so no frame data and no local path goes out."""
    out = {k: res[k] for k in AUDIT_KEYS if k in res}
    out["steps"] = [{k: s[k] for k in AUDIT_STEP_KEYS if k in s}
                    for s in res.get("steps") or [] if isinstance(s, dict)]
    out = json.loads(json.dumps(scrub(out), default=str))   # plain JSON types only
    out["stub"] = stub
    return out


# ---------- runs ----------

def _active_count() -> int:
    return sum(1 for r in _runs.values() if r["active"])


def _audit_work(run: dict, path: Path, fused: dict, stub: bool) -> None:
    """Runs on its own thread after the full read is stored. Every failure becomes a note on the run."""
    state, view, note = "failed", None, "the audit did not finish"
    try:
        if stub:
            time.sleep(STUB_AUDIT_S)
            res = stub_audit(fused)
        else:
            from clock import auditor   # imported here, so a broken auditor cannot stop the server or a run
            res = auditor.audit(path, fused, timeout_s=AUDIT_TIMEOUT_S, retries=0)
        if isinstance(res, dict) and res.get("ok"):
            view = audit_view(res, stub)
            state, note = "done", None
        else:
            err = res.get("error") if isinstance(res, dict) else None
            note = scrub(str(err or "the audit gave no answer"))[:300]
    except BaseException as e:  # noqa: BLE001 - the audit is optional, so nothing it does may surface
        note = "the audit failed: " + type(e).__name__
    finally:
        _audit_lock.release()
        with _runs_lock:
            if run["audit_state"] == "pending":     # after the audit deadline the run stays failed
                run["audit_state"], run["audit"], run["audit_note"] = state, view, note
                run["audit_t1"] = time.time()


def _start_audit(run: dict, path: Path, fused: dict, stub: bool) -> None:
    """Call with _runs_lock held, right after the full read is stored. Never raises, never waits."""
    try:
        if not audit_on():
            run["audit_note"] = "the audit is switched off"
            return
        if not _audit_lock.acquire(blocking=False):
            run["audit_note"] = "another audit was running"
            return
        run["audit_state"], run["audit_t0"] = "pending", time.time()
        try:
            threading.Thread(target=_audit_work, args=(run, path, fused, stub), daemon=True).start()
        except BaseException:  # noqa: BLE001
            _audit_lock.release()
            run["audit_state"], run["audit_note"] = "failed", "the audit could not start"
    except BaseException:  # noqa: BLE001
        pass


def _work(run_id: str, path: Path, steps: list[str], stub: bool) -> None:
    run = _runs[run_id]

    def on_fast(fast: dict) -> None:
        with _runs_lock:
            run["fast"] = public_view(fast, run_id, path.name, stub)
            if run["stage"] == "pending":
                run["stage"] = "fast"

    try:
        if stub:
            fast, fused = stub_results(path, steps)
            time.sleep(max(0.0, run["t0"] + STUB_FAST_S - time.time()))
            on_fast(fast)
            time.sleep(max(0.0, run["t0"] + STUB_FUSED_S - time.time()))
        else:
            fused = engine.run_fused(path, steps, on_fast=on_fast, tag=f"app:{run_id}")
        with _runs_lock:
            if run["stage"] != "error":     # after the deadline the run stays an error
                run["fused"] = public_view(fused, run_id, path.name, stub)
                run["stage"] = "fused"
                run["t1"] = time.time()
                _start_audit(run, path, fused, stub)
    except Exception as e:  # noqa: BLE001 - every failure becomes one plain sentence
        with _runs_lock:
            if run["stage"] != "error":
                run["stage"] = "error"
                run["error"] = plain_error(e)
                run["t1"] = time.time()
    finally:
        with _runs_lock:
            run["active"] = False


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "stub": stub_on()}


@app.post("/api/run")
async def api_run(video: UploadFile = File(...), steps: str = Form(...)) -> dict:
    step_list = parse_steps(steps)
    busy = HTTPException(429, "Two runs are already going. Wait a few seconds, then run again.")
    with _runs_lock:
        if _active_count() >= MAX_ACTIVE_RUNS:
            raise busy
    run_id = uuid.uuid4().hex[:12]
    path = await save_upload(video, run_id)
    stub = stub_on()
    with _runs_lock:
        if _active_count() >= MAX_ACTIVE_RUNS:      # another upload finished first
            path.unlink(missing_ok=True)
            raise busy
        _runs[run_id] = {"stage": "pending", "fast": None, "fused": None, "error": None,
                         "t0": time.time(), "t1": None, "active": True, "stub": stub,
                         "audit_state": "off", "audit": None, "audit_note": None,
                         "audit_t0": None, "audit_t1": None}
        for old in [k for k, r in _runs.items() if not r["active"]][:max(0, len(_runs) - MAX_KEPT_RUNS)]:
            del _runs[old]
    threading.Thread(target=_work, args=(run_id, path, step_list, stub), daemon=True).start()
    return {"run": run_id, "status": "started", "stub": stub}


@app.get("/api/result")
def api_result(run: str) -> dict:
    with _runs_lock:
        r = _runs.get(run)
        if r is None:
            raise HTTPException(404, "No run with that id. The server may have restarted. Run again.")
        now = time.time()
        if r["stage"] in ("pending", "fast") and now - r["t0"] > RUN_DEADLINE_S:
            r["stage"] = "error"
            r["t1"] = r["t0"] + RUN_DEADLINE_S
            r["error"] = (f"No full read after {RUN_DEADLINE_S} seconds. "
                          + ("The first read is shown." if r["fast"] else "Try a shorter video or fewer steps."))
        if r["audit_state"] == "pending" and now - r["audit_t0"] > AUDIT_DEADLINE_S:
            r["audit_state"] = "failed"
            r["audit_t1"] = r["audit_t0"] + AUDIT_DEADLINE_S
            r["audit_note"] = f"no audit after {AUDIT_DEADLINE_S} seconds"
        audit_s = round((r["audit_t1"] or now) - r["audit_t0"], 1) if r["audit_t0"] else None
        return {"run": run, "stage": r["stage"], "fast": r["fast"], "fused": r["fused"], "error": r["error"],
                "elapsed": round((r["t1"] or now) - r["t0"], 2), "stub": r["stub"],
                "audit_state": r["audit_state"], "audit": r["audit"], "audit_note": r["audit_note"],
                "audit_elapsed": audit_s}


@app.get("/api/latest")
def api_latest() -> dict:
    with _runs_lock:
        return {"run": next(reversed(_runs), None), "stub": stub_on()}


def public_url(request: Request) -> str:
    url = (os.environ.get("PUBLIC_URL") or "").strip()
    if not url and TUNNEL_LOG.exists():
        found = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", TUNNEL_LOG.read_text(errors="replace"))
        url = found.group(0) if found else ""
    return (url or str(request.base_url)).rstrip("/") + "/"


@app.get("/api/qr")
def api_qr(request: Request) -> Response:
    import segno

    url = public_url(request)
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="svg", scale=10, border=3, dark="#14181b", light="#ffffff",
                                    xmldecl=False, title="QR code for " + url)
    # the address also travels in a header so the page can print it under the code
    return Response(buf.getvalue(), media_type="image/svg+xml",
                    headers={"Cache-Control": "no-store", "X-Public-Url": url})


@app.get("/api/clip")
def api_clip(run: str, t: float) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{12}", run) or not math.isfinite(t) or not 0 <= t <= 36000:
        raise HTTPException(400, "Bad clip request.")
    src = next((p for p in sorted(UPLOADS.glob(run + ".*")) if p.suffix.lower() in VIDEO_EXTS), None)
    if src is None:
        raise HTTPException(404, "No video is stored for that run.")
    start = max(0.0, round(t - CLIP_BEFORE_S, 1))
    dst = CLIPS / f"{run}_{int(round(start * 10)):06d}.mp4"
    with _clip_lock:
        if not dst.exists():
            tmp = dst.with_name(f"{dst.stem}.part.mp4")
            try:
                media.cut(src, tmp, start, start + CLIP_LENGTH_S)
            except RuntimeError:
                tmp.unlink(missing_ok=True)
                raise HTTPException(500, "The clip could not be cut from the video.")
            if tmp.stat().st_size < EMPTY_CLIP_BYTES:
                tmp.unlink(missing_ok=True)
                raise HTTPException(404, "That time is past the end of the video.")
            tmp.replace(dst)
    return FileResponse(dst, media_type="video/mp4")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
