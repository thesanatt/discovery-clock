"""Web app for Discovery Clock: record or upload a video, list the steps, read a step timeline.

Start it from the project root:
    .venv/Scripts/python.exe -m uvicorn app.server:app --port 8765
Then open http://localhost:8765/

Set CLOCK_STUB=1 to get sample data without any model call. The page labels it as sample data.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import re
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clock import engine, ladder, media  # noqa: E402

UPLOADS = ROOT / "runs" / "uploads"
CLIPS = ROOT / "runs" / "clips"
STATIC = Path(__file__).resolve().parent / "static"

RUN_TIMEOUT_S = 90
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_STEPS = 20              # 20 steps x 21 questions stays inside one Decisions call
MAX_STEP_CHARS = 200
CLIP_BEFORE_S = 1.5         # the clip runs from 1.5 s before the estimate to 2.5 s after it
CLIP_LENGTH_S = 4.0
EMPTY_CLIP_BYTES = 500      # ffmpeg writes a 261 byte file when the start is past the end

app = FastAPI(title="Discovery Clock")
_run_lock = threading.Lock()    # one model call at a time
_clip_lock = threading.Lock()


def stub_on() -> bool:
    return os.environ.get("CLOCK_STUB") == "1"


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
    if ext not in (".webm", ".mp4"):
        raise HTTPException(400, "The video must be a .webm or .mp4 file.")
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
    if "timeout" in text.lower() or "timed out" in text.lower():
        return "The model did not answer in time. Run again."
    return "The run failed. " + text[:300]


def stub_result(path: Path, steps: list[str]) -> dict:
    """Sample timeline in the engine's result shape, made without a model call.

    The answers are synthetic curves pushed through the real fit. With three or more steps
    two of them are swapped, with four or more the last is not found, with five or more
    the fourth is unsure, so the page can show every state. The unsure curve sits halfway
    between the engine's two thresholds, so it stays unsure when the calibration file changes.
    """
    try:
        duration = media.duration_s(path)
    except ValueError:
        duration = 25.0     # browser recordings often carry no duration in the header
    points = inspect.signature(engine.run).parameters.get("max_points")
    grid = ladder.time_grid(duration, max_points=points.default if points is not None and isinstance(points.default, int) else 20)
    n = len(steps)
    unsure_top = (engine.DONE_AT + engine.NOT_FOUND_AT) / 2
    starts = [duration * (i + 1) / (n + 1) for i in range(n)]
    if n >= 3:
        starts[1], starts[2] = starts[2], starts[1]
    out = []
    for i, text in enumerate(steps):
        top = 0.08 if (n >= 4 and i == n - 1) else unsure_top if (n >= 5 and i == 3) else 0.93
        raw = [min(1.0, max(0.0, top / (1 + math.exp((starts[i] - t) / 0.7)) + 0.04 * math.sin(5 * t + i)))
               for t in grid]
        f = ladder.fit(grid, raw, p_any=top)
        pd = engine.p_done(f)
        found = pd >= engine.DONE_AT and f.t50 is not None
        status = "done" if found else ("not_found" if pd < engine.NOT_FOUND_AT else "unsure")
        t, lo, hi = [None if (x is None or status == "not_found") else max(0.0, x - engine.START_BIAS_S)
                     for x in (f.t50, f.lo, f.hi)]
        out.append({"index": i, "text": text, "status": status, "time": t, "lo": lo, "hi": hi,
                    "p_done": round(pd, 3), "p_end": round(f.p_end, 3), "p_any": round(top, 3),
                    "violation": round(f.violation, 3),
                    "raw": [round(x, 3) for x in f.raw], "fitted": [round(x, 3) for x in f.fitted]})
    done = [s for s in out if s["status"] == "done"]
    order = [{"earlier": b["index"], "later": a["index"]} for a in done for b in done
             if b["index"] > a["index"] and b["hi"] is not None and a["lo"] is not None and b["hi"] < a["lo"]]
    return {
        "video": {"path": str(path), "sha256": media.sha256(path), "duration": round(duration, 2)},
        "grid": grid, "steps": out, "order_issues": order,
        "call": {"seconds": 0.0, "input_tokens": 0, "n_questions": n * (len(grid) + 1)},
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "stub": stub_on()}


@app.post("/api/run")
async def api_run(video: UploadFile = File(...), steps: str = Form(...), clock: str = Form("")) -> dict:
    step_list = parse_steps(steps)
    run_id = uuid.uuid4().hex[:12]
    path = await save_upload(video, run_id)
    timecode = clock.strip().lower() in ("1", "true", "on", "yes")
    stub = stub_on()
    loop = asyncio.get_running_loop()

    if stub:
        work = lambda: stub_result(path, step_list)  # noqa: E731
    else:
        if not _run_lock.acquire(blocking=False):
            raise HTTPException(429, "Another run is still going. Wait for it to finish, then run again.")

        def work() -> dict:
            # The lock is released when the engine really returns, also after a timeout on the page side.
            try:
                return engine.run(path, step_list, timecode=timecode, tag=f"app:{run_id}")
            finally:
                _run_lock.release()

    # asyncio.wait, not wait_for: a TimeoutError raised inside the engine (a network read that
    # timed out early) must not be reported as the 90 second limit.
    job = loop.run_in_executor(None, work)
    job.add_done_callback(lambda f: f.cancelled() or f.exception())    # a late failure is already reported
    done, _ = await asyncio.wait({job}, timeout=RUN_TIMEOUT_S)
    if not done:
        raise HTTPException(504, f"No result after {RUN_TIMEOUT_S} seconds. Try a shorter video or fewer steps.")
    try:
        result = job.result()
    except Exception as e:  # noqa: BLE001 - every failure becomes one plain sentence
        raise HTTPException(502, plain_error(e))

    result["video"]["path"] = f"runs/uploads/{path.name}"     # no absolute local path leaves the server
    result.update({"run": run_id, "stub": stub, "bias_s": engine.START_BIAS_S,
                   "clip": {"before": CLIP_BEFORE_S, "length": CLIP_LENGTH_S}})
    return result


@app.get("/api/clip")
def api_clip(run: str, t: float) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{12}", run) or not math.isfinite(t) or not 0 <= t <= 36000:
        raise HTTPException(400, "Bad clip request.")
    src = next(iter(sorted(UPLOADS.glob(run + ".*"))), None)
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
