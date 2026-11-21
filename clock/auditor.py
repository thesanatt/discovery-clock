"""Auditor: a Gemini Managed Agent checks the fused read against still frames.

The fused read (clock/engine.py: run_fused) says, per step, how likely it is that the step
happened and when it started. It can credit a step whose lasting result was already in place
when the video began. The auditor is a second, independent check that runs after the fused read.

It is the Antigravity managed agent with a Google-hosted Linux sandbox. Managed agents take text
and inline images only, so the auditor works from still frames cut out of the video:

  START    about 0.3 s into the video
  END      about 0.3 s before the end
  BEFORE   1 s before a step's reported start    (only steps that have a start time)
  AFTER    2.5 s after a step's reported start

The agent (1) writes down, per step, whether the step's lasting result is in place in each frame,
(2) writes and runs a Python file in its sandbox that applies fixed rules to those observations
and to the reported times, and (3) returns the JSON that file printed. The file also prints a
SHA-256 over a one-time code and the verdicts. A language model cannot work that hash out without
running code, so a matching hash is the proof that the sandbox run really happened.

The same rules are applied again on this machine (apply_rules) to the agent's own observations,
so a bug in the agent's file shows up as "rules_agree": false.

    python -m clock.auditor VIDEO.mp4 "first step" "second step" ...
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import decide, media

AGENT = "antigravity-preview-09-2026"
MAX_TOTAL_TOKENS = 200_000
MAX_IMAGES = 14
LONG_SIDE = 768
JPEG_Q = "4"            # ffmpeg mjpeg scale 2 (best) to 31; 4 is close to JPEG quality 85
START_OFFSET_S = 0.3
END_OFFSET_S = 0.3
BEFORE_S = 1.0
AFTER_S = 2.5
POLL_S = 3.0

_LOG_LOCK = threading.Lock()
_BUDGET_LOCK = threading.Lock()
CALLS_MADE = 0          # managed-agent interactions created by this process
CALL_BUDGET = None      # set to an int to stop this process from creating more than that many


def _log(row: dict) -> None:
    line = json.dumps(row) + "\n"
    with _LOG_LOCK:
        decide.LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(decide.LOG, "a", encoding="ascii") as f:
            f.write(line)


def _scrub(e: Exception) -> str:
    text = f"{type(e).__name__}: {e}"
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.environ.get(name)
        if key:
            text = text.replace(key, "[key]")
    return text.encode("ascii", "replace").decode("ascii")[:300]


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ---------------------------------------------------------------- frames

def _frame(video_path: Path, t: float) -> bytes:
    """One JPEG at time t, long side LONG_SIDE. Empty bytes if ffmpeg gave nothing."""
    vf = f"scale='if(gt(iw,ih),{LONG_SIDE},-2)':'if(gt(iw,ih),-2,{LONG_SIDE})'"
    try:
        res = subprocess.run(
            [media.FFMPEG, "-loglevel", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(video_path),
             "-frames:v", "1", "-vf", vf, "-q:v", JPEG_Q, "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True, timeout=40)
    except Exception:  # noqa: BLE001 - a missing frame is handled by the caller
        return b""
    return res.stdout if res.returncode == 0 else b""


def _frame_near(video_path: Path, t: float, duration: float) -> tuple[float, bytes]:
    """Frame at t, stepping back a little if the video stream ends before the container does."""
    t = min(max(0.0, t), max(0.0, duration - 0.05))
    for back in (0.0, 0.4, 1.0, 2.0):
        tt = max(0.0, t - back)
        data = _frame(video_path, tt)
        if data:
            return tt, data
    return t, b""


def frame_plan(result: dict) -> list[tuple[str, float]]:
    """Labels and times of the frames the auditor gets, at most MAX_IMAGES."""
    duration = float(result["video"]["duration"])
    plan = [("START", min(START_OFFSET_S, duration / 2)), ("END", max(duration - END_OFFSET_S, duration / 2))]
    timed = [s for s in result["steps"] if s.get("time") is not None]
    # steps read as done come first if there is not room for every step
    timed.sort(key=lambda s: (s.get("status") != "done", s["index"]))
    timed = sorted(timed[:(MAX_IMAGES - 2) // 2], key=lambda s: s["index"])
    for s in timed:
        t = float(s["time"])
        plan.append((f"STEP {s['index']} BEFORE", max(0.0, t - BEFORE_S)))
        plan.append((f"STEP {s['index']} AFTER", min(duration - 0.05, t + AFTER_S)))
    return plan


def frames(video_path: str | Path, result: dict) -> list[tuple[str, bytes]]:
    """Still frames for the auditor: [(label, jpeg_bytes), ...]. The label carries the time."""
    video_path = Path(video_path)
    duration = float(result["video"]["duration"])
    out = []
    for label, t in frame_plan(result):
        tt, data = _frame_near(video_path, t, duration)
        if data:
            out.append((f"{label} (t = {tt:.1f} s)", data))
    return out


# ---------------------------------------------------------------- rules (local copy)

VALUES = ("yes", "no", "cannot tell")


def _v(x) -> str:
    x = str(x or "").strip().lower()
    return x if x in VALUES else "cannot tell"


def apply_rules(observations: list[dict], reported: list[dict]) -> tuple[dict[int, str], list[list[int]]]:
    """The audit rules, worded in PROMPT below, applied on this machine.

    observations: [{"index", "lasting", "start", "end", "before", "reversed_by"}]
    reported:     [{"index", "status", "time"}]
    Returns ({index: "confirm" | "dispute" | "cannot tell"}, [[i, j], ...] pairs out of order).
    """
    rep = {int(r["index"]): r for r in reported}
    audits: dict[int, str] = {}
    for o in observations:
        i = int(o["index"])
        status = rep.get(i, {}).get("status")
        start, end, before = _v(o.get("start")), _v(o.get("end")), _v(o.get("before"))
        lasting = str(o.get("lasting") or "none").strip().lower()
        rev = []
        for j in o.get("reversed_by") or []:
            try:
                rev.append(int(j))
            except (TypeError, ValueError):
                pass
        reversed_later = any(j > i and rep.get(j, {}).get("status") == "done" for j in rev)
        if lasting == "none" or (start == "cannot tell" and end == "cannot tell"):
            audits[i] = "cannot tell"
        elif status in ("done", "unsure"):
            if start == "yes" and before != "no":
                audits[i] = "dispute"
            elif end == "no" and not reversed_later:
                audits[i] = "dispute"
            elif start == "no" and (end == "yes" or (end == "no" and reversed_later)):
                audits[i] = "confirm"
            else:
                audits[i] = "cannot tell"
        else:
            if start == "no" and end == "yes":
                audits[i] = "dispute"
            elif start == "yes" or (end == "no" and not reversed_later):
                audits[i] = "confirm"
            else:
                audits[i] = "cannot tell"
    done = [r for r in reported if r.get("status") == "done" and r.get("time") is not None]
    pairs = [[int(a["index"]), int(b["index"])] for a in done for b in done
             if int(a["index"]) < int(b["index"]) and float(a["time"]) > float(b["time"])]
    return audits, pairs


def proof_of(nonce: str, steps: list[dict]) -> str:
    body = ";".join(f"{int(s['index'])}:{s['audit']}" for s in sorted(steps, key=lambda s: int(s["index"])))
    return hashlib.sha256((nonce + "|" + body).encode("ascii", "replace")).hexdigest()


# ---------------------------------------------------------------- prompt

PROMPT = """You are the auditor for a filmed checklist job. Another system watched the video and
reported, for each step, a status ("done", "unsure" or "not_found"), a probability and a start
time. You cannot see the video. You get still frames from it. Each frame comes right after a
line of text with its label and its time.

CHECKLIST (index. step)
{listing}

REPORTED RESULT (JSON, times in seconds from the start of the video; the video is {duration:.1f} s long)
{reported}

FRAMES IN THIS MESSAGE
{frame_list}

Do the work in this order.

PART 1. OBSERVE. For every step on the checklist:
 - "lasting": the lasting result of the step, in a few words: the state of the scene that stays
   true after the step is finished (a container open or closed, an object at a place or gone
   from a place, a part attached or removed). If the step leaves nothing behind that a photo
   could show (a gesture, a sound, a short motion), write "none".
 - "start": is that lasting result already in place in the START frame? "yes", "no" or "cannot tell".
 - "end": is it in place in the END frame? "yes", "no" or "cannot tell".
 - "before" and "after": the same question for the step's own BEFORE and AFTER frames, or
   "cannot tell" if the step has no such frames.
 - "start_note" and "end_note": what you see, under 12 words each.
 - "reversed_by": indexes of OTHER steps on the checklist whose own lasting result undoes this one
   (closing undoes opening, removing undoes placing). Usually an empty list.
Look at each frame on its own. Do not copy the reported result into your observations, and do
not assume the START frame shows the opposite of the END frame. If the lasting result is an
object put inside or behind something where a photo cannot show it, judge by the place the
object came from: still visible there means "no"; visible at START and no longer visible
anywhere at END means "yes". Use "cannot tell" when the frame does not show what you need.

PART 2. WRITE AND RUN CODE. In the sandbox, write ONE Python file, /workspace/audit.py, that
holds your observations and the reported result as plain data and applies these rules. Then run
it once with python3. Use the standard library only. Do not install anything, do not browse, and
do not list directories or read other files first.

Words used by the rules: a step is "credited" if its reported status is "done" or "unsure".
"Reversed later" means: some index in the step's reversed_by list is larger than the step's own
index and that other step has reported status "done".

For each step, the first rule that fits decides:
 A. lasting is "none", or start and end are both "cannot tell": audit "cannot tell".
 B. credited, start is "yes" and before is not "no": audit "dispute". Reason: the lasting result
    was already in place at START, so the video does not show this step being done.
 C. credited, end is "no" and the step is not reversed later: audit "dispute". Reason: credited,
    but its lasting result is absent at END and no later step on the list reverses it.
 D. credited, start is "no", and either end is "yes" or (end is "no" and the step is reversed
    later): audit "confirm".
 E. credited, anything else: audit "cannot tell".
 F. status "not_found", start is "no" and end is "yes": audit "dispute". Reason: the lasting
    result appeared during the video.
 G. status "not_found", and either start is "yes" or (end is "no" and the step is not reversed
    later): audit "confirm".
 H. status "not_found", anything else: audit "cannot tell".
Order rule: for every pair of steps i < j that both have status "done" and a start time, if the
time of i is greater than the time of j, add [i, j] to "order_pairs".

The file must also compute
  proof = hashlib.sha256(("{nonce}" + "|" + ";".join(str(index) + ":" + audit for every step in index order)).encode()).hexdigest()
and build "sheet_lines", a short plain-text sign-off sheet as a list of strings: a title line,
one line per step (index, step text, reported status, audit, reason), one line for the order
pairs (or "order: no pairs out of order"), one line with the counts of confirm, dispute and
cannot tell.

The last thing the file prints must be ONE line of JSON, made with json.dumps, shaped like this:
{{"steps": [{{"index": 0, "lasting": "...", "start": "yes", "start_note": "...", "end": "no", "end_note": "...", "before": "cannot tell", "after": "cannot tell", "reversed_by": [], "audit": "dispute", "reason": "..."}}], "order_pairs": [], "sheet_lines": ["..."], "proof": "..."}}
with one entry per checklist step, in index order. Keep every string plain ASCII.

PART 3. ANSWER. Your final answer is exactly that one line of JSON as the file printed it, and
nothing else. Do not retype the proof from memory: copy it from the output of the run.
"""


def _reported(result: dict) -> list[dict]:
    return [{"index": s["index"], "status": s["status"], "p_done": s.get("p_done"), "time": s.get("time")}
            for s in result["steps"]]


def build_input(result: dict, shots: list[tuple[str, bytes]], nonce: str) -> list[dict]:
    listing = "\n".join(f"{s['index']}. {' '.join(str(s['text']).split())}" for s in result["steps"])
    text = PROMPT.format(listing=listing, duration=float(result["video"]["duration"]),
                         reported=json.dumps(_reported(result)),
                         frame_list="\n".join(label for label, _ in shots), nonce=nonce)
    parts = [{"type": "text", "text": text}]
    for label, data in shots:
        parts.append({"type": "text", "text": f"FRAME: {label}"})
        parts.append({"type": "image", "data": base64.b64encode(data).decode("ascii"), "mime_type": "image/jpeg"})
    return parts


# ---------------------------------------------------------------- the call

def _find_json(text: str) -> dict | None:
    """The last JSON object in text that has a "steps" list."""
    dec = json.JSONDecoder()
    found = None
    pos = text.find("{")
    while pos != -1:
        try:
            obj, end = dec.raw_decode(text, pos)
        except ValueError:
            pos = text.find("{", pos + 1)
            continue
        if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
            found = obj
        pos = text.find("{", end)
    return found


def _usage(inter) -> dict:
    usage = _get(inter, "usage")
    out = {}
    for name in ("total_tokens", "total_input_tokens", "total_output_tokens", "total_thought_tokens",
                 "total_tool_use_tokens", "total_cached_tokens"):
        val = _get(usage, name)
        if isinstance(val, (int, float)):
            out[name] = int(val)
    return out


def _step_types(inter) -> dict:
    counts: dict[str, int] = {}
    for st in _get(inter, "steps") or []:
        kind = str(_get(st, "type") or type(st).__name__)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _take_budget() -> bool:
    global CALLS_MADE
    with _BUDGET_LOCK:
        if CALL_BUDGET is not None and CALLS_MADE >= CALL_BUDGET:
            return False
        CALLS_MADE += 1
        return True


def _one_call(parts: list[dict], model: str, timeout_s: float) -> tuple[str, dict]:
    """One managed-agent interaction, run in the background and polled. Returns (text, info)."""
    cl = decide.client(timeout_ms=60_000)
    t0 = time.time()
    inter = cl.interactions.create(
        agent=AGENT, input=parts, environment="remote", background=True,
        agent_config={"type": "antigravity", "model": model, "max_total_tokens": MAX_TOTAL_TOKENS})
    iid = _get(inter, "id")
    misses = 0
    while str(_get(inter, "status")) == "in_progress":
        if time.time() - t0 > timeout_s:
            try:
                cl.interactions.cancel(id=iid)
            except Exception:  # noqa: BLE001 - the timeout is what gets reported
                pass
            raise TimeoutError(f"no answer from the agent in {timeout_s:.0f} s")
        time.sleep(POLL_S)
        try:
            inter = cl.interactions.get(id=iid)
            misses = 0
        except Exception:  # noqa: BLE001 - one lost poll is not a lost audit
            misses += 1
            if misses >= 3:
                raise
    status = str(_get(inter, "status"))
    info = {"status": status, "step_types": _step_types(inter), **_usage(inter)}
    text = _get(inter, "output_text") or ""
    if status != "completed" and not text:
        raise RuntimeError(f"agent ended with status {status}")
    return str(text), info


def audit(video_path: str | Path, result: dict, model: str = "gemini-4-hydrogen", timeout_s: float = 180,
          retries: int = 1, tag: str = "audit") -> dict:
    """Audit one fused result. Never raises: on failure the answer has ok False and an error."""
    t0 = time.time()
    out = {"ok": False, "seconds": 0.0, "model": model, "steps": [], "sheet": "", "agent_ran_code": False,
           "error": None, "order_pairs": [], "rules_agree": None, "attempts": 0, "frames": []}
    try:
        video_path = Path(video_path)
        shots = frames(video_path, result)
        out["frames"] = [label for label, _ in shots]
        if not any(label.startswith("START") for label, _ in shots) or not any(label.startswith("END") for label, _ in shots):
            raise RuntimeError("could not cut the START and END frames out of the video")
        n = len(result["steps"])
        texts = {int(s["index"]): s["text"] for s in result["steps"]}
        last_err = "no call made"
        for attempt in range(retries + 1):
            if not _take_budget():
                last_err = f"{last_err}; call budget used up"
                break
            out["attempts"] = attempt + 1
            nonce = secrets.token_hex(8)
            c0 = time.time()
            row = {"ts": c0, "tag": f"{tag}:{video_path.stem}", "agent": AGENT, "model": model, "attempt": attempt,
                   "n_images": len(shots)}
            try:
                text, info = _one_call(build_input(result, shots, nonce), model, timeout_s)
                row.update(info)
                obj = _find_json(text)
                if obj is None:
                    raise ValueError("no usable JSON in the agent's answer: " + text[:120].replace("\n", " "))
                rows = {int(r["index"]): r for r in obj["steps"] if isinstance(r, dict) and "index" in r}
                if set(rows) != set(range(n)):
                    raise ValueError(f"expected verdicts for {n} steps, got indexes {sorted(rows)}")
                steps = []
                for i in range(n):
                    r = rows[i]
                    verdict = str(r.get("audit", "")).strip().lower()
                    if verdict not in ("confirm", "dispute", "cannot tell"):
                        raise ValueError(f"step {i} has no usable audit verdict")
                    steps.append({"index": i, "text": texts[i],
                                  "at_start": f"{_v(r.get('start'))}: {str(r.get('start_note') or '')[:120]}",
                                  "at_end": f"{_v(r.get('end'))}: {str(r.get('end_note') or '')[:120]}",
                                  "audit": verdict, "reason": str(r.get("reason") or "")[:300],
                                  "lasting": str(r.get("lasting") or "")[:120],
                                  "start_in_place": _v(r.get("start")), "end_in_place": _v(r.get("end")),
                                  "before": _v(r.get("before")), "after": _v(r.get("after")),
                                  "reversed_by": r.get("reversed_by") or []})
                proof_ok = str(obj.get("proof") or "").strip().lower() == proof_of(nonce, steps)
                local, pairs = apply_rules(
                    [{"index": s["index"], "lasting": s["lasting"], "start": s["start_in_place"], "end": s["end_in_place"],
                      "before": s["before"], "reversed_by": s["reversed_by"]} for s in steps], _reported(result))
                for s in steps:
                    s["local_audit"] = local.get(s["index"])
                sheet = obj.get("sheet_lines") or obj.get("sheet") or []
                if isinstance(sheet, list):
                    sheet = "\n".join(str(x) for x in sheet)
                out.update({"ok": True, "steps": steps, "sheet": str(sheet).encode("ascii", "replace").decode("ascii"),
                            "agent_ran_code": bool(proof_ok), "order_pairs": obj.get("order_pairs") or [],
                            "local_order_pairs": pairs, "rules_agree": all(s["audit"] == s["local_audit"] for s in steps),
                            "step_types": info.get("step_types"), "tokens": info.get("total_tokens"), "error": None})
                row.update({"ok": True, "seconds": round(time.time() - c0, 2), "proof_ok": bool(proof_ok)})
                _log(row)
                break
            except Exception as e:  # noqa: BLE001 - log, then retry once
                last_err = _scrub(e)
                row.update({"ok": False, "seconds": round(time.time() - c0, 2), "error": last_err})
                _log(row)
                time.sleep(1.5)
        if not out["ok"]:
            out["error"] = last_err
    except Exception as e:  # noqa: BLE001 - audit() never raises
        out["error"] = _scrub(e)
    out["seconds"] = round(time.time() - t0, 2)
    return out


def main() -> None:
    from . import engine

    if len(sys.argv) < 3:
        print('usage: python -m clock.auditor VIDEO.mp4 "first step" "second step" ...')
        raise SystemExit(2)
    video, steps = sys.argv[1], sys.argv[2:]
    res = engine.run_fused(video, steps, tag="audit_cli")
    print("FUSED READ")
    for s in res["steps"]:
        print(f"  {s['index']}. {s['text']}: {s['status']} (P {s['p_done']:.2f}, start {s['time']})")
    a = audit(video, res)
    print("AUDIT")
    print(json.dumps({k: v for k, v in a.items() if k != "sheet"}, indent=1))
    print(a["sheet"])


if __name__ == "__main__":
    main()
