"""Engine: a video and a list of steps in, a step timeline out.

Two entry points, used by the web app and by the evals:

run(video, steps)        the fast read. One Decisions call (steps x seconds), about one second.
run_fused(video, steps)  the full read. The Decisions ladder and three reasoning-model timelines
                         run in parallel; clock/fusion.py combines them with Bayes' rule.
                         Any witness that fails or is late is left out, so the fast read is
                         the built-in fallback.

Step entry (times in seconds from the start of the video):
  {"index": 0, "text": "pour water into the cup",
   "status": "done" | "not_found" | "unsure",
   "time": 8.4 | None, "lo": 7.6 | None, "hi": 9.3 | None,
   "p_done": 0.86,                         probability that the step happened at all
   "raw": [...], "fitted": [...],          ladder answers and their monotone fit, per grid time
   -- run_fused adds --
   "taus": [...], "density": [...],        posterior over the start time (peak scaled to 1)
   "witnesses": {"ladder": {"p": 0.81, "time": 8.9},
                 "hydrogen": {"happened": true, "start": 8.0}, ...}}

Result:
  {"video": {"path", "sha256", "duration"}, "grid": [...], "steps": [...],
   "order_issues": [{"earlier": 3, "later": 1}],        step 3 was seen before step 1
   "call": {"seconds", "input_tokens", "n_questions"},   the Decisions call
   "stage": "fast" | "fused",
   "witness_calls": {"hydrogen": {"ok": true, "seconds": 7.1, "model": "gemini-4-hydrogen"}, ...},
   "seconds_total": 12.3}
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from . import decide, fusion, ladder, media

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "runs" / "prep"

# Fast read: thresholds on the ladder score. eval/score.py fits them on the tuning split only
# and writes clock/calibration.json; the values below are used before that file exists.
DONE_AT = 0.5
_CAL = Path(__file__).resolve().parent / "calibration.json"
if _CAL.exists():
    DONE_AT = json.load(open(_CAL, encoding="ascii"))["done_at"]
# The fast read is a first look, shown while the witnesses are still watching. It only says
# "not found" when the ladder is far down; anything in between is "unsure" until the full read.
NOT_FOUND_AT = 0.25

# Full read: the reasoning-model witnesses. Every one sees medium resolution: at low resolution
# and 1 frame per second Hydrogen mixed up which object an action belonged to on real phone
# footage (results/field.md). Reliabilities are fitted on the tuning split (eval/fit_fusion.py).
#   name        model       thinking  resolution  frames per second
WITNESSES = {
    "hydrogen4": ("hydrogen", "low", "medium", 4.0),
    "hydrogen2": ("hydrogen", "low", "medium", 2.0),
    "neon4": ("neon", "low", "medium", 4.0),
}
WITNESS_TIMEOUT_S = 30.0     # per model call
FUSED_DEADLINE_S = 32.0      # the full read never waits longer than this for witnesses
FUSED_SOFT_DEADLINE_S = 15.0 # after this, the full read returns as soon as two witnesses have answered
FUSED_DONE_AT = 0.5
FUSED_NOT_DONE_AT = 0.2


def p_done(fit: ladder.Fit) -> float:
    """Ladder score that the step happened at all: the end of the ladder and the whole-video answer."""
    if fit.p_any is None:
        return fit.p_end
    return 0.5 * (fit.p_end + fit.p_any)


def _order_issues(steps: list[dict]) -> list[dict]:
    issues = []
    done = [s for s in steps if s["status"] == "done"]
    for a in done:
        for b in done:
            # b is listed later in the checklist but was seen clearly earlier than a
            if b["index"] > a["index"] and b["hi"] is not None and a["lo"] is not None and b["hi"] < a["lo"]:
                issues.append({"earlier": b["index"], "later": a["index"]})
    return issues


def _prepare(video_path: Path, timecode: bool = False) -> tuple[Path, float]:
    small = media.prep(video_path, WORK / f"{video_path.stem}_{'tc_' if timecode else ''}small.mp4", timecode=timecode)
    return small, media.duration_s(small)


def _ladder_pass(small: Path, duration: float, steps: list[str], client, resolution: str, max_points: int, tag: str):
    grid = ladder.time_grid(duration, max_points=max_points)
    # The long question form was the most reliable of three forms tried on the sample clips
    # (LEDGER rows 36 to 41). It costs about 116 input tokens per question.
    questions = ladder.build_questions(steps, grid)
    answers, info = decide.decide(client or decide.client(), [media.video_block(small, resolution)], questions, tag=tag)
    return grid, ladder.fit_all(steps, grid, answers), info


def _fast_steps(steps: list[str], fits: list[ladder.Fit]) -> list[dict]:
    out = []
    for i, (text, f) in enumerate(zip(steps, fits)):
        pd = p_done(f)
        if pd >= DONE_AT and f.t50 is not None:
            status = "done"
        elif pd < NOT_FOUND_AT:
            status = "not_found"
        else:
            status = "unsure"
        show = status != "not_found" and f.t50 is not None
        out.append({
            "index": i, "text": text, "status": status,
            "time": round(f.t50, 2) if show else None,
            "lo": round(f.lo, 2) if show else None,
            "hi": round(f.hi, 2) if show else None,
            "p_done": round(pd, 3), "p_end": round(f.p_end, 3),
            "p_any": None if f.p_any is None else round(f.p_any, 3),
            "rise": round(f.rise, 3), "violation": round(f.violation, 3),
            "raw": [round(x, 3) for x in f.raw], "fitted": [round(x, 3) for x in f.fitted],
        })
    return out


def run(video_path: str | Path, steps: list[str], client=None, resolution: str = "low",
        max_points: int = 20, timecode: bool = False, tag: str = "run") -> dict:
    """Fast read: one Decisions call."""
    t0 = time.time()
    video_path = Path(video_path)
    small, duration = _prepare(video_path, timecode)
    grid, fits, info = _ladder_pass(small, duration, steps, client, resolution, max_points, tag)
    out_steps = _fast_steps(steps, fits)
    return {
        "video": {"path": str(video_path), "sha256": media.sha256(video_path), "duration": round(duration, 2)},
        "grid": grid, "steps": out_steps, "order_issues": _order_issues(out_steps),
        "call": {"seconds": round(info["seconds"], 2), "input_tokens": info["input_tokens"], "n_questions": info["n_questions"]},
        "stage": "fast", "seconds_total": round(time.time() - t0, 2),
    }


def fuse_steps(steps: list[str], grid: list[int] | None, fits: list[ladder.Fit] | None, duration: float,
               reports: dict[str, list[dict]], params: dict | None = None) -> list[dict]:
    """Combine the ladder (if present) and every witness report into one entry per step."""
    params = params or fusion.load_params()
    out = []
    for i, text in enumerate(steps):
        f = fits[i] if fits else None
        w = {name: rep[i] for name, rep in reports.items() if rep and i < len(rep)}
        post = fusion.posterior(grid if f else None, f.raw if f else None, f.p_any if f else None, duration, w, params)
        pd = post["p_done"]
        status = "done" if pd >= FUSED_DONE_AT else ("not_found" if pd < FUSED_NOT_DONE_AT else "unsure")
        show = status != "not_found"
        seen = {name: {"happened": bool(o.get("happened")), "start": o.get("start")} for name, o in w.items()}
        if f:
            seen["ladder"] = {"p": round(p_done(f), 3), "time": None if f.t50 is None else round(f.t50, 2)}
        out.append({
            "index": i, "text": text, "status": status,
            "time": post["time"] if show else None, "lo": post["lo"] if show else None, "hi": post["hi"] if show else None,
            "p_done": round(pd, 3),
            "raw": [round(x, 3) for x in f.raw] if f else [], "fitted": [round(x, 3) for x in f.fitted] if f else [],
            "taus": post["taus"], "density": post["density"], "witnesses": seen,
        })
    return out


def run_fused(video_path: str | Path, steps: list[str], client=None, max_points: int = 20,
              tag: str = "fused", on_fast=None) -> dict:
    """Full read: ladder plus reasoning-model witnesses in parallel, then Bayes' rule.

    on_fast, if given, is called with the fast result as soon as the ladder returns.
    """
    from . import witness

    t0 = time.time()
    video_path = Path(video_path)
    small, duration = _prepare(video_path)
    pool = ThreadPoolExecutor(max_workers=1 + len(WITNESSES))
    ladder_job = pool.submit(_ladder_pass, small, duration, steps, client, "low", max_points, tag)

    def ask(name: str) -> tuple[list[dict], dict]:
        model, thinking, resolution, fps = WITNESSES[name]
        info: dict = {}
        rep = witness.timeline(small, steps, model=model, thinking=thinking, resolution=resolution, fps=fps,
                               timeout_s=WITNESS_TIMEOUT_S, info=info, tag=f"witness_{name}")
        return rep, info

    jobs = {name: pool.submit(ask, name) for name in WITNESSES}

    grid = fits = None
    call = {"seconds": None, "input_tokens": 0, "n_questions": 0}
    try:
        grid, fits, info = ladder_job.result(timeout=FUSED_DEADLINE_S)
        call = {"seconds": round(info["seconds"], 2), "input_tokens": info["input_tokens"], "n_questions": info["n_questions"]}
        if on_fast:
            fast_steps = _fast_steps(steps, fits)
            on_fast({"video": {"path": str(video_path), "duration": round(duration, 2)}, "grid": grid, "steps": fast_steps,
                     "order_issues": _order_issues(fast_steps), "call": call, "stage": "fast",
                     "seconds_total": round(time.time() - t0, 2)})
    except Exception as e:  # noqa: BLE001 - the witnesses can still answer without the ladder
        call["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    # Wait for every witness up to the soft deadline. After that, two witnesses are enough:
    # the fusion works on any subset, and one slow call must not hold a live demo.
    pending = set(jobs.values())
    while pending:
        elapsed = time.time() - t0
        if elapsed >= FUSED_DEADLINE_S:
            break
        answered = sum(1 for j in jobs.values() if j.done() and j.exception() is None)
        if elapsed >= FUSED_SOFT_DEADLINE_S and answered >= 2:
            break
        _, pending = wait(pending, timeout=0.5)
    reports: dict[str, list[dict]] = {}
    witness_calls: dict[str, dict] = {}
    for name, job in jobs.items():
        if not job.done():
            witness_calls[name] = {"ok": False, "error": "no answer before the deadline"}
            continue
        try:
            rep, info = job.result()
            answered = str(info.get("model") or "")
            witness_calls[name] = {"ok": True, "seconds": info.get("seconds"), "model": answered}
            # a fallback model answered: its reliability was not fitted under this name, so leave it out
            if WITNESSES[name][0] in answered:
                reports[name] = rep
            else:
                witness_calls[name]["left_out"] = "answered by a fallback model"
        except Exception as e:  # noqa: BLE001
            witness_calls[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    pool.shutdown(wait=False, cancel_futures=True)

    if fits is None and not reports:
        raise RuntimeError("No model answered: the Decisions call and every witness failed.")
    out_steps = fuse_steps(steps, grid, fits, duration, reports)
    return {
        "video": {"path": str(video_path), "sha256": media.sha256(video_path), "duration": round(duration, 2)},
        "grid": grid or [], "steps": out_steps, "order_issues": _order_issues(out_steps),
        "call": call, "stage": "fused", "witness_calls": witness_calls,
        "seconds_total": round(time.time() - t0, 2),
    }
