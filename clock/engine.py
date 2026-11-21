"""Engine: one Decisions call turns a video and a list of steps into a step timeline.

run() is the single entry point the web app and the evals both use.

Result shape (all times in seconds from the start of the video):
{
  "video": {"path": str, "sha256": str, "duration": float},
  "grid": [int, ...],
  "steps": [
    {"index": 0, "text": "pour water into the cup",
     "status": "done" | "not_found" | "unsure",
     "time": 8.4 | None, "lo": 7.6 | None, "hi": 9.3 | None,
     "p_done": 0.86,            # calibrated-style score that the step happened at all
     "violation": 0.12,         # how far the raw answers were from a monotone curve
     "raw": [...], "fitted": [...]},
    ...
  ],
  "order_issues": [{"earlier": 3, "later": 1, "p": 0.9}],   # step 3 was seen before step 1
  "call": {"seconds": float, "input_tokens": int, "n_questions": int}
}
"""
from __future__ import annotations

from pathlib import Path

from . import decide, ladder, media

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "runs" / "prep"

# Thresholds on p_done. eval/score.py fits them on the tuning split only and writes
# clock/calibration.json; the values below are the defaults used before that file exists.
DONE_AT = 0.5
NOT_FOUND_AT = 0.38
_CAL = Path(__file__).resolve().parent / "calibration.json"
if _CAL.exists():
    import json as _json
    _c = _json.load(open(_CAL, encoding="ascii"))
    DONE_AT, NOT_FOUND_AT = _c["done_at"], _c["not_found_at"]


def p_done(fit: ladder.Fit) -> float:
    """Evidence that the step happened at all: the end of the ladder and the whole-video answer agree."""
    if fit.p_any is None:
        return fit.p_end
    return 0.5 * (fit.p_end + fit.p_any)


def run(video_path: str | Path, steps: list[str], client=None, resolution: str = "low",
        max_points: int = 20, timecode: bool = False, tag: str = "run") -> dict:
    video_path = Path(video_path)
    small = media.prep(video_path, WORK / f"{video_path.stem}_{'tc_' if timecode else ''}small.mp4", timecode=timecode)
    duration = media.duration_s(small)
    grid = ladder.time_grid(duration, max_points=max_points)
    # The long question form was the most reliable of three forms tried on the sample clips
    # (LEDGER rows 36 to 41). It costs about 116 input tokens per question.
    questions = ladder.build_questions(steps, grid)
    blocks = [media.video_block(small, resolution)]
    client = client or decide.client()
    answers, info = decide.decide(client, blocks, questions, tag=tag)
    fits = ladder.fit_all(steps, grid, answers)

    out_steps = []
    for i, (text, f) in enumerate(zip(steps, fits)):
        pd = p_done(f)
        if pd >= DONE_AT and f.t50 is not None:
            status = "done"
        elif pd < NOT_FOUND_AT:
            status = "not_found"
        else:
            status = "unsure"
        show = status != "not_found" and f.t50 is not None
        out_steps.append({
            "index": i, "text": text, "status": status,
            "time": round(f.t50, 2) if show else None,
            "lo": round(f.lo, 2) if show else None,
            "hi": round(f.hi, 2) if show else None,
            "p_done": round(pd, 3), "p_end": round(f.p_end, 3),
            "p_any": None if f.p_any is None else round(f.p_any, 3),
            "rise": round(f.rise, 3), "violation": round(f.violation, 3),
            "raw": [round(x, 3) for x in f.raw], "fitted": [round(x, 3) for x in f.fitted],
        })

    order_issues = []
    done = [s for s in out_steps if s["status"] == "done"]
    for a in done:
        for b in done:
            # b is listed later in the checklist but was seen clearly earlier than a
            if b["index"] > a["index"] and b["hi"] is not None and a["lo"] is not None and b["hi"] < a["lo"]:
                order_issues.append({"earlier": b["index"], "later": a["index"]})

    return {
        "video": {"path": str(video_path), "sha256": media.sha256(video_path), "duration": round(duration, 2)},
        "grid": grid,
        "steps": out_steps,
        "order_issues": order_issues,
        "call": {"seconds": round(info["seconds"], 2), "input_tokens": info["input_tokens"], "n_questions": info["n_questions"]},
    }
