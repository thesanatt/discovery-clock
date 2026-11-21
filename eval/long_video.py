"""Push the length: run the system on long real recordings and see where it breaks.

CaptainCook4D recordings (Apache 2.0 per the project site): people cooking a recipe with a
head-mounted camera, 8 to 22 minutes each, with human labels for every recipe step, including
steps that were skipped. The benchmark clips in the README are 10 to 35 seconds; these are
14 to 40 times longer. This is a stress test on a handful of recordings, not a score.

  python -m eval.long_video 1_10 2_17
Writes results/long_video.json.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clock import decide, engine, fusion, ladder, media, witness
from eval import cc4d

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "long_video.json"
TIMEOUT_S = 240.0
# The same three witnesses as the app, plus a low-detail fallback for videos too long for 4 fps.
SETTINGS = {
    "hydrogen4": ("hydrogen", "medium", 4.0),
    "hydrogen2": ("hydrogen", "medium", 2.0),
    "neon4": ("neon", "medium", 4.0),
    "hydrogen_1fps_low": ("hydrogen", "low", None),
}


def one(rec_id: str) -> dict:
    rec = cc4d.load(rec_id)
    video = cc4d.fetch(rec_id)
    steps = [s["text"] for s in rec["steps"]]
    truth_done = [not s["skipped"] for s in rec["steps"]]
    truth_start = [s["start"] for s in rec["steps"]]
    duration = media.duration_s(video)
    out: dict = {"id": rec_id, "recipe": rec.get("recipe"), "minutes": round(duration / 60, 1), "steps": len(steps),
                 "skipped_steps": [i for i, d in enumerate(truth_done) if not d], "calls": {}}

    grid = ladder.time_grid(duration, max_points=40)
    t0 = time.time()
    fits = None
    try:
        answers, info = decide.decide(decide.client(int(TIMEOUT_S * 1000)), [media.video_block(video, "low")],
                                      ladder.build_questions(steps, grid), tag=f"long:{rec_id}")
        fits = ladder.fit_all(steps, grid, answers)
        out["calls"]["ladder"] = {"ok": True, "seconds": round(info["seconds"], 1), "input_tokens": info["input_tokens"],
                                  "questions": info["n_questions"], "grid_step_s": grid[1] - grid[0]}
    except Exception as e:  # noqa: BLE001 - a failure here is a result
        out["calls"]["ladder"] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {str(e)[:240]}"}

    def ask(name: str):
        model, resolution, fps = SETTINGS[name]
        info: dict = {}
        t = time.time()
        try:
            rep = witness.timeline(video, steps, model=model, thinking="low", resolution=resolution, fps=fps,
                                   timeout_s=TIMEOUT_S, info=info, tag=f"long_{name}")
            ok = model in str(info.get("model") or "")
            return name, (rep if ok else None), {"ok": ok, "seconds": round(time.time() - t, 1), "model": info.get("model"),
                                                 "input_tokens": info.get("input_tokens")}
        except Exception as e:  # noqa: BLE001
            return name, None, {"ok": False, "seconds": round(time.time() - t, 1), "error": f"{type(e).__name__}: {str(e)[:240]}"}

    reports = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, rep, meta in pool.map(ask, list(SETTINGS)):
            out["calls"][name] = meta
            if rep is not None:
                reports[name] = rep

    def score(said_done: list[bool], starts: list[float | None]) -> dict:
        skipped = [i for i, d in enumerate(truth_done) if not d]
        done = [i for i, d in enumerate(truth_done) if d]
        errs = [abs(starts[i] - truth_start[i]) for i in done if said_done[i] and starts[i] is not None and truth_start[i] is not None]
        return {"skipped_caught": sum(not said_done[i] for i in skipped), "skipped_total": len(skipped),
                "done_read_done": sum(said_done[i] for i in done), "done_total": len(done),
                "within_10s": sum(e <= 10 for e in errs), "within_30s": sum(e <= 30 for e in errs),
                "median_start_error_s": round(sorted(errs)[len(errs) // 2], 1) if errs else None}

    out["scores"] = {}
    for name, rep in reports.items():
        out["scores"][name] = score([r["happened"] for r in rep], [r["start"] for r in rep])
    if fits is not None:
        out["scores"]["ladder"] = score([engine.p_done(f) >= engine.DONE_AT and f.t50 is not None for f in fits], [f.t50 for f in fits])
    app_set = {k: v for k, v in reports.items() if k in engine.WITNESSES}
    if fits is not None or app_set:
        fused = engine.fuse_steps(steps, grid if fits else None, fits, duration, app_set, dict(fusion.load_params()))
        out["scores"]["fused (witnesses that answered: " + ", ".join(sorted(app_set) + (["ladder"] if fits else [])) + ")"] = \
            score([s["p_done"] >= 0.5 for s in fused], [s["time"] for s in fused])
    return out


def main() -> None:
    ids = sys.argv[1:] or ["1_10"]
    results = [one(i) for i in ids]
    json.dump(results, open(OUT, "w", encoding="ascii"), indent=1)
    for r in results:
        print(f"\n{r['id']} {r['recipe']}: {r['minutes']} min, {r['steps']} steps, skipped {r['skipped_steps']}")
        for k, v in r["calls"].items():
            print("   call", k, json.dumps(v))
        for k, v in r["scores"].items():
            print("   score", k, json.dumps(v))
    print("written", OUT)


if __name__ == "__main__":
    main()
