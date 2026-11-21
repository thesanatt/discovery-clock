"""Does the slow-motion second pass (clock/zoom.py) place steps better than the one-pass ladder?

  python -m eval.zoom_eval --split tune [--workers 4]
Runs the coarse ladder and the zoom pass on every clip of the split, writes
runs/eval/<split>_ladder2_run1.jsonl (same row shape as the ladder runs, plus "t_fine"), and
prints timing for real steps under both.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clock import decide, engine, zoom
from eval import pt_data, run_eval

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "eval"


def timing(rows: list[dict], key: str) -> dict:
    present = [r for r in rows if r["present"]]
    errs = [abs(r[key] - r["true_start"]) for r in present if r.get(key) is not None]
    n = len(present)
    return {"timed": len(errs), "of": n, "within_1s": round(sum(e <= 1 for e in errs) / n, 3),
            "within_2s": round(sum(e <= 2 for e in errs) / n, 3), "within_5s": round(sum(e <= 5 for e in errs) / n, 3),
            "median_abs_err_s": round(statistics.median(errs), 2) if errs else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    clips = pt_data.load_manifest(args.split)
    if args.limit:
        clips = clips[: args.limit]
    all_labels = sorted({seg["label"] for c in pt_data.load_manifest("tune") + pt_data.load_manifest("test") for seg in c["actions"]})
    path = OUT / f"{args.split}_ladder2_run1.jsonl"
    done = {json.loads(l)["id"] for l in open(path, encoding="ascii")} if path.exists() else set()
    client = decide.client()
    lock = threading.Lock()

    def work(clip: dict) -> None:
        if clip["id"] in done:
            return
        labels, truth = run_eval.checklist(clip, all_labels)
        steps = [run_eval.describe(l) for l in labels]
        t0 = time.time()
        try:
            small, dur = engine._prepare(Path(clip["path"]))
            grid, fits, info = engine._ladder_pass(small, dur, steps, client, "low", 20, f"eval2:{clip['id']}")
            t1 = time.time()
            ref = zoom.refine(small, dur, steps, grid, fits, client, tag=f"zoom:{clip['id']}")
            rows = []
            for label, f, r in zip(labels, fits, ref):
                pd = engine.p_done(f)
                rows.append({"label": label, "present": label in truth, "true_start": truth.get(label),
                             "status": "done" if (pd >= 0.5 and f.t50 is not None) else "not_found",
                             "time": f.t50, "t_fine": r["time"], "lo": f.lo, "hi": f.hi, "p_done": round(pd, 3),
                             "p_any": f.p_any, "p_end": round(f.p_end, 3), "violation": round(f.violation, 3),
                             "raw": [round(x, 3) for x in f.raw], "zoom_ok": r["ok"], "zoom_slow": r["slow"]})
            rec = {"id": clip["id"], "duration": dur, "rows": rows,
                   "call": {"seconds": round(info["seconds"], 2), "input_tokens": info["input_tokens"], "n_questions": info["n_questions"]},
                   "zoom_seconds": round(time.time() - t1, 2), "wall": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            rec = {"id": clip["id"], "error": f"{type(e).__name__}: {str(e)[:300]}", "rows": []}
        with lock:
            with open(path, "a", encoding="ascii") as f:
                f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, clips))
    recs = [json.loads(l) for l in open(path, encoding="ascii") if l.strip()]
    rows = [r for rec in recs for r in rec.get("rows", [])]
    zs = [rec["zoom_seconds"] for rec in recs if "zoom_seconds" in rec]
    print(f"{args.split}: {len(recs)} clips, {sum('error' in r for r in recs)} errors, zoom pass median {statistics.median(zs):.1f} s")
    print("  one pass :", json.dumps(timing(rows, "time")))
    print("  two pass :", json.dumps(timing(rows, "t_fine")))
    both = [r for r in rows if r["present"] and r.get("time") is not None and r.get("t_fine") is not None]
    better = sum(abs(r["t_fine"] - r["true_start"]) < abs(r["time"] - r["true_start"]) - 0.05 for r in both)
    worse = sum(abs(r["t_fine"] - r["true_start"]) > abs(r["time"] - r["true_start"]) + 0.05 for r in both)
    print(f"  per step: two pass closer on {better}, further on {worse}, same on {len(both) - better - worse}")


if __name__ == "__main__":
    main()
