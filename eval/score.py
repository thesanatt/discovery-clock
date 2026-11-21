"""Score evaluation runs and, on the tuning split only, fit the calibration the engine uses.

  python -m eval.score tune            fit thresholds and start bias on the tuning split, write clock/calibration.json
  python -m eval.score report          print the table for every run file found under runs/eval
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "eval"
CAL = ROOT / "clock" / "calibration.json"


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="ascii") if l.strip()]


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def rows_of(recs: list[dict]) -> list[dict]:
    return [dict(r, clip=rec["id"]) for rec in recs for r in rec.get("rows", [])]


def metrics(rows: list[dict], score_key: str = "p_done", done_at: float | None = None) -> dict:
    """Skip detection and timing. A "skip" is a step on the checklist that never happens in the clip."""
    present = [r for r in rows if r["present"]]
    phantom = [r for r in rows if not r["present"]]

    def said_done(r):
        if done_at is None:
            return r["status"] == "done"
        return (r.get(score_key) or 0.0) >= done_at

    tp = sum(1 for r in phantom if not said_done(r))       # phantom correctly flagged as not done
    fn = sum(1 for r in phantom if said_done(r))
    fp = sum(1 for r in present if not said_done(r))       # real step wrongly flagged as not done
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    timed = [r for r in present if said_done(r) and r.get("time") is not None]
    errs = [abs(r["time"] - r["true_start"]) for r in timed]
    covered = [r for r in timed if r.get("lo") is not None and r.get("hi") is not None]
    inside = sum(1 for r in covered if r["lo"] - 0.5 <= r["true_start"] <= r["hi"] + 0.5)
    return {
        "steps_present": len(present), "steps_phantom": len(phantom),
        "skip_precision": round(prec, 3), "skip_recall": round(rec, 3), "skip_f1": round(f1, 3),
        "auroc": None if auc([r.get(score_key) or 0 for r in present], [r.get(score_key) or 0 for r in phantom]) is None
        else round(auc([r.get(score_key) or 0 for r in present], [r.get(score_key) or 0 for r in phantom]), 3),
        "timed": len(timed), "within_2s": round(sum(e <= 2 for e in errs) / len(present), 3) if present else None,
        "within_5s": round(sum(e <= 5 for e in errs) / len(present), 3) if present else None,
        "median_abs_err_s": round(statistics.median(errs), 2) if errs else None,
        "interval_coverage": round(inside / len(covered), 3) if covered else None,
    }


def tune() -> None:
    files = sorted(glob.glob(str(OUT / f"tune_ladder{sys.argv[2] if len(sys.argv) > 2 else ''}_run*.jsonl")))
    if not files:
        sys.exit("no tuning runs found; run: python -m eval.run_eval --split tune --system ladder")
    rows = [r for f in files for r in rows_of(load(f))]
    best = None
    for k in range(30, 91):
        thr = k / 100
        m = metrics(rows, "p_done", thr)
        if best is None or m["skip_f1"] > best[1]["skip_f1"]:
            best = (thr, m)
    # start bias: the engine already subtracted its current bias, so add it back before refitting
    cur = json.load(open(CAL))["start_bias_s"] if CAL.exists() else 1.0
    signed = [r["time"] + cur - r["true_start"] for r in rows if r["present"] and r.get("time") is not None]
    bias = round(statistics.median(signed), 2) if signed else cur
    cal = {"done_at": best[0], "not_found_at": round(best[0] - 0.12, 2), "start_bias_s": bias,
           "fitted_on": [Path(f).name for f in files], "n_steps": len(rows), "tune_metrics": best[1]}
    json.dump(cal, open(CAL, "w", encoding="ascii"), indent=1)
    print("calibration written to", CAL)
    print(json.dumps(cal, indent=1))


def report() -> None:
    for f in sorted(glob.glob(str(OUT / "*.jsonl"))):
        recs = load(f)
        errors = sum(1 for r in recs if "error" in r)
        rows = rows_of(recs)
        if not rows:
            print(Path(f).name, "no rows", f"({errors} errors)")
            continue
        m = metrics(rows)
        secs = [r["call"]["seconds"] for r in recs if "call" in r and "seconds" in r["call"]]
        print(f"{Path(f).name}: clips {len(recs)} ({errors} errors), median call {statistics.median(secs) if secs else None}s")
        print("   ", json.dumps(m))
        if "ladder" in f:
            ab = metrics(rows, "p_any", 0.5)
            print("    yes/no ablation (whole-video question only, threshold 0.5):",
                  json.dumps({k: ab[k] for k in ("skip_precision", "skip_recall", "skip_f1", "auroc")}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"tune": tune, "report": report}[cmd]()
