"""Fit the fusion model on the tuning split and check it by leave-one-clip-out cross validation.

  python -m eval.fit_fusion            fit on all tuning clips, print cross-validated scores,
                                       write clock/fusion_params.json
Inputs (all from the tuning split, never the test split):
  runs/eval/tune_ladder_run3.jsonl     Decisions ladder answers
  runs/eval/tune_<model>_run1.jsonl    reasoning-model timelines (hydrogen, neon, carbon)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

from clock import fusion, ladder
from eval import score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "eval"
WITNESSES = ["hydrogen", "neon", "carbon", "hydrogen4", "neonmed"]
ALL_WITNESSES = WITNESSES + ["zoom"]
MAX_POINTS = 20


def load_all(split: str = "tune", ladder_run: str = "run3") -> dict:
    """clip id -> {duration, steps: {label: {present, true_start, grid, raw, p_any, w: {model: obs}}}}"""
    data: dict = {}
    for rec in score.load(str(OUT / f"{split}_ladder_{ladder_run}.jsonl")):
        if "error" in rec:
            continue
        grid = ladder.time_grid(rec["duration"], MAX_POINTS)
        steps = {}
        for r in rec["rows"]:
            steps[r["label"]] = {"present": r["present"], "true_start": r["true_start"], "grid": grid,
                                 "raw": r["raw"], "p_any": r["p_any"], "w": {}}
        data[rec["id"]] = {"duration": rec["duration"], "steps": steps}
    for name in WITNESSES:
        path = OUT / f"{split}_{name}_run1.jsonl"
        if not path.exists():
            continue
        for rec in score.load(str(path)):
            if "error" in rec or rec["id"] not in data:
                continue
            answered_by = (rec.get("call") or {}).get("model", "")
            base = {"hydrogen4": "hydrogen", "neonmed": "neon"}.get(name, name)
            if base not in answered_by:      # a fallback model answered: not this witness
                continue
            for r in rec["rows"]:
                if r["label"] in data[rec["id"]]["steps"]:
                    data[rec["id"]]["steps"][r["label"]]["w"][name] = {"happened": r["status"] == "done", "start": r["time"]}
    zoom = OUT / f"{split}_zoom_run1.jsonl"
    if zoom.exists():
        for r in score.load(str(zoom)):
            if "error" in r or r["id"] not in data or r["label"] not in data[r["id"]]["steps"]:
                continue
            if r["verdict"] == "unclear":
                continue
            yes = r["verdict"] == "yes"
            start = (r["win_start"] + r["second"]) if (yes and r["second"] is not None) else None
            data[r["id"]]["steps"][r["label"]]["w"]["zoom"] = {"happened": yes, "start": start}
    return data


def fit(clips: list[dict]) -> dict:
    """Moment and maximum-likelihood estimates of every model parameter from labeled clips."""
    before, after, a0, a1 = [], [], [], []
    step_offsets = []
    for c in clips:
        for s in c["steps"].values():
            z = fusion.logit(s["raw"])
            t = np.asarray(s["grid"], dtype=float)
            za = float(fusion.logit([s["p_any"]])[0])
            if s["present"]:
                lo = z[t <= s["true_start"] - 0.5]
                hi = z[t >= s["true_start"] + 2.0]
                before += list(lo)
                after += list(hi)
                a1.append(za)
            else:
                before += list(z)
                a0.append(za)
    mu0, mu1 = float(np.mean(before)), float(np.mean(after))
    # shared offset: spread of per-step mean residuals; noise: spread inside a step
    within0, within1 = [], []
    for c in clips:
        for s in c["steps"].values():
            z = fusion.logit(s["raw"])
            t = np.asarray(s["grid"], dtype=float)
            if s["present"]:
                side = t >= s["true_start"] + 2.0
                keep = side | (t <= s["true_start"] - 0.5)
            else:
                side = np.zeros(len(t), dtype=bool)
                keep = np.ones(len(t), dtype=bool)
            r = (z - np.where(side, mu1, mu0))[keep]
            sd = side[keep]
            if len(r) < 3:
                continue
            m = float(np.mean(r))
            step_offsets.append(m)
            within0 += list(r[~sd] - m)
            within1 += list(r[sd] - m)
    p = dict(fusion.DEFAULTS)
    p.update(mu0=mu0, mu1=mu1, s0=max(0.15, float(np.std(within0))), s1=max(0.15, float(np.std(within1))),
             sb=max(0.1, float(np.std(step_offsets))),
             a0=float(np.mean(a0)), sa0=max(0.2, float(np.std(a0))), a1=float(np.mean(a1)), sa1=max(0.2, float(np.std(a1))))
    # witnesses: hit rate, false alarm rate, timing error
    p["witness"] = {}
    for name in ALL_WITNESSES:
        said_p = [s["w"][name]["happened"] for c in clips for s in c["steps"].values() if name in s["w"] and s["present"]]
        said_a = [s["w"][name]["happened"] for c in clips for s in c["steps"].values() if name in s["w"] and not s["present"]]
        errs = [s["w"][name]["start"] - s["true_start"] for c in clips for s in c["steps"].values()
                if name in s["w"] and s["present"] and s["w"][name]["happened"] and s["w"][name]["start"] is not None]
        if len(said_p) < 10 or len(said_a) < 5 or len(errs) < 10:
            continue
        shift = float(np.median(errs))
        dev = np.abs(np.asarray(errs) - shift)
        gross = float(np.mean(dev > 4.0))
        p["witness"][name] = {
            "hit": (sum(said_p) + 1) / (len(said_p) + 2), "false_alarm": (sum(said_a) + 1) / (len(said_a) + 2),
            "shift": round(shift, 3), "scale": max(0.25, float(np.mean(dev[dev <= 4.0]))),
            "gross": min(0.5, max(0.03, gross)), "n": len(said_p) + len(said_a),
        }
    # lag of the ladder: the value that makes the true start times most likely
    best = None
    for lag in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        q = dict(p, lag=lag)
        total = 0.0
        for c in clips:
            for s in c["steps"].values():
                if not s["present"]:
                    continue
                post = fusion.posterior(s["grid"], s["raw"], s["p_any"], c["duration"], None, q)
                taus = np.asarray(post["taus"])
                dens = np.asarray(post["density"]) + 1e-6
                dens = dens / dens.sum()
                total += float(np.log(dens[int(np.argmin(np.abs(taus - s["true_start"])))]))
        if best is None or total > best[1]:
            best = (lag, total)
    p["lag"] = best[0]
    # summary form of the ladder: a score and a half-rise time per step
    u1, u0, terr, has1, has0 = [], [], [], [], []
    for c in clips:
        for s in c["steps"].values():
            f = ladder.fit(s["grid"], s["raw"], s["p_any"])
            u = float(fusion.logit([0.5 * (f.p_end + s["p_any"])])[0])
            if s["present"]:
                u1.append(u)
                has1.append(f.t50 is not None)
                if f.t50 is not None:
                    terr.append(f.t50 - s["true_start"])
            else:
                u0.append(u)
                has0.append(f.t50 is not None)
    shift = float(np.median(terr))
    dev = np.abs(np.asarray(terr) - shift)
    p["ladder_summary"] = {
        "m1": float(np.mean(u1)), "s1": max(0.2, float(np.std(u1))), "m0": float(np.mean(u0)), "s0": max(0.2, float(np.std(u0))),
        "has_time1": (sum(has1) + 1) / (len(has1) + 2), "has_time0": (sum(has0) + 1) / (len(has0) + 2),
        "shift": round(shift, 3), "scale": max(0.25, float(np.mean(dev[dev <= 4.0]))), "gross": min(0.6, max(0.03, float(np.mean(dev > 4.0)))),
    }
    return p


def predict(clips: list[dict], params: dict, use_ladder: bool, names: list[str]) -> list[dict]:
    rows = []
    for c in clips:
        for label, s in c["steps"].items():
            w = {n: s["w"][n] for n in names if n in s["w"]}
            post = fusion.posterior(s["grid"] if use_ladder else None, s["raw"] if use_ladder else None,
                                    s["p_any"] if use_ladder else None, c["duration"], w, params)
            rows.append({"label": label, "present": s["present"], "true_start": s["true_start"],
                         "p_done": post["p_done"], "time": post["time"], "lo": post["lo"], "hi": post["hi"],
                         "status": "done" if post["p_done"] >= 0.5 else "not_found"})
    return rows


def summarize(rows: list[dict], thr: float = 0.5) -> dict:
    m = score.metrics(rows, "p_done", thr)
    return {k: m[k] for k in ("skip_f1", "skip_precision", "skip_recall", "auroc", "within_2s", "within_5s", "median_abs_err_s", "interval_coverage")}


CONFIGS = [
    ("ladder only", True, []),
    ("hydrogen only", False, ["hydrogen"]),
    ("hydrogen4 only", False, ["hydrogen4"]),
    ("neon only", False, ["neon"]),
    ("hydrogen + hydrogen4 + neon", False, ["hydrogen", "hydrogen4", "neon"]),
    ("ladder + hydrogen + neon", True, ["hydrogen", "neon"]),
    ("ladder + hydrogen + hydrogen4 + neon", True, ["hydrogen", "hydrogen4", "neon"]),
]
# The shipped system. Chosen on the tuning split (LEDGER rows 55, 59, 60).
SYSTEM = ("ladder + hydrogen + hydrogen4 + neon", True, ["hydrogen", "hydrogen4", "neon"])


def main() -> None:
    data = load_all()
    ids = sorted(data)
    print(f"{len(ids)} tuning clips, {sum(len(c['steps']) for c in data.values())} steps")
    # leave-one-clip-out: parameters never see the clip they score
    cv_rows = {name: [] for name, _, _ in CONFIGS}
    for held in ids:
        params = dict(fit([data[i] for i in ids if i != held]), ladder_mode="summary")
        for name, use_ladder, names in CONFIGS:
            cv_rows[name] += predict([data[held]], params, use_ladder, names)
    print("\nLeave-one-clip-out on the tuning split (threshold 0.5):")
    for name, _, _ in CONFIGS:
        print(f"  {name:36s}", json.dumps(summarize(cv_rows[name])))
    params = fit([data[i] for i in ids])
    params["ladder_mode"] = "summary"
    params["fitted_on"] = f"{len(ids)} tuning clips"
    params["cv"] = {name: summarize(cv_rows[name]) for name, _, _ in CONFIGS}
    json.dump({"params": params}, open(fusion.PARAMS_FILE, "w", encoding="ascii"), indent=1)
    show = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in params.items() if k not in ("cv", "witness")}
    print("\nparameters:", json.dumps(show))
    for n, w in params["witness"].items():
        print("  witness", n, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in w.items()}))
    print("written", fusion.PARAMS_FILE)


if __name__ == "__main__":
    main()
