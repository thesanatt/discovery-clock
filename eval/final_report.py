"""Final scores on the held-out test split, with every parameter frozen from the tuning split.

  python -m eval.final_report            prints the table and writes runs/eval/test_report.json
  python -m eval.final_report --split tune   same table on the tuning split (in-sample, for reference)

A "skip" is a checklist step that never happens in the clip (2 planted per clip, picked by a
hash of the clip id). Precision, recall and F1 are for flagging skips. Timing is for real steps.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from clock import fusion
from eval import fit_fusion, score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "eval"
FAST = ["hydrogen4", "hydrogen2", "neon4"]      # the witnesses of the shipped system (v2)
V1 = ["hydrogen", "hydrogen4", "neon"]
VAGUE = {"Moving object(s) around", "Holding something in a state"}


def single_rows(split: str, name: str, strict: bool = True) -> tuple[list[dict], list[float], int, int]:
    """Rows from one model's own timeline. strict keeps only clips that model itself answered."""
    base = {"hydrogen4": "hydrogen", "hydrogen2": "hydrogen", "neonmed": "neon", "neon4": "neon"}.get(name, name)
    tag = fit_fusion.WITNESS_TAG if name in fit_fusion.TAGGED else ""
    path = OUT / f"{split}_{name}{tag}_run1.jsonl"
    if not path.exists():
        return [], [], 0, 0
    recs = [r for r in score.load(str(path)) if "error" not in r]
    own = [r for r in recs if base in ((r.get("call") or {}).get("model") or "")]
    use = own if strict else recs
    rows = [dict(row, clip=r["id"]) for r in use for row in r["rows"]]
    secs = [r["call"]["seconds"] for r in use if "call" in r]
    return rows, secs, len(use), len(recs) - len(own)


def fused_rows(data: dict, params: dict, use_ladder: bool, names: list[str]) -> list[dict]:
    rows = []
    for vid, c in data.items():
        for r in fit_fusion.predict([c], params, use_ladder, names):
            rows.append(dict(r, clip=vid))
    return rows


def more(rows: list[dict], thr: float | None) -> dict:
    m = score.metrics(rows, "p_done", thr)
    present = [r for r in rows if r["present"]]

    def done(r):
        return (r["p_done"] >= thr) if thr is not None else r["status"] == "done"

    errs = [abs(r["time"] - r["true_start"]) for r in present if done(r) and r.get("time") is not None]
    m["within_1s"] = round(sum(e <= 1 for e in errs) / len(present), 3) if present else None
    return m


def bootstrap(rows_a: list[dict], rows_b: list[dict], thr_a, thr_b, key: str, n: int = 1000, seed: int = 7) -> tuple[float, float, float]:
    """Paired bootstrap over clips of metric(a) - metric(b). Returns (mean, low 2.5, high 97.5)."""
    clips = sorted({r["clip"] for r in rows_a} & {r["clip"] for r in rows_b})
    by_a = {c: [r for r in rows_a if r["clip"] == c] for c in clips}
    by_b = {c: [r for r in rows_b if r["clip"] == c] for c in clips}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        pick = [rng.choice(clips) for _ in clips]
        a = [r for c in pick for r in by_a[c]]
        b = [r for c in pick for r in by_b[c]]
        diffs.append(more(a, thr_a)[key] - more(b, thr_b)[key])
    diffs.sort()
    return round(statistics.mean(diffs), 3), round(diffs[int(0.025 * n)], 3), round(diffs[int(0.975 * n)], 3)


def calibration(rows: list[dict], bins: int = 5) -> tuple[list[dict], float, float]:
    table, ece, brier = [], 0.0, 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in rows if lo <= r["p_done"] < hi or (b == bins - 1 and r["p_done"] == 1.0)]
        if not sel:
            continue
        conf = statistics.mean(r["p_done"] for r in sel)
        acc = statistics.mean(1.0 if r["present"] else 0.0 for r in sel)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel), "mean_p": round(conf, 3), "share_real": round(acc, 3)})
        ece += len(sel) / len(rows) * abs(conf - acc)
    brier = statistics.mean((r["p_done"] - (1.0 if r["present"] else 0.0)) ** 2 for r in rows)
    return table, round(ece, 3), round(brier, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "test2", "test3", "test4", "tune"])
    args = ap.parse_args()
    split = args.split
    params = dict(fusion.load_params(), ladder_mode="summary")
    data = fit_fusion.load_all(split, "run3" if split == "tune" else "run1")
    n_clips = len(data)
    n_steps = sum(len(c["steps"]) for c in data.values())
    print(f"{split} split: {n_clips} clips, {n_steps} steps "
          f"({sum(s['present'] for c in data.values() for s in c['steps'].values())} real, "
          f"{sum(not s['present'] for c in data.values() for s in c['steps'].values())} planted skips)")
    print("fusion parameters fitted on:", params.get("fitted_on"))

    table = []
    systems: dict[str, tuple[list[dict], float | None]] = {}

    lad = fused_rows(data, params, True, [])
    systems["ladder alone (Decisions API)"] = (lad, 0.5)
    for name in ["hydrogen", "neon"] + FAST + ["carbon"]:
        rows, secs, n_used, n_fallback = single_rows(split, name)
        if rows:
            label = {"hydrogen": "Hydrogen low res 1 fps, one call", "neon": "Neon low res 1 fps, one call",
                     "hydrogen4": "Hydrogen 4 fps, one call", "hydrogen2": "Hydrogen 2 fps, one call",
                     "neon4": "Neon 4 fps, one call", "carbon": "Carbon high thinking, one call"}[name]
            systems[label] = (rows, None)
            table.append((label, n_used, n_fallback, secs))
    fused = fused_rows(data, params, True, FAST)
    systems["FUSED v2: ladder + Hydrogen 4 fps + Hydrogen 2 fps + Neon 4 fps"] = (fused, 0.5)
    systems["v2 witnesses without the ladder"] = (fused_rows(data, params, False, FAST), 0.5)
    if all(any(n in s["w"] for c in data.values() for s in c["steps"].values()) for n in V1):
        systems["fused v1 (low-res Hydrogen and Neon)"] = (fused_rows(data, params, True, V1), 0.5)

    # latency of the fused system: the slowest of its parallel calls, per clip
    per_clip = {}
    for name in FAST + ["ladder"]:
        tag = fit_fusion.WITNESS_TAG if name in fit_fusion.TAGGED else ""
        path = OUT / f"{split}_{name}{tag}_run{'3' if (split == 'tune' and name == 'ladder') else '1'}.jsonl"
        if not path.exists():
            continue
        for r in score.load(str(path)):
            if "call" in r and r["call"].get("seconds") is not None:
                per_clip.setdefault(r["id"], []).append(r["call"]["seconds"])
    fused_lat = sorted(max(v) for v in per_clip.values() if len(v) == len(FAST) + 1)
    lad_secs = sorted(r["call"]["seconds"] for r in score.load(str(OUT / f"{split}_ladder_run{'3' if split == 'tune' else '1'}.jsonl")) if "call" in r)

    report = {"split": split, "clips": n_clips, "steps": n_steps, "systems": {}}
    print("\n| System | Clips | Skip F1 | Precision | Recall | AUROC | Within 1 s | Within 2 s | Within 5 s | Median error | Median latency |")
    print("|-|-|-|-|-|-|-|-|-|-|-|")
    lat = {t[0]: t[3] for t in table}
    used = {t[0]: (t[1], t[2]) for t in table}
    for label, (rows, thr) in systems.items():
        m = more(rows, thr)
        if label in lat:
            secs = sorted(lat[label])
            l_txt = f"{statistics.median(secs):.1f} s (p90 {secs[int(0.9 * len(secs)) - 1]:.0f} s)"
            clips_txt = f"{used[label][0]}" + (f" ({used[label][1]} fell back)" if used[label][1] else "")
        elif label.startswith("ladder"):
            l_txt = f"{statistics.median(lad_secs):.1f} s"
            clips_txt = str(n_clips)
        else:
            l_txt = f"{statistics.median(fused_lat):.1f} s (p90 {fused_lat[int(0.9 * len(fused_lat)) - 1]:.0f} s)" if fused_lat else "n/a"
            clips_txt = str(n_clips)
        auroc = "n/a" if thr is None else f"{m['auroc']:.3f}"
        print(f"| {label} | {clips_txt} | {m['skip_f1']:.3f} | {m['skip_precision']:.2f} | {m['skip_recall']:.2f} | {auroc} | "
              f"{m['within_1s']:.3f} | {m['within_2s']:.3f} | {m['within_5s']:.3f} | {m['median_abs_err_s']} s | {l_txt} |")
        report["systems"][label] = dict(m, latency=l_txt, clips=clips_txt)

    # paired bootstrap: fused against each single call, on the clips both scored
    print("\nPaired bootstrap over clips (1,000 resamples), fused minus single call, 95 percent interval:")
    report["bootstrap"] = {}
    for label, (rows, thr) in systems.items():
        if thr is not None:
            continue
        clips = {r["clip"] for r in rows}
        sub = [r for r in fused if r["clip"] in clips]
        out = {}
        for key in ("skip_f1", "within_2s"):
            out[key] = bootstrap(sub, rows, 0.5, None, key)
        report["bootstrap"][label] = out
        print(f"  vs {label}: F1 {out['skip_f1'][0]:+.3f} [{out['skip_f1'][1]:+.3f}, {out['skip_f1'][2]:+.3f}]   "
              f"within 2 s {out['within_2s'][0]:+.3f} [{out['within_2s'][1]:+.3f}, {out['within_2s'][2]:+.3f}]")

    tab, ece, brier = calibration(fused)
    report["calibration"] = {"bins": tab, "ece": ece, "brier": brier}
    print("\nCalibration of the fused P(done): expected calibration error", ece, "Brier", brier)
    for row in tab:
        print(f"  P in {row['bin']}: {row['n']:4d} steps, mean P {row['mean_p']:.2f}, share that really happened {row['share_real']:.2f}")

    m_all = more(fused, 0.5)
    spec = more([r for r in fused if r["label"] not in VAGUE], 0.5)
    print(f"\nInterval coverage (true start inside the reported range): {m_all['interval_coverage']}")
    print("Without the two vague classes (fixed on the tuning split):", json.dumps({k: spec[k] for k in ("skip_f1", "within_2s", "within_5s", "median_abs_err_s")}))

    # failure table: which labels the fused system gets wrong most
    wrong: dict[str, list[int]] = {}
    for r in fused:
        said = r["p_done"] >= 0.5
        bad = (said != r["present"]) or (r["present"] and said and abs(r["time"] - r["true_start"]) > 5)
        w = wrong.setdefault(r["label"], [0, 0])
        w[0] += int(bad)
        w[1] += 1
    worst = sorted(wrong.items(), key=lambda kv: (-kv[1][0], kv[0]))[:10]
    report["failures"] = [{"label": k, "wrong": v[0], "asked": v[1]} for k, v in worst]
    print("\nWhere the fused system fails most (wrong verdict, or more than 5 s off):")
    for k, v in worst:
        print(f"  {v[0]:3d} of {v[1]:3d}  {k}")
    json.dump(report, open(OUT / f"{split}_report.json", "w", encoding="ascii"), indent=1)
    print("\nsaved", OUT / f"{split}_report.json")


if __name__ == "__main__":
    main()
