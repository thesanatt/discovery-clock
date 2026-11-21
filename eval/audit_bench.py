"""The Managed Agent audit on benchmark clips: does it dispute verdicts it should not?

  python -m eval.audit_bench --split test5 [--clips 30] [--workers 3] [--budget 45]

For the first --clips clips of the split, in manifest order (fixed before anything ran):
engine.run_fused once with the clip's checklist (its labeled actions plus two planted steps that
never happen, the same rule as eval.run_eval), then clock.auditor.audit once on that result.
Writes results/audit_bench_<split>.json and prints the counts by verdict and truth.

What this can show: how often the audit disputes a verdict that was right (it should be rare),
and how often it disputes a "done" verdict on a planted step. What it cannot show: the case the
audit was built for, a result already in place before the video starts, because the benchmark
has no labels for that. The field takes cover that case (eval/audit_field.py).

Finished clips are kept, so a rerun only does what is missing.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clock import auditor, engine
from eval import audit_field, pt_data, run_eval

ROOT = Path(__file__).resolve().parent.parent
KEEP = ("index", "text", "status", "p_done", "time", "lo", "hi")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test5")
    ap.add_argument("--clips", type=int, default=30)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--budget", type=int, default=45, help="most managed-agent interactions this run may create")
    ap.add_argument("--model", default="gemini-4-hydrogen")
    args = ap.parse_args()
    out = ROOT / "results" / f"audit_bench_{args.split}.json"
    auditor.CALL_BUDGET = args.budget

    clips = pt_data.load_manifest(args.split)[: args.clips]
    all_labels = sorted({seg["label"] for c in pt_data.load_manifest("tune") + pt_data.load_manifest("test") for seg in c["actions"]})
    old = json.load(open(out, encoding="ascii"))["records"] if out.exists() else []
    records = [r for r in old if r.get("audit", {}).get("ok")]
    have = {r["take"] for r in records}
    lock = threading.Lock()

    def save() -> None:
        recs = sorted(records, key=lambda r: r["take"])
        json.dump({"split": args.split, "clips_asked": args.clips, "model": args.model,
                   "summary": audit_field.summarize(recs), "records": recs},
                  open(out, "w", encoding="ascii"), indent=1)

    def work(clip: dict) -> None:
        if clip["id"] in have:
            return
        labels, truth = run_eval.checklist(clip, all_labels)
        steps = [run_eval.describe(l) for l in labels]
        try:
            res = engine.run_fused(clip["path"], steps, tag=f"audit_bench:{clip['id']}")
            fused = {"video": {"duration": res["video"].get("duration")},
                     "steps": [{k: s.get(k) for k in KEEP} for s in res["steps"]],
                     "order_issues": res["order_issues"], "seconds_total": res["seconds_total"]}
            a = auditor.audit(clip["path"], fused, args.model, 180, 1, f"audit_bench:{clip['id']}")
        except Exception as e:  # noqa: BLE001
            print(f"{clip['id']}: failed before the audit: {type(e).__name__}: {str(e)[:160]}", flush=True)
            return
        a.pop("frames", None)
        rec = {"take": clip["id"], "model": args.model, "run": 1, "desc": "benchmark clip",
               "labels": labels, "not_done": [i for i, l in enumerate(labels) if l not in truth],
               "fused": fused, "audit": a, "ts": time.time()}
        with lock:
            records.append(rec)
            save()
        v = {s["index"]: s["audit"] for s in a.get("steps", [])}
        print(f"{clip['id']}: audit {'ok' if a['ok'] else 'FAILED ' + str(a.get('error'))[:80]} in {a['seconds']} s | "
              + " ".join(f"{s['status'][:4]}/{'Y' if s['index'] not in rec['not_done'] else 'N'}/{str(v.get(s['index'], '-'))[:4]}"
                         for s in fused["steps"]), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, clips))
    save()
    c = audit_field.summarize(sorted(records, key=lambda r: r["take"]))
    print(json.dumps(c, indent=1))
    print("interactions created by this run:", auditor.CALLS_MADE, "| written", out)


if __name__ == "__main__":
    main()
