"""Field check of the auditor: the nine phone takes, one fused read and one audit each.

  python -m eval.audit_field [--takes 1,4,6,8,9] [--model gemini-4-hydrogen] [--budget 18]
                             [--workers 4] [--reuse-fused] [--run 2]

For each take: engine.run_fused once with the six-step checklist of eval/field.py, then
clock.auditor.audit once on that result. Writes results/audit_field.json (one record per take,
model and run number; a rerun replaces the records it repeats) and prints a table and the counts.
--run N stores a repeat under run number N, so the same take can be audited more than once.

--budget is the largest number of managed-agent interactions this run may create, retries included.
--reuse-fused takes the fused read stored for the same take by an earlier run instead of a new one,
so two auditor models can be compared on the same fused result.

The takes were recorded by the builder; truth comes from eval.field.TAKES (a step index listed
there was NOT done). This is a small check on real footage, not a benchmark score.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clock import auditor, engine
from eval.field import STEPS, TAKES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "audit_field.json"
FILES = {
    "1": "0598673be4fd.mp4", "2": "59f28acbf16d.mp4", "3": "b8c33ec83154.mp4",
    "4": "19b7f1b3417f.mp4", "5": "1f3b5b67c424.mp4", "6": "05e358080485.mp4",
    "7": "203296f29ce5.mp4", "8": "81f8ea3a5269.mp4", "9": "428cb05ba955.mp4",
}
KEEP = ("index", "text", "status", "p_done", "time", "lo", "hi")


def _arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _load() -> dict:
    if OUT.exists():
        return json.load(open(OUT, encoding="ascii"))
    return {"records": []}


def summarize(records: list[dict]) -> dict:
    """The counts that matter, over the given records (one auditor model)."""
    c = {"audits": len(records), "audits_failed": 0, "code_ran": 0, "rules_agree": 0,
         "wrong_done": 0, "wrong_done_disputed": 0, "wrong_done_cannot_tell": 0, "wrong_done_confirmed": 0,
         "right_done": 0, "right_done_disputed": 0, "right_done_cannot_tell": 0, "right_done_confirmed": 0,
         "unsure_not_done": 0, "unsure_not_done_disputed": 0,
         "unsure_done": 0, "unsure_done_disputed": 0,
         "right_not_found": 0, "right_not_found_disputed": 0,
         "wrong_not_found": 0, "wrong_not_found_disputed": 0}
    secs = []
    for r in records:
        a = r["audit"]
        if not a.get("ok"):
            c["audits_failed"] += 1
            continue
        secs.append(a["seconds"])
        c["code_ran"] += bool(a.get("agent_ran_code"))
        c["rules_agree"] += bool(a.get("rules_agree"))
        verdict = {s["index"]: s["audit"] for s in a["steps"]}
        for s in r["fused"]["steps"]:
            really_done = s["index"] not in r["not_done"]
            v = verdict.get(s["index"])
            if s["status"] == "done":
                key = "right_done" if really_done else "wrong_done"
                c[key] += 1
                c[f"{key}_disputed"] += v == "dispute"
                c[f"{key}_cannot_tell"] += v == "cannot tell"
                c[f"{key}_confirmed"] += v == "confirm"
            elif s["status"] == "unsure":
                key = "unsure_done" if really_done else "unsure_not_done"
                c[key] += 1
                c[f"{key}_disputed"] += v == "dispute"
            else:
                key = "wrong_not_found" if really_done else "right_not_found"
                c[key] += 1
                c[f"{key}_disputed"] += v == "dispute"
    c["seconds_median"] = round(statistics.median(secs), 1) if secs else None
    c["seconds_max"] = round(max(secs), 1) if secs else None
    return c


def table(records: list[dict]) -> list[str]:
    lines = []
    for r in records:
        a = r["audit"]
        head = f"take {r['take']}: {r['desc']}"
        if a.get("ok"):
            head += (f" | audit {a['seconds']} s, {a.get('attempts')} call(s), sandbox run proven: "
                     f"{'yes' if a.get('agent_ran_code') else 'no'}, rules agree: {'yes' if a.get('rules_agree') else 'no'}")
        else:
            head += f" | AUDIT FAILED: {a.get('error')}"
        lines.append(head)
        verdict = {s["index"]: s for s in a.get("steps", [])}
        for s in r["fused"]["steps"]:
            really_done = s["index"] not in r["not_done"]
            v = verdict.get(s["index"], {})
            mark = ""
            if s["status"] == "done" and not really_done:
                mark = "  <- fused wrong; " + ("audit caught it" if v.get("audit") == "dispute" else "audit missed it")
            elif s["status"] == "done" and v.get("audit") == "dispute":
                mark = "  <- audit wrongly disputes a right verdict"
            elif s["status"] == "unsure" and v.get("audit") == "dispute":
                mark = "  <- audit settles an unsure step " + ("the wrong way" if really_done else "the right way")
            elif s["status"] == "not_found" and v.get("audit") == "dispute":
                mark = "  <- audit disputes not_found; " + ("it is right" if really_done else "it is wrong")
            lines.append(f"  {s['index']}. {s['text'][:40]:40} | fused {s['status']:9} {s['p_done']:.2f} | "
                         f"audit {v.get('audit', '-'):11} | truth {'done' if really_done else 'NOT done'}{mark}")
        if a.get("ok") and a.get("order_pairs"):
            lines.append(f"  order pairs listed by the audit: {a['order_pairs']}")
    return lines


def main() -> None:
    takes = [t for t in _arg("--takes", ",".join(FILES)).split(",") if t in FILES]
    model = _arg("--model", "gemini-4-hydrogen")
    auditor.CALL_BUDGET = int(_arg("--budget", "18"))
    workers = int(_arg("--workers", "4"))
    reuse = "--reuse-fused" in sys.argv
    run = int(_arg("--run", "1"))
    old = _load()
    stored = {r["take"]: r["fused"] for r in old["records"] if r.get("fused")}

    pool = ThreadPoolExecutor(max_workers=workers)
    jobs, records = {}, []
    for take in takes:
        video = ROOT / "runs" / "uploads" / FILES[take]
        desc, not_done, _, _ = TAKES[take]
        try:
            if reuse and take in stored:
                fused = stored[take]
            else:
                res = engine.run_fused(video, STEPS, tag=f"audit_field:{take}")
                fused = {"video": res["video"], "steps": [{k: s.get(k) for k in KEEP} for s in res["steps"]],
                         "order_issues": res["order_issues"], "seconds_total": res["seconds_total"]}
            fused["video"]["path"] = f"runs/uploads/{FILES[take]}"
        except Exception as e:  # noqa: BLE001
            print(f"take {take}: fused read failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        print(f"take {take}: fused read ready ({fused.get('seconds_total')} s), audit started", flush=True)
        jobs[take] = pool.submit(auditor.audit, video, fused, model, 180, 1, f"audit_field:{take}")
        records.append({"take": take, "model": model, "run": run, "file": FILES[take], "desc": desc, "not_done": not_done,
                        "fused": fused, "ts": time.time()})
    for r in records:
        r["audit"] = jobs[r["take"]].result()
        print(f"take {r['take']}: audit {'ok' if r['audit']['ok'] else 'FAILED'} in {r['audit']['seconds']} s", flush=True)
    pool.shutdown()

    done = {(r["take"], r["model"], r["run"]) for r in records}
    allrec = [r for r in old["records"] if (r["take"], r["model"], r.get("run", 1)) not in done] + records
    allrec.sort(key=lambda r: (r["model"], r["take"], r.get("run", 1)))
    summary = {m: summarize([r for r in allrec if r["model"] == m]) for m in sorted({r["model"] for r in allrec})}
    OUT.parent.mkdir(exist_ok=True)
    json.dump({"steps": STEPS, "summary": summary, "interactions_created_this_run": auditor.CALLS_MADE,
               "records": allrec}, open(OUT, "w", encoding="ascii"), indent=1)

    print()
    print("\n".join(table(records)))
    c = summarize(records)
    print()
    print(f"auditor model {model}; run {run}; takes {','.join(takes)}; managed-agent interactions created by this run: {auditor.CALLS_MADE}")
    print(f"(a) fused 'done' verdicts that are wrong: {c['wrong_done']}; the audit disputes {c['wrong_done_disputed']}, "
          f"cannot tell {c['wrong_done_cannot_tell']}, confirms {c['wrong_done_confirmed']}")
    print(f"(b) fused 'done' verdicts that are right: {c['right_done']}; the audit wrongly disputes {c['right_done_disputed']}, "
          f"cannot tell {c['right_done_cannot_tell']}, confirms {c['right_done_confirmed']}")
    print(f"    fused 'unsure' on steps not done: {c['unsure_not_done']}, disputed {c['unsure_not_done_disputed']}; "
          f"fused 'unsure' on steps done: {c['unsure_done']}, disputed {c['unsure_done_disputed']}")
    print(f"    fused 'not_found' that is right: {c['right_not_found']}, wrongly disputed {c['right_not_found_disputed']}; "
          f"fused 'not_found' that is wrong: {c['wrong_not_found']}, disputed {c['wrong_not_found_disputed']}")
    print(f"(c) seconds per audit: median {c['seconds_median']}, max {c['seconds_max']}")
    print(f"(d) audits that failed or gave no usable JSON: {c['audits_failed']} of {c['audits']}")
    print(f"(e) audits where the returned SHA-256 proof matched (the sandbox run happened): {c['code_ran']} of "
          f"{c['audits'] - c['audits_failed']}; agent rule code agreed with the local rules in {c['rules_agree']}")
    t = summary[model]
    print(f"all stored runs of {model}: {t['audits']} audits; wrong 'done' verdicts {t['wrong_done']}, disputed {t['wrong_done_disputed']}; "
          f"right 'done' verdicts {t['right_done']}, wrongly disputed {t['right_done_disputed']}; "
          f"seconds median {t['seconds_median']}, max {t['seconds_max']}; failed {t['audits_failed']}")
    print("written", OUT)


if __name__ == "__main__":
    main()
