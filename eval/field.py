"""Field check: phone recordings made at the event, each with a known script.

These are NOT part of the benchmark scores. They were recorded by the builder, so they only
show how the frozen system behaves on real phone footage of a typed checklist.

  python -m eval.field FILE=TAKE [FILE=TAKE ...] [--runs 2]

TAKE is a key of TAKES below. Writes results/field.json and results/field.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from clock import engine

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

STEPS = [
    "unzip the bag",
    "put the AirPods case in the bag",
    "put the Hello Panda box in the bag",
    "take off the watch and put it in the bag",
    "zip the bag closed",
    "knock on the table twice",
]

# take -> (what was really done, indexes of steps NOT done, whether the order was changed, note)
# The takes are numbered in the order they were recorded. The script for each take was decided
# before recording, except take 5, which is logged as what the person reported doing.
TAKES = {
    "1": ("all six steps, in order", [], False, "used to choose version 2"),
    "2": ("AirPods skipped", [1], False, "used to choose version 2"),
    "3": ("Hello Panda skipped", [2], False, ""),
    "4": ("watch skipped", [3], False, ""),
    "5": ("Hello Panda before AirPods, knock skipped", [5], True, ""),
    "6": ("bag not zipped", [4], False, ""),
    "7": ("all six, Hello Panda before AirPods", [], True, ""),
    "8": ("fake-out: AirPods picked up, moved toward the bag, put back on the table", [1], False, "hard negative"),
}


def main() -> None:
    args = [a for a in sys.argv[1:] if "=" in a]
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 2
    records = []
    for a in args:
        file, take = a.split("=", 1)
        desc, not_done, swapped, note = TAKES[take]
        for run in range(1, runs + 1):
            t0 = time.time()
            try:
                res = engine.run_fused(file, STEPS, tag=f"field:{take}:{run}")
            except Exception as e:  # noqa: BLE001
                records.append({"take": take, "run": run, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                continue
            rows = []
            for s in res["steps"]:
                truth_done = s["index"] not in not_done
                said_done = s["status"] == "done"
                rows.append({"step": s["text"], "truth": "done" if truth_done else "skipped", "status": s["status"],
                             "p_done": s["p_done"], "time": s["time"], "correct": said_done == truth_done and s["status"] != "unsure",
                             "witnesses": s["witnesses"]})
            records.append({"take": take, "run": run, "desc": desc, "note": note, "file": Path(file).name,
                            "order_expected": swapped,
                            "duration": res["video"]["duration"], "seconds": res["seconds_total"],
                            "order_issues": res["order_issues"], "witness_calls": res["witness_calls"], "rows": rows,
                            "wall": round(time.time() - t0, 1)})
            flagged = [r["step"] for r in rows if r["status"] != "done"]
            print(f"take {take} run {run}: {desc} -> flagged {flagged or 'nothing'}; order issues {res['order_issues']}; {res['seconds_total']} s", flush=True)
    OUT.mkdir(exist_ok=True)
    path = OUT / "field.json"
    old = json.load(open(path, encoding="ascii")) if path.exists() else []
    keep = [r for r in old if (r["take"], r["run"]) not in {(x["take"], x["run"]) for x in records}]
    allrec = sorted(keep + records, key=lambda r: (r["take"], r["run"]))
    json.dump(allrec, open(path, "w", encoding="ascii"), indent=1)

    lines = ["# Field check", "",
             "Phone recordings made at the event by the builder, each following a known script. They are a",
             "small extra check on real footage and are not part of the benchmark scores in the README.",
             "Takes 1 and 2 were used to choose the version 2 witness settings; later takes were not.",
             "The checklist was the same six steps every time:", ""]
    lines += [f"{i + 1}. {s}" for i, s in enumerate(STEPS)]
    lines += ["", "| Take | What was done | Run | Steps flagged as not done or unsure | Verdicts right | Order flag (expected) | Seconds |", "|-|-|-|-|-|-|-|"]
    ok_steps = total_steps = caught = planted = order_ok = order_n = 0
    for r in allrec:
        if "error" in r:
            lines.append(f"| {r['take']} | | {r['run']} | error: {r['error']} | | | |")
            continue
        flagged = [f"{x['step']} ({x['status']}, {x['p_done']:.2f})" for x in r["rows"] if x["status"] != "done"]
        right = sum(x["correct"] for x in r["rows"])
        ok_steps += right
        total_steps += len(r["rows"])
        for x in r["rows"]:
            if x["truth"] == "skipped":
                planted += 1
                caught += x["status"] == "not_found"
        expected = bool(r.get("order_expected", False))
        got = bool(r["order_issues"])
        order_n += 1
        order_ok += got == expected
        lines.append(f"| {r['take']} | {r['desc']} | {r['run']} | {'; '.join(flagged) or 'none'} | {right} of {len(r['rows'])} | "
                     f"{'yes' if got else 'no'} ({'yes' if expected else 'no'}) | {r['seconds']} |")
    lines += ["", f"Step verdicts right: {ok_steps} of {total_steps}. Skipped steps caught as not found: {caught} of {planted}. "
              f"Order flag as expected: {order_ok} of {order_n} runs."]
    (OUT / "field.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print("written", OUT / "field.md")


if __name__ == "__main__":
    main()
