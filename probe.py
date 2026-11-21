"""Go or no-go probe: does the Decisions model have a clock inside a video?

Runs on the Perception Test sample clips (labels by Google DeepMind, CC BY 4.0), which
this project did not write. For each clip, one Decisions call asks:
  1. "Is this video longer than T seconds?" for a range of T. The answers should flip at the true length.
  2. For each labeled action, "had it started by t?" for every second. The ladder should rise at the true start.
  3. The same ladder for two actions that never happen in the clip. Those should stay flat.

Usage:
  python probe.py                 plain video
  python probe.py --timecode      a running clock is burned into the frames first
  python probe.py --dry           build questions and prepare video only, no API call
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from clock import ladder, media

ROOT = Path(__file__).resolve().parent
PT = ROOT / "data" / "pt"
OUT = ROOT / "runs"


def describe(label: str) -> str:
    return f'a person doing this action: "{label}"'


def first_starts(segments: list[dict]) -> dict[str, float]:
    """First start time in seconds for each action class in a clip."""
    out: dict[str, float] = {}
    for seg in segments:
        t = seg["timestamps"][0] / 1e6
        if seg["label"] not in out or t < out[seg["label"]]:
            out[seg["label"]] = t
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timecode", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--compact", action="store_true", help="shared event list plus short questions")
    ap.add_argument("--medium", action="store_true", help="short framing, event named in each question")
    ap.add_argument("--resolution", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sample = json.load(open(PT / "sample.json", encoding="utf-8"))
    all_classes = sorted({seg["label"] for v in json.load(open(PT / "challenge_action_localisation_valid.json", encoding="utf-8")).values()
                          for seg in v["action_localisation"]})
    rng = random.Random(args.seed)
    cl = None
    if not args.dry:
        from clock import decide
        cl = decide.client()

    rows, dur_rows, absent_rows, calls = [], [], [], []
    for vid in sorted(sample)[: args.limit]:
        src = PT / "videos" / f"{vid}.mp4"
        if not src.exists():
            continue
        truth = first_starts(sample[vid]["action_localisation"])
        if not truth:
            continue
        small = media.prep(src, OUT / "prep" / f"{vid}{'_tc' if args.timecode else ''}.mp4", timecode=args.timecode)
        dur = media.duration_s(small)
        grid = ladder.time_grid(dur, max_points=40)
        present = sorted(truth)
        absent = rng.sample([c for c in all_classes if c not in truth], 2)
        steps = [describe(c) for c in present + absent]
        qs = ladder.build_compact(steps, grid) if args.compact else (ladder.build_medium(steps, grid) if args.medium else ladder.build_questions(steps, grid))
        dur_ts = list(range(5, 65, 5))
        for t in dur_ts:
            qs[f"len_{t}"] = ladder.longer_than_question(t)
        size_kb = small.stat().st_size // 1024
        print(f"{vid}: {dur:.1f}s, {size_kb} KB, {len(present)} present + 2 absent actions, {len(qs)} questions")
        if args.dry:
            continue
        t0 = time.time()
        blocks = [media.video_block(small, args.resolution)]
        if args.compact:
            blocks.insert(0, {"type": "text", "text": ladder.glossary(steps, clock_on_screen=args.timecode)})
        if args.medium:
            blocks.insert(0, {"type": "text", "text": ladder.PREAMBLE})
        answers, info = decide.decide(cl, blocks, qs, tag=f"probe:{vid}")
        calls.append({"vid": vid, **info, "wall": round(time.time() - t0, 1)})
        fits = ladder.fit_all(steps, grid, answers)
        for name, f in zip(present, fits[: len(present)]):
            err = None if f.t50 is None else abs(f.t50 - truth[name])
            rows.append({"vid": vid, "action": name, "true_start": round(truth[name], 2), "est": f.t50, "lo": f.lo, "hi": f.hi,
                         "abs_err": err, "found": f.found, "p_end": f.p_end, "p_any": f.p_any, "violation": f.violation, "raw": f.raw})
        for name, f in zip(absent, fits[len(present):]):
            absent_rows.append({"vid": vid, "action": name, "found": f.found, "p_end": f.p_end, "p_any": f.p_any, "max_raw": max(f.raw), "raw": f.raw})
        # where does "longer than T" flip?
        flip = next((t for t in dur_ts if answers[f"len_{t}"] < 0.5), None)
        dur_rows.append({"vid": vid, "true_len": round(dur, 1), "flip_at": flip, "answers": {t: round(answers[f'len_{t}'], 2) for t in dur_ts}})
        print(f"   call {info['seconds']:.1f}s, {info['input_tokens']} input tokens")

    if args.dry:
        return
    found = [r for r in rows if r["abs_err"] is not None]
    errs = [r["abs_err"] for r in found]
    summary = {
        "variant": ("timecode" if args.timecode else "plain") + ("_compact" if args.compact else "") + ("_medium" if args.medium else ""),
        "resolution": args.resolution,
        "present_events": len(rows),
        "found": sum(r["found"] for r in rows),
        "within_2s": sum(1 for e in errs if e <= 2),
        "within_5s": sum(1 for e in errs if e <= 5),
        "median_abs_err_s": round(statistics.median(errs), 2) if errs else None,
        "absent_events": len(absent_rows),
        "absent_false_finds": sum(r["found"] for r in absent_rows),
        "absent_max_p_end": round(max((r["p_end"] for r in absent_rows), default=0), 2),
        "length_flip_abs_err_s": [None if d["flip_at"] is None else round(abs(d["flip_at"] - d["true_len"]), 1) for d in dur_rows],
        "median_call_seconds": round(statistics.median([c["seconds"] for c in calls]), 1) if calls else None,
    }
    OUT.mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S")
    path = OUT / f"probe_{summary['variant']}_{args.resolution}_{stamp}.json"
    json.dump({"summary": summary, "rows": rows, "absent": absent_rows, "length": dur_rows, "calls": calls}, open(path, "w", encoding="ascii"), indent=1)
    print("\nPER EVENT (true start -> estimate [lo, hi], p_end, violation)")
    for r in rows:
        est = "none" if r["est"] is None else f"{r['est']:.1f}"
        print(f"  {r['vid']:12s} {r['action'][:44]:44s} {r['true_start']:6.1f} -> {est:>5s}  p_end={r['p_end']:.2f} viol={r['violation']:.2f}")
    print("ABSENT (should stay flat)")
    for r in absent_rows:
        print(f"  {r['vid']:12s} {r['action'][:44]:44s} p_end={r['p_end']:.2f} max_raw={r['max_raw']:.2f} p_any={r['p_any']}")
    print("LENGTH (answers to 'longer than T s')")
    for d in dur_rows:
        print(f"  {d['vid']:12s} true {d['true_len']:5.1f}s flips at {d['flip_at']}  {d['answers']}")
    print("\nSUMMARY", json.dumps(summary, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
