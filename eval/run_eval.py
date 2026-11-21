"""Held-out evaluation on the DeepMind Perception Test (labels by Google DeepMind, CC BY 4.0).

For each clip the checklist is the clip's own labeled action classes plus PHANTOMS steps that
never happen in it (picked by a hash of the clip id, never by looking at model output).
The system must say which steps happened, when each one started, and which did not happen.

Systems:
  ladder    Discovery Clock: one Decisions call, steps x seconds, isotonic fit   (clock.engine.run)
  yesno     ablation from the same call: only the whole-video yes/no answer per step, no ladder
  carbon    baseline: one gemini-4-carbon call asked for the same thing            (clock.models.baseline_timeline)

Usage:
  python -m eval.run_eval --split tune --system ladder --run 1
  python -m eval.run_eval --split test --system carbon --run 1 --limit 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from eval import pt_data

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "eval"
PHANTOMS = 2
SKIP_LABELS = {"Other"}          # not a description of anything; fixed before any test run


def describe(label: str) -> str:
    return f'a person doing this action: "{label}"'


def checklist(clip: dict, all_labels: list[str]) -> tuple[list[str], dict[str, float]]:
    """Labels to ask about (present first, then phantoms) and the true first start of each present one."""
    truth: dict[str, float] = {}
    for seg in clip["actions"]:
        if seg["label"] in SKIP_LABELS:
            continue
        if seg["label"] not in truth or seg["start"] < truth[seg["label"]]:
            truth[seg["label"]] = seg["start"]
    present = sorted(truth)
    seed = int(hashlib.sha1(clip["id"].encode()).hexdigest()[:8], 16)
    pool = [l for l in all_labels if l not in truth and l not in SKIP_LABELS]
    phantoms = random.Random(seed).sample(pool, PHANTOMS)
    return present + phantoms, truth


def run_clip(system: str, clip: dict, labels: list[str], truth: dict[str, float], client, resolution: str = "low") -> dict:
    steps = [describe(l) for l in labels]
    t0 = time.time()
    rows = []
    if system in ("ladder", "yesno"):
        from clock import engine
        res = engine.run(clip["path"], steps, client=client, resolution=resolution, tag=f"eval:{clip['id']}")
        for label, s in zip(labels, res["steps"]):
            rows.append({"label": label, "present": label in truth, "true_start": truth.get(label),
                         "status": s["status"], "time": s["time"], "lo": s["lo"], "hi": s["hi"],
                         "p_done": s["p_done"], "p_any": s.get("p_any"), "p_end": s.get("p_end"),
                         "violation": s["violation"], "raw": s["raw"]})
        call = res["call"]
    elif system in ("carbon", "neon", "hydrogen", "hydrogen4", "neonmed", "hydrogenhd", "neonhd", "neon4", "hydrogen2"):
        from clock import models
        info: dict = {}
        if system == "carbon":
            out = models.baseline_timeline(clip["path"], steps, info=info)   # high thinking, the strongest single call
        else:
            from clock import witness
            if system in ("hydrogenhd", "neonhd"):
                # same model, but it sees a 720p copy cut from the original recording, at high media resolution
                from clock import media
                raw = Path(clip["path"]).parent.parent / "valid" / (clip["id"] + ".mp4")
                hd = media.prep(raw, ROOT / "runs" / "prep_hd" / (clip["id"] + ".mp4"), height=720, fps=12)
                out = witness.timeline(hd, steps, model=system[:-2], thinking="low", resolution="high", fps=4.0,
                                       timeout_s=90.0, info=info, tag="witness_" + system)
            elif system == "neon4":        # Neon at 4 frames per second and medium resolution
                out = witness.timeline(clip["path"], steps, model="neon", thinking="low", resolution="medium", fps=4.0, info=info, tag="witness_neon4")
            elif system == "hydrogen2":    # Hydrogen at 2 frames per second and medium resolution
                out = witness.timeline(clip["path"], steps, model="hydrogen", thinking="low", resolution="medium", fps=2.0, info=info, tag="witness_hydrogen2")
            elif system == "hydrogen4":      # Hydrogen again, but 4 frames per second at medium resolution
                out = witness.timeline(clip["path"], steps, model="hydrogen", thinking="low", resolution="medium", fps=4.0, info=info, tag="witness_hydrogen4")
            elif system == "neonmed":      # Neon at medium thinking
                out = witness.timeline(clip["path"], steps, model="neon", thinking="medium", timeout_s=90.0, info=info, tag="witness_neonmed")
            else:
                out = witness.timeline(clip["path"], steps, model=system, thinking="low", info=info)
        by_index = {o["index"]: o for o in out}
        for i, label in enumerate(labels):
            o = by_index.get(i, {"happened": False, "start": None})
            rows.append({"label": label, "present": label in truth, "true_start": truth.get(label),
                         "status": "done" if o.get("happened") else "not_found", "time": o.get("start"),
                         "lo": None, "hi": None, "p_done": 1.0 if o.get("happened") else 0.0})
        # the wrapper may fall back to another model; record which one really answered
        call = {"seconds": round(time.time() - t0, 2), "model": info.get("model"), "fallback": info.get("fallback")}
    else:
        raise ValueError(system)
    return {"id": clip["id"], "duration": clip["duration"], "rows": rows, "call": call, "wall": round(time.time() - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["tune", "test", "test2", "test3", "test4"])
    ap.add_argument("--system", required=True, choices=["ladder", "carbon", "neon", "hydrogen", "hydrogen4", "neonmed", "hydrogenhd", "neonhd", "neon4", "hydrogen2"])
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resolution", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=1, help="clips in flight at once")
    ap.add_argument("--name", default="", help="suffix for the output file, for side experiments on the tuning split")
    args = ap.parse_args()

    clips = pt_data.load_manifest(args.split)
    if args.limit:
        clips = clips[: args.limit]
    all_labels = sorted({seg["label"] for c in pt_data.load_manifest("tune") + pt_data.load_manifest("test") for seg in c["actions"]})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.split}_{args.system}{('_' + args.name) if args.name else ''}_run{args.run}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in open(path, encoding="ascii") if l.strip()}
    client = None
    if args.system == "ladder":
        from clock import decide
        client = decide.client()
    print(f"{args.split} {args.system} run {args.run}: {len(clips)} clips, {len(done)} already done -> {path}")
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    todo = [c for c in clips if c["id"] not in done]

    def work(clip: dict) -> None:
        labels, truth = checklist(clip, all_labels)
        try:
            rec = run_clip(args.system, clip, labels, truth, client, args.resolution)
        except Exception as e:  # noqa: BLE001 - keep going, record the failure
            rec = {"id": clip["id"], "error": f"{type(e).__name__}: {str(e)[:300]}", "rows": []}
        with lock:
            with open(path, "a", encoding="ascii") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{clip['id']} {'ERROR ' + rec['error'] if 'error' in rec else str(rec['wall']) + 's'}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(work, todo))


if __name__ == "__main__":
    main()
