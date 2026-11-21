"""Zoom stage: Neon re-watches a few seconds around each step's fused time estimate, at higher
resolution and frame rate, and reports yes / no / unclear plus the second it starts.

  python -m eval.zoom_stage --split tune
Writes runs/eval/<split>_zoom_run1.jsonl, one row per step that was zoomed.
The window comes from the fused posterior (ladder + Hydrogen + Neon timelines) with the
parameters in clock/fusion_params.json, which are fitted on the tuning split only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clock import fusion, media, models
from eval import fit_fusion

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "eval"
CLIPS = ROOT / "runs" / "zoom"
RAW = ROOT / "data" / "pt" / "valid"
SMALL = ROOT / "data" / "pt" / "valid_small"
BEFORE_S, AFTER_S = 3.0, 4.0
MIN_P = 0.2          # steps the fusion already rules out are not zoomed
FIRST_PASS = ["hydrogen", "neon"]


def describe(label: str) -> str:
    return f'a person doing this action: "{label}"'


def cut_window(video_id: str, start: float, end: float, dst: Path) -> Path:
    """Cut [start, end] from the original recording at 540p and 12 frames per second."""
    src = RAW / f"{video_id}.mp4"
    if not src.exists():
        src = SMALL / f"{video_id}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run([media.FFMPEG, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(src),
                          "-t", f"{max(0.5, end - start):.2f}", "-vf", "scale=-2:540", "-r", "12",
                          "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                          "-c:a", "aac", "-b:a", "64k", str(dst)], capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg cut failed: {res.stderr[-200:]}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["tune", "test"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    data = fit_fusion.load_all(args.split, "run3" if args.split == "tune" else "run1")
    params = dict(fusion.load_params(), ladder_mode="summary")
    path = OUT / f"{args.split}_zoom_run1.jsonl"
    done = set()
    if path.exists():
        done = {(r["id"], r["label"]) for r in (json.loads(l) for l in open(path, encoding="ascii") if l.strip())}
    jobs = []
    for vid, c in sorted(data.items()):
        for i, (label, s) in enumerate(sorted(c["steps"].items())):
            if (vid, label) in done:
                continue
            w = {n: s["w"][n] for n in FIRST_PASS if n in s["w"]}
            post = fusion.posterior(s["grid"], s["raw"], s["p_any"], c["duration"], w, params)
            if post["p_done"] < MIN_P:
                continue
            start = max(0.0, post["time"] - BEFORE_S)
            end = min(c["duration"], post["time"] + AFTER_S)
            jobs.append((vid, i, label, start, end, post["p_done"]))
    print(f"{args.split}: {len(jobs)} steps to zoom, {len(done)} already done")
    lock = threading.Lock()

    def work(job) -> None:
        vid, i, label, start, end, p_before = job
        t0 = time.time()
        row = {"id": vid, "label": label, "win_start": round(start, 2), "win_end": round(end, 2), "p_before": p_before}
        try:
            clip = cut_window(vid, start, end, CLIPS / f"{vid}_{i}.mp4")
            info: dict = {}
            res = models.confirm_step(clip, describe(label), info=info)
            row.update(verdict=res["verdict"], second=res["second"], evidence=res["evidence"],
                       model=info.get("model"), seconds=round(time.time() - t0, 2))
        except Exception as e:  # noqa: BLE001 - record and continue
            row.update(error=f"{type(e).__name__}: {str(e)[:200]}")
        with lock:
            with open(path, "a", encoding="ascii", errors="replace") as f:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, jobs))
    rows = [json.loads(l) for l in open(path, encoding="ascii") if l.strip()]
    ok = [r for r in rows if "error" not in r]
    print(f"zoomed {len(ok)} steps, {len(rows) - len(ok)} errors;",
          {v: sum(1 for r in ok if r["verdict"] == v) for v in ("yes", "no", "unclear")})


if __name__ == "__main__":
    main()
