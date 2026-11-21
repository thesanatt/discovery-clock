"""Second, harder test set: CaptainCook4D, egocentric kitchen recordings with step and error labels.

Labels: github.com/CaptainCook4D/annotations (its LICENSE file is MIT; its README says Apache 2.0).
Videos: Box links listed in github.com/CaptainCook4D/downloader, metadata/download_links.json.
The project site, captaincook4d.github.io/captain-cook, provides the data under the Apache License 2.0.

    python -m eval.cc4d labels        # fetch the label files into data/cc4d/annotations
    python -m eval.cc4d candidates    # write eval/cc4d_candidates.json
    python -m eval.cc4d show 1_10     # print one recording as the loader returns it
    python -m eval.cc4d fetch 1_10    # download one GoPro 360p video into data/cc4d/videos

Run from the project root. All times are seconds from the start of the GoPro video.
"""
from __future__ import annotations

import csv
import hashlib
import http.client
import json
import sys
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "cc4d" / "annotations"
VIDEOS = ROOT / "data" / "cc4d" / "videos"
OUT = Path(__file__).resolve().parent / "cc4d_candidates.json"

RAW = "https://raw.githubusercontent.com/CaptainCook4D/"
LABEL_URLS = {
    "error_annotations.json": RAW + "annotations/main/annotation_json/error_annotations.json",
    "complete_step_annotations.json": RAW + "annotations/main/annotation_json/complete_step_annotations.json",
    "step_idx_description.json": RAW + "annotations/main/annotation_json/step_idx_description.json",
    "video_information.csv": RAW + "annotations/main/metadata/video_information.csv",
    "download_links.json": RAW + "downloader/main/metadata/download_links.json",
}
TIMEOUT = 60          # seconds per connection and per read
MISSING = "Missing Step"
N_EACH = 20
MB_PER_S = 0.115      # decimal MB per second of video, measured on two GoPro 360p files (920 kbit/s)


def _get(url: str, dst: Path) -> Path:
    """Download url to dst through a .part file. One retry.

    A body shorter than its Content-Length, or an .mp4 that does not start like one (a web page
    served in its place), counts as a failure, so a broken file never lands at dst.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    for attempt in (0, 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r, open(part, "wb") as f:
                want = r.headers.get("Content-Length")
                while chunk := r.read(1 << 20):   # read(n) returns short on a cut-off body, no error
                    f.write(chunk)
            got = part.stat().st_size
            if want is not None and got != int(want):
                raise OSError(f"cut-off download: {got} of {want} bytes from {url}")
            if dst.suffix == ".mp4":
                with open(part, "rb") as f:
                    if f.read(12)[4:8] != b"ftyp":
                        raise OSError(f"not an mp4 file: {url}")
            part.replace(dst)
            return dst
        except (OSError, http.client.HTTPException):   # timeouts, refused, cut-off or wrong bodies
            part.unlink(missing_ok=True)
            if attempt:
                raise
            time.sleep(2)
    return dst


def fetch_labels() -> None:
    for name in LABEL_URLS:
        _label(name)


def _label(name: str) -> Path:
    """Path of one label file. It is fetched first if it is not on disk (data/ is not in the repo)."""
    path = ANN / name
    if not path.exists():
        _get(LABEL_URLS[name], path)
    return path


@lru_cache(maxsize=None)
def _json(name: str):
    return json.loads(_label(name).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _records() -> dict:
    return {r["recording_id"]: r for r in _json("error_annotations.json")}


@lru_cache(maxsize=None)
def _durations() -> dict:
    with open(_label("video_information.csv"), encoding="utf-8", newline="") as f:
        return {row["recording_id"]: float(row["duration(sec)"]) for row in csv.DictReader(f)}


def _record(recording_id: str) -> dict:
    if not isinstance(recording_id, str) or recording_id not in _records():
        raise KeyError(f"unknown CaptainCook4D recording id: {recording_id!r}")
    return _records()[recording_id]


def _text(desc: str) -> str:
    """'Coat -Coat a cup with spray' -> 'Coat a cup with spray'. 'Add-1 egg' -> 'Add 1 egg'."""
    verb, _, rest = desc.partition("-")
    verb, rest = verb.strip(), rest.strip()
    return rest if verb.lower() in rest.lower() else f"{verb} {rest}"


@lru_cache(maxsize=None)
def recipe_steps(activity_id: int) -> tuple:
    """Step ids of one recipe, in the order of its reference recording.

    A CaptainCook4D recipe is a task graph, so it has no single official order. The reference
    is the lowest-numbered normal recording of the recipe with every step done, in time order.
    """
    full = [r for r in _records().values()
            if r["activity_id"] == activity_id and not r["is_error"]
            and all(s["start_time"] >= 0 for s in r["step_annotations"])]
    ref = min(full, key=lambda r: int(r["recording_id"].split("_")[1]))
    return tuple(s["step_id"] for s in sorted(ref["step_annotations"], key=lambda s: s["start_time"]))


def load(recording_id: str) -> dict:
    """One recording as a checklist with ground truth.

    steps follow recipe_steps(), which is the reference order and often not the order this
    recording was done in. A skipped step has start None, end None and skipped True. Three
    recipes (activity ids 2, 10, 20) repeat a step, so their checklists hold the same text more
    than once; repeats are matched to labels in time order.
    label_issues counts things that make the labels unsafe as ground truth: a recipe step with
    no label, a label left over after matching, a description that disagrees with the step
    table, or a step tagged "Missing Step" that still has a time.
    """
    rec = _record(recording_id)
    names = _json("step_idx_description.json")
    # Labels of this recording per step id: done ones first in time order, skipped ones last.
    segs: dict = {}
    issues = 0
    for s in sorted(rec["step_annotations"], key=lambda s: (s["start_time"] < 0, s["start_time"])):
        segs.setdefault(s["step_id"], []).append(s)
        tagged = any(e["tag"] == MISSING for e in s.get("errors", []))
        issues += (s["description"] != names[str(s["step_id"])]) + (tagged and s["start_time"] >= 0)
    steps = []
    for sid in recipe_steps(rec["activity_id"]):
        s = segs[sid].pop(0) if segs.get(sid) else None
        issues += s is None
        done = s is not None and s["start_time"] >= 0
        steps.append({"step_id": sid, "text": _text(names[str(sid)]),
                      "start": s["start_time"] if done else None,
                      "end": s["end_time"] if done else None,
                      "skipped": not done,
                      "errors": [e["tag"] for e in s.get("errors", [])] if s else []})
    issues += sum(len(v) for v in segs.values())
    return {"id": recording_id,
            "recipe": _json("complete_step_annotations.json")[recording_id]["activity_name"],
            "is_error": rec["is_error"],
            "duration": _durations()[recording_id],
            "url": _json("download_links.json")[recording_id].get("gopro_360p"),
            "label_issues": int(issues),
            "steps": steps}


def candidates() -> dict:
    """Pick N_EACH recordings with a skipped step and N_EACH normal ones, and write OUT."""
    pools: dict = {"missing": [], "normal": []}
    for rid in sorted(_records(), key=lambda r: hashlib.sha1(r.encode("ascii")).hexdigest()):
        rec = load(rid)
        if rec["label_issues"] or not rec["url"]:
            continue
        n_skipped = sum(s["skipped"] for s in rec["steps"])
        row = {"id": rid, "recipe": rec["recipe"], "url": rec["url"], "n_steps": len(rec["steps"]),
               "n_skipped": n_skipped, "duration_s": rec["duration"],
               "est_mb": round(rec["duration"] * MB_PER_S)}
        if any(s["skipped"] and MISSING in s["errors"] for s in rec["steps"]):
            pools["missing"].append(row)
        elif not rec["is_error"] and n_skipped == 0:
            pools["normal"].append(row)
    picked = pools["missing"][:N_EACH] + pools["normal"][:N_EACH]
    out = {
        "comment": ("Chosen by a fixed rule before any model output existed. Eligible = recordings with a "
                    "GoPro 360p link and label_issues == 0 (see eval/cc4d.py). missing = at least one step "
                    f"tagged '{MISSING}' with no time. normal = is_error false and no skipped step. Each pool "
                    f"is sorted by the hex sha1 of the recording id and the first {N_EACH} are taken. "
                    "est_mb is duration_s times a measured bitrate. Regenerate with: python -m eval.cc4d candidates"),
        "source": ("CaptainCook4D. Data under Apache License 2.0 per the project site; "
                   "annotations repo LICENSE file is MIT."),
        "n_eligible_missing": len(pools["missing"]),
        "n_eligible_normal": len(pools["normal"]),
        "est_total_mb": sum(r["est_mb"] for r in picked),
        "total_minutes": round(sum(r["duration_s"] for r in picked) / 60, 1),
        "missing": pools["missing"][:N_EACH],
        "normal": pools["normal"][:N_EACH],
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="ascii", newline="\n")
    return out


def fetch(recording_id: str) -> Path:
    """Download the GoPro 360p video of one recording unless it is already on disk."""
    _record(recording_id)                      # rejects an unknown id before any path is built
    url = _json("download_links.json")[recording_id].get("gopro_360p")
    if not url:
        raise ValueError(f"recording {recording_id} has no GoPro 360p link")
    dst = VIDEOS / f"{recording_id}_360p.mp4"
    if not dst.exists():
        _get(url, dst)
    return dst


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "labels":
        fetch_labels()
        print("labels in", ANN)
    elif cmd == "candidates":
        c = candidates()
        print(f"wrote {OUT.name}: {len(c['missing'])} missing of {c['n_eligible_missing']} eligible, "
              f"{len(c['normal'])} normal of {c['n_eligible_normal']} eligible, "
              f"about {c['est_total_mb']} MB, {c['total_minutes']} minutes of video")
    elif cmd in ("show", "fetch") and len(sys.argv) > 2:
        try:
            if cmd == "show":
                print(json.dumps(load(sys.argv[2]), indent=1))
            else:
                t0 = time.time()
                p = fetch(sys.argv[2])
                print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.1f} s")
        except (KeyError, ValueError) as e:
            sys.exit(str(e.args[0]))
    else:
        print(__doc__)
