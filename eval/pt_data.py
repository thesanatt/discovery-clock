"""Held-out evaluation set: DeepMind Perception Test, valid split (CC BY 4.0).

Labels are read from data/pt/. Videos are pulled one at a time out of the remote
zip with HTTP range requests, so the 75 GB archive is never downloaded whole.

    python -m eval.pt_data split
    python -m eval.pt_data fetch --split tune --limit 4
    python -m eval.pt_data check --split tune

Run from the project root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from remotezip import RemoteZip

from clock import media

ROOT = Path(__file__).resolve().parent.parent
PT = ROOT / "data" / "pt"
ACTIONS = PT / "challenge_action_localisation_valid.json"
SOUNDS = PT / "challenge_sound_localisation_valid.json"
RAW = PT / "valid"
SMALL = PT / "valid_small"
SPLITS = Path(__file__).resolve().parent / "splits.json"
ZIP_URL = "https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip"
TIMEOUT = (15, 60)  # connect and read timeouts in seconds, handed to requests by remotezip
MIN_ACTIONS = 4
N_TUNE, N_TEST = 30, 120
N_TEST2 = 100    # a second held-out split, the next ids in the same sha1 order, added after the first test was scored

_local = threading.local()
_lock = threading.Lock()


@lru_cache(maxsize=None)
def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- splits

def eligible() -> list[str]:
    """Ids with at least MIN_ACTIONS action segments and no cup game, in sha1(id) order."""
    ids = [vid for vid, v in _load(ACTIONS).items()
           if len(v["action_localisation"]) >= MIN_ACTIONS and v["metadata"]["is_cup_game"] == 0]
    return sorted(ids, key=lambda vid: hashlib.sha1(vid.encode("ascii")).hexdigest())


def make_splits() -> dict:
    ids = eligible()
    return {
        "comment": ("Chosen by a fixed rule before any model output existed. Eligible = Perception Test "
                    f"valid videos with at least {MIN_ACTIONS} action segments and is_cup_game == 0. "
                    "Eligible ids are sorted by the hex sha1 of the video id. "
                    f"The first {N_TUNE} are tune and the next {N_TEST} are test. "
                    "Thresholds and prompts are set on tune only. Regenerate with: python -m eval.pt_data split"),
        "source": "DeepMind Perception Test, valid split, CC BY 4.0",
        "labels": ACTIONS.name,
        "n_eligible": len(ids),
        "tune": ids[:N_TUNE],
        "test": ids[N_TUNE:N_TUNE + N_TEST],
    }


def write_splits() -> dict:
    data = make_splits()
    SPLITS.write_text(json.dumps(data, indent=1) + "\n", encoding="ascii", newline="\n")
    return data


def split_ids(split: str) -> list[str]:
    """Ids of "tune" or "test" from splits.json. Fails if the file no longer matches the rule.

    "test2" is the next N_TEST2 ids in the same order: a fresh held-out set for changes made
    after the first test split had been scored.
    """
    if split not in ("tune", "test", "test2", "test3"):
        raise ValueError(f'split must be "tune", "test", "test2" or "test3", got {split!r}')
    if split == "test2":
        return eligible()[N_TUNE + N_TEST:N_TUNE + N_TEST + N_TEST2]
    if split == "test3":   # the 100 ids after test2: for the witness settings changed after the field takes
        return eligible()[N_TUNE + N_TEST + N_TEST2:N_TUNE + N_TEST + 2 * N_TEST2]
    rule = make_splits()
    if not SPLITS.exists():
        write_splits()
    data = json.loads(SPLITS.read_text(encoding="ascii"))
    if data["tune"] != rule["tune"] or data["test"] != rule["test"]:
        raise ValueError("eval/splits.json does not match the sha1 rule; it must not be edited by hand")
    return data[split]


# ---------------------------------------------------------------- download

def _members() -> dict[str, tuple[str, int]]:
    """Video id -> (member name, size in bytes), read from the zip's file list."""
    for attempt in (1, 2):
        try:
            with RemoteZip(ZIP_URL, timeout=TIMEOUT) as z:
                return {Path(i.filename).stem: (i.filename, i.file_size)
                        for i in z.infolist() if i.filename.endswith(".mp4")}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    return {}


def _zip() -> RemoteZip:
    """One open zip per worker thread. A RemoteZip is not safe to share between threads."""
    if getattr(_local, "zip", None) is None:
        _local.zip = RemoteZip(ZIP_URL, timeout=TIMEOUT)
    return _local.zip


def _drop_zip() -> None:
    z, _local.zip = getattr(_local, "zip", None), None
    if z is not None:
        try:
            z.close()
        except Exception:
            pass


def _fetch_one(vid: str, member: tuple[str, int] | None) -> dict:
    """Download one video and write its 360p copy. Never raises; the row says what happened."""
    raw, small = RAW / f"{vid}.mp4", SMALL / f"{vid}.mp4"
    row = {"id": vid, "status": "failed", "raw_bytes": 0, "small_bytes": 0, "seconds": 0.0, "error": None}
    t0 = time.time()
    if member is None:
        row["error"] = "not in the zip file list"
        return row
    name, size = member
    part = raw.with_suffix(".part")
    tmp = small.with_name(f"{vid}.tmp.mp4")  # ffmpeg picks the format from the .mp4 ending
    worked = False
    for attempt in (1, 2):
        try:
            if not (raw.exists() and raw.stat().st_size == size):
                with _zip().open(name) as src, open(part, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)  # zipfile checks the CRC at the end of the read
                if part.stat().st_size != size:
                    raise IOError(f"got {part.stat().st_size} bytes, expected {size}")
                part.replace(raw)
                small.unlink(missing_ok=True)  # a new download makes any old small copy stale
                worked = True
            if not (small.exists() and small.stat().st_size > 0):  # an empty small copy is redone
                media.prep(raw, tmp)
                tmp.replace(small)
                worked = True
            row.update(status="ok" if worked else "skip", error=None,
                       raw_bytes=raw.stat().st_size, small_bytes=small.stat().st_size)
            break
        except Exception as e:  # one retry, on a fresh connection
            row["error"] = f"{type(e).__name__}: {e}"[:200]
            _drop_zip()
            for junk in (part, tmp):  # no half-written files are left behind
                try:
                    junk.unlink(missing_ok=True)
                except OSError:
                    pass
            if attempt == 1:
                time.sleep(2)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def fetch(ids: list[str], workers: int = 4) -> list[dict]:
    """Download ids into data/pt/valid and write 360p copies into data/pt/valid_small.

    Resumable: a raw file with the size the zip lists is kept, and so is a non-empty small copy.
    Prints one line per file and returns one status row per distinct id.
    """
    if isinstance(ids, str):
        ids = [ids]
    ids = list(dict.fromkeys(ids))  # a repeated id would put two workers on one file
    workers = max(1, workers)
    RAW.mkdir(parents=True, exist_ok=True)
    SMALL.mkdir(parents=True, exist_ok=True)
    members = _members()
    need = sum(members[v][1] for v in ids if v in members)
    print(f"fetch: {len(ids)} files, {need / 1e6:.1f} MB in the zip, {workers} workers", flush=True)
    done = [0]

    def job(vid: str) -> dict:
        row = _fetch_one(vid, members.get(vid))
        with _lock:
            done[0] += 1
            print(f"[{done[0]:>3}/{len(ids)}] {vid:<12} {row['status']:<6} raw {row['raw_bytes'] / 1e6:6.1f} MB  "
                  f"small {row['small_bytes'] / 1e6:5.2f} MB  {row['seconds']:5.1f} s"
                  + (f"  {row['error']}" if row["status"] == "failed" else ""), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(job, ids))
    good = [r for r in rows if r["status"] != "failed"]
    n = max(1, len(good))
    print(f"done: {sum(r['status'] == 'ok' for r in rows)} fetched, {sum(r['status'] == 'skip' for r in rows)} "
          f"already present, {len(rows) - len(good)} failed. Average raw {sum(r['raw_bytes'] for r in good) / n / 1e6:.1f} MB, "
          f"average small {sum(r['small_bytes'] for r in good) / n / 1e6:.2f} MB.", flush=True)
    return rows


# ---------------------------------------------------------------- manifest

def label_duration(vid: str) -> float:
    """Length in seconds according to the label file: num_frames / frame_rate."""
    m = _load(ACTIONS)[vid]["metadata"]
    return m["num_frames"] / m["frame_rate"]


@lru_cache(maxsize=None)
def _duration(path: str, size: int, mtime_ns: int) -> float:
    """Measured length of a file. Cached per file version, so repeat manifest loads skip ffmpeg."""
    return media.duration_s(path)


def _segments(rows: list[dict]) -> list[dict]:
    """Label rows as {label, start, end} in seconds, ordered by start time."""
    out = [{"label": r["label"], "start": round(r["timestamps"][0] / 1e6, 3),
            "end": round(r["timestamps"][1] / 1e6, 3)} for r in rows]
    return sorted(out, key=lambda s: (s["start"], s["end"]))


def load_manifest(split: str) -> list[dict]:
    """Rows for "tune", "test" or "all", limited to videos whose small copy is on disk.

    Each row: {id, split, path (360p copy), duration (seconds, measured from that copy),
    actions: [{label, start, end}], sounds: [{label, start, end}]}.
    """
    acts, snds = _load(ACTIONS), _load(SOUNDS)
    out = []
    for name in (("tune", "test") if split == "all" else (split,)):
        for vid in split_ids(name):
            small = SMALL / f"{vid}.mp4"
            if not small.exists():
                continue
            try:
                st = small.stat()
                dur = _duration(str(small), st.st_size, st.st_mtime_ns)
            except (OSError, ValueError) as e:  # one bad file must not take the whole split down
                print(f"load_manifest: skipped {vid}, small copy not readable ({e})", file=sys.stderr, flush=True)
                continue
            out.append({"id": vid, "split": name, "path": str(small), "duration": dur,
                        "actions": _segments(acts[vid]["action_localisation"]),
                        "sounds": _segments(snds[vid]["sound_localisation"])})
    return out


def check(split: str, tol_s: float = 0.5) -> int:
    """Compare each measured duration with num_frames / frame_rate.

    Returns the number of bad rows, or 1 when no file of the split is on disk.
    """
    bad = 0
    rows = load_manifest(split)
    for r in rows:
        want = label_duration(r["id"])
        last = max(s["end"] for s in r["actions"] + r["sounds"])
        ok = abs(r["duration"] - want) <= tol_s and last <= r["duration"] + tol_s
        bad += not ok
        print(f"{r['id']:<12} file {r['duration']:6.2f} s  labels {want:6.2f} s  diff {r['duration'] - want:+5.2f}  "
              f"last label ends {last:6.2f}  actions {len(r['actions']):>2}  sounds {len(r['sounds']):>2}  "
              f"{'ok' if ok else 'MISMATCH'}")
    print(f"{len(rows)} of {len(split_ids(split))} {split} videos on disk, {bad} outside {tol_s} s")
    return bad if rows else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Perception Test evaluation data: splits, download, checks.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split", help="write eval/splits.json from the fixed sha1 rule")
    f = sub.add_parser("fetch", help="download a split and write 360p copies")
    f.add_argument("--split", choices=["tune", "test", "test2", "test3"], required=True)
    f.add_argument("--limit", type=int, default=None, help="only the first N ids of the split")
    f.add_argument("--workers", type=int, default=4)
    c = sub.add_parser("check", help="compare file durations with the labels")
    c.add_argument("--split", choices=["tune", "test"], required=True)
    args = ap.parse_args(argv)
    if args.cmd == "split":
        data = write_splits()
        print(f"{SPLITS}: {data['n_eligible']} eligible, {len(data['tune'])} tune, {len(data['test'])} test")
        return 0
    if args.cmd == "fetch":
        if args.limit is not None and args.limit < 0:
            ap.error("--limit must be 0 or more")
        ids = split_ids(args.split)[:args.limit]
        try:
            rows = fetch(ids, workers=args.workers)
        except Exception as e:  # the zip listing failed twice; no file was touched
            print(f"fetch stopped before any download: {type(e).__name__}: {e}"[:300], flush=True)
            return 1
        return 1 if any(r["status"] == "failed" for r in rows) else 0
    return 1 if check(args.split) else 0


if __name__ == "__main__":
    raise SystemExit(main())
