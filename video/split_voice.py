"""Split one spoken recording into the nine voice lines of the demo video.

Read all nine lines of video/make_voice.py into ONE recording, with a clear pause (about two
seconds) between lines. This script finds the pauses, cuts the recording at them, and writes
vo_01.wav ... vo_09.wav (48 kHz, mono, 16-bit, trimmed and leveled like the generated voice)
into a folder that video/assemble.py can use:

    python video/split_voice.py RECORDING [--out video/out/human] [--min-gap 1.0]
                                [--keep 1,2,4,5,...] [--fit 57.5] [--denoise]
    python video/assemble.py --voice-dir video/out/human --out demo.mp4

The pause detector adapts to the room: it measures loudness in 30 ms frames and puts the line
between "speech" and "pause" halfway (in dB) between the quiet and the loud frames, so a noisy
room does not need a hand-set threshold.

--keep picks which detected segments become lines 1 to 9, in order. Use it when a line was read
twice (say the recording has ten segments and the third is a bad take: --keep 1,2,4,5,6,7,8,9,10).
--fit T speeds every line up by one common factor (at most 1.15) so that the nine lines plus the
gaps between them last T seconds or less. Pitch is kept.
--denoise applies a high-pass at 80 Hz and ffmpeg's afftdn noise reduction before cutting.

Nothing here calls a model. The words are the speaker's own.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clock import media  # noqa: E402
from video import make_voice  # noqa: E402

FRAME_S = 0.03
PAD_S = 0.15
GAP_IN_VIDEO_S = 0.20      # video/assemble.py adds this after every line
MAX_SPEEDUP = 1.15
N_LINES = 9


def _ff(args: list[str]) -> None:
    res = subprocess.run([media.FFMPEG, "-y", "-loglevel", "error", *args],
                         capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-400:])


def _pcm(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        pcm = array("h")
        pcm.frombytes(w.readframes(w.getnframes()))
    if sys.byteorder == "big":
        pcm.byteswap()
    return pcm, rate


def segments(pcm: array, rate: int, min_gap_s: float) -> tuple[list[tuple[float, float]], dict]:
    """Speech segments as (start, end) seconds, split wherever a pause lasts min_gap_s or more."""
    import math
    n = int(FRAME_S * rate)
    db = []
    for i in range(0, len(pcm) - n + 1, n):
        chunk = pcm[i:i + n]
        rms = math.sqrt(sum(v * v for v in chunk) / n)
        db.append(20 * math.log10(max(rms, 1.0) / 32768))
    if not db:
        return [], {}
    ordered = sorted(db)
    quiet = ordered[int(0.10 * (len(ordered) - 1))]
    loud = ordered[int(0.90 * (len(ordered) - 1))]
    cut = (quiet + loud) / 2
    speech = [x > cut for x in db]
    # close holes shorter than the minimum pause, so breaths inside a line do not split it
    hole = int(round(min_gap_s / FRAME_S))
    out, start, last_on = [], None, None
    for i, on in enumerate(speech):
        if on:
            if start is None:
                start = i
            elif last_on is not None and i - last_on > hole:
                out.append((start, last_on))
                start = i
            last_on = i
    if start is not None:
        out.append((start, last_on))
    # drop blips: a cough or a click is not a line
    out = [(a, b) for a, b in out if (b - a + 1) * FRAME_S >= 0.6]
    segs = [(a * FRAME_S, (b + 1) * FRAME_S) for a, b in out]
    return segs, {"quiet_db": round(quiet, 1), "loud_db": round(loud, 1), "cut_db": round(cut, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--out", default=str(ROOT / "video" / "out" / "human"))
    ap.add_argument("--min-gap", type=float, default=1.0, help="shortest pause that separates two lines, seconds")
    ap.add_argument("--keep", default="", help="which detected segments to use as lines 1..9, e.g. 1,2,4,5,6,7,8,9,10")
    ap.add_argument("--fit", type=float, default=0.0, help="speed up so lines plus gaps last at most this many seconds")
    ap.add_argument("--denoise", action="store_true")
    args = ap.parse_args()

    src = Path(args.recording)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full = out / "_full.wav"
    af = ["-af", "highpass=f=80,afftdn=nf=-25"] if args.denoise else []
    _ff(["-i", str(src), "-vn", *af, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(full)])
    pcm, rate = _pcm(full)
    segs, levels = segments(pcm, rate, args.min_gap)
    print(json.dumps({"recording_s": round(len(pcm) / rate, 2), **levels, "segments": len(segs)}))
    for i, (a, b) in enumerate(segs, 1):
        print(f"  segment {i:2d}: {a:7.2f} to {b:7.2f} s  ({b - a:5.2f} s)")

    keep = [int(x) for x in args.keep.split(",") if x.strip()] or list(range(1, len(segs) + 1))
    if len(keep) != N_LINES or any(k < 1 or k > len(segs) for k in keep):
        print(f"PROBLEM: need exactly {N_LINES} segments, have {len(keep)}. "
              f"Use --keep to choose them, or --min-gap to split differently.")
        return 1

    total_len = len(pcm) / rate
    cuts = []
    for k in keep:
        a, b = segs[k - 1]
        cuts.append((max(0.0, a - PAD_S), min(total_len, b + PAD_S)))
    raw_total = sum(b - a for a, b in cuts) + N_LINES * GAP_IN_VIDEO_S
    speed = 1.0
    if args.fit and raw_total > args.fit:
        speed = (raw_total - N_LINES * GAP_IN_VIDEO_S) / (args.fit - N_LINES * GAP_IN_VIDEO_S)
        if speed > MAX_SPEEDUP:
            print(f"PROBLEM: fitting {raw_total:.1f} s into {args.fit} s needs a {speed:.2f}x speed-up, "
                  f"more than {MAX_SPEEDUP}. Read faster or cut a line.")
            return 1

    durations = {}
    for n, (a, b) in enumerate(cuts, 1):
        dst = out / f"vo_{n:02d}.wav"
        tempo = ["-af", f"atempo={speed:.4f}"] if speed > 1.0005 else []
        _ff(["-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(full), *tempo,
             "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)])
        make_voice._trim_and_level(dst)
        durations[dst.name] = make_voice.measure(dst)["duration_s"]
    full.unlink(missing_ok=True)
    total = round(sum(durations.values()) + N_LINES * GAP_IN_VIDEO_S, 2)
    (out / "vo_durations.json").write_text(json.dumps(durations, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"speed": round(speed, 3), "video_seconds": total, "durations": durations}))
    if total > 60.0:
        print(f"PROBLEM: the video would last {total} s. The form asks for one minute. Use --fit 59.5.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
