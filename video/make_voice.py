"""Voiceover for the demo video, spoken by Gemini text-to-speech.

One WAV per script line: video/out/vo_01.wav ... vo_09.wav (48 kHz, mono, 16-bit).
The silence before the first word and after the last is cut to 0.1 s and every file is
scaled to the same mean level (-19 dB, peak never above -1 dB), so the files can be laid on
a timeline as they are. The words and the pace are the API's own.
Durations go to video/out/vo_durations.json, the full check to video/out/vo_report.json.

Call shape (Interactions API, v1beta, from the speech generation guide):
    client.interactions.create(
        model="gemini-3.8-flash-tts",
        input=[{"type": "user_input", "content": [{"type": "text", "text": LINE,
                "annotations": [{"type": "speech_metadata", "style": STYLE}]}]}],
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": VOICE}]})
The reply carries a complete 24 kHz mono WAV in interaction.output_audio.data (base64).

Each line gets a 60 second timeout, one retry on the same model, then one try on the
lite model. The same voice and the same style are used for every line.

Run from the project root:
    PYTHONPATH=. python video/make_voice.py            all nine lines
    PYTHONPATH=. python video/make_voice.py --only 3,5  redo some lines, keep the rest
    PYTHONPATH=. python video/make_voice.py --check     no API calls, measure the files
    PYTHONPATH=. python video/make_voice.py --refinish  no API calls, trim and level again
    --style "..." changes the delivery note, --out DIR writes somewhere else than video/out.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
import time
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clock import decide, media, models

OUT = Path(__file__).resolve().parent / "out"

MODEL = "gemini-3.8-flash-tts"
MODEL_LITE = "gemini-3.8-flash-lite-tts"
VOICE = "Charon"
# Pace is set through the style note; the API has no speed field. Nine-line totals, one
# take each (the first was measured before trimming, the other two after):
#   "calm, plain and even, natural conversational pace"                      75.5 s
#   "calm and plain, brisk steady pace, short pauses between sentences"      60.6 s
#   "calm and plain, quick steady pace, very short pauses between sentences" 56.1 s
# The last one leaves room for a breath between lines inside one minute. The 60.6 s take
# is kept in video/out/alt_60s (same voice, same code path) in case it sounds better.
STYLE = "calm and plain, quick steady pace, very short pauses between sentences"
TIMEOUT_S = 60
MIN_S, MAX_S = 1.5, 16.0
PAD_S = 0.10      # silence kept before the first word and after the last
SILENCE = 0.01    # samples under 1 percent of full scale (-40 dB) count as silence
MEAN_DB = -19.0   # every file is scaled to this mean level ...
PEAK_DB = -1.0    # ... unless that would push its loudest sample above this

LINES = [
    "This order shipped without the watch. The packing video already knew.",
    "Sellers film every order. Nobody watches the footage until a customer says an item is missing.",
    "Discovery Clock checks a video against a checklist. The first read takes a few seconds: "
    "one call to the Gemini Decisions API, asking seventy yes or no questions about when each "
    "step started.",
    "Then three Gemini 4 calls write their own timelines. Bayes' rule weighs each witness by "
    "how reliable it measured, and gives every step a probability and a start time.",
    "On 518 held-out clips from DeepMind's Perception Test, it catches skipped steps as well "
    "as the best single Gemini call.",
    "What it adds is odds that hold up. Of the steps it rated above 80 percent, 97 percent "
    "had happened.",
    "People try to fake a step. Version two fell for it in four of six runs. With a completion "
    "rule, version three fell for it in two of fourteen.",
    "It still makes mistakes, and the README lists them. So it decides nothing. It tells a "
    "person which four seconds to watch.",
    "Discovery Clock. Built today at the Gemini 4 NYC hackathon.",
]


def _speak(client, model: str, text: str, style: str) -> bytes:
    """One speech call. Returns the WAV bytes the API sent."""
    res = client.interactions.create(
        model=model,
        input=[{"type": "user_input", "content": [{
            "type": "text", "text": text,
            "annotations": [{"type": "speech_metadata", "style": style}],
        }]}],
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": VOICE}]},
        timeout=TIMEOUT_S,
    )
    audio = getattr(res, "output_audio", None)
    data = getattr(audio, "data", None)
    if not data:
        raise RuntimeError("reply had no audio")
    raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
    if raw[:4] != b"RIFF":
        raise RuntimeError(f"reply audio is not a WAV file (mime {getattr(audio, 'mime_type', None)})")
    return raw


def _to_48k(src: Path, dst: Path) -> None:
    res = media._run(["-y", "-loglevel", "error", "-i", str(src),
                      "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)])
    if res.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed for {src.name}: {res.stderr[-300:]}")


def _trim_and_level(path: Path) -> None:
    """Cut leading and trailing silence down to PAD_S and scale to MEAN_DB, in place."""
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        pcm = array("h")
        pcm.frombytes(w.readframes(w.getnframes()))
    if sys.byteorder == "big":
        pcm.byteswap()
    floor = int(SILENCE * 32767)
    first = next((i for i, v in enumerate(pcm) if abs(v) > floor), None)
    if first is None:
        raise RuntimeError(f"{path.name} is silent")
    last = next(i for i in range(len(pcm) - 1, -1, -1) if abs(pcm[i]) > floor)
    pad = int(PAD_S * rate)
    part = pcm[max(0, first - pad):min(len(pcm), last + 1 + pad)]
    rms = math.sqrt(sum(v * v for v in part) / len(part))
    gain = min(10 ** (MEAN_DB / 20) * 32768 / rms,
               10 ** (PEAK_DB / 20) * 32767 / max(abs(v) for v in part))
    out = array("h", (max(-32768, min(32767, round(v * gain))) for v in part))
    if sys.byteorder == "big":
        out.byteswap()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(out.tobytes())


def make_line(client, n: int, text: str, style: str, out: Path) -> dict:
    """Generate line n (1-based). Tries MODEL, MODEL again, then MODEL_LITE."""
    dst = out / f"vo_{n:02d}.wav"
    tmp = out / f"vo_{n:02d}.api.wav"
    errors = []
    for attempt, model in enumerate([MODEL, MODEL, MODEL_LITE]):
        t0 = time.time()
        try:
            tmp.write_bytes(_speak(client, model, text, style))
            _to_48k(tmp, dst)
            _trim_and_level(dst)
            tmp.unlink(missing_ok=True)
            return {"file": dst.name, "ok": True, "model": model, "attempt": attempt,
                    "seconds": round(time.time() - t0, 2), "errors": errors}
        except Exception as e:  # every failure moves on to the next try
            errors.append(f"{model}: {models._error_text(e)}")
    tmp.unlink(missing_ok=True)
    return {"file": dst.name, "ok": False, "model": None, "errors": errors}


def measure(path: Path) -> dict:
    """Exact duration from the WAV header and levels from ffmpeg volumedetect."""
    with wave.open(str(path), "rb") as w:
        rate, channels, frames = w.getframerate(), w.getnchannels(), w.getnframes()
    out = media._run(["-hide_banner", "-i", str(path), "-af", "volumedetect",
                      "-f", "null", "-"]).stderr
    peak = re.search(r"max_volume:\s*(-?[\d.]+) dB", out)
    mean = re.search(r"mean_volume:\s*(-?[\d.]+) dB", out)
    return {"duration_s": round(frames / rate, 3), "rate": rate, "channels": channels,
            "peak_db": float(peak.group(1)) if peak else None,
            "mean_db": float(mean.group(1)) if mean else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="line numbers to regenerate, e.g. 3,5")
    ap.add_argument("--style", default=STYLE, help="delivery note sent with every line")
    ap.add_argument("--check", action="store_true", help="measure existing files, no API calls")
    ap.add_argument("--refinish", action="store_true", help="trim and level existing files again")
    ap.add_argument("--out", default=str(OUT), help="output folder")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:  # what an earlier run recorded: kept for files this run does not regenerate
        old = json.loads((out / "vo_report.json").read_text(encoding="ascii"))
    except (OSError, ValueError):
        old = {}
    style = args.style if not (args.check or args.refinish) else old.get("style", args.style)

    todo = [int(x) for x in args.only.split(",") if x.strip()] or list(range(1, len(LINES) + 1))
    made: dict[str, dict] = {}
    if args.refinish:
        for path in sorted(out.glob("vo_[0-9][0-9].wav")):
            _trim_and_level(path)
    elif not args.check:
        client = decide.client(TIMEOUT_S * 1000)  # v1beta: the speech models are not on v1
        with ThreadPoolExecutor(max_workers=3) as pool:
            for row in pool.map(lambda n: make_line(client, n, LINES[n - 1], style, out), todo):
                made[row["file"]] = row
                print(json.dumps(row))

    durations, report, problems = {}, {}, []
    for n in range(1, len(LINES) + 1):
        name = f"vo_{n:02d}.wav"
        path = out / name
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        m = measure(path)
        durations[name] = m["duration_s"]
        how = made.get(name) or old.get("files", {}).get(name, {})
        report[name] = {**m, "text": LINES[n - 1],
                        **{k: how[k] for k in ("model", "attempt", "errors") if k in how}}
        if not MIN_S <= m["duration_s"] <= MAX_S:
            problems.append(f"{name}: {m['duration_s']} s is outside {MIN_S} to {MAX_S} s")
        if m["peak_db"] is None or m["peak_db"] < -30.0:
            problems.append(f"{name}: peak {m['peak_db']} dB, too quiet")
        if (m["rate"], m["channels"]) != (48000, 1):
            problems.append(f"{name}: {m['rate']} Hz, {m['channels']} ch")
    problems += [f"{k}: every try failed: {v['errors']}" for k, v in made.items() if not v["ok"]]
    total = round(sum(durations.values()), 3)

    (out / "vo_durations.json").write_text(json.dumps(durations, indent=2) + "\n", encoding="ascii")
    (out / "vo_report.json").write_text(json.dumps(
        {"voice": VOICE, "style": style, "total_s": total, "files": report,
         "problems": problems}, indent=2) + "\n", encoding="ascii")
    print(json.dumps(durations, indent=2))
    for name, r in report.items():
        print(f"{name}  {r['duration_s']:6.2f} s  peak {r['peak_db']} dB  mean {r['mean_db']} dB")
    print(f"total {total} s")
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
