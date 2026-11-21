"""Assemble the one-minute demo video from real material.

Inputs (made by the other scripts in this folder, or recorded):
  video/assets/app_1600x900.png   screenshot of the running app showing a real result
  video/assets/phone.mp4          the phone recording that result came from
  video/out/s4_math.mp4, s6_results.mp4, s7_fooled.mp4, s8_end.mp4   (video/make_visuals.py)
  video/out/vo_01.wav ... vo_09.wav                                  (video/make_voice.py, or a human recording)

  python video/assemble.py [--voice-dir video/out] [--out video/out/discovery_clock_demo.mp4]

Each voice line sets the length of its picture. Everything shown is the real app or real data.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clock import media  # noqa: E402

FF = media.FFMPEG
OUT = ROOT / "video" / "out"
ASSETS = ROOT / "video" / "assets"
W, H, FPS = 1280, 720, 30
PAPER = "0xefede6"
INK = "0x14181b"
BAD = "0xa1241a"
MONO = "C\\:/Windows/Fonts/consolab.ttf"
SANS = "C\\:/Windows/Fonts/segoeui.ttf"
SANSB = "C\\:/Windows/Fonts/seguisb.ttf"
GAP = 0.30          # silence after each line
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an"]


def run(args: list[str]) -> None:
    res = subprocess.run([FF, "-y", "-loglevel", "error", *args], capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-600:])


def text(s: str) -> str:
    """Escape a caption for ffmpeg drawtext."""
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "").replace(",", "\\,").replace("%", "\\%")


def caption(s: str, y: int, size: int = 30, font: str = SANS, color: str = INK, x: str = "60") -> str:
    return f"drawtext=fontfile='{font}':text='{text(s)}':x={x}:y={y}:fontsize={size}:fontcolor={color}"


def still(src: Path, dst: Path, dur: float, crop: str | None = None, zoom_to: float = 1.0, extra: str = "") -> None:
    """A still image shown for dur seconds, optionally cropped, with a slow push-in."""
    frames = int(round(dur * FPS))
    vf = []
    if crop:
        vf.append(f"crop={crop}")
    vf.append("scale=2560:1440:force_original_aspect_ratio=decrease,pad=2560:1440:(ow-iw)/2:(oh-ih)/2:color=" + PAPER)
    if zoom_to > 1.0:
        step = (zoom_to - 1.0) / max(frames, 1)
        vf.append(f"zoompan=z='1+{step:.6f}*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS}")
    else:
        vf.append(f"scale={W}:{H}")
    if extra:
        vf.append(extra)
    run(["-loop", "1", "-t", f"{dur:.3f}", "-i", str(src), "-vf", ",".join(vf), "-t", f"{dur:.3f}", *ENC, str(dst)])


def retime(src: Path, dst: Path, dur: float) -> None:
    """Stretch or squeeze a clip to exactly dur seconds."""
    have = media.duration_s(src)
    run(["-i", str(src), "-vf", f"setpts={dur / have:.6f}*PTS,scale={W}:{H},fps={FPS}", "-t", f"{dur:.3f}", *ENC, str(dst)])


def phone(src: Path, dst: Path, dur: float, start: float, length: float, lines: list[tuple[str, int, int, str, str]]) -> None:
    """Phone footage on the left (sped up to fit), captions on the right, on a paper background."""
    speed = length / dur
    cap = ",".join(caption(s, y, size, font, color, x="470") for s, y, size, font, color in lines)
    fc = (f"color=c={PAPER}:s={W}x{H}:r={FPS}:d={dur:.3f}[bg];"
          f"[0:v]trim=start={start:.2f}:duration={length:.2f},setpts=(PTS-STARTPTS)/{speed:.5f},"
          f"scale=-2:640,fps={FPS}[v];[bg][v]overlay=60:40:shortest=1,{cap}[o]")
    run(["-i", str(src), "-filter_complex", fc, "-map", "[o]", "-t", f"{dur:.3f}", *ENC, str(dst)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-dir", default=str(OUT))
    ap.add_argument("--out", default=str(OUT / "discovery_clock_demo.mp4"))
    args = ap.parse_args()
    vdir = Path(args.voice_dir)
    vo = [vdir / f"vo_{i:02d}.wav" for i in range(1, 10)]
    for f in vo:
        if not f.exists():
            sys.exit(f"missing voice line {f}")
    d = [media.duration_s(f) + GAP for f in vo]          # picture length for each line
    app = ASSETS / "app_1600x900.png"
    footage = ASSETS / "phone.mp4"
    res = json.load(open(ASSETS / "take4_result.json", encoding="ascii"))["fused"]
    steps = res["steps"]
    skipped = next(s for s in steps if s["status"] == "not_found")
    zip_step = next(s for s in steps if s["text"].startswith("zip"))
    seg = OUT / "seg"
    seg.mkdir(parents=True, exist_ok=True)

    # 1. cold open: the red row of the real app, slow push-in
    still(app, seg / "01.mp4", d[0], crop="1200:560:400:130", zoom_to=1.06)
    # 2. the phone recording that result came from
    phone(footage, seg / "02.mp4", d[1], 0.0, min(24.0, res["video"]["duration"] - 0.5), [
        ("PHONE RECORDING", 70, 26, MONO, INK),
        (f"{res['video']['duration']:.0f} seconds. One take.", 120, 34, SANSB, INK),
        ("Checklist typed by the seller:", 200, 28, SANS, INK),
        *[(f"{i + 1}. {s['text']}", 250 + 44 * i, 28, SANS, INK) for i, s in enumerate(steps)],
        (f"Step {skipped['index'] + 1} never happens.", 250 + 44 * len(steps) + 30, 34, SANSB, BAD),
    ])
    # 3. the whole result page
    still(app, seg / "03.mp4", d[2], zoom_to=1.0)
    # 4. how it works, animated from the same result
    retime(OUT / "s4_math.mp4", seg / "04.mp4", d[3])
    # 5 + 6. results on held-out clips
    retime(OUT / "s6_results.mp4", seg / "0506.mp4", d[4] + d[5])
    # 7. faked steps
    retime(OUT / "s7_fooled.mp4", seg / "07.mp4", d[6])
    # 8. "which four seconds to watch": the clip one row opens
    t = float(zip_step["time"])
    phone(footage, seg / "08.mp4", d[7], max(0.0, t - 1.5), 4.0, [
        ("THE CLIP ONE ROW OPENS", 70, 26, MONO, INK),
        (f"Step {zip_step['index'] + 1}: {zip_step['text']}", 130, 34, SANSB, INK),
        (f"starts at {t:.1f} s   P(done) {zip_step['p_done']:.2f}", 190, 30, MONO, INK),
        ("A person watches these four seconds", 290, 30, SANS, INK),
        ("and signs. The tool decides nothing.", 334, 30, SANS, INK),
    ])
    # 9. end card
    retime(OUT / "s8_end.mp4", seg / "09.mp4", d[8])

    order = ["01", "02", "03", "04", "0506", "07", "08", "09"]
    (seg / "list.txt").write_text("".join(f"file '{(seg / (n + '.mp4')).as_posix()}'\n" for n in order), encoding="ascii")
    run(["-f", "concat", "-safe", "0", "-i", str(seg / "list.txt"), "-c", "copy", str(seg / "picture.mp4")])

    # voice track: each line padded to its picture length
    inputs, parts = [], []
    for i, f in enumerate(vo):
        inputs += ["-i", str(f)]
        parts.append(f"[{i}:a]aresample=48000,apad=whole_dur={d[i]:.3f},atrim=0:{d[i]:.3f}[a{i}]")
    fc = ";".join(parts) + ";" + "".join(f"[a{i}]" for i in range(len(vo))) + f"concat=n={len(vo)}:v=0:a=1[a]"
    run([*inputs, "-filter_complex", fc, "-map", "[a]", "-c:a", "pcm_s16le", str(seg / "voice.wav")])
    run(["-i", str(seg / "picture.mp4"), "-i", str(seg / "voice.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
         "-shortest", "-movflags", "+faststart", args.out])
    total = media.duration_s(args.out)
    print(f"written {args.out}: {total:.1f} s; line lengths {[round(x, 1) for x in d]}")
    if total > 60.5:
        print("WARNING: over one minute. Shorten a line or raise the voice pace.")


if __name__ == "__main__":
    main()
