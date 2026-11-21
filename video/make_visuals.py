"""Render the four silent data clips of the demo video.

Run from the project root:
    PYTHONPATH=. .venv/Scripts/python.exe video/make_visuals.py            (render all, then check)
    PYTHONPATH=. .venv/Scripts/python.exe video/make_visuals.py s6_results (render one, then check)

Outputs go to video/out/ as 1280x720, 30 fps, H.264 yuv420p, no audio.

Every number on screen is read from a file in this repository:
    s4_math     video/assets/take4_result.json  (key "fused")
    s6_results  results/test4_report.json, and the clip counts of all results/test*_report.json
    s7_fooled   results/field.json, plus the two version 2 counts marked FIXED below
    s8_end      text only
Numbers are rounded half up, the same way the app prints them (JavaScript toFixed).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from clock.media import FFMPEG, duration_s  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "video" / "out"
FRAMES = OUT / "frames"

W, H, FPS = 1280, 720, 30

PAPER = "#efede6"
PANEL = "#faf9f5"
INK = "#14181b"
RULE = "#b8b4a8"
GREEN = "#0a5c41"
RED = "#a1241a"
BLUE = "#0d4a7c"
BAND = "#a9c3da"

_F = "C:/Windows/Fonts/"
MONO = font_manager.FontProperties(fname=_F + "consola.ttf")
MONO_B = font_manager.FontProperties(fname=_F + "consolab.ttf")
SANS = font_manager.FontProperties(fname=_F + "segoeui.ttf")
SANS_B = font_manager.FontProperties(fname=_F + "segoeuib.ttf")
SANS_SB = font_manager.FontProperties(fname=_F + "seguisb.ttf")

# FIXED: measured before version 3, so not in field.json (see the README section on version 3).
# Under version 2 the two fake-out takes (takes 6 and 7, three re-runs each) were read as
# done in 2 of 3 re-runs each, 4 of 6 in all.
V2_FAKE_DONE = 4
V2_FAKE_RUNS = 6
FAKE_STEPS = {"6": "AirPods", "7": "Hello Panda"}

WITNESS_LABELS = (("hydrogen4", "Hydrogen 4 fps"), ("hydrogen2", "Hydrogen 2 fps"), ("neon4", "Neon 4 fps"))
MATH_STEPS = ("zip the bag closed", "take off the watch and put it in the bag")


def fx(x: float, nd: int) -> str:
    """Round half up to nd decimals, as JavaScript toFixed does for these values."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def pct(share: float) -> str:
    return str((Decimal(repr(float(share))) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def ease(v: float) -> float:
    v = clamp01(v)
    return 1.0 - (1.0 - v) ** 3


class Canvas:
    """One figure addressed in video pixels: x right, y down, sizes in pixels."""

    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.z = 0

    def _z(self) -> int:
        # later calls draw on top of earlier ones
        self.z += 1
        return self.z

    def clear(self, bg: str = PAPER) -> None:
        self.ax.clear()
        self.z = 0
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.set_autoscale_on(False)
        self.ax.axis("off")
        self.fig.patch.set_facecolor(bg)

    def text(self, x, y, s, size, font=SANS, color=INK, ha="left", va="baseline") -> float:
        """Draw text with its baseline at y. Returns the x of its right edge in pixels."""
        t = self.ax.text(x, y, s, fontproperties=font, fontsize=size * 0.72, color=color, ha=ha, va=va,
                         zorder=self._z())
        return t.get_window_extent(self.fig.canvas.get_renderer()).x1

    def line(self, xs, ys, color=RULE, w=1.5) -> None:
        self.ax.plot(xs, ys, color=color, linewidth=w * 0.72, solid_capstyle="butt", solid_joinstyle="miter",
                     zorder=self._z())

    def rect(self, x, y, w, h, fc, ec="none", lw=0.0) -> None:
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw * 0.72,
                                    zorder=self._z()))

    def dots(self, xs, ys, color=INK, d=12.0) -> None:
        self.ax.plot(xs, ys, linestyle="none", marker="o", markersize=d * 0.72, markerfacecolor=color,
                     markeredgecolor=color, zorder=self._z())

    def fill(self, xs, ys, fc) -> None:
        self.ax.fill(xs, ys, facecolor=fc, edgecolor="none", zorder=self._z())

    def frame(self) -> bytes:
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].tobytes()


# ---------------------------------------------------------------- s4_math

SEG = 7.0
XL, XR, PR = 230, 1222, 1240  # time 0, last second of the video, panel right edge


def math_load():
    fused = json.loads((ROOT / "video" / "assets" / "take4_result.json").read_text())["fused"]
    by_text = {s["text"]: s for s in fused["steps"]}
    return fused, [by_text[t] for t in MATH_STEPS]


def math_state(t: float, n_grid: int):
    seg = 0 if t < SEG else 1
    u = t - seg * SEG
    nd = 0 if u < 0.3 else min(n_grid, int((u - 0.3) / 0.1) + 1)
    fit = round(clamp01((u - 1.8) / 0.8), 3)
    wit = tuple(round(ease((u - (2.8 + 0.35 * i)) / 0.25), 3) if u >= 2.8 + 0.35 * i else -1.0 for i in range(3))
    post = round(ease((u - 4.0) / 0.8), 3) if u >= 4.0 else -1.0
    return (seg, nd, fit, wit, post, u >= 4.9)


def math_draw(cv: Canvas, st, fused, steps) -> None:
    seg, nd, fit, wit, post, final = st
    s = steps[seg]
    grid = fused["grid"]
    dur = float(fused["video"]["duration"])

    def X(sec: float) -> float:
        return XL + (XR - XL) * sec / dur

    cv.clear(PAPER)
    cv.text(40, 42, "STEP %d OF %d" % (s["index"] + 1, len(fused["steps"])), 22, MONO_B)
    cv.text(40, 88, s["text"], 40, SANS_B)
    cv.line([40, 1240], [108, 108])

    ticks = [v for v in range(0, int(dur) + 1, 5)]

    # Decisions panel: one answer per time point, then the monotone fit
    PT, PB = 156, 366

    def Y(p: float) -> float:
        return PB - 12 - (PB - PT - 24) * p

    cv.text(XL, 140, "Decisions API: had it started by now?", 22, MONO)
    cv.rect(XL, PT, PR - XL, PB - PT, PANEL, RULE, 1.5)
    for v in ticks[1:]:
        cv.line([X(v), X(v)], [PT, PB], RULE, 1)
    for p, lab in ((0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")):
        cv.line([XL - 8, XL], [Y(p), Y(p)], RULE, 1.5)
        cv.text(XL - 14, Y(p) + 7, lab, 22, MONO, ha="right")
    cv.line([XL, PR], [Y(0.5), Y(0.5)], RULE, 1)
    if fit > 0:
        cut = grid[0] + fit * (grid[-1] - grid[0])
        xs, ys = [], []
        for i, g in enumerate(grid):
            if g <= cut:
                xs.append(X(g))
                ys.append(Y(s["fitted"][i]))
            else:
                g0, f0, f1 = grid[i - 1], s["fitted"][i - 1], s["fitted"][i]
                k = (cut - g0) / (g - g0)
                xs.append(X(cut))
                ys.append(Y(f0 + k * (f1 - f0)))
                break
        cv.line(xs, ys, BLUE, 4)
    if nd > 0:
        cv.dots([X(g) for g in grid[:nd]], [Y(v) for v in s["raw"][:nd]], INK, 12)

    # Three Gemini 4 witnesses: where each put the start, or "no"
    ST, SB = 384, 492
    cv.rect(XL, ST, PR - XL, SB - ST, PANEL, RULE, 1.5)
    for v in ticks[1:]:
        cv.line([X(v), X(v)], [ST, SB], RULE, 1)
    for i, (key, label) in enumerate(WITNESS_LABELS):
        c = ST + 18 + 36 * i
        cv.text(XL - 14, c + 7, label, 22, MONO, ha="right")
        w = wit[i]
        if w < 0:
            continue
        o = s["witnesses"][key]
        if o["happened"] and o["start"] is not None:
            x = X(o["start"])
            cv.rect(x - 3, c - 13 - (1 - w) * 30, 6, 26, BLUE)
            if w >= 0.999:
                lab = fx(o["start"], 1) + " s"
                if o["start"] > 0.8 * dur:
                    cv.text(x - 12, c + 7, lab, 22, MONO_B, BLUE, ha="right")
                else:
                    cv.text(x + 12, c + 7, lab, 22, MONO_B, BLUE)
        else:
            cv.text(XL + 14, c + 7, "no", 22, MONO_B, RED)

    # Combined posterior over the start time. Height = P(done) x density (the density has
    # peak 1), so a step that probably did not happen stays flat near the baseline.
    QT, QB = 510, 600
    cv.rect(XL, QT, PR - XL, QB - QT, PANEL, RULE, 1.5)
    for v in ticks[1:]:
        cv.line([X(v), X(v)], [QT, QB], RULE, 1)
    cv.text(XL - 14, 550, "All four", 22, MONO, ha="right")
    cv.text(XL - 14, 578, "combined", 22, MONO, ha="right")
    if post >= 0:
        k = (QB - QT - 10) * float(s["p_done"]) * post
        xs = [X(tau) for tau in s["taus"]]
        ys = [QB - k * max(0.0, min(1.0, d)) for d in s["density"]]
        cv.fill([xs[0]] + xs + [xs[-1]], [QB] + ys + [QB], BAND)
        cv.line(xs, ys, BLUE, 3)
    for v in ticks:
        cv.line([X(v), X(v)], [QB, QB + 8], RULE, 1.5)
        cv.text(X(v), 632, str(v), 22, MONO, ha="center")
    cv.text(XL - 26, 632, "seconds", 22, MONO, ha="right")

    if final:
        if s["status"] == "done" and s["time"] is not None:
            lo = s["time"] if s["lo"] is None else s["lo"]
            hi = dur if s["hi"] is None else s["hi"]
            cv.rect(X(lo), QB - 6, X(hi) - X(lo), 6, BLUE)
            cv.line([X(s["time"]), X(s["time"])], [QT + 4, QB], BLUE, 3)
            msg = "starts at %s s (%s to %s)   P(done) %s" % (fx(s["time"], 1), fx(lo, 1), fx(hi, 1),
                                                           fx(s["p_done"], 2))
            cv.text(XL, 692, msg, 34, MONO_B, GREEN)
        else:
            msg = "%s   P(done) %s" % (s["status"].replace("_", " "), fx(s["p_done"], 2))
            cv.text(XL, 692, msg, 34, MONO_B, RED)


# ---------------------------------------------------------------- s6_results

def results_load():
    rep = json.loads((ROOT / "results" / "test4_report.json").read_text())
    splits = sorted((ROOT / "results").glob("test*_report.json"))
    total = sum(json.loads(p.read_text())["clips"] for p in splits)
    single = rep["systems"]["Hydrogen 4 fps, one call"]
    fused = next(v for k, v in rep["systems"].items() if k.startswith("FUSED"))
    bins = {b["bin"]: b for b in rep["calibration"]["bins"]}
    return {"total": total, "n_splits": len(splits), "clips": rep["clips"], "single": single, "fused": fused,
            "hi": bins["0.8-1.0"], "lo": bins["0.0-0.2"]}


def results_state(t: float):
    bar = round(ease((t - 0.4) / 0.7), 3) if t >= 0.4 else -1.0
    cal = round(ease((t - 8.2) / 0.7), 3) if t >= 8.0 else -1.0
    return (bar, t >= 1.8, t >= 4.0, cal)


def results_draw(cv: Canvas, st, d) -> None:
    bar, same, beat2, cal = st
    words = {4: "four"}
    cv.clear(PAPER)
    cv.text(40, 66, "%d held-out clips, DeepMind Perception Test" % d["total"], 40, SANS_B)
    cv.text(40, 102, "%d clips across %s splits; this table is the fourth split, %d clips"
            % (d["total"], words[d["n_splits"]], d["clips"]), 22, SANS)
    cv.line([40, 1240], [122, 122])

    # beat 1: skip F1, two bars on a 0 to 1 track
    if bar >= 0:
        cv.text(40, 160, "SKIP F1, 0 TO 1", 22, MONO_B)
        x0, x1 = 400, 1040
        rows = (("Best single Gemini call", d["single"]["skip_f1"], BAND, 204),
                ("Discovery Clock", d["fused"]["skip_f1"], BLUE, 254))
        for label, v, color, c in rows:
            cv.text(40, c + 10, label, 28, SANS_SB)
            cv.rect(x0, c - 15, x1 - x0, 30, PANEL, RULE, 1.5)
            cv.rect(x0, c - 15, (x1 - x0) * float(v) * bar, 30, color)
            if bar >= 0.999:
                cv.text(x1 + 18, c + 10, fx(v, 2), 30, MONO_B)
    if same:
        cv.text(40, 316, "Same accuracy on yes or no.", 30, SANS_SB)

    # beat 2: only the fused system gives a probability
    if beat2:
        cv.line([40, 1240], [342, 342])
        cv.text(40, 388, "A single call gives no odds.", 30, SANS)
        xr = cv.text(40, 432, "Discovery Clock ranks real steps above skipped ones:", 30, SANS)
        cv.text(xr + 18, 432, "AUROC " + fx(d["fused"]["auroc"], 2), 34, MONO_B, BLUE)

    # beat 3: calibration, from the top and bottom bins
    if cal >= 0:
        cv.line([40, 1240], [460, 460])
        cv.text(40, 498, "CALIBRATION", 22, MONO_B)
        x0, x1 = 900, 1240
        rows = (("Rated 0.8 or higher: %s percent had happened", d["hi"], 556),
                ("Rated under 0.2: %s percent had happened", d["lo"], 626))
        for tpl, b, y in rows:
            cv.text(40, y, tpl % pct(b["share_real"]), 32, SANS_SB)
            cv.text(40, y + 32, "%d steps" % b["n"], 22, MONO)
            cv.rect(x0, y - 24, x1 - x0, 30, PANEL, RULE, 1.5)
            cv.rect(x0, y - 24, (x1 - x0) * float(b["share_real"]) * cal, 30, BLUE)


# ---------------------------------------------------------------- s7_fooled

def fooled_load():
    runs = json.loads((ROOT / "results" / "field.json").read_text())
    statuses = []
    for run in runs:
        word = FAKE_STEPS.get(str(run["take"]))
        if word is None:
            continue
        rows = [r for r in run["rows"] if word in r["step"]]
        assert len(rows) == 1 and rows[0]["truth"] == "skipped", (run["take"], rows)
        statuses.append(rows[0]["status"])
    assert len(statuses) == V2_FAKE_RUNS, statuses
    return statuses


def fooled_state(t: float):
    return (t >= 1.2, t >= 2.6, t >= 4.0)


def fooled_draw(cv: Canvas, st, statuses) -> None:
    v2, v3, last = st
    n = len(statuses)
    done = sum(1 for x in statuses if x == "done")
    cv.clear(PAPER)
    cv.text(40, 150, "People fake steps", 48, SANS_B)
    cv.text(40, 206, "Reach toward the bag, put it back on the table.", 30, SANS)
    cv.line([40, 1240], [244, 244])

    def squares(y: float, filled: int) -> None:
        for i in range(n):
            x = 40 + i * 38
            cv.rect(x, y - 26, 28, 28, RED if i < filled else PANEL, RED if i < filled else RULE, 1.5)

    if v2:
        squares(330, V2_FAKE_DONE)
        cv.text(290, 330, "Version 2: read as done in %d of %d runs" % (V2_FAKE_DONE, V2_FAKE_RUNS), 30, MONO_B, RED)
    if v3:
        squares(424, done)
        cv.text(290, 424, "Version 3, completion rule: read as done in %d of %d runs" % (done, n), 30, MONO_B,
                GREEN)
        parts = ["%s in %d" % (name.replace("_", " "), sum(1 for x in statuses if x == name))
                 for name in ("not_found", "unsure") if name in statuses]
        if parts:
            cv.text(290, 462, "Version 3 readings: " + ", ".join(parts), 22, MONO)
    if last:
        cv.line([40, 1240], [522, 522])
        cv.text(40, 576, "Still wrong: credits a step whose result was already in place.", 26, SANS)


# ---------------------------------------------------------------- s8_end

def end_draw(cv: Canvas) -> None:
    cv.clear(PAPER)
    cv.text(W / 2, 300, "DISCOVERY CLOCK", 76, MONO_B, ha="center")
    cv.line([W / 2 - 310, W / 2 + 310], [330, 330])
    cv.text(W / 2, 384, "Which step was skipped, and when.", 36, SANS, ha="center")
    cv.text(W / 2, 462, "github.com/thesanatt/discovery-clock", 30, MONO_B, BLUE, ha="center")
    cv.text(W / 2, 640, "Built Nov 21 2026 at the Gemini 4 NYC hackathon. "
            "A review aid: a person checks every flag.", 22, SANS, ha="center")


# ---------------------------------------------------------------- render and check

def encode(name: str, seconds: float, state_fn, draw_fn) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / (name + ".mp4")
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(FPS), "-i", "-", "-an", "-vf",
           "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p",
           "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-movflags", "+faststart", str(dst)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cv = Canvas()
    last_state, buf = None, b""
    for i in range(int(round(seconds * FPS))):
        st = state_fn(i / FPS)
        if st != last_state:
            draw_fn(cv, st)
            buf = cv.frame()
            last_state = st
        proc.stdin.write(buf)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed for " + name)
    plt.close(cv.fig)
    return dst


def check(name: str, seconds: float) -> bool:
    """Duration within 0.1 s, 1280x720, 30 fps, and three frames written as PNG."""
    src = OUT / (name + ".mp4")
    info = subprocess.run([FFMPEG, "-hide_banner", "-i", str(src)], capture_output=True, text=True,
                          errors="replace").stderr
    line = next((x for x in info.splitlines() if "Video:" in x), "")
    dur = duration_s(src)
    ok = (abs(dur - seconds) <= 0.1 and "%dx%d" % (W, H) in line and re.search(r"\b30 fps", line) is not None
          and "h264" in line and "yuv420p" in line)
    FRAMES.mkdir(parents=True, exist_ok=True)
    last = seconds - 1.0 / FPS
    for tag, t in (("start", 0.0), ("middle", seconds / 2), ("end", last)):
        png = FRAMES / ("%s_%s.png" % (name, tag))
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-vf",
                        "select=eq(n\\,%d)" % int(round(t * FPS)), "-frames:v", "1", str(png)], check=True)
    print("%-11s %s  duration %.2f s (target %.1f)  %s" % (name, "OK " if ok else "BAD", dur, seconds,
                                                          line.strip()[:110]))
    return ok


def main(argv: list[str]) -> int:
    fused, steps = math_load()
    res = results_load()
    statuses = fooled_load()
    clips = {
        "s4_math": (2 * SEG, lambda t: math_state(t, len(fused["grid"])),
                    lambda cv, st: math_draw(cv, st, fused, steps)),
        "s6_results": (12.0, results_state, lambda cv, st: results_draw(cv, st, res)),
        "s7_fooled": (6.0, fooled_state, lambda cv, st: fooled_draw(cv, st, statuses)),
        "s8_end": (4.0, lambda t: 0, lambda cv, st: end_draw(cv)),
    }
    names = [a for a in argv if a in clips] or list(clips)
    good = True
    for name in names:
        seconds, state_fn, draw_fn = clips[name]
        encode(name, seconds, state_fn, draw_fn)
        good = check(name, seconds) and good
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
