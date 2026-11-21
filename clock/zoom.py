"""Second ladder pass: zoom in time.

The Decisions model looks at a video one frame per second and its timestamps are whole seconds,
so a ladder over the whole video cannot place an event better than its grid step: 2 seconds on
a 30 second clip, 33 seconds on a 21 minute recording.

The second pass cuts a short window around each step's first estimate and SLOWS IT DOWN before
asking the same ladder again. A 6 second window slowed 4 times is a 24 second clip: one frame
per second of the clip is one frame per quarter second of the scene, and "by 0:07" in the clip
means 1.75 seconds into the window. The fine estimate is mapped back to real time.

One Decisions call per step, run in parallel, about a second each.
"""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import decide, ladder, media

TARGET_CLIP_S = 24.0      # length of the slowed clip
MAX_SLOW = 4.0
MIN_HALF_WINDOW_S = 2.0
WINDOW_STEPS = 1.5        # half-width of the window, in coarse grid steps


def slow_clip(src: Path, dst: Path, start: float, end: float, slow: float) -> Path:
    """Cut [start, end] from src and stretch it in time by the factor slow (sound included)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tempo = []
    k = slow
    while k > 2.0 + 1e-9:          # atempo accepts 0.5 to 2.0 per stage
        tempo.append("atempo=0.5")
        k /= 2.0
    tempo.append(f"atempo={1.0 / k:.4f}")
    res = subprocess.run(
        [media.FFMPEG, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
         "-vf", f"setpts={slow:.4f}*PTS", "-af", ",".join(tempo), "-r", "10",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "48k", str(dst)],
        capture_output=True, text=True, errors="replace", timeout=120)
    if res.returncode != 0 or not dst.exists():
        # a source without sound: try again with no audio filter
        res = subprocess.run(
            [media.FFMPEG, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
             "-vf", f"setpts={slow:.4f}*PTS", "-an", "-r", "10", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
             "-pix_fmt", "yuv420p", str(dst)], capture_output=True, text=True, errors="replace", timeout=120)
        if res.returncode != 0 or not dst.exists():
            raise RuntimeError(f"ffmpeg slow clip failed: {res.stderr[-300:]}")
    return dst


def refine(small: Path, duration: float, steps: list[str], grid: list[int], fits: list[ladder.Fit],
           client=None, tag: str = "zoom", workdir: Path | None = None) -> list[dict]:
    """Fine start time for every step whose coarse ladder rose.

    Returns one dict per step: {"time": refined seconds or None, "coarse": coarse seconds or None,
    "window": [a, b] or None, "slow": factor, "resolution_s": seconds of the scene per fine grid step,
    "raw": fine answers, "ok": bool}. A step whose fine curve does not rise keeps its coarse time.
    """
    client = client or decide.client()
    workdir = workdir or (small.parent / "zoom")
    step_s = float(grid[1] - grid[0]) if len(grid) > 1 else float(grid[0])

    def one(i: int) -> dict:
        f = fits[i]
        out = {"time": f.t50, "coarse": f.t50, "window": None, "slow": 1.0, "resolution_s": step_s, "raw": [], "ok": False}
        if f.t50 is None:
            return out
        half = max(MIN_HALF_WINDOW_S, WINDOW_STEPS * step_s)
        # undo the coarse bias so the window is centred on the half-rise itself
        centre = f.t50 + ladder.BIAS_STEPS * step_s
        a, b = max(0.0, centre - half), min(duration, centre + half)
        if b - a < 1.0:
            return out
        slow = max(1.0, min(MAX_SLOW, TARGET_CLIP_S / (b - a)))
        try:
            clip = slow_clip(small, workdir / f"{small.stem}_{i}.mp4", a, b, slow)
            clip_len = media.duration_s(clip)
            fine_grid = ladder.time_grid(clip_len, max_points=int(TARGET_CLIP_S) + 2)
            qs = {f"t{j}": ladder.by_time_question(steps[i], t) for j, t in enumerate(fine_grid)}
            answers, _ = decide.decide(client, [media.video_block(clip, "low")], qs, tag=f"{tag}:{i}")
            raw = [answers[f"t{j}"] for j in range(len(fine_grid))]
            ff = ladder.fit(fine_grid, raw, None)
            out.update(window=[round(a, 2), round(b, 2)], slow=round(slow, 2), resolution_s=round(1.0 / slow, 3),
                       raw=[round(x, 3) for x in raw], ok=True)
            if ff.t50 is not None:
                out["time"] = round(min(b, max(a, a + ff.t50 / slow)), 2)
            return out
        except Exception as e:  # noqa: BLE001 - the coarse time stands
            out["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            return out

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(steps)))) as pool:
        return list(pool.map(one, range(len(steps))))
