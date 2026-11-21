"""The ladder: turn yes/no probabilities over time thresholds into a time with error bars.

For one event we ask, at each grid time t, "had it already started by t?". The answers
should rise from 0 to 1 at the moment it starts. Isotonic regression makes the curve
monotone; where it crosses 0.5 is the estimate, and the 0.2 to 0.8 span is the error bar.
A curve that never rises is a "not found".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression


def mmss(t: float) -> str:
    t = int(round(t))
    return f"{t // 60}:{t % 60:02d}"


def time_grid(duration: float, max_points: int = 40) -> list[int]:
    """Whole-second thresholds from the first second to the end of the video."""
    end = max(1, int(math.floor(duration)))
    step = max(1, int(math.ceil(end / max_points)))
    grid = list(range(step, end + 1, step))
    if grid[-1] != end:
        grid.append(end)
    return grid


def by_time_question(desc: str, t: float) -> dict:
    return {
        "type": "noul",
        "instructions": (
            f"Consider only the part of the video from 0:00 to {mmss(t)} and ignore everything after {mmss(t)}. "
            f"In that part, has this already started to happen: {desc}? "
            f"Answer yes only if it clearly started at or before {mmss(t)}."
        ),
    }


def anywhere_question(desc: str) -> dict:
    return {"type": "noul", "instructions": f"At any point in the whole video, does this happen: {desc}?"}


def longer_than_question(t: float) -> dict:
    return {"type": "noul", "instructions": f"Is this video longer than {int(t)} seconds?"}


def build_questions(steps: list[str], grid: list[int]) -> dict[str, dict]:
    """One by-time question per step per grid time, plus one whole-video question per step."""
    qs: dict[str, dict] = {}
    for i, desc in enumerate(steps):
        qs[f"s{i}_any"] = anywhere_question(desc)
        for j, t in enumerate(grid):
            qs[f"s{i}_t{j}"] = by_time_question(desc, t)
    return qs


# Compact form: the event list goes into one shared text block, and each question is a few tokens.
# The verbose form costs about 100 tokens per question; this one costs about 12.

def glossary(steps: list[str], clock_on_screen: bool = False) -> str:
    lines = ["You will be asked many short yes/no questions about the video."]
    if clock_on_screen:
        lines.append("A clock burned into the top-left corner shows the elapsed time.")
    lines.append("Events:")
    lines += [f"E{i + 1}: {desc}" for i, desc in enumerate(steps)]
    lines.append('"Had E3 started by 0:07?" means: looking only at the video from 0:00 to 0:07 and ignoring '
                 "everything after, had event E3 clearly started to happen.")
    return "\n".join(lines) + "\n"


def build_compact(steps: list[str], grid: list[int]) -> dict[str, dict]:
    qs: dict[str, dict] = {}
    for i, _ in enumerate(steps):
        qs[f"s{i}_any"] = {"type": "noul", "instructions": f"Does E{i + 1} happen at any point in the video?"}
        for j, t in enumerate(grid):
            qs[f"s{i}_t{j}"] = {"type": "noul", "instructions": f"Had E{i + 1} started by {mmss(t)}?"}
    return qs


# Medium form: each question names its event in full but drops the long framing.
# The verbose form costs about 110 input tokens per question, this one about 65.
PREAMBLE = ("You will be asked many yes/no questions about the video. A question of the form "
            '"By 0:07, had this clearly started: X?" means: look only at the video from 0:00 to 0:07, '
            "ignore everything after 0:07, and answer yes only if X had clearly started to happen by then.\n")


def build_medium(steps: list[str], grid: list[int]) -> dict[str, dict]:
    qs: dict[str, dict] = {}
    for i, desc in enumerate(steps):
        qs[f"s{i}_any"] = {"type": "noul", "instructions": f"At any point in the whole video, does this happen: {desc}?"}
        for j, t in enumerate(grid):
            qs[f"s{i}_t{j}"] = {"type": "noul", "instructions": f"By {mmss(t)}, had this clearly started: {desc}?"}
    return qs


@dataclass
class Fit:
    grid: list[int]
    raw: list[float]
    fitted: list[float]
    t50: float | None          # estimated start, None when the curve never reaches 0.5
    lo: float | None           # where the curve crosses 0.2
    hi: float | None           # where the curve crosses 0.8
    p_end: float               # fitted probability at the last grid time
    p_any: float | None        # answer to the whole-video question, when asked
    violation: float           # total downward movement in the raw answers (0 = already monotone)
    found: bool = field(default=False)


def _cross(grid: list[int], fitted: np.ndarray, level: float) -> float | None:
    """First time the monotone curve reaches level, linearly interpolated. The curve starts at t=0, p=0."""
    ts = np.concatenate([[0.0], np.asarray(grid, dtype=float)])
    ps = np.concatenate([[0.0], fitted])
    idx = np.where(ps >= level)[0]
    if len(idx) == 0:
        return None
    k = int(idx[0])
    if k == 0:
        return 0.0
    p0, p1, t0, t1 = ps[k - 1], ps[k], ts[k - 1], ts[k]
    if p1 == p0:
        return float(t1)
    return float(t0 + (level - p0) / (p1 - p0) * (t1 - t0))


def fit(grid: list[int], probs: list[float], p_any: float | None = None, found_at: float = 0.5) -> Fit:
    raw = np.asarray(probs, dtype=float)
    fitted = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True).fit_transform(np.asarray(grid, dtype=float), raw)
    drops = np.clip(raw[:-1] - raw[1:], 0, None)
    f = Fit(grid=list(grid), raw=[float(x) for x in raw], fitted=[float(x) for x in fitted],
            t50=_cross(grid, fitted, 0.5), lo=_cross(grid, fitted, 0.2), hi=_cross(grid, fitted, 0.8),
            p_end=float(fitted[-1]), p_any=p_any, violation=float(drops.sum()))
    f.found = f.t50 is not None and f.p_end >= found_at
    return f


def fit_all(steps: list[str], grid: list[int], answers: dict[str, float]) -> list[Fit]:
    out = []
    for i, _ in enumerate(steps):
        probs = [answers.get(f"s{i}_t{j}", float("nan")) for j in range(len(grid))]
        if any(math.isnan(p) for p in probs):
            raise KeyError(f"missing answers for step {i}")
        out.append(fit(grid, probs, answers.get(f"s{i}_any")))
    return out
