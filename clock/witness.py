"""Second witness: a Gemini 4 reasoning model watches the video and writes its own timeline.

The Decisions ladder (clock/ladder.py) is the first witness: fast and calibrated, but coarse.
A reasoning model is the second: sharper on timing, slower, and it can be confidently wrong.
clock/fusion.py combines the two.

timeline() uses the same prompt and schema as the Carbon baseline in clock/models.py, so the
only difference between the baseline and a witness is which model answers and how hard it thinks.
"""
from __future__ import annotations

from pathlib import Path

from . import media, models


def timeline(video_path: str | Path, steps: list[str], model: str = "neon", thinking: str = "low",
             resolution: str = "low", timeout_s: float = 60.0, info: dict | None = None,
             fps: float | None = None, tag: str | None = None) -> list[dict]:
    """One call: for each step, did it happen and when did it start.

    Returns [{"index": int, "happened": bool, "start": float | None}, ...] in step order.
    model is "neon", "hydrogen" or "carbon". Read info["model"] to see which model really answered.
    """
    steps = [" ".join(str(s).split()) for s in steps]
    n = len(steps)
    video_path = Path(video_path)
    duration = media.duration_s(video_path)

    def parse(out: dict) -> list[dict]:
        rows = out["steps"]
        got = {r["index"]: r for r in rows}
        if set(got) != set(range(n)):
            if len(rows) != n:
                raise ValueError(f"expected {n} entries, got {len(rows)}")
            got = dict(enumerate(rows))
        result = []
        for i in range(n):
            happened = bool(got[i]["happened"])
            start = got[i].get("start") if happened else None
            if start is not None:
                start = min(max(0.0, float(start)), duration)
            result.append({"index": i, "happened": happened, "start": start})
        return result

    listing = "\n".join(f"{i}. {text}" for i, text in enumerate(steps))
    video = media.video_block(video_path, resolution)
    if fps:
        video["processing"] = {"type": "static", "fps": fps}   # the default is 1 frame per second
    blocks = [video, {"type": "text", "text": models.BASELINE_PROMPT.format(duration=duration, steps=listing)}]
    return models._call(model, blocks, models.BASELINE_SCHEMA, parse, thinking, timeout_s,
                        f"{tag or 'witness_' + model}:{video_path.stem}", info)
