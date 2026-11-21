"""Video preparation with ffmpeg: downscale, optional burned-in clock, hashes."""
from __future__ import annotations

import base64
import hashlib
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# A font that ships with Windows. ffmpeg filter syntax needs the drive colon escaped.
FONT = "C\\:/Windows/Fonts/arial.ttf"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([FFMPEG, *args], capture_output=True, text=True, errors="replace")


def duration_s(path: str | Path) -> float:
    """Length of a media file in seconds, read from the ffmpeg banner."""
    out = _run(["-hide_banner", "-i", str(path)]).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if not m:
        raise ValueError(f"no duration found for {path}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def prep(src: str | Path, dst: str | Path, height: int = 360, fps: int = 10,
         timecode: bool = False, audio: bool = True) -> Path:
    """Write a small H.264 copy of src. With timecode=True a running clock is drawn on every frame."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # height is the SHORT side: a portrait phone video keeps 360 px of width instead of shrinking to 202
    vf = f"scale='if(gt(iw,ih),-2,{height})':'if(gt(iw,ih),{height},-2)'"
    if timecode:
        vf += (f",drawtext=fontfile='{FONT}':text='%{{pts\\:hms}}':x=8:y=8:fontsize=26:"
               "fontcolor=white:box=1:boxcolor=black@0.75:boxborderw=6")
    args = ["-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p"]
    args += ["-c:a", "aac", "-b:a", "48k", "-ac", "1"] if audio else ["-an"]
    res = _run([*args, str(dst)])
    if res.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed for {src}: {res.stderr[-400:]}")
    return dst


def cut(src: str | Path, dst: str | Path, start: float, end: float) -> Path:
    """Cut [start, end] seconds out of src, re-encoded so the cut is frame accurate."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    res = _run(["-y", "-loglevel", "error", "-ss", f"{max(0.0, start):.2f}", "-i", str(src),
                "-t", f"{max(0.1, end - start):.2f}", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", str(dst)])
    if res.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg cut failed for {src}: {res.stderr[-400:]}")
    return dst


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def video_block(path: str | Path, resolution: str = "low") -> dict:
    """Inline video content block for the Decisions and Interactions APIs."""
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "video", "data": data, "mime_type": "video/mp4", "resolution": resolution}
