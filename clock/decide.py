"""Thin wrapper around the Gemini Decisions API: timeout, one retry, call log."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "runs" / "calls.jsonl"
DECISION_MODEL = "gemini-decision-1"
MAX_QUESTIONS = 1000


def client(timeout_ms: int = 90_000) -> genai.Client:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("No GEMINI_API_KEY found. Put it in .env (gitignored).")
    return genai.Client(api_key=key, http_options={"api_version": "v1beta", "timeout": timeout_ms})


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _log(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="ascii") as f:
        f.write(json.dumps(row) + "\n")


def decide(cl: genai.Client, blocks: list[dict], questions: dict[str, dict], tag: str = "",
           retries: int = 1) -> tuple[dict[str, float], dict]:
    """Ask all yes/no questions about the same input. Returns ({key: P(yes)}, call info).

    More than 1,000 questions are split across calls that share the same input.
    """
    keys = list(questions)
    answers: dict[str, float] = {}
    info = {"tag": tag, "calls": 0, "seconds": 0.0, "input_tokens": 0, "n_questions": len(keys)}
    for start in range(0, len(keys), MAX_QUESTIONS):
        part = {k: questions[k] for k in keys[start:start + MAX_QUESTIONS]}
        last_err = None
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                res = cl.decisions.create(model=DECISION_MODEL, input=blocks, questions=part)
                dt = time.time() - t0
                got = _get(res, "answers") or {}
                for k in part:
                    a = _get(got, k) if isinstance(got, dict) else getattr(got, k, None)
                    answers[k] = float(_get(a, "noul"))
                usage = _get(res, "usage")
                tokens = int(_get(usage, "total_input_tokens", 0) or 0)
                info["calls"] += 1
                info["seconds"] += dt
                info["input_tokens"] += tokens
                _log({"ts": time.time(), "tag": tag, "model": DECISION_MODEL, "ok": True, "attempt": attempt,
                      "seconds": round(dt, 2), "input_tokens": tokens, "n_questions": len(part)})
                last_err = None
                break
            except Exception as e:  # noqa: BLE001 - log every failure, retry once
                dt = time.time() - t0
                last_err = e
                _log({"ts": time.time(), "tag": tag, "model": DECISION_MODEL, "ok": False, "attempt": attempt,
                      "seconds": round(dt, 2), "error": f"{type(e).__name__}: {str(e)[:300]}"})
                time.sleep(1.5)
        if last_err is not None:
            raise last_err
    return answers, info
