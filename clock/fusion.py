"""Two witnesses and Bayes' rule.

Witness 1, the ladder. For one step we hold the Decisions answers y_j to "had it started by t_j?".
In logit space z_j = logit(y_j) we model them as a change point at the unknown start time tau:

    z_j = b + mu0 + e_j          if t_j <  tau + lag      (not yet started)
    z_j = b + mu1 + e_j          if t_j >= tau + lag      (started)

with independent noise e_j ~ N(0, s0^2) or N(0, s1^2) and one offset b ~ N(0, sb^2) shared by all
answers of the step, because some steps read high or low across the board. The whole-video
answer joins the same vector with its own two levels. Under "never happened" (H0) every ladder
answer sits at the low level. The shared offset makes the covariance diagonal plus rank one, so
each likelihood is closed form (Sherman-Morrison).

More witnesses, each a reasoning model's timeline: it says happened or not, and a start time s.
    P(says happened | H1) = hit,   P(says happened | H0) = false_alarm
    s - tau ~ (1 - gross) * Laplace(shift, scale) + gross * Uniform(0, T)

Posterior. With a flat prior on tau over the video and prior odds for H1:
    P(tau | data, H1)  is proportional to  L_ladder(tau) * product over witnesses of L_w(tau)
    P(H1 | data) = prior * sum_tau(...) / (prior * sum_tau(...) + (1 - prior) * L0)
The start estimate is the posterior median, the range is a central posterior interval, and
"not done" is P(H1 | data) below a threshold.

Every parameter is fitted on the tuning split by eval/fit_fusion.py and stored in
clock/fusion_params.json. Nothing here looks at the test split.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PARAMS_FILE = Path(__file__).resolve().parent / "fusion_params.json"

DEFAULTS = {
    "mu0": -1.2, "s0": 0.45, "mu1": 1.4, "s1": 0.6, "sb": 0.5, "lag": 0.5,
    "a0": -1.3, "sa0": 0.7, "a1": 0.9, "sa1": 1.0,
    "witness": {},
    "prior": 0.5, "done_at": 0.5, "not_done_at": 0.3, "range_scale": 1.0,
    # The witnesses watch the same video, so their errors are not independent and the raw
    # posterior is overconfident. Dividing the log odds by a temperature (fitted on the tuning
    # split by cross-validated likelihood) corrects that without changing any ranking.
    "temperature": 1.0,
}
TAU_STEP = 0.25


def load_params() -> dict:
    p = dict(DEFAULTS)
    if PARAMS_FILE.exists():
        p.update(json.load(open(PARAMS_FILE, encoding="ascii")).get("params", {}))
    return p


def logit(y) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=float), 0.01, 0.99)
    return np.log(y / (1 - y))


def _gauss_offset_loglik(r: np.ndarray, d: np.ndarray, sb: float) -> float:
    """log N(r; 0, diag(d) + sb^2 * ones) by Sherman-Morrison."""
    inv = 1.0 / d
    a = float(np.sum(inv))
    b = float(np.sum(r * inv))
    quad = float(np.sum(r * r * inv)) - (sb * sb * b * b) / (1.0 + sb * sb * a)
    logdet = float(np.sum(np.log(d))) + math.log(1.0 + sb * sb * a)
    return -0.5 * (quad + logdet + len(r) * math.log(2 * math.pi))


def posterior(grid: list[int] | None, raw: list[float] | None, p_any: float | None, duration: float,
              witnesses: dict | None = None, params: dict | None = None, ladder_time: float | None = None) -> dict:
    """Posterior over the start time of one step, and the probability that it happened at all.

    witnesses maps a model name ("hydrogen", "neon", "carbon") to its report for this step,
    {"happened": bool, "start": float | None}. Each witness has its own fitted reliability.
    Pass grid=None and raw=None to leave the ladder out (witnesses only).
    ladder_time, if given, is the ladder's start time from the two-pass zoom (clock/zoom.py) and
    replaces the one-pass half-rise time.
    """
    p = params or load_params()
    use_ladder = grid is not None and raw is not None
    t = np.asarray(grid if use_ladder else [], dtype=float)
    z = logit(raw) if use_ladder else np.zeros(0)
    use_any = p_any is not None
    za = float(logit([p_any])[0]) if use_any else 0.0
    taus = np.arange(0.0, max(duration, t[-1] if use_ladder else 0.0) + 1e-9, TAU_STEP)

    def ladder_ll(after: np.ndarray, happened: bool) -> float:
        mean = np.where(after, p["mu1"], p["mu0"])
        var = np.where(after, p["s1"] ** 2, p["s0"] ** 2)
        r, d = z - mean, var
        if use_any:
            r = np.append(r, za - (p["a1"] if happened else p["a0"]))
            d = np.append(d, (p["sa1"] if happened else p["sa0"]) ** 2)
        if len(r) == 0:
            return 0.0
        return _gauss_offset_loglik(r, d, p["sb"])

    T = max(duration, 1.0)
    lw = p.get("ladder_summary")
    if use_ladder and p.get("ladder_mode", "changepoint") == "summary" and lw:
        # Summary form of witness 1. The ladder is reduced to two numbers: a score u (logit of the
        # mean of its last fitted value and the whole-video answer) and the half-rise time t_hat
        # from the isotonic fit. u is Gaussian under each hypothesis; t_hat errs like a witness.
        from . import ladder as _ladder
        f = _ladder.fit(list(grid), list(raw), p_any)
        u = float(logit([0.5 * (f.p_end + (p_any if use_any else f.p_end))])[0])

        def norm_ll(x: float, m: float, s: float) -> float:
            return -0.5 * ((x - m) / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)

        ll0 = norm_ll(u, lw["m0"], lw["s0"])
        ll_tau = np.full(len(taus), norm_ll(u, lw["m1"], lw["s1"]))
        t_hat = ladder_time if (ladder_time is not None and f.t50 is not None) else f.t50
        if t_hat is not None:
            ll0 += math.log(lw["has_time0"]) + math.log(1.0 / T)
            lap = np.exp(-np.abs(t_hat - taus - lw["shift"]) / lw["scale"]) / (2 * lw["scale"])
            ll_tau = ll_tau + math.log(lw["has_time1"]) + np.log((1 - lw["gross"]) * lap + lw["gross"] / T)
        else:
            ll0 += math.log(1 - lw["has_time0"])
            ll_tau = ll_tau + math.log(1 - lw["has_time1"])
    else:
        ll0 = ladder_ll(np.zeros(len(t), dtype=bool), happened=False)
        ll_tau = np.array([ladder_ll(t >= tau + p["lag"], happened=True) for tau in taus])
    for name, obs in (witnesses or {}).items():
        wp = p.get("witness", {}).get(name)
        if wp is None or obs is None:
            continue
        if obs.get("happened"):
            ll0 += math.log(wp["false_alarm"])
            s = obs.get("start")
            if s is not None:
                ll0 += math.log(1.0 / T)
                lap = np.exp(-np.abs(s - taus - wp["shift"]) / wp["scale"]) / (2 * wp["scale"])
                ll_tau = ll_tau + math.log(wp["hit"]) + np.log((1 - wp["gross"]) * lap + wp["gross"] / T)
            else:
                ll_tau = ll_tau + math.log(wp["hit"])
        else:
            ll0 += math.log(1 - wp["false_alarm"])
            ll_tau = ll_tau + math.log(1 - wp["hit"])

    # flat prior over tau; log-sum-exp for the evidence of H1
    m = float(np.max(ll_tau))
    w = np.exp(ll_tau - m)
    ll1 = m + math.log(float(np.mean(w)))
    log_odds = math.log(p["prior"] / (1 - p["prior"])) + ll1 - ll0
    log_odds = log_odds / max(1.0, float(p.get("temperature", 1.0)))
    p_done = 1.0 / (1.0 + math.exp(-max(min(log_odds, 60.0), -60.0)))

    post = w / float(np.sum(w))
    cdf = np.cumsum(post)

    def q(level: float) -> float:
        return float(taus[min(int(np.searchsorted(cdf, level)), len(taus) - 1)])

    med = q(0.5)
    half_lo, half_hi = med - q(0.1), q(0.9) - med
    k = p.get("range_scale", 1.0)
    return {
        "p_done": round(p_done, 4), "log_odds": round(log_odds, 3),
        "time": round(med, 2),
        "lo": round(max(0.0, med - k * half_lo - TAU_STEP), 2),
        "hi": round(min(float(taus[-1]), med + k * half_hi + TAU_STEP), 2),
        "taus": [round(float(x), 2) for x in taus[::2]],
        "density": [round(float(x), 4) for x in (post[::2] / max(float(post.max()), 1e-12))],
    }
