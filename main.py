"""gta6-trophy-rarity-forecast-01 entry: an adaptive ensemble forecaster (Python 3.13, stdlib only).

Forecasts, for every trophy of Grand Theft Auto VI on PS5, five quantiles of the percent
of Exophase-tracked players who will have earned it at the case's target, from the reads
of Exophase's trophy page in the case's history.

Why this shape, and what the data says
---------------------------------------
A trophy's rate is a ratio: rate = earners / players_tracked. The two halves move
differently in launch week, and the practice data plus a live read of the page show it:

- players_tracked explodes after launch (638 at the first probe, 3,868 at the first
  refresh a few hours later, tens of thousands by the next evening) and then its growth
  decelerates. Every refresh that adds a wave of new players dilutes most rates: on the
  live page the platinum fell 0.63% -> 0.03% and the chapter trophies fell 5-20x in
  about ten hours, while the one trophy everyone gets on starting stayed at ~100%.
- earners only grow, and they grow at a trophy's own pace: early story and easy
  activity trophies catch up fast as the mass of players plays through the opening
  hours; the platinum, the golds and the grinds climb slowly.

So the median is built from the two halves separately, each with a saturating curve
in hours since release, and then ensembled with the house curve baseline's log-time
trend, which is the right shape while the tracked population is still growing fast:

- N(t): ln N = L_inf - (L_inf - L0) * exp(-t / tau_N), tau_N and L_inf grid-searched
  on the history's distinct states (consecutive identical reads carry no information
  and are collapsed first). The forecast N is blended 30% toward no further growth,
  in log space, because the fit cannot see past the last refresh.
- earners_i(t): e = E_inf * (1 - exp(-t / tau_tier)), tau pooled per tier (platinum,
  gold, silver, bronze behave differently), E_inf fitted per trophy by least squares.
  Earners are anchored at the last observed count and can never exceed N.
- trend: per trophy, the house curve baseline's least-squares line of logit rate on
  ln(1 + hours since release) over the last 24 hours of reads, read at the target.
- ensemble: median = alpha * saturating + (1 - alpha) * trend, in logit space, where
  alpha adapts to how fast N grew over the last 24 hours of history: 0.75 when N has
  nearly stopped growing (trust the saturating model), 0.25 when it is still growing
  faster than 0.6 log units a day (trust the trend), linear between.
- spread: the baseline random-walk term (0.5 * sqrt(horizon in days)) combined with
  the saturating fit's uncertainty and half the distance between the two models'
  medians (when the models disagree, the forecast is honestly less sure), then
  scaled by 0.8 and capped at 3 logit units. The scale was set on backtests, below.

Backtests (honest numbers)
---------------------------
The practice cases cannot validate a growth model: their histories are seven or
eight identical reads, so every model reduces to persistence on them (loss about
1.51 on the 90-minute practice, like the baselines). Two backtests were run instead:

- Live backtest: train on p01's history (checkpoint 03:30Z), predict the rates the
  live Exophase page showed at 13:26Z the same day (a real 10-hour horizon with a
  real dilution surge). Persistence scores 2.39; this model scores 2.39, the same
  within noise -- with a flat history there is no signal to beat persistence on, and
  the backtest only confirms the program does not crash or do anything exotic there.
- Synthetic long-horizon backtest: a cohort generator (players join on a saturating
  curve; a player earns a trophy with a per-trophy timescale; Exophase refreshes
  every few hours) cut to a 2-day history and a 4.6-day target, over 12 parameter
  settings (week-1 tracked 80k to 2M, N timescales 8 to 36 hours, refresh 2 to 8
  hours, earner timescales scaled 0.6x to 1.5x). Mean pinball loss: persistence
  0.59, trend baseline 0.33, this model 0.28; worst case: persistence 0.87, trend
  0.42, this model 0.35. The model beats both baselines on mean and on worst case,
  but the generator is a guess, so these numbers are evidence about the estimator,
  not a prediction of the real score.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P keeps this
file's directory OFF sys.path, so a module shipped beside main.py imports only after
the sys.path line below, which puts the directory first; -s drops user site-packages.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a time,
    with only that case's file there): JSON {"schema": "makefun/gta6-trophies-case/v1",
    "id", "at", "horizon_minutes", "target", "levels", "game", "trophies", "history",
    "notes"}, where `trophies` lists the case's trophies {id, name, description, tier,
    points, secret} and `history` every read before `at`, oldest first, each {slot, at,
    players_tracked, trophies_earned, rates}, `rates` in the order of `trophies`
    (null where that read lacked the trophy);
  - write <id>.out to the output directory: {"trophies": [{"id": 1, "q": [q05, q25,
    q50, q75, q95]}, ...]}, one row for each of the case's trophies, each id once, no
    other key, each q five finite numbers from 0 to 100 (percent), non-decreasing;
    at most 65,536 bytes. Anything else is an invalid case, scored as if every
    quantile equalled the trophy's rate in the last read, and an entry with an invalid
    case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in
for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=practice-input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (the scorer's Z, and the baselines').
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# The spread of a rate's move, in logit units per square-root day (the baselines' SIGMA).
SIGMA = 0.5
# The largest spread a forecast may give, logit units (the house's cap).
SPREAD_MAX = 3.0
# The saturating model's weight toward no further growth of N (log space).
BLEND_N = 0.3
# The spread scale, set on the synthetic backtests (see the module docstring).
SPREAD_SCALE = 0.8
# The saturating model's weight at the two ends of N's recent growth (log units/day).
ALPHA_SATURATED = 0.75
ALPHA_GROWING = 0.25
GROWTH_LOW = 0.15
GROWTH_HIGH = 0.60


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg > 700:
        return 100.0
    if lg < -700:
        return 0.0
    return 100 / (1 + math.exp(-lg))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def _norm_ppf(p: float) -> float:
    """The standard normal's quantile at p (Acklam's approximation), for levels other
    than the contest's five, where the exact Z above is used instead."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _z_for(levels) -> list:
    """The normal quantile for each requested level: the exact contest Z where the
    level matches, Acklam's approximation otherwise."""
    out = []
    for lv in levels:
        hit = None
        for known, z in zip(LEVELS, Z):
            if abs(lv - known) < 1e-9:
                hit = z
                break
        out.append(hit if hit is not None else _norm_ppf(lv))
    return out


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _release(case: dict) -> datetime:
    """Hours are counted from the game's release (midnight UTC on release_date)."""
    try:
        day = case["game"]["release_date"]
        return datetime.fromisoformat(day + "T00:00:00+00:00")
    except Exception:
        hist = case.get("history") or []
        if hist:
            return _parse_ts(hist[0]["at"])
        return _parse_ts(case["at"])


def _hours_since(ts: str, origin: datetime) -> float:
    return (_parse_ts(ts) - origin).total_seconds() / 3600.0


def _distinct_states(case: dict):
    """Collapse consecutive identical (players_tracked, rates) reads: they carry no
    new information, and the refresh jumps between them are the signal."""
    states = []
    for h in case["history"]:
        key = (h["players_tracked"], tuple(h["rates"]))
        if states and states[-1][0] == key:
            continue
        states.append((key, h["at"]))
    return states


def _fit_saturating(ts, ys):
    """Fit y(t) = Y_inf - (Y_inf - y0) * exp(-t / tau) to points (t, y) by grid search
    over tau, with Y_inf solved by least squares at each tau. Returns (Y_inf, tau,
    sse) or None when there are too few points. y0 is the first observed value."""
    if len(ts) < 2:
        return None
    y0 = ys[0]
    best = None
    for k in range(28):
        tau = 1.0 * (1.35 ** k)  # about 1 hour to about 400 hours
        g = [1 - math.exp(-t / tau) for t in ts]
        num = sum((y - y0 * (1 - gg)) * gg for y, gg in zip(ys, g))
        den = sum(gg * gg for gg in g)
        if den < 1e-12:
            continue
        y_inf = num / den
        if y_inf < max(ys):
            y_inf = max(ys)  # the asymptote cannot sit below what was already seen
        sse = sum((y - (y_inf * gg + y0 * (1 - gg))) ** 2 for y, gg in zip(ys, g))
        if best is None or sse < best[2]:
            best = (y_inf, tau, sse)
    return best


def _trend_medians(case: dict, origin: datetime):
    """The house curve baseline's medians: per trophy, a least-squares line of logit
    rate on ln(1 + hours since release) over the last 24 hours of reads, read at the
    target. Fewer than 3 reads fall back to the last rate."""
    at = _parse_ts(case["at"])
    hist = [h for h in case["history"]
            if (at - _parse_ts(h["at"])).total_seconds() <= 24 * 3600]
    x_tgt = math.log1p(max(_hours_since(case["target"], origin), 0.0))
    meds = []
    for i in range(len(case["trophies"])):
        pts = [(math.log1p(max(_hours_since(h["at"], origin), 0.0)), logit(h["rates"][i]))
               for h in hist if h["rates"][i] is not None]
        if len(pts) < 3:
            r = None
            for h in reversed(case["history"]):
                if h["rates"][i] is not None:
                    r = h["rates"][i]
                    break
            meds.append(logit(r if r is not None else 0))
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        b = sxy / sxx if sxx > 1e-12 else 0.0
        meds.append((my - b * mx) + b * x_tgt)
    return meds


def _recent_n_growth(case: dict) -> float:
    """N's daily log-growth over the last 24 hours of distinct states (0 when there
    are fewer than two states in that window)."""
    states = _distinct_states(case)
    if len(states) < 2:
        return 0.0
    at = _parse_ts(case["at"])
    pts = []
    for (key, a) in states:
        if (at - _parse_ts(a)).total_seconds() <= 24 * 3600 + 1:
            pts.append((_parse_ts(a), math.log(max(key[0], 1))))
    if len(pts) < 2:
        return 0.0
    dt = (pts[-1][0] - pts[0][0]).total_seconds() / 86400.0
    if dt < 1e-6:
        return 0.0
    return (pts[-1][1] - pts[0][1]) / dt


def _saturating_forecast(case: dict, origin: datetime):
    """The saturating halves: N(t) and earners_i(t). Returns (median logits, spread
    per trophy, debug dict)."""
    trophies = case["trophies"]
    hist = case["history"]
    last = hist[-1]
    d_days = case["horizon_minutes"] / 1440.0
    t_tgt = max(_hours_since(case["target"], origin), 0.0)
    states = _distinct_states(case)
    ts = [max(_hours_since(a, origin), 0.0) for _, a in states]
    n_last = last["players_tracked"]
    # --- N(t): saturating in log space, blended toward no further growth.
    ln_n = [math.log(max(s[0][0], 1)) for s in states]
    n_hat = float(n_last)
    fit_n = _fit_saturating(ts, ln_n) if len(states) >= 3 else None
    if fit_n is not None:
        l_inf, tau_n, _sse = fit_n
        n_hat = math.exp(l_inf - (l_inf - ln_n[0]) * math.exp(-t_tgt / tau_n))
    n_hat = max(n_hat, float(n_last))
    if BLEND_N > 0:
        n_hat = math.exp((1 - BLEND_N) * math.log(n_hat) + BLEND_N * math.log(max(n_last, 1)))
    n_hat = max(n_hat, 1.0)
    # --- earners: tau pooled per tier, E_inf fitted per trophy by least squares.
    tier_tau = {}
    for tier in {t["tier"] for t in trophies}:
        idxs = [i for i, t in enumerate(trophies) if t["tier"] == tier]
        best = None
        for k in range(28):
            tau = 1.0 * (1.35 ** k)
            sse_tot = 0.0
            cnt = 0
            for i in idxs:
                pts = []
                for (key, _a), t in zip(states, ts):
                    r = key[1][i]
                    if r is None:
                        continue
                    pts.append((t, max(r * key[0] / 100.0, 0.0)))
                if len(pts) < 2:
                    continue
                g = [1 - math.exp(-tt / tau) for tt, _ in pts]
                num = sum(e * gg for (_, e), gg in zip(pts, g))
                den = sum(gg * gg for gg in g)
                if den < 1e-12:
                    continue
                e_inf = num / den
                sse_tot += sum((e - e_inf * gg) ** 2 for (_, e), gg in zip(pts, g))
                cnt += len(pts)
            if cnt == 0:
                continue
            if best is None or sse_tot < best[1]:
                best = (tau, sse_tot)
        if best is not None:
            tier_tau[tier] = best[0]
    meds = []
    spreads = []
    for i, t in enumerate(trophies):
        tau = tier_tau.get(t["tier"], 72.0)
        pts = []
        for (key, _a), tt in zip(states, ts):
            r = key[1][i]
            if r is None:
                continue
            pts.append((tt, max(r * key[0] / 100.0, 0.0)))
        e_last = max((last["rates"][i] or 0) * n_last / 100.0, 0.0)
        if len(pts) >= 2:
            g = [1 - math.exp(-tt / tau) for tt, _ in pts]
            num = sum(e * gg for (_, e), gg in zip(pts, g))
            den = sum(gg * gg for gg in g)
            e_inf = num / den if den > 1e-12 else e_last
            e_hat = e_inf * (1 - math.exp(-t_tgt / tau))
            resid = math.sqrt(sum((e - e_inf * gg) ** 2 for (_, e), gg in zip(pts, g))
                              / max(len(pts) - 1, 1))
            rel = resid / max(e_last, 1.0)
        else:
            e_hat = e_last
            rel = 0.5
        e_hat = min(max(e_hat, e_last), n_hat)  # earners only grow, and never exceed tracked
        r_hat = 100.0 * e_hat / n_hat
        meds.append(logit(r_hat))
        s = math.sqrt((SIGMA * math.sqrt(d_days)) ** 2 + min(rel, 1.0) ** 2 * 0.25)
        spreads.append(min(max(s, SIGMA * math.sqrt(d_days)), 2.0))
    return meds, spreads, {"n_hat": n_hat, "n_last": n_last, "tier_tau": tier_tau}


def forecast_case(case: dict, name: str) -> dict:
    trophies = case["trophies"]
    levels = case.get("levels") or LEVELS
    z = _z_for(levels)
    hist = case.get("history") or []
    if not hist:
        # No history at all: say almost nothing, with a wide spread. Never crash.
        rows = [{"id": t["id"],
                 "q": [round4(inv_logit(logit(0.5) + zz * 2.0)) for zz in z]}
                for t in trophies]
        return {"trophies": rows}
    origin = _release(case)
    d_days = case["horizon_minutes"] / 1440.0
    rw = SIGMA * math.sqrt(d_days)
    sat_med, sat_sp, _dbg = _saturating_forecast(case, origin)
    tr_med = _trend_medians(case, origin)
    # How fast N grew over the last day decides which half of the ensemble to trust.
    g = _recent_n_growth(case)
    if g <= GROWTH_LOW:
        alpha = ALPHA_SATURATED
    elif g >= GROWTH_HIGH:
        alpha = ALPHA_GROWING
    else:
        alpha = ALPHA_SATURATED - (ALPHA_SATURATED - ALPHA_GROWING) * (g - GROWTH_LOW) / (GROWTH_HIGH - GROWTH_LOW)
    # A trophy never observed at all falls back to its tier's last median rate.
    tier_last = {}
    for tier in {t["tier"] for t in trophies}:
        vals = []
        for i, t in enumerate(trophies):
            if t["tier"] != tier:
                continue
            for h in reversed(hist):
                if h["rates"][i] is not None:
                    vals.append(h["rates"][i])
                    break
        if vals:
            tier_last[tier] = sum(vals) / len(vals)
    rows = []
    for i, t in enumerate(trophies):
        observed = any(h["rates"][i] is not None for h in hist)
        if observed:
            lg = alpha * sat_med[i] + (1 - alpha) * tr_med[i]
            disagree = abs(sat_med[i] - tr_med[i])
            s = math.sqrt(max(sat_sp[i], rw) ** 2 + (0.5 * disagree) ** 2) * SPREAD_SCALE
        else:
            lg = logit(tier_last.get(t["tier"], 0.5))
            s = 2.0
        s = min(s, SPREAD_MAX)
        qs = [round4(inv_logit(lg + zz * s)) for zz in z]
        qs = [min(max(q, 0.01), 99.99) for q in qs]
        for j in range(1, len(qs)):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
