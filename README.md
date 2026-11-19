[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Yugen-11/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Yugen-11/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Forecasts, for every one of GTA VI's 55 PS5 trophies, five quantiles of the percent of
Exophase-tracked players who will have earned it at the case's target — a week after
launch — from the house's reads of Exophase's trophy page. Entry for the make.fun
`gta6-trophy-rarity-forecast-01` bounty. Scored by mean pinball loss on the logit
scale; a prize needs a loss strictly below the best house entry's.

## The model

A trophy's rate is a ratio — `rate = earners / players_tracked` — and the two halves
move differently in launch week. The program models them separately, each with a
saturating curve in hours since release, and ensembles that with the house curve
baseline's log-time trend:

- **Players tracked, N(t):** `ln N = L_inf - (L_inf - L0) * exp(-t / tau_N)`,
  tau_N and L_inf grid-searched on the history's distinct states (consecutive
  identical reads are collapsed first; the refresh jumps between them are the
  signal). The forecast N is blended 30% toward no further growth, in log space.
- **Earners, e_i(t):** `e = E_inf * (1 - exp(-t / tau_tier))`, tau pooled per tier
  (platinum, gold, silver, bronze behave differently), E_inf fitted per trophy by
  least squares. Earners only grow and can never exceed N.
- **Trend:** per trophy, the house curve baseline's least-squares line of logit rate
  on `ln(1 + hours since release)` over the last 24 hours of reads, read at the
  target.
- **Ensemble:** median = `alpha * saturating + (1 - alpha) * trend`, in logit
  space. Alpha adapts to how fast N grew over the last 24 hours of history: 0.75
  when N has nearly stopped growing, 0.25 when it is still growing faster than 0.6
  log units a day, linear between.
- **Spread:** the baseline random-walk term (`0.5 * sqrt(horizon in days)`) combined
  with the saturating fit's uncertainty and half the distance between the two
  models' medians, scaled by 0.8 and capped at 3 logit units.

Why this shape: the practice data and a live read of the Exophase page show the
tracked population exploding after launch (638 at the first probe, 3,868 at the
first refresh, tens of thousands by the next evening) and then decelerating, while
every refresh dilutes most rates — the platinum fell 0.63% to 0.03% and the chapter
trophies fell 5–20x in about ten hours, while the trophy everyone gets on starting
stayed at ~100%. Earners only grow, at a trophy's own pace.

## Backtests

The practice cases cannot validate a growth model — their histories are seven or
eight identical reads, so every model reduces to persistence on them (loss about
1.51 on the 90-minute practice, like the baselines). Two backtests were run instead:

- **Live backtest:** train on p01's history (checkpoint 03:30Z), predict the rates
  the live Exophase page showed at 13:26Z the same day (a real 10-hour horizon
  with a real dilution surge). Persistence scores 2.39; this model scores 2.39 —
  with a flat history there is no signal to beat persistence on.
- **Synthetic long-horizon backtest:** a cohort generator (players join on a
  saturating curve; a player earns a trophy with a per-trophy timescale; Exophase
  refreshes every few hours) cut to a 2-day history and a 4.6-day target, over 12
  parameter settings (week-1 tracked 80k to 2M, N timescales 8 to 36 hours, refresh
  2 to 8 hours, earner timescales scaled 0.6x to 1.5x). Mean pinball loss:
  persistence 0.59, trend baseline 0.33, this model 0.28; worst case: persistence
  0.87, trend 0.42, this model 0.35. The generator is a guess, so these numbers are
  evidence about the estimator, not a prediction of the real score.

## How to run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=practice-input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `<id>.in` in
`MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` to
`MAKEFUN_OUTPUT_DIR` (default `/output`): `{"trophies": [{"id": n, "q": [q05,
q25, q50, q75, q95]}, ...]}`, one row per trophy, five finite non-decreasing
quantiles in 0–100. Runs in about 0.07 seconds on the practice cases — far under
the 8,000 ms per-case cap — and never reads `practice-input/` at runtime; that
directory is only for local testing. Edge cases (one-read history, null rates,
empty history) all produce valid output instead of crashing.

## Files

- `main.py` — the whole entry: the forecaster, run as `python3 /bot/main.py`.
- `practice-input/` — the kit's practice cases `p00.in` and `p01.in` (launch night,
  90-minute horizons) plus `NOTICE.txt`.
- `LICENSE` — MIT.

<sub>Entered on [make.fun](https://make.fun/?ref=4ab68p3v&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
