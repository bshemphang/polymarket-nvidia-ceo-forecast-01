# Polymarket NVIDIA CEO Forecast

Minute-by-minute 4-hour-ahead quantile forecasts of the Yes prices of
Polymarket's "Next CEO of NVIDIA?" event (24 markets), for the make.fun
bounty of the same name.

Entry point: `main.py` — Python 3.13, standard library only, offline,
deterministic, no clock, no randomness. About 1 second per case.

## How it works

The 24 markets are mutually exclusive outcomes of one event, so their
Yes prices should sum to ~1. The forecast exploits that constraint:

1. **Median path — mean reversion to the renormalised share.**
   Each market's median is pulled toward its cross-market share
   `s_i = p_i / sum_j p_j` as an Ornstein-Uhlenbeck process,
   `m_i(h) = s_i + (p_i0 - s_i) * exp(-h / 700)`, and the medians are
   rescaled each horizon so their sum reverts toward 1 at the same
   rate. A small AR(1) bounce correction (lag-1 autocorrelation of
   10-minute changes, about -0.33 in history) adjusts the first
   ~30 minutes. The pull is disabled when the input sum is outside
   [0.5, 3.0] (a partial market set, where "shares" are meaningless).
2. **Spread — jump-aware volatility.** Per-market per-minute sigma is
   the median of the RMS of changes at 1-, 10- and 60-minute scales
   over the recent history (RMS, not MAD: the prices are stepwise —
   long flat stretches punctuated by jumps — so most minute changes
   are exactly zero and MAD collapses to 0). A cross-market backstop
   (30% of the median sigma) keeps frozen markets honest, and a mild
   `sqrt(p(1-p))` level factor scales volatility down near 0 and 1.
   Horizon growth is strongly sub-diffusive,
   `sigma(h) = sig * sqrt(h) * (h/10)^-0.70` for `h > 10`, because
   jumps partially revert: realised 240-minute moves are only ~1.7x
   the 10-minute moves, not ~4.9x as a random walk would imply.
3. **Quantiles — Laplace, not normal.** The change distribution has a
   sharper peak and fatter tails than a normal, so quantiles use
   Laplace scores `(-1.628, -0.490, 0, +0.490, +1.628)` in units of
   sigma. Quantiles are clipped to `[0.001, 0.999]` and forced
   monotone.

## Usage

```
python3 main.py                       # every $MAKEFUN_INPUT_DIR/*.in
                                      # -> $MAKEFUN_OUTPUT_DIR/*.out
                                      # (defaults /input and /output)
python3 main.py in.in out.out         # one case, for local testing
```

## Input / output format (verified against the kit)

The `.in` / `.out` format is the kit's real schema
(`makefun/polymarket-case/v1`, see the kit's `RULES.txt`), verified
2026-11-13 against the kit's six practice cases (`p00`–`p05`).

Input (`/input/<id>.in`): one JSON object, keys sorted, no whitespace:
`schema`, `competition`, `case` (id), `at` (checkpoint, ISO 8601 UTC),
`horizon_minutes` (240), `quantiles` ([0.05, 0.25, 0.5, 0.75, 0.95]),
`event` (`slug`, `title`) and `markets`, a list in the universe's order,
each `id` (conditionId), `name`, `slug`, `yes_token`, `start` (minute
of its first price) and `prices`, one number per minute from `start`
to `at` inclusive, so its last is the price at the checkpoint. Prices
are on a 0–1 scale. Markets that started later hold fewer prices; each
market's series is used on its own length. The program reads
`$MAKEFUN_INPUT_DIR` when set, else `/input/`, and processes every
`*.in` file it finds, in name order.

Output (`/output/<id>.out`): UTF-8 JSON, one object with exactly one
key, `forecasts`: a list with exactly one row for each market of the
case, each exactly `{"id": <the market's id>, "q": <rows>}`. `q` is
exactly 240 rows, row h-1 the forecast for minute at+h, each row
exactly five numbers, the quantiles at 0.05, 0.25, 0.5, 0.75 and 0.95
in that order, each finite, in [0, 1] and non-decreasing. Quantiles
are rounded to 6 decimals. Output is at most ~260 KB per case (limit
1,048,576 bytes). The program writes to `$MAKEFUN_OUTPUT_DIR` when
set, else `/output/`.

## Practice evaluation

Scored 2026-11-13 on the kit's six practice cases (`p00`–`p05`) against
Polymarket's own public price history (the same source the house
recorder uses: `GET /clob/v1/prices-history`, fidelity 1, 1-minute
grid, last point at or before each minute carried forward). Mean
pinball loss, in percentage points, over every market, minute and
quantile of every case (lower is better):

| model | mean pinball loss | median error | 90% coverage |
|---|---|---|---|
| **This entry** | **0.307528** | 0.759405 | 0.9103 |
| House baseline (persistence + `0.0025 * sqrt(h)` spread) | 0.363589 | 0.767503 | 0.9496 |

The baseline figures reproduce the kit's published practice numbers
exactly (loss 0.363589, median error 0.767503, coverage 0.949642),
which validates the fetch and the scoring. The entry beats the house
baseline by ~15% on the practice. The practice is no guide to the
final (the final is graded on the 48 hours after the deadline), and
the house model forecast is the other bar to beat; but the entry is
valid on every practice case, deterministic (two runs byte-identical)
and fast (~0.13 s per case, limit 8 s).

The builder's earlier calibration harness (`eval/`: `evaluate.py`,
`practice_grid.json`, six CSV cases) is kept outside this repository
as a backup; it is not part of the entry. Calibration notes
(Polymarket history, 2026-11-10 to 13):

- The 24 named markets' Yes prices sum to ~1.00 at 2026-11-13 10:25
  UTC (Ian Buck 0.305, Colette Kress 0.146, Rene Haas 0.128, ...),
  confirming they are the mutually exclusive set. The gamma API lists
  107 markets for the event; the other 83 are zero-volume placeholders
  ("Candidate S", ..., "Other") at 0.50 and are not part of the 24.
- The sum was far from 1 earlier (1.58 at 2026-11-12 06:00, ~4.9
  mean over Nov 10–11 when stale 0.485 opening prices dominated) and
  deflated toward 1 over ~1 day. Regression of future changes on
  (share − price): 5% of the gap closes in 60 min, 14% in 240 min.
- Realised horizon volatility is strongly sub-diffusive:
  std of price changes 0.024 (10 min), 0.031 (60 min), 0.040 (240 min)
  in the recent regime — far below `sqrt(h)` scaling.
- Lag-1 autocorrelation of 10-minute changes ≈ −0.33 (bounce /
  partial jump reversion).
- Caveats: only 6 practice cases from 3 days of one event, all in a
  deflationary window (sum falling toward 1); the fitted growth
  exponent (−0.70) is more negative than the directly fitted −0.33
  because pinball rewards narrower intervals when most minutes do not
  move. If the final cases contain a news shock, losses will be
  dominated by that case (practice case 3, the Colette Kress spike,
  scored 1.09 pp for every model). Constants were chosen near, not at,
  the practice optimum to limit overfitting.

## What is still needed before submission

- **I/O format: verified 2026-11-13** against the kit's practice
  cases (see above). The entry runs on the real schema, is valid on
  every practice case, deterministic and ~0.13 s per case (limit 8 s).
- Upload this directory to GitHub and submit the repo URL on make.fun
  (max 10 submissions/day, 10 total). No submission has been made.
  Deadline: 2026-11-13 21:00 UTC.

## License

MIT — see `LICENSE`.
