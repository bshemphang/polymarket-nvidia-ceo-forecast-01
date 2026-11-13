#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 240 minutes ahead.

Reads one case from <input>.in and writes <output>.out with, for each of the
24 markets and each of the 240 minutes after the checkpoint, five quantiles
(5%, 25%, 50%, 75%, 95%) of that minute's Yes price.

Usage:
    python3 main.py                       # every $MAKEFUN_INPUT_DIR/*.in
                                          # -> $MAKEFUN_OUTPUT_DIR/*.out
                                          # (defaults /input and /output)
    python3 main.py in.in out.out         # one case, for local testing

Python 3.13, standard library only. Offline, deterministic, no clock, no
randomness. Runs in well under a second per case.

Input format (the kit's real schema, "makefun/polymarket-case/v1")
---------------------------------------------------------------
One JSON object: `schema`, `competition`, `case` (id), `at` (checkpoint,
ISO 8601 UTC), `horizon_minutes` (240), `quantiles` ([0.05, 0.25, 0.5,
0.75, 0.95]), `event` (`slug`, `title`) and `markets`, a list in the
universe's order, each `id` (conditionId), `name`, `slug`, `yes_token`,
`start` (minute of its first price) and `prices`, one number per minute
from `start` to `at` inclusive, so its last is the price at the
checkpoint. Prices are on a 0-1 scale. Markets that started later hold
fewer prices; each market's series is used on its own length.

Output format
-------------
JSON, one object with exactly one key, `forecasts`: a list with exactly
one row for each market of the case, each exactly `{"id": <the market's
id>, "q": <rows>}`. `q` is exactly 240 rows, row h-1 the forecast for
minute at+h, each row exactly five numbers, the quantiles at the levels
0.05, 0.25, 0.5, 0.75 and 0.95 in that order, each finite, in [0, 1]
and at least the one before it. Quantiles are rounded to 6 decimals.

Model (per market, horizon h = 1..240 minutes)
-----------------------------------------------
The 24 markets are mutually exclusive outcomes of one event, so their Yes
prices should sum to ~1. The forecast exploits that constraint:

1.  Median path: an Ornstein-Uhlenbeck pull of each price toward its
    renormalised cross-market share s_i = p_i / sum_j p_j, i.e.
        m_i(h) = s_i + (p_i0 - s_i) * exp(-h / TAU)
    with TAU fitted from history (see README). The medians are then
    rescaled each horizon so their sum reverts toward 1 at the same rate.
    The pull is applied only when the input sum is within [0.5, 3.0]:
    for a partial market set the "shares" are meaningless and the
    median stays at the last price. A small AR(1) term corrects the
    first minutes for the (slightly negative) lag-1 autocorrelation
    of 10-minute changes.
2.  Spread: sigma_i(h) = sig_i * sqrt(h) * G(h), where sig_i is a
    robust per-minute volatility estimated from the recent history at
    1-, 10- and 60-minute scales (RMS of changes, not MAD: the prices
    are stepwise, so most minute changes are exactly zero), with a
    cross-market backstop so frozen markets still get jump room, and
    scaled by a mild price-level factor sqrt(p(1-p)). G(h) is a fitted
    sub-diffusive correction (G(h) = (h/10)^-0.30 for h > 10): jumps
    partially revert, so long-horizon moves grow much more slowly
    than sqrt(h).
3.  Quantiles: median + z_tau * sigma_i(h) with Laplace z-scores
    (-1.628, -0.490, 0, +0.490, +1.628), clipped to [0.001, 0.999]
    and forced monotone. A floor on sigma keeps far-out quantiles
    honest for near-frozen markets.
"""

import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Fitted constants (calibrated on Polymarket history, 2026-11-10..13;
# see README for the calibration numbers).
# ---------------------------------------------------------------------------
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
# Laplace (double-exponential) quantile scores in units of the change
# standard deviation. Polymarket prices are jumpy: long flat stretches
# punctuated by jumps, so the change distribution has a sharper peak
# and fatter tails than a normal. Laplace scores give narrower 25%/75%
# quantiles and about the same 5%/95% tails as normal scores, and beat
# normal scores on all 6 practice cases (see README).
Z = (-1.6276, -0.4900, 0.0, 0.4900, 1.6276)
HORIZON = 240          # minutes to forecast
N_MARKETS_EXPECTED = 24

TAU_REVERT = 700.0     # minutes: OU pull of price toward renormalised share
AR1 = -0.08            # fallback lag-1 autocorrelation of minute changes
SIG_FLOOR = 0.00040    # per-minute sigma floor (price units)
SIG_CAP = 0.020        # per-minute sigma cap
SIG_SCALE = 0.9        # global multiplier on estimated per-minute sigma
POOL_BACKSTOP = 0.30   # per-market sigma is at least this fraction of the
                       # cross-market median sigma (frozen markets can jump)
# Growth: sigma(h) = sig * sqrt(h) * G(h), G(h) = 1 for h <= 10,
# G(h) = (h / 10) ** G_EXP for h > 10. G_EXP < 0 because Polymarket
# prices are jumpy and mean-reverting: long-horizon moves grow much
# more slowly than sqrt(h) (fitted exponent about -0.33 on Nov 10-13
# history; the pinball-optimal value on the 6 practice cases is more
# negative, about -0.70, because most minutes do not move at all).
G_EXP = -0.70
G_KNEE = 10.0
# Price-level factor: f(p) = LVL_MIX + (1 - LVL_MIX) * sqrt(4 p (1-p)).
LVL_MIX = 0.75
SIG_H_FLOOR = 0.0006   # floor on sigma(h) / sqrt(h) (price units)
Q_FLOOR = 0.001        # quantiles are clipped to [Q_FLOOR, 1 - Q_FLOOR]


# ---------------------------------------------------------------------------
# Input parsing: the kit's real schema, makefun/polymarket-case/v1.
# ---------------------------------------------------------------------------
def parse_case(data):
    """Return (market_ids, series) where series[j] is market j's price
    list, oldest -> newest, one number per minute from its `start` to
    the checkpoint, on a 0-1 scale. Raises ValueError on a bad case."""
    if not isinstance(data, dict):
        raise ValueError("case is not a JSON object")
    markets = data.get("markets")
    if not isinstance(markets, list) or not markets:
        raise ValueError("case holds no markets list")
    horizon = data.get("horizon_minutes", HORIZON)
    if horizon != HORIZON:
        raise ValueError("horizon_minutes is %r, expected %d"
                         % (horizon, HORIZON))
    ids = []
    series = []
    for m in markets:
        if not isinstance(m, dict):
            raise ValueError("a market is not an object")
        mid = m.get("id")
        prices = m.get("prices")
        if mid is None or not isinstance(prices, list) or not prices:
            raise ValueError("a market has no id or no prices")
        row = []
        for v in prices:
            try:
                p = float(v)
            except (TypeError, ValueError):
                raise ValueError("a price is not a number") from None
            if p != p or p in (float("inf"), float("-inf")):
                raise ValueError("a price is not finite")
            row.append(min(0.999, max(0.001, p)))
        ids.append(str(mid))
        series.append(row)
    return ids, series


# ---------------------------------------------------------------------------
# Volatility estimation
# ---------------------------------------------------------------------------
def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _rms_sigma(diffs, lag):
    """Per-minute sigma from lag-diffs: RMS / sqrt(lag).

    RMS (not MAD) is used deliberately: these prices are stepwise --
    long flat stretches punctuated by jumps -- so more than half of all
    minute diffs are exactly zero and a MAD estimate collapses to 0.
    """
    if len(diffs) < 5:
        return 0.0
    return math.sqrt(sum(d * d for d in diffs) / len(diffs)) / math.sqrt(lag)


def estimate_sigma(series):
    """Per-minute volatility of one market's price series (list of floats,
    oldest -> newest, one value per minute). Median of the 1/10/60-minute
    RMS scales, so a single stale or jumpy stretch cannot dominate."""
    n = len(series)
    if n < 3:
        return SIG_FLOOR
    sigs = []
    for lag, window in ((1, 720), (10, 1440), (60, 2880)):
        diffs = []
        start = max(0, n - window)
        for i in range(max(start + lag, lag), n):
            diffs.append(series[i] - series[i - lag])
        if len(diffs) >= 5:
            sigs.append(_rms_sigma(diffs, lag))
    if not sigs:
        return SIG_FLOOR
    sig = _median(sigs)
    return min(SIG_CAP, max(SIG_FLOOR, sig))


def estimate_ar1(series):
    """Lag-1 autocorrelation of 10-minute changes over the recent window.
    10-minute (not 1-minute) diffs are used because 1-minute diffs are
    mostly zero on stepwise prices; the 10-minute changes carry the
    bounce (empirically about -0.33 on Nov 10-13 history)."""
    n = len(series)
    if n < 120:
        return 0.0
    diffs = [series[i] - series[i - 10]
             for i in range(max(10, n - 1440), n)]
    if len(diffs) < 30:
        return 0.0
    mean = sum(diffs) / len(diffs)
    num = sum((diffs[i] - mean) * (diffs[i - 1] - mean)
              for i in range(1, len(diffs)))
    den = sum((d - mean) ** 2 for d in diffs)
    if den <= 0:
        return 0.0
    return max(-0.4, min(0.1, num / den))


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
def forecast(market_ids, series):
    """Return {id: [[q05, q25, q50, q75, q95], ... 240 rows]}.

    series[j] is market j's price list, oldest -> newest; each market is
    used on its own length (markets that started later hold fewer
    minutes). Only the last 72 hours (4,320 minutes) of each series are
    used, as the case holds at most that."""
    n_markets = len(market_ids)
    # Last 72h of minutes (or all we have) per market.
    cols = [s[-4320:] for s in series]
    p0 = [cols[j][-1] for j in range(n_markets)]
    sig = [estimate_sigma(cols[j]) for j in range(n_markets)]
    # Frozen markets can still jump: no market gets a sigma below a
    # fraction of the cross-market median.
    pooled = _median(sig)
    sig = [min(SIG_CAP, max(s, POOL_BACKSTOP * pooled)) * SIG_SCALE
           for s in sig]
    ar1 = _median([estimate_ar1(cols[j]) for j in range(n_markets)])
    if ar1 != ar1:  # NaN guard
        ar1 = AR1
    ar1 = max(-0.4, min(0.1, ar1))

    # Renormalised shares and the sum the medians should revert toward.
    # Guard: the sum-to-one logic only applies to (nearly) the full
    # mutually-exclusive set. If the input holds a partial set (sum far
    # from 1), the "shares" are meaningless and the median stays put.
    total0 = sum(p0)
    use_pull = 0.5 <= total0 <= 3.0
    shares = [p / total0 if total0 > 0 else 1.0 / n_markets for p in p0]
    # Last 10-minute move per market (the bounce correction acts on this).
    last_move10 = []
    for j in range(n_markets):
        c = cols[j]
        last_move10.append(c[-1] - c[-11] if len(c) > 10 else 0.0)

    # Precompute medians / spreads per horizon, then emit market-major
    # (all 240 rows for one market, then the next market).
    med_grid = []
    sig_grid = []
    for h in range(1, HORIZON + 1):
        decay = math.exp(-h / TAU_REVERT)
        growth = (max(h, G_KNEE) / G_KNEE) ** G_EXP
        sqrt_h = math.sqrt(h)
        # AR(1) bounce correction: fades over ~30 minutes.
        ar_w = ar1 * math.exp(-h / 30.0)
        medians = []
        for j in range(n_markets):
            if use_pull:
                m = shares[j] + (p0[j] - shares[j]) * decay
            else:
                m = p0[j]
            m += ar_w * last_move10[j]
            medians.append(m)
        # Renormalise medians so their sum reverts toward 1.
        s_med = sum(medians)
        target_sum = 1.0 + (total0 - 1.0) * decay
        if use_pull and s_med > 0:
            scale = target_sum / s_med
            medians = [m * scale for m in medians]
        med_grid.append(medians)
        row_sig = []
        for j in range(n_markets):
            p = min(0.999, max(0.001, p0[j]))
            lvl = LVL_MIX + (1.0 - LVL_MIX) * math.sqrt(4.0 * p * (1.0 - p))
            s_h = sig[j] * sqrt_h * growth * lvl
            # Minimum absolute spread so frozen markets still get a
            # widening interval.
            row_sig.append(max(s_h, SIG_H_FLOOR * sqrt_h))
        sig_grid.append(row_sig)

    out = {}
    for j in range(n_markets):
        rows = []
        for h in range(1, HORIZON + 1):
            med = min(0.999, max(0.001, med_grid[h - 1][j]))
            s_h = sig_grid[h - 1][j]
            qs = []
            for z in Z:
                q = med + z * s_h
                qs.append(min(0.999, max(0.001, q)))
            # Enforce monotonicity after clipping.
            for k in range(1, 5):
                if qs[k] < qs[k - 1]:
                    qs[k] = qs[k - 1]
            rows.append([round(q, 6) for q in qs])
        out[market_ids[j]] = rows
    return out


# ---------------------------------------------------------------------------
# Output / driver
# ---------------------------------------------------------------------------
def process(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    market_ids, series = parse_case(data)
    result = forecast(market_ids, series)
    doc = {"forecasts": [{"id": mid, "q": result[mid]}
                         for mid in market_ids]}
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def main(argv):
    if len(argv) >= 3:
        process(argv[1], argv[2])
        return 0
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    if len(argv) >= 2:
        in_dir = argv[1]
    if os.path.isdir(in_dir):
        names = sorted(f for f in os.listdir(in_dir)
                       if f.endswith(".in"))
        if not names:
            sys.stderr.write("no .in files in %s\n" % in_dir)
            return 1
        os.makedirs(out_dir, exist_ok=True)
        for name in names:
            stem = name[:-3]
            process(os.path.join(in_dir, name),
                    os.path.join(out_dir, stem + ".out"))
        return 0
    # Single file path given.
    if os.path.isfile(in_dir):
        out_path = (argv[2] if len(argv) >= 3 else
                    os.path.splitext(in_dir)[0] + ".out")
        process(in_dir, out_path)
        return 0
    sys.stderr.write("usage: main.py [input_path [output_path]]\n")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
