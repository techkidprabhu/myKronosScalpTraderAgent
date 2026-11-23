---
name: scalp-starter
description: STARTER scalp method for 1-5 minute charts (3-minute by default) built on the four core scalping models - momentum/breakout (opening-range breakout), trend pullback (VWAP / 9-EMA), mean reversion (Bollinger + RSI extremes in a range) and liquidity sweep (stop-runs through prior highs/lows) - long and short, with R-based targets, VWAP-cross and time exits. A placeholder until Prabhu's own scalp rules replace it. Use when deciding whether to take, manage or exit a short intraday scalp, or which scalping model fits the current market.
version: 0.2
---

# Scalp Starter Skill

**This is a starter.** It is a common, widely used way to scalp, written so
the scalp agent has something to trade from day one. Replace it with your
own rules (or edit the numbers in section 6) once you have them.

How this file is used:
- **By an AI:** follow sections 1-8 as the decision procedure.
- **By the scalp agent:** with Command Center -> Who decides trades -> Skill
  file (or Skill + Kronos), the agent's built-in `scalp_vwap_orb` engine
  trades it using the settings block in section 9. Your risk limits (max
  loss per trade, daily limits, per-stock limits, flat-by-close) and the
  scalp management (time stop, break-even stop) still apply.

---------------------------------------------------------------------------

# Core scalping models

Most scalping strategies on 1-minute to 5-minute charts fall into four
primary execution models:

| Model | Core mechanism | Common tools & triggers | Best market condition |
|---|---|---|---|
| **Momentum / Breakout** | Enters on volume surges as price clears key intraday levels to ride short-term order flow. | Opening range breakouts, volume spikes, Level 2 / tape speed. | High-volatility session opens (e.g. London/NY overlap). |
| **Trend Pullback** | Buys minor dips or sells brief rallies within an established short-term trend. | 9/21 EMA crosses, Volume Weighted Average Price (VWAP) bounces. | Clean, directional intraday trends. |
| **Mean Reversion** | Fades overextended price spikes back toward an average value. | Bollinger Bands, RSI extremes (30/70), exhaustion candles. | Ranging, choppy, or low-catalyst sessions. |
| **Liquidity Sweep** | Enters after price briefly breaks support/resistance to trigger stops, then snaps back. | Wyckoff springs/upthrusts, Fair Value Gaps (FVG), rejection wicks. | Stop-hunting zones around prior highs/lows. |

In this file: Momentum / Breakout = section 2 (`orb`), Trend Pullback =
section 3 (`pullback`), Mean Reversion = section 4 (`mean_reversion`),
Liquidity Sweep = section 5 (`sweep`). Section 6 says which to use when.
`setups:` in section 9 decides which ones the agent trades.

What the agent checks itself, and what stays with you: it reads candles
only, so Level 2 / tape speed (model 1) and Fair Value Gaps (model 4) are
judgement calls for a person or an AI reading the chart - use them as extra
confirmation, the agent doesn't wait for them.

---------------------------------------------------------------------------

# 1. Chart and session

- 3-minute candles (`timeframe: MIN_3`, the default). 1-minute and 5-minute
  work too: set `orb_minutes` to at least one candle and scale the candle
  counts (a 5-minute chart needs fewer, a 1-minute chart more).
- Regular session only. VWAP restarts every day at the open.
- 9 EMA (fast) and 20 EMA (slow) show the short-term trend (`ema_slow: 21`
  for the 9/21 version).
- Bollinger Bands (20, 2), RSI (14) and ADX (14) for mean reversion.

# 2. Model 1: Momentum / Breakout - opening-range breakout (ORB)

1. Mark the high and low of the first 3-minute candle (the opening range).
2. **Long:** the first candle that closes above the range high, above VWAP,
   on at least 1.5x the average volume of the last 20 candles.
   **Short:** the first close below the range low, below VWAP, on the same
   volume.
3. Only in the first 90 minutes of the session.
4. Entry at the signal candle's close. Stop at the middle of the opening
   range. Target 1.5x the risk.
5. Best right after a high-volatility open. On IG forex and indices the
   strongest is the London/NY overlap (about 08:00-11:00 New York time); the
   agent's ORB uses each market's own session open.

# 3. Model 2: Trend Pullback - VWAP / 9 EMA pullback

1. **Uptrend:** 9 EMA above 20 EMA and price above VWAP.
   **Downtrend:** the mirror.
2. Price pulls back to VWAP or the 9 EMA (within 0.05%).
3. The next candle closes back in the trend's direction, beyond the 9 EMA:
   green for longs, red for shorts.
4. Entry at that candle's close. Stop just beyond the pull-back's low (longs)
   or high (shorts). Target 1.5x the risk.

# 4. Model 3: Mean Reversion - Bollinger + RSI fade

Only in a **ranging** market: ADX (14) below 20 before the spike. In a
trend, fading moves is the wrong model - use section 3 instead.

1. **Long:** a candle closes **below the lower Bollinger Band** (20, 2) with
   RSI (14) at **30 or lower** - an overextended drop.
   **Short:** a close **above the upper band** with RSI **70 or higher**.
2. The next candle is the exhaustion signal: it closes **back inside the
   band**, green for longs, red for shorts.
3. Entry at that candle's close. Stop just beyond the spike's low (longs) or
   high (shorts). Target: the **middle band** (the 20-period average) -
   only take it if that is at least 1x the risk away.

# 5. Model 4: Liquidity Sweep - spring / upthrust

Stops cluster just beyond obvious levels: the **prior session's high and
low**, and the **recent swing high and low** (last 20 candles).

1. **Long (spring):** a candle trades **below** one of those lows - running
   the stops - then **closes back above it**, with a lower rejection wick at
   least half the candle and the close in the candle's upper half.
   **Short (upthrust):** the mirror above a prior high.
2. Entry at that candle's close. Stop just beyond the wick. Target 1.5x
   the risk.
3. Extra confirmation (judgement, not checked by the agent): a Fair Value
   Gap left by the move back - a three-candle gap between candle 1's high
   and candle 3's low (longs) - shows real buying stepped in.

# 6. Which model when

- **First hour after the open, volume surging:** Momentum / Breakout.
- **Clean trend (9 EMA above/below 20 EMA, price on one side of VWAP):**
  Trend Pullback. Don't fade it.
- **Choppy, ranging, no news (ADX below 20):** Mean Reversion. Skip breakouts
  - they tend to fail in ranges.
- **Price poking through yesterday's high/low or an obvious swing level:**
  Liquidity Sweep - wait for the close back inside, never anticipate it.
- If two models disagree on the same candle, stand aside.

# 7. Exits

- Stop or target, whichever comes first.
- At 1R in profit, move the stop to break-even (the agent does this).
- Breakout and pullback trades: close if a candle closes back across VWAP
  against the trade. (Not for mean-reversion or sweep trades - they often
  start on the far side of VWAP.)
- Close after 10 candles (30 minutes) if neither stop nor target was hit
  (time stop).
- Flat before the close: scalps never hold overnight.

# 8. When not to trade

- The first 5 minutes of the session (opening print).
- The last 20 minutes.
- After the day's loss limit or trade limit.
- A stock with 3 losing scalps today.
- Mean reversion in a trend; breakouts in a dead range.

---------------------------------------------------------------------------

# 9. Settings the agent trades with

The agent reads only this block from the file. Change a number and save:
the next scan uses it, no restart.

```yaml
strategy: scalp_vwap_orb
# Which models the agent trades. both = orb + pullback (the original two).
# Add the others by name, e.g. [orb, pullback, mean_reversion, sweep], or all.
setups: both
sides: both               # both | long | short (shorts need risk.allow_short)
orb_minutes: 3            # opening range: the first 3-minute candle
orb_window_minutes: 90    # ORB signals only this long after the open
orb_stop: mid             # mid of the range | candle (beyond the breakout candle)
volume_mult: 1.5          # signal volume vs the average of the last volume_period candles
volume_period: 20         # 20 x 3-minute candles = the last hour
ema_fast: 9
ema_slow: 20
pullback_touch_pct: 0.05  # how close to VWAP / the 9 EMA counts as a touch
stop_buffer_pct: 0.02
target_r: 1.5
max_wait_candles: 2       # cancel an unfilled entry after 2 candles (6 minutes)
max_hold_candles: 10      # close after 10 candles (30 minutes) in the trade
exit_on_vwap_cross: true
use_trailing_stop: true   # the agent's trailing stop on these trades
min_quality: 0            # 1 = only signals on 2x volume or more
# Mean reversion (section 4)
bb_period: 20
bb_std: 2.0
rsi_period: 14
rsi_low: 30
rsi_high: 70
adx_period: 14
mr_max_adx: 20            # only when ADX is below this before the spike (ranging); 0 = no check
mr_min_r: 1.0             # the middle band must be at least this many R away
# Liquidity sweep (section 5)
sweep_levels: both        # both | prior_day | swing
sweep_lookback: 20        # swing high / low over the last 20 candles
sweep_wick_ratio: 0.5     # rejection wick at least half the candle
```
