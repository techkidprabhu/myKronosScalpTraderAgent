---
name: scalp-starter
description: STARTER scalp method for 3-minute charts - opening-range breakout (first 3-minute candle) and VWAP / 9-EMA pullback, long and short, 1.5R targets, VWAP-cross and time exits. A placeholder until Prabhu's own scalp rules replace it. Use when deciding whether to take, manage or exit a short intraday scalp.
version: 0.1
---

# Scalp Starter Skill

**This is a starter.** It is a common, widely used way to scalp, written so
the scalp agent has something to trade from day one. Replace it with your
own rules (or edit the numbers in section 6) once you have them.

How this file is used:
- **By an AI:** follow sections 1-5 as the decision procedure.
- **By the scalp agent:** with Command Center -> Who decides trades -> Skill
  file, the agent's built-in `scalp_vwap_orb` engine trades it using the
  settings block in section 6. Kronos is not called. Your risk limits (max
  loss per trade, daily limits, per-stock limits, flat-by-close) and the
  scalp management (time stop, break-even stop) still apply.

---------------------------------------------------------------------------

# 1. Chart and session

- 3-minute candles (`timeframe: MIN_3`, the default). 1-minute works too
  (`MIN_1`): then set `orb_minutes: 1` or more and longer candle counts.
- Regular session only. VWAP restarts every day at the open.
- 9 EMA (fast) and 20 EMA (slow) show the short-term trend.

# 2. Setup A - Opening-range breakout (ORB)

1. Mark the high and low of the first 3-minute candle (the opening range).
2. **Long:** the first candle that closes above the range high, above VWAP,
   on at least 1.5x the average volume of the last 20 candles.
   **Short:** the first close below the range low, below VWAP, on the same
   volume.
3. Only in the first 90 minutes of the session.
4. Entry at the signal candle's close. Stop at the middle of the opening
   range. Target 1.5x the risk.

# 3. Setup B - VWAP / 9 EMA pullback

1. **Uptrend:** 9 EMA above 20 EMA and price above VWAP.
   **Downtrend:** the mirror.
2. Price pulls back to VWAP or the 9 EMA (within 0.05%).
3. The next candle closes back in the trend's direction, beyond the 9 EMA:
   green for longs, red for shorts.
4. Entry at that candle's close. Stop just beyond the pull-back's low (longs)
   or high (shorts). Target 1.5x the risk.

# 4. Exits

- Stop or target, whichever comes first.
- At 1R in profit, move the stop to break-even (the agent does this).
- Close if a candle closes back across VWAP against the trade.
- Close after 10 candles (30 minutes) if neither stop nor target was hit
  (time stop).
- Flat before the close: scalps never hold overnight.

# 5. When not to trade

- The first 5 minutes of the session (opening print).
- The last 20 minutes.
- After the day's loss limit or trade limit.
- A stock with 3 losing scalps today.

---------------------------------------------------------------------------

# 6. Settings the agent trades with

The agent reads only this block from the file. Change a number and save:
the next scan uses it, no restart.

```yaml
strategy: scalp_vwap_orb
setups: both              # both | orb | pullback
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
```
