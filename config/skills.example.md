# My trading rules

These rules run after the indicator check and before Kronos. A rule can only
stop a signal; it never creates one. They are switched on with
`skills.enabled: true` in config.yaml (or Config & Risk on the dashboard).
`skills.mode: log` records what they would have stopped without stopping
anything - a good way to try a new rule for a few days.

Your own copy lives in config/skills.md. Until it exists, the agent reads
this example. Edit on the dashboard (Strategies & Tactics -> Your rules)
or in any text editor; changes are picked up without a restart.

How to write a rule:
- a "## " heading is the rule's name
- `applies to:` buy, sell or both
- `patterns:` optional - only for these patterns (comma separated)
- `block if:` stop the signal when this is true (several lines = any of them)
- `only if:` stop the signal unless this is true (several lines = all of them)
- "and" joins conditions on one line; `why:` is your note; `off` skips the rule
The measures you can use are listed on the dashboard next to the editor.

These starting rules are common price-action habits. Change the numbers to
your own, switch off what you don't want with `- off`, and add your own.

## Don't chase far above VWAP
- applies to: buy
- block if: price_vs_vwap_pct > 1.5
- why: Buying well above the day's average price often buys the top of the move.

## Don't chase far below VWAP
- applies to: sell
- block if: price_vs_vwap_pct < -1.5
- why: The same for shorts.

## Trade with the 200 EMA
- applies to: buy
- only if: price_vs_ema200_pct > 0
- why: Long setups below the 200 EMA fight the bigger trend.

## Short only below the 200 EMA
- applies to: sell
- only if: price_vs_ema200_pct < 0
- why: The same for shorts.

## Skip climax candles
- block if: candle_range_atr > 2.5
- why: A signal candle more than 2.5 times the usual range is often exhaustion, and the stop ends up far away.

## Engulfing needs volume
- patterns: bullish_engulfing, bearish_engulfing
- block if: volume_ratio < 0.8
- why: An engulfing candle on below-average volume is a weak reversal.

## Inside bars need a trend
- patterns: inside_bar
- block if: adx < 20
- why: An inside bar is a continuation bet; without a trend there is nothing to continue.

## No buys straight into resistance
- applies to: buy
- block if: room_to_high_pct < 0.3 and trend_vs_trade is against
- why: Buying just under the recent high against the trend leaves no room to the target.

## Don't buy big gap-ups
- applies to: buy
- block if: gap_pct > 3
- why: After a large gap up, the first move is often a fill back down.

## Don't short big gap-downs
- applies to: sell
- block if: gap_pct < -3
- why: The same for shorts.

## Avoid the last half hour
- block if: minutes_to_close < 30
- why: Late entries have little time to work, and intraday mode closes everything before the close.
- off
