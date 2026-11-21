"""
Indicator confirmation stage: pattern -> INDICATORS -> Kronos -> risk.

After patterns.py flags a candidate, this checks it against standard
candle indicators (EMA 9/20/50/200 stack, ADX, Supertrend, RSI, MACD,
Stochastic, Bollinger, Donchian, volume/OBV/VWAP) from
src/kronos_indicators. It runs before Kronos, so setups that fight the
indicator picture are dropped without paying for a forecast.

Pure computation, with no broker calls and no model calls. Like
patterns.py it only *filters* signals. It never sizes or approves
orders; risk.py stays the final gate.

Config (config.yaml -> `indicators:`); a missing section means disabled:

    indicators:
      enabled: true
      mode: filter            # filter = can reject; log = record only, never blocks
      min_alignment: 0        # bias score in the trade's direction, -100..100
      require_ma_trend: false # fast MA must be on the trade's side of the slow MA
      ma_fast: 20
      ma_slow: 50
      ma_kind: ema            # sma | ema | wma | hma | vwma
      max_rsi_for_buy: 75     # skip BUYs into an overbought RSI (null = off)
      min_rsi_for_sell: 25    # skip SELLs into an oversold RSI (null = off)
      min_bars: 60            # fewer closed bars -> gate passes with a note
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.kronos_indicators import ma_crossover, technical_snapshot

logger = logging.getLogger(__name__)

INTRADAY_TIMEFRAMES = {"MIN_1", "MIN_3", "MIN_5", "MIN_15", "HOUR_1"}


@dataclass
class IndicatorConfig:
    enabled: bool = False
    mode: str = "filter"
    min_alignment: float = 0.0
    require_ma_trend: bool = False
    ma_fast: int = 20
    ma_slow: int = 50
    ma_kind: str = "ema"
    max_rsi_for_buy: float | None = 75.0
    min_rsi_for_sell: float | None = 25.0
    min_bars: int = 60

    @classmethod
    def from_dict(cls, raw: dict | None) -> "IndicatorConfig":
        raw = dict(raw or {})
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        unknown = set(raw) - set(known)
        if unknown:
            logger.warning("Ignoring unknown indicators settings: %s", sorted(unknown))
        cfg = cls(**known)
        cfg.mode = str(cfg.mode).lower()
        if cfg.mode not in ("filter", "log"):
            raise ValueError(f"indicators.mode must be 'filter' or 'log', got {cfg.mode!r}")
        return cfg


@dataclass
class IndicatorVerdict:
    passed: bool
    alignment: float | None              # bias score signed toward the trade (+ = agrees)
    reasons: list[str] = field(default_factory=list)   # why it would be rejected
    summary: dict = field(default_factory=dict)        # compact numbers for audit / Telegram
    skipped: str | None = None           # set when the gate could not run

    def short_text(self) -> str:
        """One line for Telegram."""
        if self.skipped:
            return f"indicators skipped ({self.skipped})"
        s = self.summary
        parts = [f"bias {s.get('bias')} {self.alignment:+.0f}"]
        if s.get("rsi_14") is not None:
            parts.append(f"RSI {s['rsi_14']:.0f}")
        if s.get("adx") is not None:
            parts.append(f"ADX {s['adx']:.0f}")
        if s.get("ma_state"):
            parts.append(f"{s['ma_pair']} {'up' if s['ma_state'] == 'fast_above_slow' else 'down'}")
        return ", ".join(parts)


def _to_market_time(df: pd.DataFrame, tz) -> pd.DataFrame:
    """Bars arrive indexed by naive-UTC bar start (see bars_to_df). Convert
    to exchange-local time so the session VWAP resets at the local date."""
    if tz is None or df.index.tz is not None:
        return df
    out = df.copy()
    out.index = out.index.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    return out


def evaluate(df: pd.DataFrame, direction: str, cfg: IndicatorConfig,
             timeframe: str = "DAY", tz=None) -> IndicatorVerdict:
    """Check a pattern's `direction` ("BUY"/"SELL") against the indicators
    on the closed bars in `df` (columns open/high/low/close[/volume])."""
    if len(df) < cfg.min_bars:
        return IndicatorVerdict(True, None, skipped=f"only {len(df)} bars, need {cfg.min_bars}")

    sign = 1 if direction == "BUY" else -1
    bars = _to_market_time(df, tz)
    snap = technical_snapshot(bars, intraday=timeframe in INTRADAY_TIMEFRAMES)
    if "error" in snap:
        return IndicatorVerdict(True, None, skipped=snap["error"])
    mac = ma_crossover(bars, cfg.ma_fast, cfg.ma_slow, cfg.ma_kind, lookback=10)

    alignment = sign * (snap["bias_score"] or 0.0)
    rsi = snap["momentum"]["rsi_14"]
    reasons = []

    if alignment < cfg.min_alignment:
        reasons.append(f"indicator bias {snap['bias']} ({alignment:+.0f} toward {direction}, "
                       f"need {cfg.min_alignment:+.0f})")
    if cfg.require_ma_trend:
        want = "fast_above_slow" if sign > 0 else "fast_below_slow"
        if mac["state"] != want:
            reasons.append(f"{mac['pair']} trend against {direction}")
    if rsi is not None:
        if sign > 0 and cfg.max_rsi_for_buy is not None and rsi >= cfg.max_rsi_for_buy:
            reasons.append(f"RSI {rsi:.0f} overbought for a BUY (max {cfg.max_rsi_for_buy:.0f})")
        if sign < 0 and cfg.min_rsi_for_sell is not None and rsi <= cfg.min_rsi_for_sell:
            reasons.append(f"RSI {rsi:.0f} oversold for a SELL (min {cfg.min_rsi_for_sell:.0f})")

    summary = {
        "bias": snap["bias"], "bias_score": snap["bias_score"], "alignment": alignment,
        "rsi_14": rsi, "adx": snap["trend"]["adx"], "trend_strength": snap["trend"]["strength"],
        "supertrend_dir": snap["trend"]["supertrend_dir"],
        "ema_20": snap["trend"]["ema_20"], "ema_50": snap["trend"]["ema_50"], "ema_200": snap["trend"]["ema_200"],
        "macd_hist": snap["momentum"]["macd_hist"], "atr_14": snap["volatility"]["atr_14"],
        "atr_pct": snap["volatility"]["atr_pct"], "squeeze": snap["volatility"]["squeeze"],
        "ma_pair": mac["pair"], "ma_state": mac["state"],
        "ma_recent_crosses": mac["recent_crosses"],
        "signals": [f"{'+' if s['direction'] > 0 else '-'}{s['signal']}" for s in snap["signals"]],
    }
    passed = not reasons or cfg.mode == "log"
    return IndicatorVerdict(passed, alignment, reasons, summary)
