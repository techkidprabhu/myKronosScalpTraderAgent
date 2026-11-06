"""
Deterministic candlestick pattern detection over an OHLC DataFrame.

Expects a DataFrame with columns: open, high, low, close (indexed by
time, most recent last). Each detector returns a signal for the LAST
row only - call after a candle has closed.

This is intentionally simple and rule-based - the point of the pipeline
is that pattern detection here is fast/cheap and fully inspectable, and
Kronos (kronos_agent.py) is used only to confirm/reject candidates it
flags, not to generate signals from scratch.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class PatternSignal:
    name: str
    direction: str  # "BUY" or "SELL"
    strength: float  # 0-1, rule-based confidence


def _body(row):
    return abs(row["close"] - row["open"])


def _range(row):
    return row["high"] - row["low"]


def _is_bullish(row):
    return row["close"] > row["open"]


def detect_engulfing(df: pd.DataFrame) -> PatternSignal | None:
    if len(df) < 2:
        return None
    prev, cur = df.iloc[-2], df.iloc[-1]
    if _is_bullish(cur) and not _is_bullish(prev):
        if cur["open"] <= prev["close"] and cur["close"] >= prev["open"]:
            return PatternSignal("bullish_engulfing", "BUY", 0.7)
    if not _is_bullish(cur) and _is_bullish(prev):
        if cur["open"] >= prev["close"] and cur["close"] <= prev["open"]:
            return PatternSignal("bearish_engulfing", "SELL", 0.7)
    return None


def detect_hammer(df: pd.DataFrame) -> PatternSignal | None:
    cur = df.iloc[-1]
    rng = _range(cur)
    if rng == 0:
        return None
    body = _body(cur)
    lower_wick = min(cur["open"], cur["close"]) - cur["low"]
    upper_wick = cur["high"] - max(cur["open"], cur["close"])
    if lower_wick >= 2 * body and upper_wick <= 0.2 * rng and body <= 0.35 * rng:
        return PatternSignal("hammer", "BUY", 0.6)
    return None


def detect_shooting_star(df: pd.DataFrame) -> PatternSignal | None:
    cur = df.iloc[-1]
    rng = _range(cur)
    if rng == 0:
        return None
    body = _body(cur)
    upper_wick = cur["high"] - max(cur["open"], cur["close"])
    lower_wick = min(cur["open"], cur["close"]) - cur["low"]
    if upper_wick >= 2 * body and lower_wick <= 0.2 * rng and body <= 0.35 * rng:
        return PatternSignal("shooting_star", "SELL", 0.6)
    return None


def detect_inside_bar(df: pd.DataFrame) -> PatternSignal | None:
    if len(df) < 2:
        return None
    prev, cur = df.iloc[-2], df.iloc[-1]
    if cur["high"] < prev["high"] and cur["low"] > prev["low"]:
        # Direction is a breakout bet in the direction of the prior trend leg
        direction = "BUY" if _is_bullish(prev) else "SELL"
        return PatternSignal("inside_bar", direction, 0.45)
    return None


DETECTORS = [detect_engulfing, detect_hammer, detect_shooting_star, detect_inside_bar]


def scan(df: pd.DataFrame) -> list[PatternSignal]:
    """Run all detectors against the latest closed candle and return every
    pattern that fired (usually zero or one, occasionally more)."""
    signals = []
    for detector in DETECTORS:
        result = detector(df)
        if result:
            signals.append(result)
    return signals
