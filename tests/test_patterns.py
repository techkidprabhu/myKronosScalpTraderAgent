import pandas as pd
import pytest

from src.patterns import (
    detect_engulfing,
    detect_hammer,
    detect_inside_bar,
    detect_shooting_star,
    scan,
)


def test_bullish_engulfing():
    # Day 1: Bearish candle (open 100, close 90)
    # Day 2: Bullish candle engulfing day 1 (open 88, close 102)
    df = pd.DataFrame([
        {"open": 100.0, "high": 102.0, "low": 89.0, "close": 90.0},
        {"open": 88.0, "high": 104.0, "low": 87.0, "close": 102.0},
    ])
    sig = detect_engulfing(df)
    assert sig is not None
    assert sig.name == "bullish_engulfing"
    assert sig.direction == "BUY"


def test_bearish_engulfing():
    # Day 1: Bullish candle (open 90, close 100)
    # Day 2: Bearish candle engulfing day 1 (open 102, close 88)
    df = pd.DataFrame([
        {"open": 90.0, "high": 101.0, "low": 89.0, "close": 100.0},
        {"open": 102.0, "high": 103.0, "low": 87.0, "close": 88.0},
    ])
    sig = detect_engulfing(df)
    assert sig is not None
    assert sig.name == "bearish_engulfing"
    assert sig.direction == "SELL"


def test_hammer():
    # Small body at top of range, long lower wick
    # open 100, close 102 (body 2), high 102.5 (upper wick 0.5), low 92 (lower wick 8)
    df = pd.DataFrame([
        {"open": 100.0, "high": 102.5, "low": 92.0, "close": 102.0},
    ])
    sig = detect_hammer(df)
    assert sig is not None
    assert sig.name == "hammer"
    assert sig.direction == "BUY"


def test_inside_bar():
    # Day 1: High 100, Low 80, Close 95 (bullish)
    # Day 2: Inside range (High 92, Low 85)
    df = pd.DataFrame([
        {"open": 85.0, "high": 100.0, "low": 80.0, "close": 95.0},
        {"open": 88.0, "high": 92.0, "low": 85.0, "close": 90.0},
    ])
    sig = detect_inside_bar(df)
    assert sig is not None
    assert sig.name == "inside_bar"
    assert sig.direction == "BUY"


def test_scan_multiple():
    df = pd.DataFrame([
        {"open": 100.0, "high": 102.0, "low": 89.0, "close": 90.0},
        {"open": 88.0, "high": 104.0, "low": 87.0, "close": 102.0},
    ])
    signals = scan(df)
    assert len(signals) >= 1
    assert signals[0].direction == "BUY"
