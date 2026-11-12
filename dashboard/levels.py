"""
Support and resistance levels for the chart (display only - no trading
decision reads these).

A level is a price where the market has turned more than once:
  1. find swing highs and swing lows (a bar whose high / low is the extreme
     of the `left` bars before and `right` bars after it);
  2. merge swing points that sit within a tolerance of each other (a share
     of the ATR, so it adapts to how much the symbol moves);
  3. rank each merged level by how often it was touched and how recently;
  4. keep the strongest few above the last close (resistance) and below it
     (support).

Because it is recomputed from the latest candles on every chart refresh, a
level that price breaks through changes sides: old resistance becomes
support, and levels that stop mattering drop out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low, prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    val = tr.rolling(period, min_periods=min(period, len(df))).mean().iloc[-1]
    return float(val) if np.isfinite(val) else 0.0


def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[tuple[int, float, str]]:
    """[(bar index, price, 'high' | 'low'), ...] in bar order."""
    highs, lows = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    out = []
    for i in range(left, len(df) - right):
        win_h, win_l = highs[i - left:i + right + 1], lows[i - left:i + right + 1]
        if highs[i] == win_h.max() and (win_h == highs[i]).sum() == 1:
            out.append((i, float(highs[i]), "high"))
        if lows[i] == win_l.min() and (win_l == lows[i]).sum() == 1:
            out.append((i, float(lows[i]), "low"))
    return out


def support_resistance(df: pd.DataFrame, left: int = 3, right: int = 3, tolerance_atr: float = 0.5,
                       per_side: int = 3) -> list[dict]:
    """Up to `per_side` support and `per_side` resistance levels, nearest
    first on each side. Each: price, kind, touches, strength (1-3),
    first_index / last_index (bars of the first and latest touch) and
    distance_pct from the last close."""
    n = len(df)
    if n < left + right + 10:
        return []
    close = float(df["close"].iloc[-1])
    if not np.isfinite(close) or close <= 0:
        return []
    tol = max(tolerance_atr * _atr(df), close * 0.001)
    pivots = sorted(swing_points(df, left, right), key=lambda p: p[1])
    if not pivots:
        return []

    clusters, current = [], [pivots[0]]
    for p in pivots[1:]:
        if p[1] - np.mean([c[1] for c in current]) <= tol:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)

    levels = []
    for c in clusters:
        price = float(np.mean([p[1] for p in c]))
        if abs(price - close) < tol * 0.2:          # price is sitting on it - neither side yet
            continue
        idx = [p[0] for p in c]
        touches = len(c)
        score = touches + 1.5 * max(idx) / n        # more touches and more recent = stronger
        levels.append({
            "price": round(price, 4), "kind": "resistance" if price > close else "support",
            "touches": touches, "score": round(score, 3),
            "strength": 3 if touches >= 4 else 2 if touches >= 2 else 1,
            "first_index": int(min(idx)), "last_index": int(max(idx)),
            "distance_pct": round((price - close) / close * 100, 2),
        })

    out = []
    for kind in ("resistance", "support"):
        side = sorted((lv for lv in levels if lv["kind"] == kind), key=lambda lv: lv["score"], reverse=True)[:per_side]
        out += sorted(side, key=lambda lv: abs(lv["price"] - close))
    return out
