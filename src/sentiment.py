"""
Market Sentiment Meter - what is the broad market doing right now?

Reads a few benchmark ETFs for the broker's market (S&P 500 / Nasdaq 100 /
Russell 2000 for US, the STI for SGX, Nifty 50 / Bank Nifty for NSE, ...)
from the broker you're already connected to - the same price data a
TradingView chart shows - and classifies each into one regime:

    Up trend    price above a rising EMA20 above EMA50, and ADX >= adx_trend
    Down trend  the mirror image
    Stable      no trend and low volatility (narrow Bollinger width / ATR)
    Sideways    no clean trend, normal-to-high volatility - a choppy range

plus a needle score from -100 (strongly bearish) to +100 (strongly
bullish): 65% trend (EMA20 slope, damped when ADX is low) and 35% the
indicator library's short-term bias score (EMA stack, Supertrend, RSI,
MACD, breakouts...). Breadth adds the share of the agent's own active
symbols trading above their EMA50.

Display only: nothing here feeds the trading pipeline (patterns ->
indicator gate -> Kronos -> risk).
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from src.kronos_indicators import indicators as ind
from src.kronos_indicators import technical_snapshot

logger = logging.getLogger(__name__)

REGIMES = ("Up trend", "Down trend", "Sideways", "Stable")

# Benchmarks per market: liquid ETFs every broker can quote (indices often
# need extra quote permissions). Override in config.yaml -> sentiment.benchmarks.
DEFAULT_BENCHMARKS = {
    "US": [("US.SPY", "S&P 500"), ("US.QQQ", "Nasdaq 100"), ("US.IWM", "Russell 2000")],
    "SG": [("SG.ES3", "STI (SPDR)"), ("SG.G3B", "STI (Nikko AM)")],
    "HK": [("HK.02800", "Hang Seng (Tracker Fund)"), ("HK.03033", "Hang Seng Tech (CSOP)")],
    "CN": [("SH.510300", "CSI 300 ETF"), ("SH.510500", "CSI 500 ETF")],
    "NSE": [("NSE.NIFTYBEES", "Nifty 50"), ("NSE.BANKBEES", "Bank Nifty"), ("NSE.JUNIORBEES", "Nifty Next 50")],
    "BSE": [("NSE.NIFTYBEES", "Nifty 50"), ("NSE.BANKBEES", "Bank Nifty")],
}

DEFAULTS = {
    "enabled": True,
    "timeframe": "DAY",        # daily candles = the market's trend; MIN_15 / HOUR_1 for intraday mood
    "lookback_bars": 250,
    "refresh_minutes": 15,
    "adx_trend": 20,           # ADX at or above this = trending
    "slope_min_pct": 0.5,      # EMA20 must move this % over 10 bars to count as rising / falling
    "stable_bb_percentile": 35,  # Bollinger width in the lowest 35% of the last 120 bars = calm
    "breadth_symbols": 15,     # how many active symbols to check for breadth (0 = off)
    "benchmarks": {},          # market -> [symbols] (or [{symbol, name}]) to override the defaults
}


def settings(cfg: dict | None) -> dict:
    return {**DEFAULTS, **((cfg or {}).get("sentiment") or {})}


def benchmarks_for(market: str, s: dict) -> list[tuple[str, str]]:
    custom = (s.get("benchmarks") or {}).get(market)
    if custom:
        out = []
        for b in custom:
            if isinstance(b, dict):
                out.append((str(b["symbol"]).upper(), str(b.get("name") or b["symbol"])))
            else:
                out.append((str(b).upper(), str(b)))
        return out
    return list(DEFAULT_BENCHMARKS.get(market, []))


def _num(x) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) or np.isinf(f) else f


def classify(df: pd.DataFrame, adx_trend: float = 20, slope_min_pct: float = 0.5,
             stable_bb_percentile: float = 35, intraday: bool = False) -> dict:
    """Regime of one instrument from its closed bars (columns open, high,
    low, close[, volume]; oldest first)."""
    if len(df) < 60:
        return {"error": f"only {len(df)} bars - need at least 60"}
    snap = technical_snapshot(df, intraday=intraday)
    if "error" in snap:
        return {"error": snap["error"]}
    close = df["close"].astype(float)
    ema20, ema50 = ind.ema(close, 20), ind.ema(close, 50)
    c, e20, e50 = float(close.iloc[-1]), float(ema20.iloc[-1]), float(ema50.iloc[-1])
    slope = (e20 / float(ema20.iloc[-11]) - 1) * 100 if len(ema20) > 11 else 0.0
    ret_20 = (c / float(close.iloc[-21]) - 1) * 100 if len(close) > 21 else None
    adx = _num(snap["trend"]["adx"]) or 0.0
    bb_pct = _num(snap["volatility"]["bb_width_percentile_120"])
    atr_pct = _num(snap["volatility"]["atr_pct"])
    bias = _num(snap["bias_score"]) or 0.0
    # Needle: mostly the trend (EMA20 slope, damped when ADX says there's no
    # trend), partly the indicator library's short-term bias score.
    trend = 100 * float(np.tanh(slope / 2)) * min(1.0, adx / 25) ** 2
    score = max(-100.0, min(100.0, 0.65 * trend + 0.35 * bias))

    up = c > e50 and e20 > e50 and slope >= slope_min_pct
    down = c < e50 and e20 < e50 and slope <= -slope_min_pct
    if adx >= adx_trend and up:
        regime, why = "Up trend", f"above rising EMA20 > EMA50, ADX {adx:.0f}"
    elif adx >= adx_trend and down:
        regime, why = "Down trend", f"below falling EMA20 < EMA50, ADX {adx:.0f}"
    elif bb_pct is not None and bb_pct <= stable_bb_percentile:
        regime, why = "Stable", f"no trend (ADX {adx:.0f}), volatility low (BB width {bb_pct:.0f}th pct)"
    else:
        regime, why = "Sideways", f"no clean trend (ADX {adx:.0f}), range-bound"
    if regime in ("Stable", "Sideways"):
        score = max(-50.0, min(50.0, score))   # no trend: the needle can only lean
    # How clearly the regime holds: trend strength for trends, calmness for Stable.
    if regime in ("Up trend", "Down trend"):
        conf = min(1.0, 0.5 + (adx - adx_trend) / 30 + min(abs(slope), 3) / 12)
    elif regime == "Stable":
        conf = min(1.0, 0.5 + (stable_bb_percentile - (bb_pct or 0)) / (2 * stable_bb_percentile))
    else:
        conf = min(1.0, 0.4 + (adx_trend - min(adx, adx_trend)) / (2 * adx_trend))
    return {
        "regime": regime, "reason": why, "score": round(score, 1), "bias_score": round(bias, 1),
        "confidence": round(max(conf, 0.0), 2),
        "close": round(c, 4), "ema20": round(e20, 4), "ema50": round(e50, 4),
        "ema20_slope_pct": round(slope, 2), "adx": round(adx, 1),
        "return_20_bars_pct": round(ret_20, 2) if ret_20 is not None else None,
        "atr_pct": round(atr_pct, 2) if atr_pct is not None else None,
        "bb_width_percentile": round(bb_pct, 0) if bb_pct is not None else None,
        "rsi": _num(snap["momentum"]["rsi_14"]), "last_bar": snap.get("last_bar"),
    }


def overall(results: list[dict]) -> dict:
    """Combine the benchmarks: the most common regime (ties -> the one of the
    benchmark with the strongest score), average needle score."""
    ok = [r for r in results if "regime" in r]
    if not ok:
        return {"regime": None, "score": None, "agreement": 0, "of": 0}
    counts = Counter(r["regime"] for r in ok)
    top = counts.most_common()
    best_n = top[0][1]
    tied = [reg for reg, n in top if n == best_n]
    if len(tied) > 1:
        leader = max((r for r in ok if r["regime"] in tied), key=lambda r: abs(r["score"]))
        regime = leader["regime"]
    else:
        regime = tied[0]
    score = round(float(np.mean([r["score"] for r in ok])), 1)
    return {"regime": regime, "score": score, "agreement": counts[regime], "of": len(ok)}


def breadth(frames: dict[str, pd.DataFrame]) -> dict | None:
    """Share of symbols closing above their EMA50."""
    above = total = 0
    for df in frames.values():
        if df is None or len(df) < 50:
            continue
        close = df["close"].astype(float)
        total += 1
        above += int(close.iloc[-1] > ind.ema(close, 50).iloc[-1])
    if not total:
        return None
    pct = round(100 * above / total)
    label = "broad strength" if pct >= 60 else "broad weakness" if pct <= 40 else "mixed"
    return {"above_ema50_pct": pct, "above": above, "of": total, "label": label}
