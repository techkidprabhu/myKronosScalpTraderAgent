"""
Single-stock focus: instead of scanning every screener pick, choose the ONE
cleanest volume breakout and scalp only that until its momentum fades.

The screener still picks its shortlist (liquid, tight spread, moving). On
every candle close, focus mode checks that shortlist with the rule below and
locks onto the best qualifying symbol. Only that symbol is scanned for
entries; every open trade is still managed. One agent position at a time.

The volume-breakout rule (all must pass, on the last closed 1-5 min candle):

  1. Level        - resistance = highest high of the previous `level_bars`
                    candles (support = lowest low, for shorts).
  2. Breakout     - the candle CLOSES beyond the level by `min_break_pct`, and
                    closes in the top half of its own range (bottom half for
                    shorts) - a candle that pokes through and falls back is a
                    fakeout.
  3. Volume spike - candle volume >= `min_volume_ratio` (1.5x) times the
                    average of the previous `volume_period` (20) candles.
  4. Accumulation - over those 20 candles, volume on up candles divided by
                    volume on down candles >= `min_accumulation` (volume
                    expanding on rises, fading on dips).
  5. Momentum     - close above session VWAP; EMA(9) above EMA(21); RSI(14)
                    between `rsi_min` and `rsi_max` (strong, not exhausted).
                    Mirrored for shorts.
  6. Session      - markets with a regular session (stocks, index CFDs): only
                    in the first `prime_minutes` after the open. 24-hour
                    markets have no window.

Best score wins (0-100): volume spike 45, accumulation 20, clean break 15,
close near the high 10, RSI in the sweet spot 10.

The focus is released (and the next candle re-screens) when, with no trade
open on it: the price closes back through the breakout level (failed
breakout), volume fades (last 3 candles under `fade_ratio` x average), or
`max_wait_bars` candles pass without an entry.

Risk: with `stop_below_level`, an entry taken on the breakout itself puts its
stop just beyond the breakout level (`stop_buffer_pct` past it) and its
target at `target_r` x that risk. Every other risk limit still applies.

The on/off switch lives in data/focus_mode.json so the Command Center and
Settings can flip it without a restart; missing file = off.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SWITCH_PATH = ROOT / "data" / "focus_mode.json"
STATE_KEY = "focus_state"
SIDES = ("long", "both")

DEFAULTS = {
    "level_bars": 20,
    "volume_period": 20,
    "min_volume_ratio": 1.5,
    "strong_volume_ratio": 3.0,
    "min_break_pct": 0.0,
    "min_accumulation": 1.0,
    "require_vwap": True,
    "require_ema": True,
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi_period": 14,
    "rsi_min": 55.0,
    "rsi_max": 80.0,
    "sides": "long",
    "prime_minutes": 120,
    "max_wait_bars": 6,
    "fade_ratio": 0.8,
    "entry_on_breakout": True,
    "stop_below_level": True,
    "stop_buffer_pct": 0.05,
    "target_r": 2.0,
}


def settings(config: dict) -> dict:
    raw = ((config.get("screener") or {}).get("focus")) or {}
    s = {**DEFAULTS, **{k: v for k, v in raw.items() if v is not None and k in DEFAULTS}}
    s["sides"] = str(s["sides"]).lower() if str(s["sides"]).lower() in SIDES else "long"
    for k in ("level_bars", "volume_period", "ema_fast", "ema_slow", "rsi_period", "prime_minutes", "max_wait_bars"):
        s[k] = max(0 if k == "prime_minutes" else 1, int(s[k]))
    return s


# ---- the on/off switch -----------------------------------------------------

def read_switch(path: Path | None = None) -> dict:
    path = Path(path or SWITCH_PATH)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {"enabled": bool(raw.get("enabled")), "changed_ts": raw.get("changed_ts")}
    except (OSError, ValueError):
        return {"enabled": False, "changed_ts": None}


def write_switch(enabled: bool, path: Path | None = None) -> dict:
    path = Path(path or SWITCH_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"enabled": bool(enabled), "changed_ts": time.time()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)
    return value


# ---- indicators --------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    u, w = float(up.iloc[-1]), float(dn.iloc[-1])
    if w == 0:
        return 100.0 if u > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + u / w)


def _session_vwap(df: pd.DataFrame) -> float | None:
    """VWAP of the last candle's calendar day (the session so far)."""
    day = df[df.index.normalize() == df.index[-1].normalize()] if isinstance(df.index, pd.DatetimeIndex) else df
    vol = day["volume"].astype(float)
    if vol.sum() <= 0:
        return None
    tp = (day["high"] + day["low"] + day["close"]) / 3
    return float((tp * vol).sum() / vol.sum())


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    v = float(tr.tail(period).mean())
    return v if v == v and v > 0 else 0.0


# ---- the rule ------------------------------------------------------------------

def evaluate(df: pd.DataFrame, cfg: dict) -> dict:
    """Check the last closed candle against the volume-breakout rule.
    Returns {"qualified", "direction", "level", "score", "why", ...}."""
    c = {**DEFAULTS, **(cfg or {})}
    need = max(c["level_bars"], c["volume_period"], c["ema_slow"], c["rsi_period"]) + 2
    out = {"qualified": False, "direction": None, "level": None, "score": 0.0, "why": None}
    if df is None or len(df) < need:
        out["why"] = f"not enough candles ({0 if df is None else len(df)} of {need})"
        return out
    last = df.iloc[-1]
    prev = df.iloc[-(c["level_bars"] + 1):-1]
    vol_prev = df["volume"].iloc[-(c["volume_period"] + 1):-1].astype(float)
    avg_vol = float(vol_prev.mean())
    if not avg_vol > 0 or not float(last["volume"]) > 0:
        out["why"] = "no volume data for this market"
        return out
    close, high, low = float(last["close"]), float(last["high"]), float(last["low"])
    resistance, support = float(prev["high"].max()), float(prev["low"].min())
    vol_ratio = float(last["volume"]) / avg_vol
    window = df.iloc[-(c["volume_period"] + 1):-1]
    up_vol = float(window.loc[window["close"] > window["open"], "volume"].sum())
    dn_vol = float(window.loc[window["close"] < window["open"], "volume"].sum())
    ema_f = float(df["close"].ewm(span=c["ema_fast"], adjust=False).mean().iloc[-1])
    ema_s = float(df["close"].ewm(span=c["ema_slow"], adjust=False).mean().iloc[-1])
    rsi = _rsi(df["close"].astype(float), c["rsi_period"])
    vwap = _session_vwap(df)
    rng = high - low
    loc = (close - low) / rng if rng > 0 else 0.5
    brk = c["min_break_pct"] / 100.0

    if close > resistance * (1 + brk):
        direction, level = "BUY", resistance
        accum = up_vol / dn_vol if dn_vol > 0 else (3.0 if up_vol > 0 else 0.0)
        bar_ok, vwap_ok, ema_ok = loc >= 0.5, vwap is None or close > vwap, ema_f > ema_s
        rsi_ok, rsi_fit = c["rsi_min"] <= rsi <= c["rsi_max"], rsi
        mid = (c["rsi_min"] + c["rsi_max"]) / 2
    elif c["sides"] == "both" and close < support * (1 - brk):
        direction, level = "SELL", support
        accum = dn_vol / up_vol if up_vol > 0 else (3.0 if dn_vol > 0 else 0.0)
        bar_ok, vwap_ok, ema_ok = loc <= 0.5, vwap is None or close < vwap, ema_f < ema_s
        lo, hi = 100 - c["rsi_max"], 100 - c["rsi_min"]
        rsi_ok, rsi_fit = lo <= rsi <= hi, rsi
        mid = (lo + hi) / 2
    else:
        out.update(why=f"no breakout (close {close:.6g}, level {resistance:.6g})", level=round(resistance, 6),
                   volume_ratio=round(vol_ratio, 2))
        return out

    out.update(direction=direction, level=round(level, 6), volume_ratio=round(vol_ratio, 2),
               accumulation=round(accum, 2), rsi=round(rsi, 1), vwap=round(vwap, 6) if vwap else None,
               close=round(close, 6))
    checks = [
        (vol_ratio >= c["min_volume_ratio"], f"volume only {vol_ratio:.1f}x average (need {c['min_volume_ratio']}x)"),
        (bar_ok, "breakout candle closed away from its extreme - fakeout risk"),
        (accum >= c["min_accumulation"], f"volume not building with the move ({accum:.2f} < {c['min_accumulation']})"),
        (vwap_ok or not c["require_vwap"], "price on the wrong side of VWAP"),
        (ema_ok or not c["require_ema"], f"EMA {c['ema_fast']}/{c['ema_slow']} not aligned"),
        (rsi_ok, f"RSI {rsi:.0f} outside {c['rsi_min']:.0f}-{c['rsi_max']:.0f}" if direction == "BUY"
         else f"RSI {rsi:.0f} outside {100 - c['rsi_max']:.0f}-{100 - c['rsi_min']:.0f}"),
    ]
    for ok, why in checks:
        if not ok:
            out["why"] = why
            return out
    atr = _atr(df) or rng or abs(close) * 0.001
    clean = min(abs(close - level) / atr, 1.0) if atr else 0.0
    near_extreme = loc if direction == "BUY" else 1 - loc
    sweet = max(0.0, 1 - abs(rsi_fit - mid) / max((c["rsi_max"] - c["rsi_min"]) / 2, 1))
    score = (45 * min(vol_ratio, c["strong_volume_ratio"]) / c["strong_volume_ratio"]
             + 20 * min(accum, 3.0) / 3.0 + 15 * clean + 10 * near_extreme + 10 * sweet)
    out.update(qualified=True, score=round(float(score), 1), why="volume breakout")
    return out


def still_valid(df: pd.DataFrame, focus: dict, cfg: dict) -> str | None:
    """Why a focus with no open trade should be dropped, or None to keep it."""
    c = {**DEFAULTS, **(cfg or {})}
    if df is None or df.empty:
        return None
    close = float(df["close"].iloc[-1])
    level = float(focus.get("level") or 0)
    if focus.get("direction") == "BUY" and level and close < level:
        return "failed breakout: closed back below the level"
    if focus.get("direction") == "SELL" and level and close > level:
        return "failed breakdown: closed back above the level"
    vols = df["volume"].astype(float)
    base = float(vols.iloc[-(c["volume_period"] + 4):-3].mean()) if len(vols) > c["volume_period"] + 3 else 0.0
    if base > 0 and float(vols.tail(3).mean()) < c["fade_ratio"] * base:
        return "momentum stalled: volume faded"
    if int(focus.get("bars_waited") or 0) >= c["max_wait_bars"]:
        return f"no entry within {c['max_wait_bars']} candles"
    return None


def breakout_stop(entry: float, focus: dict, cfg: dict) -> float | None:
    """The stop distance just beyond the breakout level, or None if the level
    is on the wrong side of the entry."""
    c = {**DEFAULTS, **(cfg or {})}
    level = float(focus.get("level") or 0)
    if not level:
        return None
    buf = c["stop_buffer_pct"] / 100.0
    stop = level * (1 - buf) if focus.get("direction") == "BUY" else level * (1 + buf)
    d = entry - stop if focus.get("direction") == "BUY" else stop - entry
    return d if d > 0 else None


def pick(results: dict[str, dict]) -> str | None:
    """The best qualified symbol from {symbol: evaluate(...)}."""
    ok = [(r["score"], s) for s, r in results.items() if r.get("qualified")]
    return max(ok)[1] if ok else None


def in_prime_window(session_minutes, cfg: dict) -> bool:
    """True inside the first `prime_minutes` after the open; markets with no
    regular session (None) are always in it."""
    c = {**DEFAULTS, **(cfg or {})}
    if not c["prime_minutes"] or not session_minutes:
        return True
    since_open = session_minutes[0]
    return since_open is None or 0 <= float(since_open) <= c["prime_minutes"]


__all__ = ["DEFAULTS", "settings", "read_switch", "write_switch", "evaluate", "still_valid",
           "breakout_stop", "pick", "in_prime_window", "STATE_KEY"]
