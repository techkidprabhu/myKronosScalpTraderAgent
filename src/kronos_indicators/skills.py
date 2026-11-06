"""
Indicator *skills* for the Kronos swing-trading agent.

Each skill is a plain function that takes OHLCV candles (Kronos format) and
returns a small JSON-serialisable dict that an LLM subagent can reason over.

`IndicatorToolkit` wraps them as function-calling tools:
    toolkit = IndicatorToolkit(candle_provider=my_broker.get_candles,
                               forecast_provider=my_kronos.forecast)
    tools = toolkit.tool_schemas()          # pass to Hermes / Claude / OpenAI-style APIs
    result = toolkit.call(name, arguments)  # run the tool the model picked
"""
from __future__ import annotations

import json
import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import indicators as ind


# --------------------------------------------------------------------------- #
# small utils
# --------------------------------------------------------------------------- #
def _r(x, nd: int = 4):
    """Round and turn NaN/inf into None so results are JSON-safe."""
    if x is None:
        return None
    if isinstance(x, (np.floating, float, np.integer, int)):
        x = float(x)
        return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)
    return x


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return _r(obj)


def _crosses(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 where fast crosses above slow on that bar, -1 where it crosses below, else 0."""
    diff = np.sign(fast - slow)
    prev = diff.shift(1)
    up = (diff > 0) & (prev <= 0)
    down = (diff < 0) & (prev >= 0)
    return pd.Series(np.where(up, 1, np.where(down, -1, 0)), index=fast.index)


# --------------------------------------------------------------------------- #
# skill 1: raw indicator values
# --------------------------------------------------------------------------- #
INDICATOR_SPECS = {
    # name: (callable(df, **params) -> Series|DataFrame, default params)
    "sma": (lambda df, period=20: ind.sma(df["close"], period), {"period": 20}),
    "ema": (lambda df, period=20: ind.ema(df["close"], period), {"period": 20}),
    "wma": (lambda df, period=20: ind.wma(df["close"], period), {"period": 20}),
    "hma": (lambda df, period=20: ind.hma(df["close"], period), {"period": 20}),
    "vwma": (lambda df, period=20: ind.vwma(df, period), {"period": 20}),
    "rsi": (lambda df, period=14: ind.rsi(df["close"], period), {"period": 14}),
    "macd": (lambda df, fast=12, slow=26, signal=9: ind.macd(df["close"], fast, slow, signal), {}),
    "stochastic": (lambda df, k=14, d=3, smooth=3: ind.stochastic(df, k, d, smooth), {}),
    "roc": (lambda df, period=10: ind.roc(df["close"], period), {"period": 10}),
    "adx": (lambda df, period=14: ind.adx(df, period), {"period": 14}),
    "supertrend": (lambda df, period=10, mult=3.0: ind.supertrend(df, period, mult), {}),
    "atr": (lambda df, period=14: ind.atr(df, period), {"period": 14}),
    "bollinger": (lambda df, period=20, k=2.0: ind.bollinger(df["close"], period, k), {}),
    "keltner": (lambda df, period=20, mult=1.5: ind.keltner(df, period, mult), {}),
    "donchian": (lambda df, period=20: ind.donchian(df, period), {"period": 20}),
    "obv": (lambda df: ind.obv(df), {}),
    "vwap": (lambda df: ind.vwap(df), {}),
    "mfi": (lambda df, period=14: ind.mfi(df, period), {"period": 14}),
    "volume_ratio": (lambda df, period=20: ind.volume_ratio(df, period), {"period": 20}),
}


def get_indicators(df: pd.DataFrame, indicators: list[dict], last_n: int = 5) -> dict:
    """
    indicators: [{"name": "ema", "period": 50}, {"name": "rsi"}, {"name": "macd", "fast": 8}]
    Returns the last `last_n` values of each.
    """
    df = ind.prepare(df)
    out = {"last_close": _r(df["close"].iloc[-1]), "last_bar": df.index[-1].isoformat(), "values": {}}
    for spec in indicators:
        spec = dict(spec)
        name = spec.pop("name").lower()
        if name not in INDICATOR_SPECS:
            out["values"][name] = {"error": f"unknown indicator; available: {sorted(INDICATOR_SPECS)}"}
            continue
        fn, _ = INDICATOR_SPECS[name]
        try:
            res = fn(df, **spec)
        except Exception as e:  # bad params from the model, missing volume, ...
            out["values"][name] = {"error": str(e)}
            continue
        label = name + ("_" + "_".join(str(v) for v in spec.values()) if spec else "")
        tail = res.tail(last_n)
        if isinstance(tail, pd.Series):
            out["values"][label] = [_r(v) for v in tail]
        else:
            out["values"][label] = {c: [_r(v) for v in tail[c]] for c in tail.columns}
    return _clean(out)


# --------------------------------------------------------------------------- #
# skill 2: moving-average crossover scan
# --------------------------------------------------------------------------- #
def ma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50, kind: str = "ema",
                 lookback: int = 30) -> dict:
    """Current fast/slow MA relationship and any crosses in the last `lookback` bars."""
    df = ind.prepare(df)
    f = ind.moving_average(df, fast, kind)
    s = ind.moving_average(df, slow, kind)
    x = _crosses(f, s)
    recent = x.iloc[-lookback:]
    events = [
        {"bar": ts.isoformat(), "bars_ago": int(len(df) - 1 - df.index.get_loc(ts)),
         "type": "bullish_cross" if v > 0 else "bearish_cross"}
        for ts, v in recent[recent != 0].items()
    ]
    spread = (f.iloc[-1] - s.iloc[-1]) / s.iloc[-1] * 100 if s.iloc[-1] else np.nan
    spread_prev = (f.iloc[-6] - s.iloc[-6]) / s.iloc[-6] * 100 if len(df) > 6 and s.iloc[-6] else np.nan
    state = "fast_above_slow" if f.iloc[-1] > s.iloc[-1] else "fast_below_slow"
    label = {(20, 50): None, (50, 200): "golden_cross" if state == "fast_above_slow" else "death_cross"}.get((fast, slow))
    return _clean({
        "pair": f"{kind.upper()}{fast}/{kind.upper()}{slow}",
        "state": state,
        "regime": label,
        "fast": f.iloc[-1], "slow": s.iloc[-1],
        "spread_pct": spread,
        "spread_change_5bars_pct": spread - spread_prev if not np.isnan(spread_prev) else None,
        "converging": bool(abs(spread) < abs(spread_prev)) if not np.isnan(spread_prev) else None,
        "price_vs_fast_pct": (df["close"].iloc[-1] / f.iloc[-1] - 1) * 100,
        "price_vs_slow_pct": (df["close"].iloc[-1] / s.iloc[-1] - 1) * 100,
        "recent_crosses": events,
    })


# --------------------------------------------------------------------------- #
# skill 3: full technical snapshot with a transparent bias score
# --------------------------------------------------------------------------- #
def technical_snapshot(df: pd.DataFrame, intraday: bool = False) -> dict:
    """
    One call that summarises trend / momentum / volatility / volume / levels and
    lists every signal that fired with its weight, so the LLM can see *why*.
    bias_score is a heuristic in [-100, 100], not a probability.
    """
    a = ind.compute_all(df, intraday=intraday)
    if len(a) < 60:
        return {"error": f"need at least 60 bars, got {len(a)}"}
    L, P = a.iloc[-1], a.iloc[-2]
    c = L["close"]
    signals: list[dict] = []

    def add(name, direction, weight, detail):
        signals.append({"signal": name, "direction": direction, "weight": weight, "detail": detail})

    # ---- trend ----
    has200 = not np.isnan(L["ema_200"])
    if has200:
        add("price_vs_ema200", 1 if c > L["ema_200"] else -1, 15,
            f"close {'above' if c > L['ema_200'] else 'below'} EMA200 by {abs(c / L['ema_200'] - 1) * 100:.1f}%")
    stack_up = L["ema_9"] > L["ema_20"] > L["ema_50"] and (not has200 or L["ema_50"] > L["ema_200"])
    stack_dn = L["ema_9"] < L["ema_20"] < L["ema_50"] and (not has200 or L["ema_50"] < L["ema_200"])
    if stack_up:
        add("ema_stack", 1, 15, "EMAs stacked bullish (9>20>50" + (">200)" if has200 else ")"))
    elif stack_dn:
        add("ema_stack", -1, 15, "EMAs stacked bearish (9<20<50" + ("<200)" if has200 else ")"))
    x2050 = _crosses(a["ema_20"], a["ema_50"]).iloc[-10:]
    if (x2050 != 0).any():
        v = int(x2050[x2050 != 0].iloc[-1])
        ago = int(len(x2050) - 1 - np.flatnonzero(x2050.to_numpy())[-1])
        add("ema20_50_cross", v, 10, f"EMA20 crossed {'above' if v > 0 else 'below'} EMA50 {ago} bars ago")
    if has200:
        x50200 = _crosses(a["sma_50"], a["sma_200"]).iloc[-20:]
        if (x50200 != 0).any():
            v = int(x50200[x50200 != 0].iloc[-1])
            add("golden_cross" if v > 0 else "death_cross", v, 10, "SMA50/SMA200 cross in last 20 bars")
    if not np.isnan(L["st_dir"]):
        flipped = L["st_dir"] != P["st_dir"]
        add("supertrend", int(L["st_dir"]), 10 if flipped else 5,
            f"Supertrend {'up' if L['st_dir'] > 0 else 'down'}" + (" (just flipped)" if flipped else ""))
    trend_strength = "strong" if L["adx"] >= 25 else "weak" if L["adx"] < 20 else "moderate"

    # ---- momentum ----
    r = L["rsi_14"]
    if r >= 70:
        add("rsi_overbought", -1, 5, f"RSI {r:.0f} overbought")
    elif r <= 30:
        add("rsi_oversold", 1, 5, f"RSI {r:.0f} oversold")
    elif r > 50:
        add("rsi_bull_zone", 1, 5, f"RSI {r:.0f} above 50")
    else:
        add("rsi_bear_zone", -1, 5, f"RSI {r:.0f} below 50")
    mx = _crosses(a["macd"], a["macd_signal"]).iloc[-5:]
    if (mx != 0).any():
        v = int(mx[mx != 0].iloc[-1])
        add("macd_cross", v, 10, f"MACD crossed {'above' if v > 0 else 'below'} signal in last 5 bars")
    else:
        add("macd_hist", 1 if L["macd_hist"] > 0 else -1, 5,
            f"MACD hist {'positive' if L['macd_hist'] > 0 else 'negative'}, "
            f"{'rising' if L['macd_hist'] > P['macd_hist'] else 'falling'}")
    if L["stoch_k"] < 20 and L["stoch_k"] > L["stoch_d"]:
        add("stoch_bull_turn", 1, 5, "Stochastic turning up from oversold")
    elif L["stoch_k"] > 80 and L["stoch_k"] < L["stoch_d"]:
        add("stoch_bear_turn", -1, 5, "Stochastic turning down from overbought")

    # ---- volatility ----
    bb_w_pct = (a["bb_width"].iloc[-120:].rank(pct=True).iloc[-1]) * 100
    squeeze = bb_w_pct <= 10
    if c > L["bb_upper"]:
        add("bb_breakout_up", 1, 5, "close above upper Bollinger band")
    elif c < L["bb_lower"]:
        add("bb_breakout_down", -1, 5, "close below lower Bollinger band")
    if c >= a["dc_upper"].iloc[-2]:
        add("donchian_breakout_up", 1, 10, "new 20-bar high")
    elif c <= a["dc_lower"].iloc[-2]:
        add("donchian_breakout_down", -1, 10, "new 20-bar low")

    # ---- volume ----
    vol = {}
    if "vol_ratio" in a.columns:
        vr = L["vol_ratio"]
        obv_slope = a["obv"].iloc[-1] - a["obv"].iloc[-10]
        vol = {"volume_ratio_20": vr, "obv_10bar_slope": "up" if obv_slope > 0 else "down", "mfi_14": L["mfi_14"]}
        day_dir = 1 if c > P["close"] else -1
        if vr >= 1.5:
            add("volume_surge", day_dir, 5, f"volume {vr:.1f}x average on a {'up' if day_dir > 0 else 'down'} bar")
        if "vwap" in a.columns and not np.isnan(L["vwap"]):
            add("vwap", 1 if c > L["vwap"] else -1, 5, f"close {'above' if c > L['vwap'] else 'below'} VWAP")

    # ---- score ----
    total_w = sum(s["weight"] for s in signals) or 1
    score = 100 * sum(s["direction"] * s["weight"] for s in signals) / total_w
    if trend_strength == "weak":
        score *= 0.7  # trend signals are less reliable when ADX is low
    bias = "bullish" if score >= 25 else "bearish" if score <= -25 else "neutral"

    piv = ind.pivot_points(a[["open", "high", "low", "close"]].iloc[:-1]) if len(a) > 1 else {}
    atr_ = L["atr_14"]
    return _clean({
        "last_bar": a.index[-1].isoformat(),
        "close": c,
        "bias": bias,
        "bias_score": score,
        "trend": {
            "ema_20": L["ema_20"], "ema_50": L["ema_50"], "ema_200": L["ema_200"],
            "adx": L["adx"], "plus_di": L["plus_di"], "minus_di": L["minus_di"],
            "strength": trend_strength, "supertrend_dir": L["st_dir"], "supertrend_line": L["st_line"],
        },
        "momentum": {"rsi_14": r, "macd": L["macd"], "macd_signal": L["macd_signal"],
                     "macd_hist": L["macd_hist"], "stoch_k": L["stoch_k"], "stoch_d": L["stoch_d"]},
        "volatility": {"atr_14": atr_, "atr_pct": atr_ / c * 100, "bb_pct_b": L["bb_pct_b"],
                       "bb_width_percentile_120": bb_w_pct, "squeeze": bool(squeeze)},
        "volume": vol,
        "levels": {"donchian_high_20": L["dc_upper"], "donchian_low_20": L["dc_lower"],
                   "bb_upper": L["bb_upper"], "bb_lower": L["bb_lower"], **piv},
        "risk_hint": {"long_stop_2atr": c - 2 * atr_, "short_stop_2atr": c + 2 * atr_,
                      "long_target_3atr": c + 3 * atr_, "short_target_3atr": c - 3 * atr_},
        "signals": signals,
    })


# --------------------------------------------------------------------------- #
# skill 4: check a Kronos forecast against the indicators
# --------------------------------------------------------------------------- #
def confirm_kronos_forecast(history: pd.DataFrame, forecast: pd.DataFrame,
                            fast: int = 20, slow: int = 50, kind: str = "ema") -> dict:
    """
    history:  the candles you fed Kronos
    forecast: KronosPredictor.predict(...) output (open/high/low/close[/volume], future index)

    Returns forecast direction, whether today's indicator bias agrees, and what the
    indicators would look like if the forecast played out (projected MA crosses,
    projected RSI), plus an ATR-based sanity check on the forecast move size.
    """
    h = ind.prepare(history)
    f = ind.prepare(forecast)
    if "volume" not in f.columns and "volume" in h.columns:
        f["volume"] = h["volume"].iloc[-20:].mean()
    cols = [c for c in h.columns if c in ("open", "high", "low", "close", "volume")]
    joined = pd.concat([h[cols], f[[c for c in cols if c in f.columns]]])
    n_h, n_f = len(h), len(f)

    last = h["close"].iloc[-1]
    end = f["close"].iloc[-1]
    ret_pct = (end / last - 1) * 100
    f_dir = 1 if ret_pct > 0 else -1 if ret_pct < 0 else 0
    max_up = (f["high"].max() / last - 1) * 100
    max_dn = (f["low"].min() / last - 1) * 100

    snap = technical_snapshot(h)
    ind_dir = {"bullish": 1, "bearish": -1}.get(snap.get("bias"), 0)

    fm = ind.moving_average(joined, fast, kind)
    sm = ind.moving_average(joined, slow, kind)
    x = _crosses(fm, sm).iloc[n_h:]
    proj_cross = [{"bars_ahead": int(i + 1), "type": "bullish_cross" if v > 0 else "bearish_cross"}
                  for i, v in enumerate(x.to_numpy()) if v != 0]
    proj_rsi = ind.rsi(joined["close"]).iloc[-1]
    proj_st = ind.supertrend(joined)["direction"].iloc[-1]

    atr_ = ind.atr(h).iloc[-1]
    expected_move_atr = abs(end - last) / atr_ if atr_ else np.nan
    # a typical random-walk move over n bars is ~ ATR * sqrt(n); flag forecasts far outside that
    plausibility = "normal" if expected_move_atr <= 2 * math.sqrt(n_f) else "unusually_large"

    if f_dir == 0 or ind_dir == 0:
        verdict = "no_confluence"
    elif f_dir == ind_dir:
        verdict = "confluence_long" if f_dir > 0 else "confluence_short"
    else:
        verdict = "conflict"

    return _clean({
        "forecast": {"bars": n_f, "last_close": last, "forecast_close": end, "return_pct": ret_pct,
                     "direction": f_dir, "max_up_pct": max_up, "max_down_pct": max_dn,
                     "move_in_atr": expected_move_atr, "plausibility": plausibility},
        "indicators_now": {"bias": snap.get("bias"), "bias_score": snap.get("bias_score"),
                           "adx": snap.get("trend", {}).get("adx"), "rsi_14": snap.get("momentum", {}).get("rsi_14")},
        "if_forecast_plays_out": {"projected_ma_crosses": proj_cross, "ma_pair": f"{kind.upper()}{fast}/{kind.upper()}{slow}",
                                  "projected_rsi_14": proj_rsi, "projected_supertrend_dir": proj_st},
        "verdict": verdict,
        "risk_hint": snap.get("risk_hint"),
    })


# --------------------------------------------------------------------------- #
# function-calling wrapper
# --------------------------------------------------------------------------- #
_TF = {"type": "string", "description": "Candle timeframe, e.g. '1d', '1h', '15m', '5m'"}
_SYM = {"type": "string", "description": "Instrument symbol as used by the broker, e.g. 'RELIANCE', 'D05', 'AAPL'"}
_BARS = {"type": "integer", "description": "How many candles to load (default 300; 260+ needed for EMA200)"}


class IndicatorToolkit:
    """
    candle_provider(symbol, timeframe, bars) -> DataFrame of OHLCV candles
    forecast_provider(symbol, timeframe, pred_len, history_df) -> DataFrame (optional; your Kronos wrapper)
    """

    def __init__(self, candle_provider: Callable, forecast_provider: Optional[Callable] = None):
        self.candles = candle_provider
        self.forecast = forecast_provider

    # ---- schemas (OpenAI / Ollama / Hermes format; see to_anthropic() for Claude) ----
    def tool_schemas(self) -> list[dict]:
        tools = [
            {"name": "technical_snapshot",
             "description": "Full technical read of a symbol: trend (EMA 9/20/50/200, ADX, Supertrend), momentum "
                            "(RSI, MACD, Stochastic), volatility (ATR, Bollinger squeeze), volume (OBV, MFI), key levels, "
                            "every fired signal with its weight, a bias score from -100 to 100, and ATR stop/target hints. "
                            "Use this first when evaluating a swing setup.",
             "parameters": {"type": "object", "properties": {"symbol": _SYM, "timeframe": _TF, "bars": _BARS,
                            "intraday": {"type": "boolean", "description": "Add session VWAP (intraday bars only)"}},
                            "required": ["symbol", "timeframe"]}},
            {"name": "ma_crossover",
             "description": "Moving-average crossover check for a fast/slow pair: current state, spread, whether the "
                            "averages are converging, and crosses in the recent lookback window.",
             "parameters": {"type": "object", "properties": {
                 "symbol": _SYM, "timeframe": _TF, "bars": _BARS,
                 "fast": {"type": "integer", "description": "Fast period (default 20)"},
                 "slow": {"type": "integer", "description": "Slow period (default 50)"},
                 "kind": {"type": "string", "enum": ["sma", "ema", "wma", "hma", "vwma"], "description": "MA type (default ema)"},
                 "lookback": {"type": "integer", "description": "Bars to scan for crosses (default 30)"}},
                 "required": ["symbol", "timeframe"]}},
            {"name": "get_indicators",
             "description": f"Raw recent values for specific indicators. Available: {', '.join(sorted(INDICATOR_SPECS))}. "
                            "Pass parameters per indicator, e.g. {\"name\":\"ema\",\"period\":50} or {\"name\":\"macd\",\"fast\":8,\"slow\":21,\"signal\":5}.",
             "parameters": {"type": "object", "properties": {
                 "symbol": _SYM, "timeframe": _TF, "bars": _BARS,
                 "indicators": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}},
                                "required": ["name"], "additionalProperties": True}},
                 "last_n": {"type": "integer", "description": "How many recent values to return (default 5)"}},
                 "required": ["symbol", "timeframe", "indicators"]}},
        ]
        if self.forecast is not None:
            tools.append(
                {"name": "confirm_kronos_forecast",
                 "description": "Run the Kronos forecast for a symbol and check it against the indicators: forecast "
                                "direction and size, whether current indicator bias agrees (confluence or conflict), MA "
                                "crosses and RSI the forecast would produce, and an ATR plausibility check.",
                 "parameters": {"type": "object", "properties": {
                     "symbol": _SYM, "timeframe": _TF, "bars": _BARS,
                     "pred_len": {"type": "integer", "description": "Bars to forecast (default 10)"},
                     "fast": {"type": "integer"}, "slow": {"type": "integer"},
                     "kind": {"type": "string", "enum": ["sma", "ema", "wma", "hma"]}},
                     "required": ["symbol", "timeframe"]}})
        return [{"type": "function", "function": t} for t in tools]

    def to_anthropic(self) -> list[dict]:
        """Same tools in Anthropic Messages API format."""
        return [{"name": t["function"]["name"], "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]} for t in self.tool_schemas()]

    # ---- dispatch ----
    def call(self, name: str, arguments) -> dict:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        try:
            symbol, tf = args.pop("symbol"), args.pop("timeframe")
            bars = int(args.pop("bars", 300) or 300)
            df = self.candles(symbol, tf, bars)
            if df is None or len(df) == 0:
                return {"error": f"no candles for {symbol} {tf}"}
            if name == "technical_snapshot":
                res = technical_snapshot(df, intraday=bool(args.get("intraday", False)))
            elif name == "ma_crossover":
                res = ma_crossover(df, **{k: v for k, v in args.items() if k in ("fast", "slow", "kind", "lookback")})
            elif name == "get_indicators":
                res = get_indicators(df, args.get("indicators", []), int(args.get("last_n", 5)))
            elif name == "confirm_kronos_forecast" and self.forecast is not None:
                pred_len = int(args.pop("pred_len", 10))
                fc = self.forecast(symbol, tf, pred_len, df)
                res = confirm_kronos_forecast(df, fc, **{k: v for k, v in args.items() if k in ("fast", "slow", "kind")})
            else:
                return {"error": f"unknown tool {name}"}
            return {"symbol": symbol, "timeframe": tf, **res}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
