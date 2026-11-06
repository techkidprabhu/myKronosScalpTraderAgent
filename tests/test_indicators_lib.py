import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from src.kronos_indicators import IndicatorToolkit, confirm_kronos_forecast, indicators as ind, ma_crossover, technical_snapshot


def synth(n=400, seed=0, drift=0.0005, freq="B"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(1e5, 5e5, n).astype(float)
    ts = pd.date_range("2024-01-01 09:15", periods=n, freq=freq)
    return pd.DataFrame({"timestamps": ts, "open": open_, "high": high, "low": low, "close": close,
                         "volume": vol, "amount": vol * close})


def test_sma_ema_known_values():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert ind.sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    # EMA span 3 -> alpha 0.5, seeded on first value
    e = s.ewm(alpha=0.5, adjust=False).mean()
    assert np.allclose(ind.ema(s, 3).dropna(), e.iloc[2:])
    assert np.isclose(ind.wma(s, 3).iloc[-1], (3 * 1 + 4 * 2 + 5 * 3) / 6)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 50, dtype=float))
    assert ind.rsi(up).dropna().eq(100).all()
    df = synth()
    r = ind.rsi(df["close"]).dropna()
    assert r.between(0, 100).all()


def test_rsi_matches_wilder_reference():
    # classic StockCharts RSI worked example. Their table shows 70.53, 66.32, ... because it
    # rounds the intermediate averages; exact Wilder math (SMA seed, then RMA) gives these:
    closes = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                        45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64])
    r = ind.rsi(closes, 14).dropna().round(2).tolist()
    assert np.allclose(r, [70.46, 66.25, 66.48, 69.35, 66.29, 57.92], atol=0.01)
    # first value = 100 - 100 / (1 + avg_gain/avg_loss) over the first 14 changes
    d = closes.diff()[1:15]
    g, l = d.clip(lower=0).mean(), (-d.clip(upper=0)).mean()
    assert np.isclose(r[0], round(100 - 100 / (1 + g / l), 2))


def test_atr_supertrend_adx_shapes():
    df = ind.prepare(synth())
    assert (ind.atr(df).dropna() > 0).all()
    st = ind.supertrend(df)
    assert set(st["direction"].dropna().unique()) <= {1.0, -1.0}
    a = ind.adx(df).dropna()
    assert a["adx"].between(0, 100).all()


def test_ma_crossover_detects_cross():
    # down then up -> must produce a bullish cross of EMA5/EMA20
    close = np.r_[np.linspace(120, 90, 60), np.linspace(90, 130, 60)]
    df = pd.DataFrame({"timestamps": pd.date_range("2024-01-01", periods=120, freq="B"),
                       "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                       "volume": 1e5})
    res = ma_crossover(df, fast=5, slow=20, lookback=60)
    assert res["state"] == "fast_above_slow"
    assert any(e["type"] == "bullish_cross" for e in res["recent_crosses"])


def test_snapshot_is_json_and_trend_aware():
    bull = technical_snapshot(synth(drift=0.004, seed=1))
    bear = technical_snapshot(synth(drift=-0.004, seed=1))
    json.dumps(bull)
    assert bull["bias_score"] > bear["bias_score"]
    assert bull["bias"] == "bullish" and bear["bias"] == "bearish"
    assert all({"signal", "direction", "weight", "detail"} <= set(s) for s in bull["signals"])


def test_intraday_vwap():
    df = synth(n=300, freq="5min")
    snap = technical_snapshot(df, intraday=True)
    assert any(s["signal"] == "vwap" for s in snap["signals"])


def test_confirm_forecast_confluence_and_projection():
    hist = synth(drift=0.004, seed=2)
    last = hist["close"].iloc[-1]
    path = last * np.linspace(1.002, 1.04, 10)
    fc = pd.DataFrame({"open": path, "high": path * 1.005, "low": path * 0.995, "close": path},
                      index=pd.date_range(hist["timestamps"].iloc[-1], periods=11, freq="B")[1:])
    res = confirm_kronos_forecast(hist, fc)
    json.dumps(res)
    assert res["forecast"]["direction"] == 1
    assert res["verdict"] in ("confluence_long", "no_confluence")


def test_toolkit_dispatch():
    data = synth()
    tk = IndicatorToolkit(lambda s, tf, b: data.tail(b),
                          lambda s, tf, n, h: pd.DataFrame(
                              {"open": [h["close"].iloc[-1]] * n, "high": [h["close"].iloc[-1] * 1.01] * n,
                               "low": [h["close"].iloc[-1] * 0.99] * n, "close": [h["close"].iloc[-1] * 0.98] * n},
                              index=pd.date_range("2030-01-01", periods=n, freq="B")))
    names = [t["function"]["name"] for t in tk.tool_schemas()]
    assert names == ["technical_snapshot", "ma_crossover", "get_indicators", "confirm_kronos_forecast"]
    assert len(tk.to_anthropic()) == 4
    r = tk.call("get_indicators", json.dumps({"symbol": "X", "timeframe": "1d",
                                              "indicators": [{"name": "ema", "period": 50}, {"name": "macd"}, {"name": "nope"}]}))
    assert "ema_50" in r["values"] and "macd" in r["values"] and "error" in r["values"]["nope"]
    assert "error" not in tk.call("ma_crossover", {"symbol": "X", "timeframe": "1d", "fast": 50, "slow": 200})
    assert tk.call("confirm_kronos_forecast", {"symbol": "X", "timeframe": "1d", "pred_len": 5})["forecast"]["direction"] == -1
    assert "error" in tk.call("technical_snapshot", {"symbol": "X"})  # missing timeframe handled
