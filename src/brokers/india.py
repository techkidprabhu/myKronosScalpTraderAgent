"""
Helpers shared by the Indian broker adapters (Zerodha Kite, Angel One).

Both brokers return candles stamped with the bar START in IST and cap how
many days one history request may span, so bars are fetched in chunks and
converted to the agent's format: epoch-ms UTC bar start, oldest first.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta

import pandas as pd


IST = "Asia/Kolkata"
SESSION_MINUTES = 375          # 09:15-15:30
COLUMNS = ["time", "open", "high", "low", "close", "volume"]
# agent timeframe -> (minutes per broker bar, bars grouped into one agent bar)
TIMEFRAME_BARS = {"MIN_1": (1, 1), "MIN_5": (5, 1), "MIN_15": (15, 1), "HOUR_1": (60, 1),
                  "HOUR_4": (60, 4), "DAY": (None, 1)}


def history_window(resolution: str, num_points: int) -> timedelta:
    """Calendar span that comfortably covers `num_points` closed bars,
    weekends and holidays included."""
    minutes, group = TIMEFRAME_BARS[resolution]
    want = num_points * group + group
    if minutes is None:
        days = want * 1.6 + 10
    else:
        days = want * minutes / SESSION_MINUTES * 1.6 + 7
    return timedelta(days=int(days) + 1)


def fetch_chunked(fetch, start: datetime, end: datetime, max_days: int, pause: float = 0.35) -> list:
    """Call fetch(chunk_start, chunk_end) over [start, end] in spans the
    broker accepts, pausing between calls for its rate limit."""
    rows, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=max_days), end)
        rows += fetch(cursor, chunk_end) or []
        cursor = chunk_end
        if cursor < end and pause:
            time.sleep(pause)
    return rows


def to_frame(rows: list, resolution: str, num_points: int, symbol: str) -> pd.DataFrame:
    """rows: [(timestamp, open, high, low, close, volume), ...] with
    timestamps in IST (aware or naive) -> the agent's bar DataFrame."""
    if not rows:
        return pd.DataFrame(columns=COLUMNS + ["symbol"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["ts"])
    ts = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
    df["ts"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.drop_duplicates("ts").sort_values("ts")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    group = TIMEFRAME_BARS[resolution][1]
    if group > 1:
        local_day = df["ts"].dt.tz_localize("UTC").dt.tz_convert(IST).dt.date
        idx = df.groupby(local_day).cumcount() // group
        df = df.groupby([local_day, idx], sort=False).agg(
            ts=("ts", "first"), open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        ).reset_index(drop=True).sort_values("ts")
    df = df.tail(num_points)
    out = pd.DataFrame({
        "time": (df["ts"].astype("datetime64[ns]").astype("int64") // 10**6).to_numpy(),
        **{c: df[c].to_numpy() for c in ("open", "high", "low", "close", "volume")},
    })
    out["symbol"] = symbol
    return out


def round_to_instrument(price: float, tick: float | None, market: str, mode: str = "nearest") -> float:
    """Round to the instrument's own published tick (exact), else to the
    exchange band table. mode: nearest | down | up."""
    if not tick or tick <= 0:
        from src.market import tick_size
        tick = tick_size(price, market)
    q = price / tick
    q = math.floor(q + 1e-9) if mode == "down" else math.ceil(q - 1e-9) if mode == "up" else round(q)
    decimals = max(0, -int(math.floor(math.log10(tick)))) + 1
    return round(q * tick, decimals)
