"""
Exchange rules the agent must respect before any order goes out:
trading sessions, price tick sizes, and which bar is still forming.

Pure functions only - no broker calls - so they're easy to unit test.

Exchange holidays and early closes come from src/market_calendar.py:
NYSE/Nasdaq rules are computed for every year (Good Friday, observed
days, 1 pm closes); SGX/HKEX are weekend-only unless dates are added in
config/holidays.yaml. China A-shares are weekend-only.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from src import market_calendar as cal

SESSIONS = {
    # US regular trading hours (pre/post-market not traded by the agent)
    "US": (ZoneInfo("America/New_York"), [(time(9, 30), time(16, 0))]),
    # HKEX: morning + afternoon continuous sessions
    "HK": (ZoneInfo("Asia/Hong_Kong"), [(time(9, 30), time(12, 0)), (time(13, 0), time(16, 0))]),
    # SGX securities market: morning + afternoon session with a lunch break
    "SG": (ZoneInfo("Asia/Singapore"), [(time(9, 0), time(12, 0)), (time(13, 0), time(17, 0))]),
    # China A-shares (SSE/SZSE)
    "CN": (ZoneInfo("Asia/Shanghai"), [(time(9, 30), time(11, 30)), (time(13, 0), time(15, 0))]),
    # India - NSE / BSE cash market, normal session (holidays: market_calendar NSE list)
    "NSE": (ZoneInfo("Asia/Kolkata"), [(time(9, 15), time(15, 30))]),
    "BSE": (ZoneInfo("Asia/Kolkata"), [(time(9, 15), time(15, 30))]),
    # IG around the clock: each market keeps its own hours (src/ig_hours.py);
    # this entry only gives the risk day (New York midnight) and a weekday clock.
    "IG24": (ZoneInfo("America/New_York"), [(time(0, 0), time(23, 59, 59))]),
}

# NSE cash-market tick bands (NSE circulars, in force since 15 Apr 2025):
# (upper price bound, tick). NSE sets each stock's band monthly from the
# previous month's close, so near a boundary the live price can sit in the
# other band - the Zerodha / Angel adapters re-round to the instrument's own
# published tick before sending an order.
IN_TICKS = [(250.0, 0.01), (1000.0, 0.05), (5000.0, 0.10), (10000.0, 0.50),
            (20000.0, 1.00), (float("inf"), 5.00)]

BAR_DURATION = {
    "MIN_1": pd.Timedelta(minutes=1),
    "MIN_3": pd.Timedelta(minutes=3),
    "MIN_5": pd.Timedelta(minutes=5),
    "MIN_15": pd.Timedelta(minutes=15),
    "HOUR_1": pd.Timedelta(hours=1),
    "HOUR_4": pd.Timedelta(hours=4),
    "DAY": pd.Timedelta(days=1),
}

# HKEX spread table: (upper price bound, tick)
HK_SPREADS = [
    (0.25, 0.001), (0.50, 0.005), (10.0, 0.01), (20.0, 0.02), (100.0, 0.05),
    (200.0, 0.1), (500.0, 0.2), (1000.0, 0.5), (2000.0, 1.0), (5000.0, 2.0),
    (float("inf"), 5.0),
]


def market_timezone(market: str) -> ZoneInfo:
    if market not in SESSIONS:
        raise ValueError(f"Unknown market {market!r}; expected one of {sorted(SESSIONS)}")
    return SESSIONS[market][0]


def trading_day_timezone(cfg: dict | None, market: str) -> ZoneInfo:
    """Where "today" starts and ends for P&L, the daily loss limit, the
    profit target and the dashboard's Today figures: `trading_day_timezone`
    in config.yaml (e.g. Asia/Singapore), else the market's own time zone.
    Matters for markets that trade around the clock (IG), where the market's
    midnight can fall in the middle of your day."""
    name = str((cfg or {}).get("trading_day_timezone") or "").strip()
    if name:
        try:
            return ZoneInfo(name)
        except Exception:  # noqa: BLE001 - a typo falls back, the settings check reports it
            pass
    return market_timezone(market)


def is_market_open(market: str, now: datetime | None = None) -> bool:
    tz = market_timezone(market)
    local = (now or datetime.now(tz)).astimezone(tz)
    if local.weekday() >= 5:
        return False
    exchange = cal.MARKET_EXCHANGE.get(market)
    sessions = SESSIONS[market][1]
    if exchange:
        day = local.date()
        if cal.holiday(exchange, day):
            return False
        early = cal.EXCHANGES[exchange].early_close
        if early and cal.early_close(exchange, day):
            sessions = [(s, min(e, early)) for s, e in sessions if s < early]
    t = local.time()
    return any(start <= t < end for start, end in sessions)


def tick_size(price: float, market: str = "US") -> float:
    """Minimum price increment for a stock at `price` on `market`."""
    if market == "SG":
        # <S$0.20 -> 0.001, S$0.20 to <S$1.00 -> 0.005, >=S$1.00 -> 0.01
        if price < 0.20:
            return 0.001
        if price < 1.00:
            return 0.005
        return 0.01
    if market in ("NSE", "BSE"):
        for bound, tick in IN_TICKS:
            if price <= bound if bound != 250.0 else price < bound:
                return tick
    if market == "HK":
        for bound, tick in HK_SPREADS:
            if price < bound:
                return tick
    if market == "CN":
        return 0.01
    # US: $0.01 at/above $1, $0.0001 below
    return 0.01 if price >= 1.00 else 0.0001


def round_to_tick(price: float, market: str = "US") -> float:
    tick = tick_size(price, market)
    # round() twice: once to the nearest tick count, once to strip float noise
    # like 71.55999999999999 that the broker would reject.
    return round(round(price / tick) * tick, 6)


def drop_unclosed_bar(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Brokers return the still-forming bar as the last row. Pattern
    detection must only ever see closed candles, so drop the last row if
    its period hasn't ended yet. Index is naive UTC bar START time (see
    MoomooClient.get_historical_bars, which converts moomoo's end-stamped
    intraday bars to start times)."""
    if df.empty:
        return df
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    if df.index[-1] + BAR_DURATION[timeframe] > now:
        return df.iloc[:-1]
    return df
