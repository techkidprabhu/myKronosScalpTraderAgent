"""
Trading hours of IG's around-the-clock markets, by asset class.

IG's 24-hour CFD markets follow the global futures / interbank week, in New
York time:

  forex        Sunday 17:00 -> Friday 17:00, continuous
  indices      Sunday 18:00 -> Friday 17:00, daily break 17:00-18:00
  commodities  same as indices
  crypto       always open

These hours drive two things: whether a market is open for new entries
(together with IG's own live status - TRADEABLE or not - which also covers
holidays and instruments that close early) and the flat-before-close rule,
which closes a trade `risk.flat_by_minutes_before_close` minutes before its
own market's next close (the daily break or the weekend).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
CLASSES = ("forex", "indices", "commodities", "crypto")
MARKET = "IG24"           # the broker "market" code for around-the-clock IG trading

_FAR = timedelta(days=3650)


def _windows(cls: str, now_et: datetime) -> list[tuple[datetime, datetime]]:
    """Open windows around `now_et` (New York time), oldest first."""
    if cls == "crypto":
        return [(now_et - _FAR, now_et + _FAR)]
    today = now_et.date()
    sunday = today - timedelta(days=(today.weekday() + 1) % 7)       # this week's Sunday
    out = []
    for week in (-1, 0, 1):
        sun = sunday + timedelta(days=7 * week)
        fri = sun + timedelta(days=5)
        if cls == "forex":
            out.append((datetime.combine(sun, time(17), ET), datetime.combine(fri, time(17), ET)))
        else:                                   # indices, commodities: one window a day, Sun-Thu evenings
            for i in range(5):
                d = sun + timedelta(days=i)
                out.append((datetime.combine(d, time(18), ET), datetime.combine(d + timedelta(days=1), time(17), ET)))
    return out


def _now(now: datetime | None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(ET)


def current_window(cls: str, now: datetime | None = None) -> tuple[datetime, datetime] | None:
    t = _now(now)
    for start, end in _windows(cls, t):
        if start <= t < end:
            return start, end
    return None


def is_open(cls: str, now: datetime | None = None) -> bool:
    return current_window(cls, now) is not None


def session_minutes(cls: str, now: datetime | None = None) -> tuple[float, float] | None:
    """(minutes since this market opened, minutes to its next close), or None
    while it is closed. Crypto never closes: (inf, inf)."""
    t = _now(now)
    win = current_window(cls, t)
    if win is None:
        return None
    if cls == "crypto":
        return float("inf"), float("inf")
    return (t - win[0]).total_seconds() / 60, (win[1] - t).total_seconds() / 60


def next_open(cls: str, now: datetime | None = None) -> datetime | None:
    t = _now(now)
    if current_window(cls, t) is not None:
        return t
    opens = [s for s, _ in _windows(cls, t) if s > t]
    return min(opens) if opens else None


def classify(instrument_type: str | None, epic: str = "", hint: str | None = None) -> str | None:
    """Asset class of an IG market: from the navigation branch it was found
    in (hint), else its instrumentType, else its epic."""
    if hint in CLASSES:
        return hint
    it = str(instrument_type or "").upper()
    e = str(epic or "").upper()
    if "CRYPTO" in it or ".BITCOIN." in e or e.startswith("CS.D.BITCOIN") or ".ETHUSD." in e or ".BTCUSD." in e:
        return "crypto"
    if it == "CURRENCIES" or e.startswith("CS.D."):
        return "forex"
    if it == "INDICES" or e.startswith("IX.D."):
        return "indices"
    if it == "COMMODITIES" or e.startswith(("CC.D.", "MT.D.", "EN.D.")):
        return "commodities"
    return None
