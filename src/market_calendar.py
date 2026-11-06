"""
Exchange calendars and live session state for NASDAQ and NSE (plus SGX /
HKEX with weekend-only rules).

  NASDAQ / NYSE  holidays and 1 pm early closes computed from the NYSE
                 rules, so every year is covered (checked against the
                 published 2026 and 2027 schedules)
  NSE            the circular list (NSE/CMTR/71775 + NSE/CMTR/72260 for
                 2026); later years are added to config/holidays.yaml when
                 NSE publishes them - until then only weekends are known
                 and the status says so

Extra dates for any exchange can be added in config/holidays.yaml:

    NSE:
      "2027-01-26": Republic Day
    NASDAQ:
      "2027-01-09": National day of mourning
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXTRA_PATH = ROOT / "config" / "holidays.yaml"


@dataclass(frozen=True)
class Exchange:
    code: str
    name: str
    city: str
    tz: str
    tz_label: str
    open: time
    close: time
    early_close: time | None = None
    pre: tuple | None = None          # (start, end) local
    post: tuple | None = None
    lunch: tuple | None = None        # midday break (SGX/HKEX)
    holidays_known_until: int | None = None


EXCHANGES: dict[str, Exchange] = {
    "NASDAQ": Exchange("NASDAQ", "NASDAQ", "New York", "America/New_York", "ET", time(9, 30), time(16, 0),
                       early_close=time(13, 0), pre=(time(4, 0), time(9, 30)), post=(time(16, 0), time(20, 0)),
                       holidays_known_until=9999),
    "NSE": Exchange("NSE", "NSE", "Mumbai", "Asia/Kolkata", "IST", time(9, 15), time(15, 30),
                    pre=(time(9, 0), time(9, 15)), post=(time(15, 40), time(16, 0)), holidays_known_until=2026),
    "SGX": Exchange("SGX", "SGX", "Singapore", "Asia/Singapore", "SGT", time(9, 0), time(17, 0),
                    pre=(time(8, 30), time(9, 0)), lunch=(time(12, 0), time(13, 0))),
    "HKEX": Exchange("HKEX", "HKEX", "Hong Kong", "Asia/Hong_Kong", "HKT", time(9, 30), time(16, 0),
                     pre=(time(9, 0), time(9, 30)), lunch=(time(12, 0), time(13, 0))),
}

# broker market code -> the exchange whose session drives the agent
MARKET_EXCHANGE = {"US": "NASDAQ", "NSE": "NSE", "BSE": "NSE", "SG": "SGX", "HK": "HKEX"}

NSE_HOLIDAYS = {  # capital market segment, NSE/CMTR/71775 (12 Dec 2025) + NSE/CMTR/72260 (12 Jan 2026)
    "2026-01-15": "Municipal Corporation Election, Maharashtra",
    "2026-01-26": "Republic Day", "2026-03-03": "Holi", "2026-03-26": "Shri Ram Navami",
    "2026-03-31": "Shri Mahavir Jayanti", "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Baba Saheb Ambedkar Jayanti", "2026-05-01": "Maharashtra Day",
    "2026-05-28": "Bakri Id", "2026-06-26": "Muharram", "2026-09-14": "Ganesh Chaturthi",
    "2026-10-02": "Mahatma Gandhi Jayanti", "2026-10-20": "Dussehra",
    "2026-11-10": "Diwali-Balipratipada", "2026-11-24": "Prakash Gurpurb Sri Guru Nanak Dev",
    "2026-12-25": "Christmas",
}
NSE_SPECIAL = {"2026-11-08": "Muhurat trading (Diwali, timing per NSE circular)"}


# ---------------------------------------------------------------- US rules
def _nth_weekday(y, m, weekday, n):
    d = date(y, m, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(y, m, weekday):
    d = date(y, m + 1, 1) - timedelta(days=1) if m < 12 else date(y, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(y):  # anonymous Gregorian algorithm
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    return date(y, month, (h + l - 7 * m + 114) % 31 + 1)


def _observed(d):
    return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d


@lru_cache(maxsize=16)
def us_calendar(y: int) -> tuple[dict, dict]:
    """(holidays {iso: name}, early_closes {iso: name}) per NYSE rules."""
    h = {}
    ny = date(y, 1, 1)
    if ny.weekday() != 5:  # a Saturday New Year is not observed on Dec 31
        h[_observed(ny)] = "New Year's Day"
    h[_nth_weekday(y, 1, 0, 3)] = "Martin Luther King Jr. Day"
    h[_nth_weekday(y, 2, 0, 3)] = "Washington's Birthday"
    h[_easter(y) - timedelta(days=2)] = "Good Friday"
    h[_last_weekday(y, 5, 0)] = "Memorial Day"
    if y >= 2022:
        h[_observed(date(y, 6, 19))] = "Juneteenth"
    h[_observed(date(y, 7, 4))] = "Independence Day"
    h[_nth_weekday(y, 9, 0, 1)] = "Labor Day"
    thanks = _nth_weekday(y, 11, 3, 4)
    h[thanks] = "Thanksgiving Day"
    h[_observed(date(y, 12, 25))] = "Christmas Day"
    e = {thanks + timedelta(days=1): "Day after Thanksgiving"}
    j4 = date(y, 7, 4)
    if 1 <= j4.weekday() <= 4:  # Jul 4 on Tue-Fri -> Jul 3 closes at 1 pm
        e[j4 - timedelta(days=1)] = "Independence Day eve"
    x24 = date(y, 12, 24)
    if x24.weekday() <= 3:     # Christmas Eve Mon-Thu
        e[x24] = "Christmas Eve"
    return ({d.isoformat(): n for d, n in h.items()}, {d.isoformat(): n for d, n in e.items()})


@lru_cache(maxsize=1)
def _extra() -> dict:
    if EXTRA_PATH.exists():
        try:
            return {k: {str(d): n for d, n in (v or {}).items()} for k, v in (yaml.safe_load(EXTRA_PATH.read_text()) or {}).items()}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def holiday(ex: str, d: date) -> str | None:
    iso = d.isoformat()
    if iso in _extra().get(ex, {}):
        return _extra()[ex][iso]
    if ex == "NASDAQ":
        return us_calendar(d.year)[0].get(iso)
    if ex == "NSE":
        return NSE_HOLIDAYS.get(iso)
    return None


def early_close(ex: str, d: date) -> str | None:
    return us_calendar(d.year)[1].get(d.isoformat()) if ex == "NASDAQ" else None


def is_trading_day(ex: str, d: date) -> bool:
    return d.weekday() < 5 and not holiday(ex, d)


def _at(d: date, t: time, tz: ZoneInfo) -> datetime:
    return datetime.combine(d, t, tzinfo=tz)


def _close_time(ex: Exchange, d: date) -> time:
    return ex.early_close if (ex.early_close and early_close(ex.code, d)) else ex.close


def next_open(code: str, now: datetime | None = None) -> datetime:
    ex = EXCHANGES[code]
    tz = ZoneInfo(ex.tz)
    now = (now or datetime.now(timezone.utc)).astimezone(tz)
    d = now.date()
    for i in range(15):
        day = d + timedelta(days=i)
        if is_trading_day(code, day) and _at(day, ex.open, tz) > now:
            return _at(day, ex.open, tz)
    return _at(d + timedelta(days=15), ex.open, tz)


def session(code: str, now: datetime | None = None, extended: bool = False) -> dict:
    """Live state of one exchange. state: open | lunch | pre | post | closed | holiday | weekend."""
    ex = EXCHANGES[code]
    tz = ZoneInfo(ex.tz)
    now = (now or datetime.now(timezone.utc)).astimezone(tz)
    d, t = now.date(), now.time()
    hol = holiday(code, d)
    early = early_close(code, d)
    close_t = _close_time(ex, d)
    nxt = next_open(code, now)
    out = {"code": code, "name": ex.name, "city": ex.city, "tz": ex.tz, "tz_label": ex.tz_label,
           "local": now.isoformat(), "date": d.isoformat(), "holiday": hol, "early_close": early,
           "special": NSE_SPECIAL.get(d.isoformat()) if code == "NSE" else None,
           "regular": f"{ex.open:%H:%M}-{close_t:%H:%M}", "next_open": nxt.timestamp(),
           "closes_at": None, "calendar_note": None}
    if ex.holidays_known_until and d.year > ex.holidays_known_until:
        out["calendar_note"] = f"{ex.name} {d.year} holiday list not loaded - add it to config/holidays.yaml"
    elif not ex.holidays_known_until:
        out["calendar_note"] = "weekends only - add holidays to config/holidays.yaml"
    if d.weekday() >= 5:
        state, label = "weekend", "Weekend"
    elif hol:
        state, label = "holiday", f"Holiday · {hol}"
    elif ex.open <= t < close_t:
        if ex.lunch and ex.lunch[0] <= t < ex.lunch[1]:
            state, label = "lunch", "Lunch break"
        else:
            state, label = "open", "Open" + (" · early close" if early else "")
        out["closes_at"] = _at(d, close_t, tz).timestamp()
    elif ex.pre and ex.pre[0] <= t < ex.pre[1]:
        state, label = "pre", "Pre-market" if code == "NASDAQ" else "Pre-open"
    elif ex.post and t >= close_t and (close_t if early else ex.post[0]) <= t < (time(17, 0) if early else ex.post[1]):
        state, label = "post", "After-hours" if code == "NASDAQ" else "Closing session"
    else:
        state, label = "closed", "Closed"
    out.update(state=state, label=label,
               active=state == "open" or (extended and state in ("pre", "post")))
    out["next_change"] = out["closes_at"] if state in ("open", "lunch") else out["next_open"]
    return out


def upcoming_holidays(code: str, n: int = 4, today: date | None = None) -> list[dict]:
    today = today or datetime.now(ZoneInfo(EXCHANGES[code].tz)).date()
    out, d = [], today
    for _ in range(400):
        if len(out) >= n:
            break
        h, e = holiday(code, d), early_close(code, d)
        if (h and d.weekday() < 5) or e:
            out.append({"date": d.isoformat(), "name": h or f"{e} (closes 1 pm)", "early": bool(e and not h)})
        d += timedelta(days=1)
    return out


def exchange_for_market(market: str) -> str:
    return MARKET_EXCHANGE.get(market.upper(), "NASDAQ")
