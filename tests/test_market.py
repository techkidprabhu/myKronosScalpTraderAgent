from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.market import drop_unclosed_bar, is_market_open, round_to_tick, tick_size

SGT = ZoneInfo("Asia/Singapore")


@pytest.mark.parametrize("hhmm,expected", [
    ((8, 59), False),
    ((9, 0), True),
    ((11, 59), True),
    ((12, 30), False),  # lunch break
    ((13, 0), True),
    ((16, 59), True),
    ((17, 0), False),
    ((23, 53), False),  # when the live orders in the audit log were placed
])
def test_sgx_sessions(hhmm, expected):
    monday = datetime(2026, 11, 2, *hhmm, tzinfo=SGT)
    assert is_market_open("SG", monday) is expected


def test_sgx_closed_on_weekend():
    saturday = datetime(2026, 11, 7, 10, 0, tzinfo=SGT)
    assert is_market_open("SG", saturday) is False


def test_session_check_converts_timezones():
    # 02:00 UTC = 10:00 SGT
    assert is_market_open("SG", datetime(2026, 11, 2, 2, 0, tzinfo=ZoneInfo("UTC"))) is True


def test_unknown_market_raises():
    with pytest.raises(ValueError):
        is_market_open("XX")


@pytest.mark.parametrize("price,tick", [(0.15, 0.001), (0.20, 0.005), (0.99, 0.005), (1.00, 0.01), (71.7, 0.01)])
def test_sgx_tick_sizes(price, tick):
    assert tick_size(price, "SG") == tick


def test_round_to_tick_removes_float_noise():
    assert round_to_tick(71.55999999999999, "SG") == 71.56
    assert round_to_tick(4.260000000000001, "SG") == 4.26
    assert round_to_tick(0.537, "SG") == 0.535
    assert round_to_tick(0.1234, "SG") == 0.123


def _bars(index):
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=pd.DatetimeIndex(index))


def test_drop_unclosed_bar_drops_forming_bar():
    now = pd.Timestamp("2026-11-03 06:00")
    df = _bars(["2026-11-03 00:00", "2026-11-03 04:00"])  # 04:00 bar runs to 08:00
    out = drop_unclosed_bar(df, "HOUR_4", now=now)
    assert list(out.index) == [pd.Timestamp("2026-11-03 00:00")]


def test_drop_unclosed_bar_keeps_closed_bar():
    now = pd.Timestamp("2026-11-03 08:00")
    df = _bars(["2026-11-03 00:00", "2026-11-03 04:00"])
    assert len(drop_unclosed_bar(df, "HOUR_4", now=now)) == 2


@pytest.mark.parametrize("price,tick", [
    (0.2, 0.001), (0.3, 0.005), (5.0, 0.01), (15.0, 0.02), (50.0, 0.05),
    (150.0, 0.1), (400.0, 0.2), (600.0, 0.5), (1500.0, 1.0),
])
def test_hk_spread_table(price, tick):
    assert tick_size(price, "HK") == tick


def test_us_ticks():
    assert tick_size(150.0, "US") == 0.01
    assert tick_size(0.5, "US") == 0.0001
    assert round_to_tick(187.3449999, "US") == 187.34


def test_hk_session_lunch_break():
    hkt = ZoneInfo("Asia/Hong_Kong")
    assert is_market_open("HK", datetime(2026, 11, 2, 10, 0, tzinfo=hkt)) is True
    assert is_market_open("HK", datetime(2026, 11, 2, 12, 30, tzinfo=hkt)) is False


def test_us_session():
    et = ZoneInfo("America/New_York")
    assert is_market_open("US", datetime(2026, 11, 2, 9, 29, tzinfo=et)) is False
    assert is_market_open("US", datetime(2026, 11, 2, 15, 59, tzinfo=et)) is True


ET = ZoneInfo("America/New_York")


@pytest.mark.parametrize("day,name", [
    ((2026, 11, 26), "Thanksgiving Day"),
    ((2026, 12, 25), "Christmas Day"),
    ((2027, 3, 26), "Good Friday"),
    ((2026, 7, 3), "Independence Day"),   # Jul 4 2026 is a Saturday -> observed Friday
    ((2027, 6, 18), "Juneteenth"),        # Jun 19 2027 is a Saturday
])
def test_us_holidays_closed_all_day(day, name):
    from src.market_calendar import holiday
    from datetime import date
    assert holiday("NASDAQ", date(*day)) == name
    assert is_market_open("US", datetime(*day, 11, 0, tzinfo=ET)) is False


def test_us_early_close_at_1pm():
    # Day after Thanksgiving 2026
    assert is_market_open("US", datetime(2026, 11, 27, 12, 59, tzinfo=ET)) is True
    assert is_market_open("US", datetime(2026, 11, 27, 13, 0, tzinfo=ET)) is False


def test_us_saturday_new_year_not_observed_on_friday():
    # Jan 1 2028 is a Saturday: NYSE does not close on Fri Dec 31 2027
    assert is_market_open("US", datetime(2027, 12, 31, 11, 0, tzinfo=ET)) is True


def test_next_open_skips_weekend_and_holiday():
    from src.market_calendar import next_open
    # Wed 25 Nov 2026 after the close -> Thu is Thanksgiving -> Fri 27 Nov 09:30
    nxt = next_open("NASDAQ", datetime(2026, 11, 25, 17, 0, tzinfo=ET))
    assert nxt == datetime(2026, 11, 27, 9, 30, tzinfo=ET)


def test_session_states_for_dashboard():
    from src.market_calendar import session
    ist = ZoneInfo("Asia/Kolkata")
    assert session("NASDAQ", datetime(2026, 11, 2, 10, 0, tzinfo=ET))["state"] == "open"
    assert session("NASDAQ", datetime(2026, 11, 2, 7, 0, tzinfo=ET))["state"] == "pre"
    assert session("NASDAQ", datetime(2026, 11, 26, 11, 0, tzinfo=ET))["state"] == "holiday"
    assert session("NSE", datetime(2026, 11, 10, 11, 0, tzinfo=ist))["state"] == "holiday"
    assert session("NSE", datetime(2026, 11, 9, 11, 0, tzinfo=ist))["state"] == "open"
