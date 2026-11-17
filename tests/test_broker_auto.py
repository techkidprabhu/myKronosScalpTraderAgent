"""Auto broker: the dashboard follows whichever market is open."""
from datetime import datetime, timezone

from src import market_calendar as cal

BROKERS = [{"name": "moomoo", "market": "US", "enabled": True}, {"name": "zerodha", "market": "NSE", "enabled": True}]


def at(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def test_follows_india_then_the_us_through_the_day():
    day = [(at(2026, 11, 18, 2), "zerodha", "next"),    # 07:30 IST: India opens next
           (at(2026, 11, 18, 5), "zerodha", "open"),    # 10:30 IST
           (at(2026, 11, 18, 12), "moomoo", "next"),    # India closed, US opens at 14:30 UTC
           (at(2026, 11, 18, 16), "moomoo", "open"),    # 11:00 ET
           (at(2026, 11, 18, 22), "zerodha", "next")]   # US closed: India is next
    for now, name, state in day:
        p = cal.auto_pick(BROKERS, now)
        assert (p["name"], p["state"]) == (name, state), (now, p)
    assert "NSE is open" == cal.auto_pick(BROKERS, at(2026, 11, 18, 5))["reason"]


def test_weekends_holidays_and_disabled_brokers():
    # Saturday: Monday's Indian open comes before the US one
    assert cal.auto_pick(BROKERS, at(2026, 11, 21, 12))["name"] == "zerodha"
    # US Thanksgiving (26 Nov 2026): India trades, then the next open is India again
    p = cal.auto_pick(BROKERS, at(2026, 11, 26, 16))
    assert p["name"] == "zerodha" and p["state"] == "next"
    # a disabled broker is ignored while another is enabled
    only_us = [{**BROKERS[0]}, {**BROKERS[1], "enabled": False}]
    assert cal.auto_pick(only_us, at(2026, 11, 18, 5))["name"] == "moomoo"
    # nothing enabled: still picks among all; unknown markets never win
    assert cal.auto_pick([{**b, "enabled": False} for b in BROKERS], at(2026, 11, 18, 5))["name"] == "zerodha"
    assert cal.auto_pick([], at(2026, 11, 18, 5)) is None


def test_brokers_api_reports_the_pick(client):  # noqa: F811
    d = client.get("/api/brokers", headers=AUTH).get_json()
    assert d["auto"]["name"] in {b["name"] for b in d["brokers"]} and d["auto"]["reason"]


from tests.test_dashboard_multibroker import AUTH, client  # noqa: E402,F401
