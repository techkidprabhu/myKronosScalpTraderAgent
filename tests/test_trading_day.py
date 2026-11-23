"""'Today' can start at midnight in your own time zone (trading_day_timezone)."""
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.audit_log import AuditLog
from src.market import trading_day_timezone
from src.risk import RiskConfig, RiskManager
from tests.test_dashboard_multibroker import AUTH, client  # noqa: F401 - fixture


def test_default_is_the_markets_time_zone():
    assert str(trading_day_timezone({}, "US")) == "America/New_York"
    assert str(trading_day_timezone({"trading_day_timezone": ""}, "IG24")) == "America/New_York"


def test_your_own_time_zone_when_set():
    assert str(trading_day_timezone({"trading_day_timezone": "Asia/Singapore"}, "IG24")) == "Asia/Singapore"
    assert str(trading_day_timezone({"trading_day_timezone": "Mars/Base"}, "US")) == "America/New_York"   # typo


def test_daily_limits_reset_at_that_midnight(tmp_path):
    rm = RiskManager(RiskConfig(max_open_positions=3, stop_loss_points=1, take_profit_points=2, max_daily_loss=100,
                                min_minutes_between_trades=0, max_stake_per_point=100), store=AuditLog(str(tmp_path / "a.db")),
                     timezone="Asia/Singapore")
    rm._roll_day()
    assert rm.state.trading_day == datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat()


def seed_trades(path):
    a = AuditLog(str(path))
    sg = ZoneInfo("Asia/Singapore")
    midnight_sg = datetime.now(sg).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    for ts, pnl in ((midnight_sg - 3600, -59.25), (midnight_sg + 60, 223.20)):   # yesterday SGT, today SGT
        a.conn.execute("INSERT INTO trade_results (ts, epic, deal_id, pnl, details) VALUES (?,?,?,?,?)",
                       (ts, "CC.D.CL.UNG.IP", "D", pnl, "{}"))
    a.conn.commit()
    return midnight_sg


def set_day_tz(monkeypatch, name):
    import dashboard.app as dash
    real = dash.load_config
    monkeypatch.setattr(dash, "load_config", lambda *a, **k: {**(real(*a, **k) or {}), "trading_day_timezone": name})


def test_command_center_and_performance_today_agree(client, monkeypatch):  # noqa: F811
    set_day_tz(monkeypatch, "Asia/Singapore")
    midnight_sg = seed_trades(client.tmp / "audit.db")
    if time.time() < midnight_sg + 120:
        pytest.skip("too close to midnight in Singapore")
    ov = client.get("/api/overview", headers=AUTH).get_json()
    assert ov["results"]["realized_today"] == 223.20
    assert ov["day_starts"]["tz"] == "Asia/Singapore" and ov["day_starts"]["label"].startswith("00:00")
    pf = client.get("/api/performance?days=today", headers=AUTH).get_json()
    assert pf["summary"]["net"] == pytest.approx(223.20)
    pf30 = client.get("/api/performance?days=30", headers=AUTH).get_json()
    assert pf30["summary"]["net"] == pytest.approx(163.95)


def test_settings_reject_a_misspelt_time_zone():
    from dashboard import schema
    assert schema.coerce("trading_day_timezone", "Asia/Singapore") == "Asia/Singapore"
    assert schema.coerce("trading_day_timezone", "") is None
    with pytest.raises(ValueError, match="unknown time zone"):
        schema.coerce("trading_day_timezone", "Asia/Singapor")
