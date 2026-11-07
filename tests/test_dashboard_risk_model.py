"""Risk Model screen: live risk state endpoint, and edits saved through the
settings API land in config.yaml."""
import time
from datetime import datetime

import pytest
import yaml

import dashboard.app as dash
from src.audit_log import AuditLog
from src.market import market_timezone
from tests.test_dashboard_multibroker import AUTH, client  # noqa: F401
from tests.test_manual_orders import LiveClient


def _today():
    return datetime.now(market_timezone(dash._spec_market(dash.current_spec()))).date().isoformat()


def test_risk_state_reports_today(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dash, "_quote_cache", {})
    monkeypatch.setattr(dash, "get_broker", lambda name=None: LiveClient("moomoo", "US"))
    cooldown = dash.load_config().get("risk", {}).get("min_minutes_between_trades", 0)
    audit = AuditLog(str(dash.DB_PATH))
    audit.set_kv("risk_state", {"daily_pnl": -42.5, "trading_halted": True, "trading_day": _today(),
                                "last_trade_time": {"US.AAPL": time.time() - 60, "US.OLD": time.time() - 10 ** 7}})
    audit.record_pending_entry("US.AAPL", "BUY", 5, 1.0, 0.9, 1.2, entry_order_id="1")
    audit.close()
    d = client.get("/api/risk_state?symbol=US.AAPL", headers=AUTH).get_json()
    assert d["daily_pnl"] == -42.5 and d["halted"] is True
    assert d["open_positions"] == 1 and d["positions"][0]["symbol"] == "US.AAPL"
    if cooldown > 1:
        assert [c["symbol"] for c in d["cooldowns"]] == ["US.AAPL"]
        assert 0 < d["cooldowns"][0]["minutes_left"] <= cooldown
    assert d["reference_price"] == 1.25


def test_yesterdays_loss_is_not_shown_as_today(client):  # noqa: F811
    audit = AuditLog(str(dash.DB_PATH))
    audit.set_kv("risk_state", {"daily_pnl": -500.0, "trading_halted": True, "trading_day": "2000-01-01",
                                "last_trade_time": {}})
    audit.close()
    d = client.get("/api/risk_state", headers=AUTH).get_json()
    assert d["daily_pnl"] == 0.0 and d["halted"] is False and d["reference_price"] is None


def test_no_agent_state_yet(client):  # noqa: F811
    d = client.get("/api/risk_state", headers=AUTH).get_json()
    assert d["daily_pnl"] == 0.0 and d["open_positions"] == 0 and d["cooldowns"] == []


def test_risk_model_save_round_trip(client):  # noqa: F811
    changes = {"risk.stop_loss_pct": 0.8, "risk.max_open_positions": 4,
               "trailing_stop.enabled": True, "trailing_stop.trail_distance": 1.5}
    r = client.post("/api/settings", json={"scope": "shared", "changes": changes}, headers=AUTH)
    assert r.status_code == 200, r.get_json()
    cfg = yaml.safe_load(dash.CONFIG_PATH.read_text())
    assert cfg["risk"]["stop_loss_pct"] == 0.8 and cfg["risk"]["max_open_positions"] == 4
    assert cfg["trailing_stop"]["enabled"] is True and cfg["trailing_stop"]["trail_distance"] == 1.5
    bad = client.post("/api/settings", json={"scope": "shared", "changes": {"risk.max_daily_loss": 0}}, headers=AUTH)
    assert bad.status_code == 400


def test_page_has_risk_model_tab(client):  # noqa: F811
    html = client.get("/", headers=AUTH).get_data(as_text=True)
    assert 'id="tab-risk-model"' in html and "switchTab('risk-model')" in html
