"""Audit trail shows the price at decision time and the live price now."""
import time

import pytest

import dashboard.app as dash
from src.audit_log import AuditLog
from tests.test_dashboard_multibroker import AUTH, client  # noqa: F401
from tests.test_manual_orders import LiveClient


@pytest.fixture
def live(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dash, "_quote_cache", {})
    c = LiveClient("moomoo", "US")
    monkeypatch.setattr(dash, "get_broker", lambda name=None: c)
    return c


def _decisions(client):  # noqa: F811
    return {d["pattern"]: d for d in client.get("/api/decisions", headers=AUTH).get_json()["decisions"]}


def test_decision_and_live_prices(client, live):  # noqa: F811
    audit = AuditLog(str(dash.DB_PATH))
    now = time.time()
    # rejected BUY: price comes from the snapshot taken while evaluating it
    audit.conn.execute("INSERT INTO snapshots (ts, symbol, price, stage, details) VALUES (?, ?, ?, ?, ?)",
                       (now - 5, "US.AAPL", 1.00, "pattern_detected", "{}"))
    audit.conn.commit()
    audit.log_decision("US.AAPL", "hammer", "BUY", 0.8, False, 0.4, None, "low", "rejected")
    # opened SELL: price is the order's entry price
    audit.log_decision("US.MSFT", "shooting_star", "SELL", 0.9, True, 0.9, True, "", "opened",
                       deal_id="7", details={"entry": 1.50, "stop": 1.6, "target": 1.3})
    # no snapshot and no entry: no decision price, still a live price
    audit.log_decision("US.NVDA", "doji", "BUY", 0.5, None, None, None, "indicators", "rejected")
    audit.close()

    d = _decisions(client)
    assert d["hammer"]["decision_price"] == 1.00 and d["hammer"]["current_price"] == 1.25
    assert d["hammer"]["move_pct"] == pytest.approx(25.0)          # BUY, price rose: with the signal
    assert d["shooting_star"]["decision_price"] == 1.50
    assert d["shooting_star"]["move_pct"] == pytest.approx(16.67)  # SELL, price fell: with the signal
    assert d["doji"]["decision_price"] is None and d["doji"]["current_price"] == 1.25
    assert d["doji"]["move_pct"] is None


def test_old_snapshot_is_not_used(client, live):  # noqa: F811
    audit = AuditLog(str(dash.DB_PATH))
    audit.conn.execute("INSERT INTO snapshots (ts, symbol, price, stage, details) VALUES (?, ?, ?, ?, ?)",
                       (time.time() - 7200, "US.AAPL", 9.0, "pattern_detected", "{}"))
    audit.conn.commit()
    audit.log_decision("US.AAPL", "hammer", "BUY", 0.8, False, 0.4, None, "low", "rejected")
    audit.close()
    assert _decisions(client)["hammer"]["decision_price"] is None


def test_broker_offline_keeps_decision_price(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dash, "get_broker", lambda name=None: None)
    audit = AuditLog(str(dash.DB_PATH))
    audit.log_decision("US.MSFT", "engulfing", "BUY", 0.9, True, 0.9, True, "", "opened",
                       deal_id="1", details={"entry": 2.0})
    audit.close()
    d = _decisions(client)["engulfing"]
    assert d["decision_price"] == 2.0 and d["current_price"] is None and d["move_pct"] is None


def test_quotes_are_shared_between_tables(client, live, monkeypatch):  # noqa: F811
    calls = []
    orig = live.get_snapshots
    monkeypatch.setattr(live, "get_snapshots", lambda syms: calls.append(list(syms)) or orig(syms))
    audit = AuditLog(str(dash.DB_PATH))
    audit.log_decision("US.X", "hammer", "BUY", 0.8, False, 0.4, None, "low", "rejected")
    audit.close()
    client.get("/api/positions", headers=AUTH)   # holds US.X
    client.get("/api/decisions", headers=AUTH)
    assert calls == [["US.X"]]   # quoted once, reused by the audit trail
