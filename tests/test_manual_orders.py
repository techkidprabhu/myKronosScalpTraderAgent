"""Manual buy / sell / close from the dashboard, and live prices in the
positions table."""
import pandas as pd
import pytest

import dashboard.app as dash
from src.audit_log import AuditLog
from tests.test_orchestrator import SYMBOL, FakeBroker, make_worker, market_open  # noqa: F401
from tests.test_dashboard_multibroker import AUTH, client, FakeClient  # noqa: F401


# ---- orchestrator side ---------------------------------------------------

def test_manual_buy_creates_agent_managed_pending_entry(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    worker, audit, _ = make_worker(tmp_path, broker)
    msg = worker.manual_entry("BUY", 25)
    pos = audit.get_active_position(SYMBOL)
    assert pos["status"] == "pending_entry" and pos["quantity"] == 25 and pos["direction"] == "BUY"
    assert "submitted" in msg
    # Kronos is not consulted for a manual order
    worker.kronos_agent.confirm_direction.assert_not_called()
    # once filled it gets the standard bracket, anchored to the fill
    broker.fill(int(pos["entry_order_id"]), price=71.0)
    broker.held = 25
    worker.monitor_position()
    pos = audit.get_active_position(SYMBOL)
    assert pos["status"] == "open" and pos["current_stop"] < 71.0 < pos["target"]


def test_manual_buy_refused_when_already_exposed(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    broker.held = 10
    worker, audit, _ = make_worker(tmp_path, broker)
    with pytest.raises(RuntimeError, match="already"):
        worker.manual_entry("BUY", 5)
    assert audit.get_active_position(SYMBOL) is None


def test_manual_close_of_managed_position_books_the_exit(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.manual_entry("BUY", 10)
    pos = audit.get_active_position(SYMBOL)
    broker.fill(int(pos["entry_order_id"]), price=71.0)
    broker.held = 10
    worker.monitor_position()
    broker.price = 72.0
    msg = worker.manual_close()
    assert "closed" in msg
    assert audit.get_active_position(SYMBOL) is None
    assert broker.held == 0
    reasons = [r[0] for r in audit.conn.execute("SELECT details FROM trade_results")]
    assert any('"manual"' in r for r in reasons)


def test_manual_close_of_unmanaged_holding(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    broker.held = 7
    worker, audit, _ = make_worker(tmp_path, broker)
    # protected by default: a holding the agent did not open is never sold
    with pytest.raises(RuntimeError, match="protected"):
        worker.manual_close()
    assert broker.held == 7
    worker.global_cfg["unmanaged_positions"] = {"allow_manual_close": True}
    assert "SELL 7" in worker.manual_close()
    assert broker.held == 0


def test_queue_is_claimed_once_and_stale_requests_expire(tmp_path):
    audit = AuditLog(str(tmp_path / "a.db"))
    old = audit.queue_manual_order("US.AAPL", "BUY", 5)
    audit.conn.execute("UPDATE manual_orders SET ts = ts - 1000 WHERE id=?", (old,))
    new = audit.queue_manual_order("US.AAPL", "CLOSE", None)
    claimed = audit.claim_manual_orders()
    assert [r["id"] for r in claimed] == [new]
    assert audit.claim_manual_orders() == []
    rows = {r["id"]: r for r in audit.recent_manual_orders()}
    assert rows[old]["status"] == "failed" and "expired" in rows[old]["result"]
    assert rows[new]["status"] == "working"


def test_orchestrator_processes_queue(tmp_path, market_open):  # noqa: F811
    from unittest.mock import MagicMock
    import src.orchestrator as orch
    broker = FakeBroker()
    worker, audit, _ = make_worker(tmp_path, broker)
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    o.audit_log, o.workers = audit, {SYMBOL: worker}
    o._monitor_worker = MagicMock()
    ok_id = audit.queue_manual_order(SYMBOL, "BUY", 3)
    bad_id = audit.queue_manual_order(SYMBOL, "BUY", 3)     # second entry on the same symbol -> refused
    o.process_manual_orders()
    rows = {r["id"]: r for r in audit.recent_manual_orders()}
    assert rows[ok_id]["status"] == "done", rows[ok_id]["result"]
    assert rows[bad_id]["status"] == "failed" and "already" in rows[bad_id]["result"]
    o._monitor_worker.assert_not_called()
    # a market-prefixed request finds the worker for the bare symbol
    o.workers = {"AAPL": worker}
    assert o._worker_for("US.AAPL") is worker


# ---- dashboard side ------------------------------------------------------

class LiveClient(FakeClient):
    quote = 1.25

    def get_snapshots(self, symbols):
        return pd.DataFrame([{"code": s, "last_price": self.quote, "volume": 1} for s in symbols])


@pytest.fixture
def live(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dash, "_quote_cache", {})
    c = LiveClient("moomoo", "US")
    monkeypatch.setattr(dash, "get_broker", lambda name=None: c)
    return c


def test_positions_use_live_quote_not_cached_broker_price(client, live):  # noqa: F811
    data = client.get("/api/positions", headers=AUTH).get_json()
    p = data["positions"][0]
    assert p["current_price"] == 1.25 and p["price_source"] == "live"
    assert p["unrealized_pnl"] == pytest.approx(0.25)    # (1.25 - 1.0) * 1
    assert data["updated_at"] > 0


def test_positions_fall_back_when_quotes_fail(client, live, monkeypatch):  # noqa: F811
    def boom(symbols):
        raise RuntimeError("quote rights")
    monkeypatch.setattr(live, "get_snapshots", boom)
    p = client.get("/api/positions", headers=AUTH).get_json()["positions"][0]
    assert p["current_price"] == 1.0 and p["price_source"] == "broker"


def test_manual_order_endpoint_queues_and_validates(client, live):  # noqa: F811
    r = client.post("/api/manual_order", json={"symbol": "us.aapl", "action": "buy", "quantity": 5}, headers=AUTH)
    assert r.status_code == 200 and r.get_json()["ok"]
    for bad in ({"symbol": "", "action": "BUY", "quantity": 1},
                {"symbol": "X", "action": "HOLD", "quantity": 1},
                {"symbol": "X", "action": "BUY", "quantity": 0},
                {"symbol": "X", "action": "SELL", "quantity": "lots"}):
        assert client.post("/api/manual_order", json=bad, headers=AUTH).status_code == 400
    assert client.post("/api/manual_order", json={"symbol": "X", "action": "CLOSE"}, headers=AUTH).status_code == 200
    rows = client.get("/api/positions", headers=AUTH).get_json()["manual_orders"]
    assert [(r["action"], r["symbol"], r["status"]) for r in rows] == [
        ("CLOSE", "X", "queued"), ("BUY", "US.AAPL", "queued")]
