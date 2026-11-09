"""Chart trade marks: entries, exits and the live position for one symbol."""
import time

import dashboard.app as dash
from src.audit_log import AuditLog
from tests.test_dashboard_multibroker import AUTH, client  # noqa: F401


def test_trade_marks_for_one_symbol(client):  # noqa: F811
    a = AuditLog(str(dash.DB_PATH))
    a.log_decision("US.AAPL", "hammer", "BUY", 0.9, True, 0.78, True, "ok", "opened", deal_id="1",
                   details={"entry": 296.0, "quantity": 50})
    a.log_decision("US.MSFT", "hammer", "BUY", 0.9, True, 0.7, True, "ok", "opened", deal_id="2")
    a.log_decision("AAPL", "manual", "SELL", None, None, None, None, "manual override", "opened", deal_id="3",
                   details={"entry": 300.0, "quantity": 5})
    a.log_trade_result("US.AAPL", "1", 120.0, {"reason": "target", "exit": 298.4, "quantity": 50})
    a.log_trade_result("US.MSFT", "2", -10.0, {"reason": "stop", "exit": 400.0})
    a.record_pending_entry("US.AAPL", "BUY", 20, 297.0, 295.0, 301.0, entry_order_id="9")
    a.close()

    d = client.get("/api/trade_marks?symbol=US.AAPL", headers=AUTH).get_json()
    assert [(e["direction"], e["price"], e["manual"]) for e in d["entries"]] == [("BUY", 296.0, False), ("SELL", 300.0, True)]
    assert d["entries"][0]["quantity"] == 50 and d["entries"][0]["confidence"] == 0.78
    assert [(x["pnl"], x["price"], x["reason"]) for x in d["exits"]] == [(120.0, 298.4, "target")]
    p = d["position"]
    assert p["status"] == "pending_entry" and p["current_stop"] == 295.0 and p["target"] == 301.0 and p["trail_active"] is False


def test_trade_marks_since_and_validation(client):  # noqa: F811
    a = AuditLog(str(dash.DB_PATH))
    a.log_trade_result("US.AAPL", "1", 5.0, {"exit": 1.0})
    a.close()
    later = time.time() + 60
    d = client.get(f"/api/trade_marks?symbol=AAPL&since={later}", headers=AUTH).get_json()
    assert d["exits"] == [] and d["position"] is None
    assert client.get("/api/trade_marks", headers=AUTH).status_code == 400
