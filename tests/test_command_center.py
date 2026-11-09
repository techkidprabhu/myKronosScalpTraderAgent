"""Command Center: today's overview and the live server-sent-event stream."""
import json
import time

import pytest

import dashboard.app as dash
from src.audit_log import AuditLog
from tests.test_dashboard_multibroker import AUTH, client  # noqa: F401


@pytest.fixture
def seeded(client):  # noqa: F811
    a = AuditLog(str(dash.DB_PATH))
    for _ in range(5):
        a.log_snapshot("US.AAPL", 100.0, "scanning", {})
    a.log_decision("US.AAPL", "hammer", "BUY", 0.8, None, None, None, "RSI 74 overbought", "rejected",
                   details={"indicators": {"rsi": 74}})
    a.log_decision("US.MSFT", "engulfing", "BUY", 0.9, False, 0.41, None, "kronos confidence 41% < 60%", "rejected")
    a.log_decision("US.TSLA", "engulfing", "BUY", 0.9, True, 0.8, False, "reward:risk 1.20 below min_reward_risk 1.5",
                   "rejected")
    a.log_decision("US.NVDA", "hammer", "BUY", 0.9, True, 0.78, True, "ok", "opened", deal_id="7")
    a.log_decision("US.AMD", "manual", "BUY", None, None, None, None, "manual override", "opened", deal_id="8")
    a.log_trade_result("US.META", "5", -48.2, {"reason": "stop"})
    a.log_trade_result("US.AAPL", "6", 120.0, {"reason": "target"})
    a.record_pending_entry("US.NVDA", "BUY", 20, 186.0, 183.0, 192.0, entry_order_id="7")
    a.log_snapshot("US.QQQ", None, "flat_by_cancel_entry", {})
    a.close()
    return client


def test_overview_pipeline_and_results(seeded):
    d = seeded.get("/api/overview", headers=AUTH).get_json()
    p = d["pipeline"]
    assert (p["scanned"], p["patterns"], p["indicators"], p["kronos"], p["risk"], p["opened"]) == (5, 4, 1, 1, 1, 1)
    assert {r["stage"] for r in d["top_reasons"]} == {"indicators", "kronos", "risk"}
    r = d["results"]
    assert r["realized_today"] == pytest.approx(71.8) and r["closed_today"] == 2
    assert r["win_rate_30d"] == 50.0 and r["profit_factor"] == pytest.approx(2.49, abs=0.01)
    assert r["curve_30d"] == [-48.2, 71.8]
    assert d["risk"]["open_risk"] == 60.0 and d["risk"]["open_positions"] == 1


def test_overview_activity_is_newest_first_and_labelled(seeded):
    acts = seeded.get("/api/overview", headers=AUTH).get_json()["activity"]
    assert [a["ts"] for a in acts] == sorted((a["ts"] for a in acts), reverse=True)
    kinds = {a["kind"] for a in acts}
    assert {"indicators", "kronos", "risk", "opened", "manual", "win", "loss", "cancel"} <= kinds
    texts = " | ".join(a["text"] for a in acts)
    assert "Risk rejected US.TSLA" in texts and "Entered US.NVDA hammer BUY · Kronos 78%" in texts
    assert "Entry cancelled (flat by close): US.QQQ" in texts


def test_overview_with_empty_database(client):  # noqa: F811
    d = client.get("/api/overview", headers=AUTH).get_json()
    assert d["pipeline"]["patterns"] == 0 and d["results"]["win_rate_30d"] is None and d["activity"] == []


def _events(raw: str) -> dict:
    out = {}
    for block in raw.split("\n\n"):
        lines = dict(l.split(": ", 1) for l in block.splitlines() if ": " in l and not l.startswith(":"))
        if "event" in lines:
            out[lines["event"]] = json.loads(lines["data"])
    return out


def test_stream_sends_overview_and_positions(seeded):
    r = seeded.get("/api/stream?once=1", headers=AUTH)
    assert r.mimetype == "text/event-stream" and r.headers["Cache-Control"] == "no-cache"
    raw = r.get_data(as_text=True)
    assert raw.startswith("retry: 3000")
    ev = _events(raw)
    assert ev["overview"]["pipeline"]["opened"] == 1
    assert "positions" in ev["positions"]


def test_stream_needs_login(client):  # noqa: F811
    assert client.get("/api/stream?once=1").status_code == 401


def test_page_opens_on_command_center(client):  # noqa: F811
    html = client.get("/", headers=AUTH).get_data(as_text=True)
    assert 'id="tab-command-center" class="tab-panel active"' in html
    assert "new EventSource(" in html
