"""Performance tab and decision analytics."""
import time

import pytest

import dashboard.app as dash
from src.audit_log import AuditLog
from tests.test_dashboard_multibroker import AUTH, client  # noqa: F401

DAY = 86400


def _trade(a, deal, symbol, pattern, conf, entry, stop, qty, pnl, reason, opened_ago, closed_ago):
    a.log_decision(symbol, pattern, "BUY", 0.9, True, conf, True, "ok", "opened", deal_id=deal,
                   details={"entry": entry, "stop": stop, "quantity": qty})
    a.log_trade_result(symbol, deal, pnl, {"reason": reason, "entry": entry, "exit": entry + pnl / qty, "quantity": qty})
    now = time.time()
    a.conn.execute("UPDATE decisions SET ts = ? WHERE deal_id = ?", (now - opened_ago, deal))
    a.conn.execute("UPDATE trade_results SET ts = ? WHERE deal_id = ?", (now - closed_ago, deal))
    a.conn.commit()


@pytest.fixture
def history(client):  # noqa: F811
    a = AuditLog(str(dash.DB_PATH))
    # risk per trade = |entry - stop| * qty = 1.0 * 50 = 50
    _trade(a, "1", "US.AAPL", "hammer", 0.65, 100, 99, 50, 100.0, "target", 5 * DAY, 5 * DAY - 3600)       # +2R
    _trade(a, "2", "US.AAPL", "hammer", 0.72, 100, 99, 50, -50.0, "stop", 4 * DAY, 4 * DAY - 1800)         # -1R
    _trade(a, "3", "US.MSFT", "engulfing", 0.85, 200, 198, 25, -50.0, "stop", 3 * DAY, 3 * DAY - 600)      # -1R
    _trade(a, "4", "US.MSFT", "engulfing", 0.91, 200, 198, 25, 75.0, "trailing_stop", 2 * DAY, 2 * DAY - 900)  # +1.5R
    _trade(a, "5", "US.NVDA", "manual", None, 50, 49, 10, 5.0, "manual", 40 * DAY, 40 * DAY - 60)          # outside 30d
    a.close()
    return client


def test_performance_summary(history):
    d = history.get("/api/performance?days=30", headers=AUTH).get_json()
    s = d["summary"]
    assert s["trades"] == 4 and s["net"] == 75.0 and s["win_rate"] == 50.0
    assert s["profit_factor"] == 1.75 and s["expectancy"] == 18.75 and s["payoff"] == 1.75
    assert s["avg_r"] == pytest.approx(0.38, abs=0.01) and s["r_trades"] == 4
    assert s["max_drawdown"] == -100.0 and s["max_drawdown_pct"] == -100.0
    assert (s["win_streak"], s["loss_streak"]) == (1, 2)
    assert s["avg_hold_sec"] == round((3600 + 1800 + 600 + 900) / 4)
    assert [e["cum"] for e in d["equity"]] == [100.0, 50.0, 0.0, 75.0]
    assert [x["dd"] for x in d["drawdown"]] == [0.0, -50.0, -100.0, -25.0]


def test_performance_breakdowns(history):
    d = history.get("/api/performance?days=30", headers=AUTH).get_json()
    pat = {g["name"]: g for g in d["by_pattern"]}
    assert pat["hammer"]["net"] == 50.0 and pat["hammer"]["avg_r"] == 0.5
    assert pat["engulfing"]["win_rate"] == 50.0
    assert {g["name"] for g in d["by_reason"]} == {"target", "stop", "trailing stop"}
    assert sum(d["r_hist"]["counts"]) == 4
    calib = {c["bucket"]: c for c in d["confidence"]}
    assert calib["90–100%"]["win_rate"] == 100.0 and calib["80–90%"]["win_rate"] == 0.0
    assert len(d["daily"]) == 4 and d["trades"][0]["symbol"] == "US.MSFT"   # newest first
    assert d["trades"][0]["r"] == 1.5


def test_all_time_includes_manual_and_empty_db(history, client):  # noqa: F811
    d = history.get("/api/performance?days=all", headers=AUTH).get_json()
    assert d["summary"]["trades"] == 5 and any(g["name"] == "manual" for g in d["by_pattern"])
    assert history.get("/api/performance?days=x", headers=AUTH).status_code == 400


def test_empty_performance(client):  # noqa: F811
    d = client.get("/api/performance", headers=AUTH).get_json()
    assert d["summary"]["trades"] == 0 and d["summary"]["win_rate"] is None and d["equity"] == []


def test_decision_stats_funnel(client):  # noqa: F811
    a = AuditLog(str(dash.DB_PATH))
    a.log_decision("US.A", "hammer", "BUY", 0.8, None, None, None, "RSI 74 overbought", "rejected", details={"indicators": {}})
    a.log_decision("US.B", "hammer", "BUY", 0.8, False, 0.45, None, "kronos confidence 45% < 60%", "rejected")
    a.log_decision("US.C", "engulfing", "BUY", 0.9, True, 0.82, False, "reward:risk 1.2 below 1.5 (cap)", "rejected")
    a.log_decision("US.D", "engulfing", "BUY", 0.9, True, 0.77, True, "ok", "opened", deal_id="9")
    a.log_decision("US.E", "manual", "BUY", None, None, None, None, "manual override", "opened", deal_id="10")
    a.close()
    d = client.get("/api/decision_stats?days=7", headers=AUTH).get_json()
    assert [f["count"] for f in d["funnel"]] == [4, 3, 2, 1, 1]
    assert d["totals"] == {"opened": 1, "indicators": 1, "kronos": 1, "risk": 1, "error": 0}
    assert d["confidence_hist"]["confirmed"][8] == 1 and d["confidence_hist"]["confirmed"][7] == 1
    assert d["confidence_hist"]["rejected"][4] == 1
    assert {r["reason"] for r in d["top_reasons"]} >= {"reward:risk 1.2 below 1.5"}
    eng = next(p for p in d["by_pattern"] if p["name"] == "engulfing")
    assert eng["signals"] == 2 and eng["open_rate"] == 50.0
    rows = client.get("/api/decisions?limit=10", headers=AUTH).get_json()["decisions"]
    assert {r["epic"]: r["stage"] for r in rows} == {"US.A": "indicators", "US.B": "kronos", "US.C": "risk",
                                                     "US.D": "opened", "US.E": "manual"}


def test_entry_match_needs_same_symbol_and_earlier_time(client):  # noqa: F811
    a = AuditLog(str(dash.DB_PATH))
    _trade(a, "7", "US.AAPL", "hammer", 0.7, 100, 99, 10, 20.0, "target", 2 * DAY, 2 * DAY - 600)
    a.log_decision("US.TSLA", "doji", "BUY", 0.9, True, 0.9, True, "ok", "opened", deal_id="7",
                   details={"entry": 5, "stop": 4, "quantity": 1})        # same order id, other broker/symbol
    a.close()
    t = client.get("/api/performance?days=30", headers=AUTH).get_json()["trades"][0]
    assert t["pattern"] == "hammer" and t["hold_sec"] == 600 and t["r"] == 2.0
