"""Kronos scorecard: each verdict checked against the close N candles later."""
import json
import sqlite3

import pytest

from src.audit_log import AuditLog
from src import kronos_scorecard as ks
from tests.test_dashboard_multibroker import AUTH, client  # noqa: F401 - fixture

BAR = 180.0
T0 = 1_780_000_000.0


def seed(path, verdicts, horizon=3, prices=None):
    """verdicts: (symbol, direction, confirmed, confidence, price now, price `horizon` candles later)."""
    a = AuditLog(str(path))
    db = a.conn
    for k, (sym, d, ok, conf, p0, p1) in enumerate(verdicts):
        t = T0 + k * 10 * BAR
        for i in range(horizon + 1):                     # one 'scanning' close per candle
            price = p0 if i == 0 else (p1 if i == horizon else (p0 + p1) / 2)
            db.execute("INSERT INTO snapshots (ts, symbol, price, stage, details) VALUES (?,?,?,?,?)",
                       (t + i * BAR - 1, sym, price, "scanning", "{}"))
        db.execute("INSERT INTO decisions (ts, epic, pattern, pattern_direction, pattern_strength, kronos_confirmed, "
                   "kronos_confidence, risk_allowed, risk_reason, action, deal_id, details) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (t, sym, "engulfing", d, 0.8, int(ok), conf, None, None, "opened" if ok and conf >= .62 else "rejected",
                    None, json.dumps({"paths_confidence": 0.9 if p1 > p0 else 0.1})))
    db.commit()
    return a


def test_passed_and_rejected_verdicts_are_checked_n_candles_later(tmp_path):
    a = seed(tmp_path / "a.db", [
        ("US.A", "BUY", True, 1.0, 100, 101),     # agreed, right  (+1%)
        ("US.B", "SELL", True, 0.67, 50, 49),     # agreed, right  (price fell)
        ("US.C", "BUY", True, 0.67, 10, 9.9),     # agreed, wrong
        ("US.D", "BUY", False, 0.33, 20, 19),     # disagreed, price fell: Kronos was right to say no
    ])
    out = ks.score(a.conn, horizon=3, bar_seconds=BAR, threshold=0.62, now=T0 + 50 * BAR, days=30)
    g = {x["name"]: x for x in out["groups"]}
    assert g["All verdicts"]["scored"] == 4
    agreed, disagreed = out["groups"][1], out["groups"][2]
    assert (agreed["signals"], agreed["right"], agreed["right_pct"]) == (3, 2, 66.7)
    assert (disagreed["signals"], disagreed["right"], disagreed["right_pct"]) == (1, 0, 0.0)
    assert out["edge_points"] == 66.7
    assert agreed["avg_move_pct"] == pytest.approx((1 + 2 - 1) / 3, abs=1e-3)
    b = {x["name"]: x["signals"] for x in out["buckets"]}
    assert b == {"Below 40%": 1, "40-59%": 0, "60-79%": 2, "80% and up": 1}


def test_verdicts_without_a_later_candle_stay_unchecked(tmp_path):
    a = seed(tmp_path / "a.db", [("US.A", "BUY", True, 1.0, 100, 101)], horizon=3)
    a.conn.execute("DELETE FROM snapshots WHERE ts > ?", (T0 + BAR,))     # symbol dropped by the screener
    a.conn.commit()
    out = ks.score(a.conn, horizon=3, bar_seconds=BAR, threshold=0.62, now=T0 + 50 * BAR)
    assert out["groups"][0]["signals"] == 1 and out["groups"][0]["scored"] == 0 and out["edge_points"] is None


def test_a_recorded_shadow_confidence_can_be_scored_instead(tmp_path):
    a = seed(tmp_path / "a.db", [("US.A", "BUY", False, 0.33, 100, 101), ("US.B", "BUY", True, 0.67, 100, 99)])
    out = ks.score(a.conn, horizon=3, bar_seconds=BAR, threshold=0.62, now=T0 + 50 * BAR, conf_key="paths_confidence")
    agreed, disagreed = out["groups"][1], out["groups"][2]
    # The shadow measure agreed with the rise and disagreed with the fall: price went the
    # agreed signal's way (100%) and against the disagreed one (0%) - both good calls.
    assert agreed["right_pct"] == 100.0 and disagreed["right_pct"] == 0.0 and out["edge_points"] == 100.0


def test_endpoint_reports_scorecard_and_timing(client):  # noqa: F811
    import time
    a = AuditLog(str(client.tmp / "audit.db"))
    now = time.time()
    db = a.conn
    for i, price in enumerate([100, 100.5, 101, 102, 102, 102, 102]):
        db.execute("INSERT INTO snapshots (ts, symbol, price, stage, details) VALUES (?,?,?,?,?)",
                   (now - 3000 + i * 300 - 1, "US.A", price, "scanning", "{}"))
    db.execute("INSERT INTO decisions (ts, epic, pattern_direction, kronos_confirmed, kronos_confidence, action) "
               "VALUES (?,?,?,?,?,?)", (now - 3000, "US.A", "BUY", 1, 0.8, "opened"))
    db.commit()
    a.set_kv("kronos_timing", {"day": "x", "calls": 2, "symbols": 3, "total_ms": 3000, "max_ms": 2000,
                               "last_ms": 1000, "last_symbols": 1, "avg_ms_per_symbol": 1000})
    d = client.get("/api/kronos_scorecard?days=7", headers=AUTH).get_json()
    assert d["horizon"] == 5 and d["timeframe"] == "MIN_5" and d["verdicts"] == 1     # from the test config
    assert d["groups"][1]["right"] == 1 and d["timing"]["calls"] == 2
    assert client.get("/api/kronos_scorecard?days=0", headers=AUTH).status_code == 400
