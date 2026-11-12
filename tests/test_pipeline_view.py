"""Decision Pipeline card: the newest signal's whole trip, not just the
newest snapshot (which is nearly always 'scanning')."""
import json
import sqlite3
import time

from dashboard.app import _pipeline_latest
from src.audit_log import AuditLog


def db(tmp_path):
    a = AuditLog(str(tmp_path / "a.db"))
    conn = sqlite3.connect(str(tmp_path / "a.db"))
    conn.row_factory = sqlite3.Row
    return a, conn


def test_full_trip_is_collected_and_later_scans_do_not_hide_it(tmp_path):
    a, conn = db(tmp_path)
    a.log_snapshot("NSE.TCS", 100.0, "scanning")
    a.log_snapshot("NSE.TCS", 100.0, "pattern_detected", {"pattern": "hammer", "direction": "BUY", "strength": 0.6})
    a.log_snapshot("NSE.TCS", 100.0, "indicator_check", {"passed": True, "alignment": 40})
    a.log_snapshot("NSE.TCS", 100.0, "kronos_result", {"confirmed": True, "confidence": 0.8})
    a.log_snapshot("NSE.TCS", 100.0, "risk_check", {"allowed": True, "reason": "ok"})
    a.log_decision("NSE.TCS", "hammer", "BUY", 0.6, True, 0.8, True, "ok", "error",
                   details={"error": "account dormant"})
    for sym in ("NSE.INFY", "NSE.ITC", "NSE.SBIN"):                 # later scans of other symbols
        a.log_snapshot(sym, 50.0, "scanning")
    a.log_snapshot("NSE.TCS", 101.0, "scanning")                     # TCS's next bar
    p = _pipeline_latest(conn)
    c = p["cycle"]
    assert c["symbol"] == "NSE.TCS"
    assert set(c["stages"]) == {"pattern_detected", "indicator_check", "kronos_result", "risk_check"}
    assert c["stages"]["kronos_result"]["confidence"] == 0.8
    assert c["decision"]["action"] == "error" and c["decision"]["error"] == "account dormant"
    assert p["scan"]["symbols"] == 4


def test_prefers_the_selected_symbol_and_handles_no_signal(tmp_path):
    a, conn = db(tmp_path)
    assert _pipeline_latest(conn) == {"scan": None, "cycle": None}
    a.log_snapshot("NSE.TCS", 1.0, "pattern_detected", {"pattern": "hammer", "direction": "BUY", "strength": 0.6})
    a.log_snapshot("NSE.ITC", 1.0, "pattern_detected", {"pattern": "inside_bar", "direction": "SELL", "strength": 0.5})
    assert _pipeline_latest(conn)["cycle"]["symbol"] == "NSE.ITC"
    assert _pipeline_latest(conn, "NSE.TCS")["cycle"]["symbol"] == "NSE.TCS"
    assert _pipeline_latest(conn, "NSE.LT")["cycle"]["symbol"] == "NSE.ITC"   # no signal: newest overall
