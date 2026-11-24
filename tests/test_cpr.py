"""CPR step 1: the levels, Ochoa's two-day relationships, width classes,
the per-session scorecard, the agent's once-a-day candle collector and the
dashboard endpoint. Nothing here trades."""
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import cpr
from src.audit_log import AuditLog
from src.orchestrator import Orchestrator


def lv(tc, bc):
    return {"TC": tc, "BC": bc, "P": (tc + bc) / 2}


def test_levels_formulas_and_the_swap():
    l = cpr.levels(110, 90, 104)
    assert l["P"] == pytest.approx(101.3333, rel=1e-4) and l["BC"] == 100 and l["TC"] == pytest.approx(102.6667, rel=1e-4)
    assert l["R1"] == pytest.approx(2 * l["P"] - 90) and l["S1"] == pytest.approx(2 * l["P"] - 110)
    assert l["R2"] == pytest.approx(l["P"] + 20) and l["S2"] == pytest.approx(l["P"] - 20)
    assert l["width_pct"] == pytest.approx((l["TC"] - l["BC"]) / l["P"] * 100)
    low_close = cpr.levels(110, 90, 94)          # close under the midpoint: TC computes below BC
    assert low_close["TC"] > low_close["BC"] and low_close["TC"] == 100


@pytest.mark.parametrize("cur,prev,want", [
    (lv(106, 104), lv(102, 100), "higher_value"),
    (lv(98, 96), lv(102, 100), "lower_value"),
    (lv(103, 101), lv(102, 100), "overlapping_higher"),
    (lv(101, 99), lv(102, 100), "overlapping_lower"),
    (lv(101.5, 100.5), lv(102, 100), "inside_value"),
    (lv(103, 99), lv(102, 100), "outside_value"),
    (lv(102.1, 100.1), lv(102, 100), "unchanged"),
])
def test_two_day_relationships(cur, prev, want):
    assert cpr.relationship(cur, prev) == want


def test_width_class_is_relative_to_the_markets_own_history():
    hist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert cpr.width_class(0.1, hist) == "narrow"
    assert cpr.width_class(0.45, hist) == "normal"
    assert cpr.width_class(0.9, hist) == "wide"
    assert cpr.width_class(0.1, hist[:3]) == "unknown"


def daily(n=40, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) + rng.uniform(0.2, 1.5, n)
    lo = np.minimum(o, c) - rng.uniform(0.2, 1.5, n)
    idx = pd.date_range("2026-09-01", periods=n, freq="D")
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c}, index=idx)


def test_build_days_scores_each_session_against_its_own_cpr():
    d = daily()
    days = cpr.build_days(d)
    assert len(days) == len(d)                                # one per session + the current one
    assert days[-1]["current"] and days[-1]["day"] is None and days[-1]["based_on"] == "2026-10-10"
    r = days[5]                                               # the 7th session, CPR from the 6th
    src, day = d.iloc[5], d.iloc[6]
    l = cpr.levels(src["high"], src["low"], src["close"])
    assert r["day"] == "2026-09-07" and r["P"] == pytest.approx(l["P"], abs=1e-6)
    assert r["reached_r1"] == (day["high"] >= l["R1"]) and r["closed_above"] == (day["close"] > l["TC"])
    assert r["touched"] == (day["low"] <= l["TC"] and day["high"] >= l["BC"])
    assert days[0]["relationship"] is None and days[1]["relationship"] in cpr.LABELS
    assert days[3]["width_class"] == "unknown" and days[30]["width_class"] in ("narrow", "normal", "wide")


def test_trend_day_needs_a_big_range_and_a_close_at_an_extreme():
    idx = pd.date_range("2026-09-01", periods=17, freq="D")
    rows = [(100, 101, 99, 100)] * 15 + [(100, 101, 99, 100), (100, 104, 99.8, 103.9)]
    d = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    last = cpr.build_days(d)[-2]
    assert last["trend_day"] and last["range_x"] == pytest.approx(4.2 / 2, rel=1e-3)


def test_scorecard_groups_and_rates():
    days = [
        {"day": "a", "width_class": "narrow", "relationship": "higher_value", "range_x": 2.0, "trend_day": True,
         "reached_r1": True, "reached_s1": False, "touched": False, "closed_above": True, "closed_below": False},
        {"day": "b", "width_class": "wide", "relationship": "lower_value", "range_x": 0.5, "trend_day": False,
         "reached_r1": False, "reached_s1": False, "touched": True, "closed_above": False, "closed_below": False},
        {"day": None, "width_class": "narrow", "relationship": "inside_value", "current": True},
    ]
    s = cpr.scorecard(days)
    assert s["all"]["days"] == 2 and s["all"]["trend_days_pct"] == 50.0 and s["all"]["avg_range_x"] == 1.25
    assert [g["class"] for g in s["by_width"]] == ["narrow", "wide"]
    assert s["by_width"][0]["trend_days_pct"] == 100.0 and s["by_width"][1]["touched_pct"] == 100.0
    assert {g["relationship"] for g in s["by_relationship"]} == {"higher_value", "lower_value"}
    assert cpr.scorecard([])["all"]["days"] == 0


def test_acceptance_needs_consecutive_closes_after_the_opening_minutes():
    idx = pd.date_range("2026-11-24 09:00", periods=8, freq="5min")
    band = {"TC": 101.0, "BC": 100.0, "P": 100.5}
    up = pd.DataFrame({"close": [101.5, 101.6, 100.5, 100.6, 100.8, 100.9, 101.2, 101.3]}, index=idx)
    assert cpr.acceptance(up, band, session_start="2026-11-24 09:00") == "above"
    one = up.copy()
    one.iloc[-2, 0] = 100.9
    assert cpr.acceptance(one, band, session_start="2026-11-24 09:00") == "inside"
    early = up.iloc[:3]                                       # all inside the first 15 minutes
    assert cpr.acceptance(early, band, session_start="2026-11-24 09:00") == "inside"
    dn = pd.DataFrame({"close": [99.0] * 8}, index=idx)
    assert cpr.acceptance(dn, band, session_start="2026-11-24 09:00") == "below"
    assert cpr.location(101.2, band) == "above" and cpr.location(100.2, band) == "inside"


def test_settings_merge_and_clamp():
    s = cpr.settings({"cpr": {"enabled": True, "width_lookback": 0, "bogus": 1}})
    assert s["enabled"] is True and s["width_lookback"] == 2 and "bogus" not in s
    assert cpr.settings({})["enabled"] is False


# ---------------------------------------------------------------- the agent's collector

class DailyBroker:
    def __init__(self):
        self.calls = []

    def get_historical_bars(self, symbol, tf, n):
        self.calls.append((symbol, tf, n))
        d = daily(n + 1)
        d.index = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n + 1, freq="D")   # last = today, unclosed
        return pd.DataFrame({"time": d.index.astype("int64") // 10**6, **{c: d[c].values for c in d}, "volume": 0.0})


def make_orch(tmp_path, enabled=True, symbols=("A", "B")):
    o = Orchestrator.__new__(Orchestrator)
    o.config = {"cpr": {"enabled": enabled, "backfill_days": 40}}
    o.audit_log = AuditLog(str(tmp_path / "audit.db"))
    o.broker = DailyBroker()
    o.workers = {s: SimpleNamespace(symbol=s) for s in symbols}
    return o


def test_collector_backfills_once_then_waits_for_tomorrow(tmp_path):
    o = make_orch(tmp_path)
    now = time.time()
    o._collect_daily_bars(now)
    assert o.broker.calls == [("A", "DAY", 40), ("B", "DAY", 40)]
    assert o.audit_log.daily_bar_count("A") == 40                       # today's unclosed candle dropped
    o._collect_daily_bars(now + 600)
    assert len(o.broker.calls) == 2                                      # done for today
    o.audit_log.set_kv("cpr_fetched:A", "2000-01-01")
    o._collect_daily_bars(now + 1200)
    assert o.broker.calls[-1] == ("A", "DAY", 5)                         # history known: last 5 days only


def test_collector_is_off_by_default_and_spreads_requests(tmp_path):
    o = make_orch(tmp_path, enabled=False)
    o._collect_daily_bars(time.time())
    assert o.broker.calls == []
    o = make_orch(tmp_path / "x", symbols=("A", "B", "C", "D", "E", "F"))
    o._collect_daily_bars(time.time())
    assert len(o.broker.calls) == Orchestrator.CPR_SYMBOLS_PER_CHECK


def test_a_failed_fetch_waits_an_hour(tmp_path):
    o = make_orch(tmp_path, symbols=("A",))
    o.broker.get_historical_bars = lambda *a: (_ for _ in ()).throw(RuntimeError("no data"))
    now = time.time()
    o._collect_daily_bars(now)
    assert o.audit_log.daily_bar_count("A") == 0 and o._cpr_failed["A"] == now


# ---------------------------------------------------------------- dashboard

from tests.test_dashboard_multibroker import AUTH, client  # noqa: E402,F401,F811 - fixture


def test_dashboard_cpr_endpoint(client):  # noqa: F811
    a = AuditLog(str(client.tmp / "audit.db"))
    a.upsert_daily_bars("US.A", daily(40))
    a.conn.execute("INSERT INTO snapshots (ts, symbol, price, stage, details) VALUES (?,?,?,?,?)",
                   (time.time(), "US.A", 1000.0, "scanning", "{}"))
    a.conn.commit()
    d = client.get("/api/cpr", headers=AUTH).get_json()
    assert d["enabled"] is False and len(d["symbols"]) == 1
    s = d["symbols"][0]
    assert s["symbol"] == "US.A" and s["based_on"] == "2026-10-10" and s["location"] == "above"
    assert s["TC"] >= s["BC"] and s["history_days"] == 40
    assert d["scorecard"]["all"]["days"] == 39


def test_dashboard_cpr_endpoint_before_any_candles(client):  # noqa: F811
    d = client.get("/api/cpr", headers=AUTH).get_json()
    assert d["symbols"] == [] and d["scorecard"]["all"]["days"] == 0
