"""Single-stock focus: the volume-breakout rule, the switch, and how the
scan loop uses it (only the focused symbol scanned, one position at a time,
the breakout entry's stop just beyond the level)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src import focus
from src.audit_log import AuditLog
from src.orchestrator import Orchestrator
from src.patterns import PatternSignal


def bars(n=40, breakout=True, spike=3.0, close_frac=0.9, trend=True, day="2026-11-24 14:30"):
    """n 5-minute candles drifting up under a 101 ceiling, then a breakout
    candle on `spike` x volume."""
    idx = pd.date_range(day, periods=n + 1, freq="5min")
    base = 100.0 + 0.4 * np.sin(np.arange(n) / 4.0) + np.linspace(0, 0.3, n) if trend else np.full(n, 100.0)
    o = base - 0.03
    c = base + np.where(np.arange(n) % 3 == 2, -0.08, 0.03)     # mostly up candles
    hi = np.minimum(np.maximum(o, c) + 0.1, 100.7)
    lo = np.minimum(o, c) - 0.1
    v = np.where(c > o, 1200.0, 800.0)
    if breakout:
        lo_b, hi_b = 100.55, 100.95
        cl = lo_b + (hi_b - lo_b) * close_frac
        row = (100.6, hi_b, lo_b, cl, float(v.mean() * spike))
    else:
        row = (100.5, 100.8, 100.4, 100.7, float(v.mean()))
    df = pd.DataFrame({"open": [*o, row[0]], "high": [*hi, row[1]], "low": [*lo, row[2]],
                       "close": [*c, row[3]], "volume": [*v, row[4]]}, index=idx)
    return df


def test_a_clean_volume_breakout_qualifies():
    r = focus.evaluate(bars(), focus.DEFAULTS)
    assert r["qualified"], r
    assert r["direction"] == "BUY" and r["level"] == 100.7
    assert r["volume_ratio"] >= 1.5 and 0 < r["score"] <= 100


def test_weak_volume_is_a_fakeout():
    r = focus.evaluate(bars(spike=1.2), focus.DEFAULTS)
    assert not r["qualified"] and "volume only" in r["why"]


def test_a_candle_that_closes_back_near_its_low_is_rejected():
    df = bars()
    df.iloc[-1, df.columns.get_loc("high")] = 101.4
    df.iloc[-1, df.columns.get_loc("close")] = 100.75
    r = focus.evaluate(df, focus.DEFAULTS)
    # closes at 100.75: still past the 100.7 level, but near the bottom of its range
    assert not r["qualified"] and "fakeout" in r["why"]


def test_no_breakout_no_pick_and_no_volume_data_is_said():
    assert "no breakout" in focus.evaluate(bars(breakout=False), focus.DEFAULTS)["why"]
    df = bars()
    df["volume"] = 0.0
    assert focus.evaluate(df, focus.DEFAULTS)["why"] == "no volume data for this market"
    assert "not enough candles" in focus.evaluate(bars(n=10), focus.DEFAULTS)["why"]


def test_rsi_window_and_shorts_only_when_asked():
    hot = focus.evaluate(bars(), {**focus.DEFAULTS, "rsi_max": 56})
    assert not hot["qualified"] and "RSI" in hot["why"]
    down = bars()
    flip = down.copy()
    for col in ("open", "high", "low", "close"):
        flip[col] = 200 - down[col]
    flip["high"], flip["low"] = 200 - down["low"], 200 - down["high"]
    assert not focus.evaluate(flip, focus.DEFAULTS)["qualified"]             # long only by default
    r = focus.evaluate(flip, {**focus.DEFAULTS, "sides": "both"})
    assert r["qualified"] and r["direction"] == "SELL" and r["level"] == 99.3


def test_pick_takes_the_best_score_and_prime_window():
    assert focus.pick({"A": {"qualified": True, "score": 60}, "B": {"qualified": True, "score": 80},
                       "C": {"qualified": False, "score": 0}}) == "B"
    assert focus.pick({"A": {"qualified": False}}) is None
    assert focus.in_prime_window((30, 300), focus.DEFAULTS)
    assert not focus.in_prime_window((150, 200), focus.DEFAULTS)
    assert focus.in_prime_window(None, focus.DEFAULTS)                       # 24-hour market
    assert focus.in_prime_window((300, 60), {**focus.DEFAULTS, "prime_minutes": 0})


def test_release_rules():
    f = {"direction": "BUY", "level": 100.7, "bars_waited": 1}
    df = bars()
    assert focus.still_valid(df, f, focus.DEFAULTS) is None
    back = df.copy()
    back.iloc[-1, back.columns.get_loc("close")] = 100.6
    assert "failed breakout" in focus.still_valid(back, f, focus.DEFAULTS)
    assert "no entry within 6" in focus.still_valid(df, {**f, "bars_waited": 6}, focus.DEFAULTS)
    quiet = df.copy()
    quiet.iloc[-3:, quiet.columns.get_loc("volume")] = 100.0
    assert "volume faded" in focus.still_valid(quiet, f, focus.DEFAULTS)


def test_breakout_stop_sits_just_past_the_level():
    d = focus.breakout_stop(101.5, {"direction": "BUY", "level": 101.0}, focus.DEFAULTS)
    assert abs(d - (101.5 - 101.0 * 0.9995)) < 1e-9
    assert focus.breakout_stop(100.5, {"direction": "BUY", "level": 101.0}, focus.DEFAULTS) is None
    s = focus.breakout_stop(98.5, {"direction": "SELL", "level": 99.0}, focus.DEFAULTS)
    assert abs(s - (99.0 * 1.0005 - 98.5)) < 1e-9


def test_switch_defaults_off_and_round_trips(tmp_path):
    p = tmp_path / "f.json"
    assert focus.read_switch(p)["enabled"] is False
    focus.write_switch(True, p)
    assert focus.read_switch(p)["enabled"] is True
    p.write_text("not json")
    assert focus.read_switch(p)["enabled"] is False


def test_settings_merge_and_clamp():
    s = focus.settings({"screener": {"focus": {"min_volume_ratio": 2.0, "sides": "SHORT", "bogus": 1}}})
    assert s["min_volume_ratio"] == 2.0 and s["sides"] == "long" and "bogus" not in s
    assert focus.settings({})["prime_minutes"] == 120


# ---------------------------------------------------------------- the scan loop

def make_orch(tmp_path, frames):
    o = Orchestrator.__new__(Orchestrator)
    o.config = {"timeframe": "MIN_5", "polling_interval_seconds": 15}
    o.audit_log = AuditLog(str(tmp_path / "audit.db"))
    o.kronos_agent = MagicMock()
    o.broker = SimpleNamespace(get_historical_bars=lambda sym, tf, n: frames[sym])
    o.workers = {}
    for sym in frames:
        w = MagicMock()
        w.symbol, w.name, w.monitor_only, w.trade_outside_hours = sym, sym, False, True
        w.timeframe, w.lookback_bars, w.forecast_horizon = "MIN_5", 120, 3
        w.bar_duration = pd.Timedelta(minutes=5)
        w.kronos_opts = None
        w._session_minutes.return_value = (30, 300)
        w.prepare_signal.return_value = None
        o.workers[sym] = w
    return o


def frames_for(**dfs):
    return {k: v.reset_index().rename(columns={"index": "time"}) for k, v in dfs.items()}


def test_focus_off_scans_every_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr("src.orchestrator.bars_to_df", lambda b: b.set_index("time"))
    monkeypatch.setattr("src.orchestrator.drop_unclosed_bar", lambda df, tf: df)
    o = make_orch(tmp_path, frames_for(A=bars(), B=bars(breakout=False)))
    o.poll_workers()
    assert all(w.prepare_signal.called for w in o.workers.values())


def test_focus_on_scans_only_the_breakout_and_its_stop_is_the_level(tmp_path, monkeypatch):
    monkeypatch.setattr("src.orchestrator.bars_to_df", lambda b: b.set_index("time"))
    monkeypatch.setattr("src.orchestrator.drop_unclosed_bar", lambda df, tf: df)
    focus.write_switch(True)
    o = make_orch(tmp_path, frames_for(A=bars(breakout=False), B=bars(), C=bars(spike=1.1)))
    b = o.workers["B"]
    b.prepare_signal.side_effect = lambda extra=None: {"df": bars(), "signal": extra, "last_close": 100.91}
    o.kronos_agent.confirm_direction.return_value = (True, 0.9)
    b.kronos_agent = o.kronos_agent
    o.poll_workers()
    o.workers["A"].prepare_signal.assert_not_called()
    o.workers["C"].prepare_signal.assert_not_called()
    sig = b.prepare_signal.call_args.args[0]
    assert isinstance(sig, PatternSignal) and sig.name == "volume_breakout" and sig.direction == "BUY"
    ctx = b.finish_signal.call_args.args[0]
    stop = ctx["plan_fixed"]["stop_distance"]
    assert abs(stop - (100.91 - 100.7 * 0.9995)) < 1e-9
    assert abs(ctx["plan_fixed"]["target_distance"] - 2 * stop) < 1e-9
    state = o.audit_log.get_kv(focus.STATE_KEY)
    assert state["focus"]["symbol"] == "B" and state["candidates"][0]["symbol"] == "B"
    # next candle: B stays the focus and is scanned without a new breakout signal
    b.prepare_signal.reset_mock()
    b.prepare_signal.side_effect = None
    b.prepare_signal.return_value = None
    o.poll_workers()
    assert b.prepare_signal.call_args.args == ()
    o.workers["A"].prepare_signal.assert_not_called()


def test_nothing_qualifies_nothing_is_scanned(tmp_path, monkeypatch):
    monkeypatch.setattr("src.orchestrator.bars_to_df", lambda b: b.set_index("time"))
    monkeypatch.setattr("src.orchestrator.drop_unclosed_bar", lambda df, tf: df)
    focus.write_switch(True)
    o = make_orch(tmp_path, frames_for(A=bars(breakout=False), B=bars(spike=1.0)))
    o.poll_workers()
    assert not any(w.prepare_signal.called for w in o.workers.values())
    st = o.audit_log.get_kv(focus.STATE_KEY)
    assert st["focus"] is None and len(st["candidates"]) == 2


def test_an_open_trade_owns_the_focus(tmp_path, monkeypatch):
    monkeypatch.setattr("src.orchestrator.bars_to_df", lambda b: b.set_index("time"))
    monkeypatch.setattr("src.orchestrator.drop_unclosed_bar", lambda df, tf: df)
    focus.write_switch(True)
    o = make_orch(tmp_path, frames_for(A=bars(), B=bars()))
    o.audit_log.get_active_positions = lambda: [{"symbol": "A"}]
    o.poll_workers()
    o.workers["A"].prepare_signal.assert_called_once_with()
    o.workers["B"].prepare_signal.assert_not_called()
    assert o.audit_log.get_kv(focus.STATE_KEY)["focus"]["in_trade"] is True


def test_off_again_clears_the_published_state(tmp_path, monkeypatch):
    monkeypatch.setattr("src.orchestrator.bars_to_df", lambda b: b.set_index("time"))
    monkeypatch.setattr("src.orchestrator.drop_unclosed_bar", lambda df, tf: df)
    focus.write_switch(True)
    o = make_orch(tmp_path, frames_for(A=bars()))
    o.poll_workers()
    focus.write_switch(False)
    o.poll_workers()
    assert o.audit_log.get_kv(focus.STATE_KEY)["enabled"] is False


def test_live_switch_is_not_saved_to_config():
    from dashboard import schema
    import pytest
    with pytest.raises(ValueError, match="switched live"):
        schema.coerce("focus_mode", True)
    assert schema.coerce("screener.focus.min_volume_ratio", "2") == 2.0
