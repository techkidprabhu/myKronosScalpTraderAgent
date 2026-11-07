"""Day-trader risk rules (from the Day Trader skill pack), added switched off.

Every rule is checked twice: off -> the agent behaves exactly as before,
on -> the rule does what the config says."""
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import src.orchestrator as orch
from src.risk import open_risk_at_stops
from src import market_calendar as cal
from src.risk import RiskConfig, RiskManager, RiskState
from tests.test_orchestrator import (SYMBOL, FakeBroker, engulfing_bars, make_worker,  # noqa: F401
                                     market_open, open_position)


def rm(**kw):
    base = dict(max_open_positions=5, max_stake_per_point=1000, stop_loss_points=1.0, take_profit_points=2.0,
                max_daily_loss=500.0, min_minutes_between_trades=0)
    base.update(kw)
    return RiskManager(RiskConfig(**base))


# ---- off by default ------------------------------------------------------

def test_defaults_change_nothing():
    r = rm(stop_loss_pct=0.5, take_profit_pct=1.0)
    plan = r.plan_entry(100.0, 10)
    assert plan.allowed and plan.quantity == 10 and plan.sized_by == "configured quantity"
    assert (plan.stop_distance, plan.target_distance) == r.bracket_distances(100.0) == (0.5, 1.0)
    assert r.can_trade("US.AAPL", 10, 0, session_minutes=(1, 1)) == (True, "ok")
    assert not r.config.plans_distances


def test_old_saved_state_without_new_fields_loads():
    class Store:
        def __init__(self): self.kv = {"risk_state": {"daily_pnl": -5.0, "last_trade_time": {}, "trading_halted": False,
                                                       "trading_day": "2000-01-01", "obsolete": 1}}
        def get_kv(self, k): return self.kv.get(k)
        def set_kv(self, k, v): self.kv[k] = v
    r = RiskManager(RiskConfig(1, 10, 1, 2, 100, 0), store=Store())
    assert r.state.trades_today == 0


# ---- stops and targets ---------------------------------------------------

def test_atr_stop_and_r_multiple_target():
    r = rm(stop_loss_pct=0.5, stop_atr_multiple=1.5, target_r_multiple=2.0)
    assert r.bracket_distances(100.0, atr=0.8) == pytest.approx((1.2, 2.4))
    assert r.bracket_distances(100.0) == pytest.approx((0.5, 1.0))   # no ATR -> % stop, still 2R
    assert r.config.plans_distances


def test_atr_stop_without_atr_is_refused():
    plan = rm(stop_atr_multiple=1.5).plan_entry(100.0, 10, atr=None)
    assert not plan.allowed and "ATR" in plan.reason


def test_min_stop_and_min_reward_risk():
    plan = rm(stop_loss_pct=0.02, take_profit_pct=1.0, min_stop_pct=0.05).plan_entry(100.0, 10)
    assert not plan.allowed and "too tight" in plan.reason
    plan = rm(stop_loss_points=1.0, take_profit_points=0.8, min_reward_risk=1.0).plan_entry(100.0, 10)
    assert not plan.allowed and "reward:risk 0.80" in plan.reason
    assert rm(stop_loss_points=1.0, take_profit_points=1.0, min_reward_risk=1.0).plan_entry(100.0, 10).allowed


# ---- sizing ----------------------------------------------------------------

def test_risk_based_sizing_and_caps():
    plan = rm(max_loss_per_trade=50).plan_entry(100.0, 10)           # 50 / 1.0
    assert plan.quantity == 50 and plan.sized_by == "max loss per trade" and plan.max_loss == 50
    plan = rm(max_loss_per_trade=50, max_stake_per_point=30).plan_entry(100.0, 10)
    assert plan.quantity == 30 and plan.sized_by == "max quantity per position"
    plan = rm(max_loss_per_trade=250, lot_size=100).plan_entry(100.0, 10)
    assert plan.quantity == 200                                       # whole lots, rounded down
    plan = rm(max_loss_per_trade=0.5).plan_entry(100.0, 10)
    assert not plan.allowed and "too small" in plan.reason


def test_account_caps():
    acct = {"total_assets": 10_000.0, "cash": 3_000.0}
    plan = rm(max_loss_per_trade=500, max_position_pct=20).plan_entry(100.0, 10, account=acct)
    assert plan.quantity == 20 and plan.sized_by == "max position % of account"
    plan = rm(max_loss_per_trade=500, cash_buffer_pct=10).plan_entry(100.0, 10, account=acct)
    assert plan.quantity == 20 and plan.sized_by == "cash after buffer"    # (3000 - 1000) / 100
    plan = rm(max_position_pct=20).plan_entry(100.0, 10, account=None)
    assert not plan.allowed and "account value unavailable" in plan.reason


# ---- day limits -------------------------------------------------------------

def test_daily_cap_counts_open_risk():
    r = rm(max_loss_per_trade=60, max_daily_loss=200, daily_loss_includes_open_risk=True)
    r.state.daily_pnl = -100.0
    r.state.trading_day = datetime.now(r.tz).date().isoformat()
    assert not r.plan_entry(100.0, 10, open_risk=50).allowed           # -100 - 50 - 60 = -210
    assert r.plan_entry(100.0, 10, open_risk=30).allowed                # -190
    off = rm(max_loss_per_trade=60, max_daily_loss=200)
    off.state.daily_pnl = -100.0
    off.state.trading_day = r.state.trading_day
    assert off.plan_entry(100.0, 10, open_risk=500).allowed             # rule off: realised P&L only


def test_trades_per_day_and_reset():
    r = rm(max_trades_per_day=2)
    r.mark_trade_opened("US.A"); r.mark_trade_opened("US.B")
    ok, why = r.can_trade("US.C", 10, 0)
    assert not ok and "max_trades_per_day" in why
    r.state.trading_day = "2000-01-01"                                  # next exchange day
    assert r.can_trade("US.C", 10, 0)[0] and r.state.trades_today == 0


def test_blocked_symbols_match_with_or_without_prefix():
    r = rm(blocked_symbols=["TSLA", "US.GME"])
    assert not r.can_trade("US.TSLA", 10, 0)[0]
    assert not r.can_trade("GME", 10, 0)[0]
    assert r.can_trade("US.AAPL", 10, 0)[0]


def test_entry_window():
    r = rm(no_entries_first_minutes=15, no_entries_last_minutes=30)
    assert "first 15 min" in r.can_trade("A", 10, 0, session_minutes=(10, 300))[1]
    assert "last 30 min" in r.can_trade("A", 10, 0, session_minutes=(300, 20))[1]
    assert r.can_trade("A", 10, 0, session_minutes=(60, 200))[0]
    assert r.can_trade("A", 10, 0, session_minutes=None)[0]             # clock unknown: don't block
    flat = rm(flat_by_minutes_before_close=5, no_entries_last_minutes=None)
    assert "last 5 min" in flat.can_trade("A", 10, 0, session_minutes=(300, 4))[1]


def test_sized_stake_skips_fixed_quantity_checks():
    r = rm(lot_size=100, max_loss_per_trade=50)
    assert r.can_trade("A", None, 0) == (True, "ok")


def test_session_minutes_early_close_and_weekend():
    ny = ZoneInfo("America/New_York")
    assert cal.session_minutes("NASDAQ", datetime(2026, 11, 9, 9, 40, tzinfo=ny)) == (10.0, 380.0)
    assert cal.session_minutes("NASDAQ", datetime(2026, 11, 27, 12, 50, tzinfo=ny)) == (200.0, 10.0)  # 1 pm close
    assert cal.session_minutes("NASDAQ", datetime(2026, 11, 27, 13, 10, tzinfo=ny)) is None
    assert cal.session_minutes("NASDAQ", datetime(2026, 11, 8, 11, 0, tzinfo=ny)) is None             # Sunday


def test_open_risk_at_stops():
    positions = [
        {"direction": "BUY", "entry_price": 100, "current_stop": 99, "quantity": 10},    # 10
        {"direction": "SELL", "entry_price": 50, "current_stop": 51, "quantity": 20},    # 20
        {"direction": "BUY", "entry_price": 100, "current_stop": 101, "quantity": 10},   # profit locked: 0
    ]
    assert open_risk_at_stops(positions) == pytest.approx(30.0)


# ---- the agent's order path ---------------------------------------------------

def _entry_qty(broker):
    return [o["quantity"] for o in broker.orders.values()][0]


def test_agent_sizes_entry_from_max_loss(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, risk = make_worker(tmp_path, broker)
    risk.config.max_loss_per_trade = 120          # stop 0.5 -> 240 shares -> 200 (lot 100)
    worker.poll_once()
    assert _entry_qty(broker) == 200
    pos = audit.get_active_position(SYMBOL)
    assert pos["quantity"] == 200
    assert risk.state.trades_today == 1


def test_agent_unchanged_with_rules_off(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, risk = make_worker(tmp_path, broker)
    worker.poll_once()
    assert _entry_qty(broker) == 100                # the configured quantity
    pos = audit.get_active_position(SYMBOL)
    assert pos["initial_stop"] == pytest.approx(71.2) and pos["target"] == pytest.approx(72.7)


def test_agent_rejects_on_reward_risk_and_logs_it(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, risk = make_worker(tmp_path, broker)
    risk.config.min_reward_risk = 3.0               # 1.0 / 0.5 = 2R
    worker.poll_once()
    assert broker.orders == {} and audit.get_active_position(SYMBOL) is None
    row = audit.conn.execute("SELECT action, risk_reason FROM decisions ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] == "rejected" and "reward:risk 2.00" in row[1]


def test_agent_entry_window_blocks(tmp_path, market_open, monkeypatch):  # noqa: F811
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, risk = make_worker(tmp_path, broker)
    risk.config.no_entries_first_minutes = 15
    monkeypatch.setattr(cal, "session_minutes", lambda code, now=None: (5.0, 400.0))
    worker.poll_once()
    assert broker.orders == {}


def test_atr_bracket_survives_the_fill(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, risk = make_worker(tmp_path, broker)
    risk.config.stop_atr_multiple = 2.0
    risk.config.target_r_multiple = 2.0
    worker.poll_once()
    pos = audit.get_active_position(SYMBOL)
    stop_d = pos["entry_price"] - pos["initial_stop"]
    assert stop_d > 0 and (pos["target"] - pos["entry_price"]) == pytest.approx(2 * stop_d, abs=0.011)
    broker.fill(int(pos["entry_order_id"]), price=71.9)           # fills above the signal close
    broker.held = 100
    worker.monitor_position()
    pos = audit.get_active_position(SYMBOL)
    assert pos["status"] == "open"
    assert 71.9 - pos["initial_stop"] == pytest.approx(stop_d, abs=0.011)    # same distance, from the fill
    assert pos["target"] - 71.9 == pytest.approx(2 * stop_d, abs=0.011)


def test_flat_by_close_exits_agent_positions(tmp_path, market_open, monkeypatch):  # noqa: F811
    broker = FakeBroker()
    worker, audit, risk, _, _ = open_position(tmp_path, broker)
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    o.risk_manager, o.broker, o.audit_log, o.workers = risk, broker, audit, {SYMBOL: worker}
    monkeypatch.setattr(cal, "session_minutes", lambda code, now=None: (400.0, 30.0))
    o.flat_by_close()                                            # rule off
    assert audit.get_active_position(SYMBOL) is not None
    risk.config.flat_by_minutes_before_close = 10
    o.flat_by_close()                                            # 30 min left: not yet
    assert audit.get_active_position(SYMBOL) is not None
    monkeypatch.setattr(cal, "session_minutes", lambda code, now=None: (420.0, 8.0))
    o.flat_by_close()
    assert audit.get_active_position(SYMBOL) is None and broker.held == 0
    reason = audit.conn.execute("SELECT details FROM trade_results ORDER BY rowid DESC LIMIT 1").fetchone()[0]
    assert "flat_by_close" in reason


def test_flat_by_close_cancels_unfilled_entry(tmp_path, market_open, monkeypatch):  # noqa: F811
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, risk = make_worker(tmp_path, broker)
    worker.poll_once()
    entry_id = audit.get_active_position(SYMBOL)["entry_order_id"]
    risk.config.flat_by_minutes_before_close = 10
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    o.risk_manager, o.broker, o.audit_log, o.workers = risk, broker, audit, {SYMBOL: worker}
    monkeypatch.setattr(cal, "session_minutes", lambda code, now=None: (420.0, 8.0))
    o.flat_by_close()
    assert int(entry_id) in [int(c) for c in broker.cancelled]


def test_cross_checks():
    from dashboard.schema import cross_check
    base = {"timeframe": "MIN_5", "lookback_bars": 100, "risk": {"max_daily_loss": 100}}
    assert cross_check(base) == []
    assert any("Flat-by-close" in e for e in cross_check({**base, "timeframe": "DAY",
                                                          "risk": {"flat_by_minutes_before_close": 5}}))
    assert any("one stop-out" in e for e in cross_check({**base, "risk": {"max_daily_loss": 100, "max_loss_per_trade": 150}}))
    assert any("ATR period" in e for e in cross_check({**base, "lookback_bars": 10,
                                                       "risk": {"stop_atr_multiple": 1.5, "atr_period": 14}}))
    assert any("no time to trade" in e for e in cross_check({**base, "risk": {"no_entries_first_minutes": 200,
                                                                               "no_entries_last_minutes": 200}}))


def test_risk_state_reports_trades_and_open_risk(tmp_path, monkeypatch):
    import dashboard.app as dash
    from src.audit_log import AuditLog
    from tests.test_dashboard_multibroker import AUTH  # noqa: F401
    monkeypatch.setattr(dash, "DB_PATH", tmp_path / "a.db")
    monkeypatch.setattr(dash, "_db_path", lambda: tmp_path / "a.db", raising=False)
    audit = AuditLog(str(tmp_path / "a.db"))
    tz = dash.market_timezone(dash._spec_market(dash.current_spec()))
    audit.set_kv("risk_state", {"daily_pnl": 0.0, "trading_halted": False, "trades_today": 2,
                                "trading_day": datetime.now(tz).date().isoformat(), "last_trade_time": {}})
    audit.record_pending_entry("US.AAPL", "BUY", 10, 100.0, 99.0, 102.0, entry_order_id="1")
    audit.close()
    with dash.app.test_request_context("/api/risk_state"):
        d = dash.get_risk_state().get_json()
    assert d["trades_today"] == 2 and d["open_risk"] == 10.0
