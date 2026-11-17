"""Daily profit target: no new entries once reached; everything else carries on."""
import json

import pytest

from src.risk import RiskConfig, RiskManager


class Store:
    def __init__(self):
        self.kv = {}

    def get_kv(self, k, default=None):
        return self.kv.get(k, default)

    def set_kv(self, k, v):
        self.kv[k] = json.loads(json.dumps(v))


def rm(store=None, **kw):
    base = dict(max_open_positions=10, max_stake_per_point=1000, stop_loss_points=1.0, take_profit_points=2.0,
                max_daily_loss=5000.0, min_minutes_between_trades=0)
    base.update(kw)
    return RiskManager(RiskConfig(**base), store=store or Store(), timezone="America/New_York")


def test_amount_target_stops_new_entries_only():
    r = rm(daily_profit_target=300)
    r.record_trade_result(+200, "US.AMD")
    assert r.can_trade("US.NVDA", None, 0)[0]
    r.record_trade_result(+150, "US.NVDA")
    ok, why = r.can_trade("US.TSLA", None, 0)
    assert not ok and why.startswith("daily profit target reached (today +350.00, target 300.00)")
    assert "analysis continues" in why and r.state.profit_locked
    r.record_trade_result(-200, "US.AMD")           # an open trade stopping out later keeps the day locked
    assert r.state.profit_locked and not r.can_trade("US.TSLA", None, 0)[0]


def test_percent_target_needs_the_start_of_day_value():
    r = rm(daily_profit_target_pct=1.0)
    r.record_trade_result(+500, "US.AMD")
    assert r.wants_start_equity() and r.can_trade("US.NVDA", None, 0)[0]   # not known yet: no target applied
    r.set_start_equity(40_500)                       # value now includes today's +500
    assert r.state.start_equity == 40_000 and not r.wants_start_equity()
    assert r.profit_target() == pytest.approx(400)
    assert not r.can_trade("US.NVDA", None, 0)[0]
    r.set_start_equity(99_999)                       # recorded once per day
    assert r.state.start_equity == 40_000


def test_lower_of_amount_and_percent_applies_and_raising_it_resumes():
    store = Store()
    r = rm(store, daily_profit_target=1000, daily_profit_target_pct=1.0)
    r.set_start_equity(50_000)
    assert r.profit_target() == 500
    r.record_trade_result(+600, "US.AMD")
    assert not r.can_trade("US.NVDA", None, 0)[0]
    # target raised in the config, agent restarted: entries resume
    r2 = rm(store, daily_profit_target=1000)
    assert r2.can_trade("US.NVDA", None, 0)[0] and not r2.state.profit_locked
    # switched off entirely: also resumes
    r3 = rm(store, daily_profit_target=100)
    assert not r3.can_trade("US.NVDA", None, 0)[0]
    assert rm(store).can_trade("US.NVDA", None, 0)[0]


def test_off_by_default_and_new_day_resets():
    r = rm()
    r.record_trade_result(+10_000, "US.AMD")
    assert r.can_trade("US.NVDA", None, 0)[0] and r.profit_target() is None
    r2 = rm(daily_profit_target=100)
    r2.record_trade_result(+150, "US.AMD")
    assert not r2.can_trade("US.NVDA", None, 0)[0]
    r2.state.trading_day = "2000-01-01"
    assert r2.can_trade("US.NVDA", None, 0)[0] and r2.state.start_equity is None


def test_kronos_still_runs_after_the_target(tmp_path, monkeypatch):
    """The pipeline still evaluates and logs the signal; only the order is refused."""
    import src.orchestrator as orch
    from tests.test_orchestrator import FakeBroker, engulfing_bars, make_worker
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, risk = make_worker(tmp_path, broker)
    risk.config.daily_profit_target = 100
    risk.record_trade_result(+250, "US.AMD")
    worker.poll_once()
    assert worker.kronos_agent.confirm_direction.called          # Kronos evaluated as usual
    assert broker.kinds() == []                                  # no order
    cur = audit.conn.execute("SELECT kronos_confidence, risk_allowed, risk_reason FROM decisions ORDER BY id DESC")
    conf, allowed, reason = cur.fetchone()
    assert conf == 0.9 and allowed == 0 and reason.startswith("daily profit target reached")


def test_dashboard_reports_the_target(client, monkeypatch):  # noqa: F811
    import dashboard.app as app_mod
    view = app_mod._profit_target_view({"start_equity": 40000, "profit_locked": True, "daily_pnl": 450},
                                       {"daily_profit_target_pct": 1.0}, stale=False)
    assert view["profit_target"] == 400 and view["profit_locked"] and view["closed_pnl_today"] == 450
    assert app_mod._profit_target_view({"profit_locked": True}, {"daily_profit_target": 50}, stale=True)["profit_locked"] is False
    d = client.get("/api/risk_state", headers=AUTH).get_json()
    assert "profit_target" in d and "profit_locked" in d


from tests.test_dashboard_multibroker import AUTH, client  # noqa: E402,F401
