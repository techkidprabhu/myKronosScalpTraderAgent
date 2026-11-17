"""Per-stock daily limits, percent trailing stop, and ATR trailing that now receives an ATR."""
import json

import pytest

from src.risk import RiskConfig, RiskManager
from src.trailing_stop import TrailConfig, compute_update


class Store:
    def __init__(self):
        self.kv = {}

    def get_kv(self, k, default=None):
        return self.kv.get(k, default)

    def set_kv(self, k, v):
        self.kv[k] = json.loads(json.dumps(v))


def rm(**kw):
    base = dict(max_open_positions=10, max_stake_per_point=1000, stop_loss_points=1.0, take_profit_points=2.0,
                max_daily_loss=5000.0, min_minutes_between_trades=0)
    base.update(kw)
    return RiskManager(RiskConfig(**base), store=Store(), timezone="America/New_York")


def test_per_stock_loss_amount_blocks_only_that_stock():
    r = rm(max_symbol_loss_per_day=150)
    r.record_trade_result(-100, "US.INTC")
    assert r.can_trade("US.INTC", None, 0)[0]
    r.record_trade_result(-60, "INTC")                   # same stock, written without the market prefix
    ok, why = r.can_trade("US.INTC", None, 0)
    assert not ok and "INTC lost 160.00 today (per-stock limit 150.00)" in why
    assert r.can_trade("US.NVDA", None, 0)[0]            # other stocks keep trading
    r.record_trade_result(+50, "US.INTC")                # a later win nets it back above the limit
    assert r.can_trade("US.INTC", None, 0)[0]


def test_per_stock_losing_trades_count():
    r = rm(max_losses_per_symbol_per_day=2)
    r.record_trade_result(-10, "US.AMD")
    r.record_trade_result(+30, "US.AMD")                 # wins don't count
    assert r.can_trade("US.AMD", None, 0)[0]
    r.record_trade_result(-5, "US.AMD")
    ok, why = r.can_trade("US.AMD", None, 0)
    assert not ok and "2 losing trades today (per-stock limit 2)" in why


def test_limits_are_off_by_default_and_reset_each_day():
    r = rm()
    for _ in range(5):
        r.record_trade_result(-500, "US.TSLA")
    assert r.can_trade("US.TSLA", None, 0)[0]            # nothing set: no per-stock limit
    r2 = rm(max_losses_per_symbol_per_day=1)
    r2.record_trade_result(-1, "US.TSLA")
    assert not r2.can_trade("US.TSLA", None, 0)[0]
    r2.state.trading_day = "2000-01-01"                  # a new exchange day
    assert r2.can_trade("US.TSLA", None, 0)[0] and r2.state.symbol_losses == {}


def test_state_survives_a_restart():
    store = Store()
    cfg = RiskConfig(max_open_positions=10, max_stake_per_point=1000, stop_loss_points=1.0, take_profit_points=2.0,
                     max_daily_loss=5000.0, min_minutes_between_trades=0, max_symbol_loss_per_day=100)
    a = RiskManager(cfg, store=store, timezone="America/New_York")
    a.record_trade_result(-120, "US.INTC")
    b = RiskManager(cfg, store=store, timezone="America/New_York")
    assert not b.can_trade("US.INTC", None, 0)[0]


# ---------------------------------------------------------------- trailing

def test_percent_trailing_behaves_the_same_on_cheap_and_expensive_stocks():
    cfg = TrailConfig(trigger_points=2.0, trail_distance=3.0, trigger_pct=0.5, trail_pct=0.4)
    for price in (60.0, 500.0):
        # +0.4%: not yet
        u = compute_update("BUY", price, price * 1.004, price * 0.995, price, False, price * 1.01, cfg)
        assert not u.trail_active
        # +0.6%: trailing on, stop 0.4% under the best price - above the original 0.5% stop
        u = compute_update("BUY", price, price * 1.006, price * 0.995, price, False, price * 1.01, cfg)
        assert u.trail_active and u.stop_changed
        assert u.new_stop == pytest.approx(price * 1.006 * 0.996)
    # with the dollar settings, a $60 stock never reaches a $2 trigger before its 1% target
    old = TrailConfig(trigger_points=2.0, trail_distance=3.0)
    u = compute_update("BUY", 60.0, 60.59, 59.7, 60.0, False, 60.6, old)
    assert not u.trail_active


def test_percent_trailing_for_shorts():
    cfg = TrailConfig(trigger_points=0, trail_distance=0, trigger_pct=0.5, trail_pct=0.4)
    u = compute_update("SELL", 100.0, 99.3, 100.5, 100.0, False, 99.0, cfg)
    assert u.trail_active and u.new_stop == pytest.approx(99.3 * 1.004)


def test_atr_distance_wins_when_an_atr_is_given():
    cfg = TrailConfig(trigger_points=0.5, trail_distance=3.0, use_atr=True, atr_multiplier=2.0, trail_pct=0.4)
    u = compute_update("BUY", 100.0, 101.0, 99.0, 100.0, False, 110.0, cfg, atr=0.2)
    assert u.new_stop == pytest.approx(101.0 - 0.4)
    u = compute_update("BUY", 100.0, 101.0, 99.0, 100.0, False, 110.0, cfg, atr=None)   # no ATR yet: the % distance
    assert u.new_stop == pytest.approx(101.0 * 0.996)


def test_agent_passes_atr_and_counts_per_stock_losses(tmp_path, monkeypatch):
    """The worker now hands the candles' ATR to the trailing stop (use_atr
    did nothing before), and every closed trade is counted per stock."""
    import src.orchestrator as orch
    from tests.test_orchestrator import FakeBroker, engulfing_bars, make_worker, SYMBOL
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, risk = make_worker(tmp_path, broker, trailing_stop={"enabled": True, "use_atr": True,
                                      "atr_multiplier": 2.0, "trigger_pct": 0.1, "trail_pct": 0.3})
    assert worker.trail_config.trigger_pct == 0.1 and worker.trail_config.trail_pct == 0.3
    worker.poll_once()
    assert worker._trail_atr and worker._trail_atr > 0
    seen = []
    real = orch.compute_update

    def spy(*a, **k):
        seen.append(k.get("atr"))
        return real(*a, **k)
    monkeypatch.setattr(orch, "compute_update", spy)
    entry_id = int(audit.get_active_position(SYMBOL)["entry_order_id"])
    broker.fill(entry_id, price=71.7)
    broker.held = 100
    worker.monitor_position()           # places the stop and target
    worker.monitor_position()           # then trails
    assert seen and seen[-1] == worker._trail_atr
    risk.record_trade_result(-25.0, SYMBOL)
    assert list(risk.state.symbol_losses.values()) == [1]
