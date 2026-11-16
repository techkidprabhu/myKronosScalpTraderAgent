import time
import pytest

from src.risk import RiskConfig, RiskManager


@pytest.fixture
def risk_manager():
    config = RiskConfig(
        max_open_positions=3,
        max_stake_per_point=100.0,
        stop_loss_points=0.15,
        take_profit_points=0.30,
        max_daily_loss=200.0,
        min_minutes_between_trades=10,
    )
    return RiskManager(config)


def test_can_trade_default(risk_manager):
    allowed, reason = risk_manager.can_trade("D05.SI", 50.0, 0)
    assert allowed is True
    assert reason == "ok"


def test_max_open_positions(risk_manager):
    allowed, reason = risk_manager.can_trade("D05.SI", 50.0, 3)
    assert allowed is False
    assert "max open positions" in reason


def test_stake_exceeded(risk_manager):
    allowed, reason = risk_manager.can_trade("D05.SI", 150.0, 0)
    assert allowed is False
    assert "stake exceeds" in reason


def test_cooldown_enforced(risk_manager):
    risk_manager.mark_trade_opened("D05.SI")
    allowed, reason = risk_manager.can_trade("D05.SI", 50.0, 1)
    assert allowed is False
    assert "min_minutes_between_trades" in reason

    # Different symbol is not on cooldown
    allowed2, reason2 = risk_manager.can_trade("Z74.SI", 50.0, 1)
    assert allowed2 is True


def test_daily_loss_halt(risk_manager):
    risk_manager.record_trade_result(-250.0)
    assert risk_manager.state.trading_halted is True

    allowed, reason = risk_manager.can_trade("D05.SI", 50.0, 0)
    assert allowed is False
    assert "daily loss cap breached" in reason

    # Reset
    risk_manager.reset_daily()
    assert risk_manager.state.trading_halted is False
    allowed, reason = risk_manager.can_trade("D05.SI", 50.0, 0)
    assert allowed is True


def _config(**overrides):
    params = dict(
        max_open_positions=3,
        max_stake_per_point=1000.0,
        stop_loss_points=0.15,
        take_profit_points=0.30,
        max_daily_loss=200.0,
        min_minutes_between_trades=240,
        lot_size=100,
    )
    params.update(overrides)
    return RiskConfig(**params)


def test_state_survives_restart(tmp_path):
    from src.audit_log import AuditLog

    store = AuditLog(str(tmp_path / "audit.db"))
    rm = RiskManager(_config(), store=store)
    rm.mark_trade_opened("D05.SI")
    rm.record_trade_result(-250.0)

    restarted = RiskManager(_config(), store=AuditLog(str(tmp_path / "audit.db")))
    assert restarted.state.trading_halted is True
    assert restarted.state.daily_pnl == -250.0
    assert "D05.SI" in restarted.state.last_trade_time


def test_cooldown_survives_restart(tmp_path):
    from src.audit_log import AuditLog

    RiskManager(_config(), store=AuditLog(str(tmp_path / "audit.db"))).mark_trade_opened("D05.SI")
    restarted = RiskManager(_config(), store=AuditLog(str(tmp_path / "audit.db")))
    allowed, reason = restarted.can_trade("D05.SI", 100, 0)
    assert allowed is False
    assert "min_minutes_between_trades" in reason


def test_new_trading_day_resets_halt(tmp_path):
    from src.audit_log import AuditLog

    store = AuditLog(str(tmp_path / "audit.db"))
    store.set_kv("risk_state", {"daily_pnl": -500.0, "last_trade_time": {}, "trading_halted": True,
                                "trading_day": "2000-01-01"})
    rm = RiskManager(_config(), store=store)
    assert rm.state.trading_halted is False
    assert rm.state.daily_pnl == 0.0


def test_short_rejected_by_default():
    allowed, reason = RiskManager(_config()).can_trade("Z74.SI", 100, 0, direction="SELL")
    assert allowed is False
    assert "short" in reason


def test_short_allowed_when_enabled():
    allowed, _ = RiskManager(_config(allow_short=True)).can_trade("Z74.SI", 100, 0, direction="SELL")
    assert allowed is True


def test_existing_position_blocks_entry():
    allowed, reason = RiskManager(_config()).can_trade("D05.SI", 100, 1, has_position=True)
    assert allowed is False
    assert "already have a position" in reason


def test_odd_lot_rejected():
    allowed, reason = RiskManager(_config()).can_trade("D05.SI", 10, 0)
    assert allowed is False
    assert "lot_size" in reason


def test_percent_brackets_override_points():
    from src.risk import RiskConfig, RiskManager
    cfg = RiskConfig(max_open_positions=5, max_stake_per_point=200, stop_loss_points=4.0,
                     take_profit_points=8.0, max_daily_loss=600, min_minutes_between_trades=15,
                     stop_loss_pct=0.5, take_profit_pct=1.0)
    rm = RiskManager(cfg)
    stop, target = rm.bracket_distances(400.0)
    assert round(stop, 6) == 2.0 and round(target, 6) == 4.0
    assert rm.bracket_distances() == (4.0, 8.0)  # no price -> fixed points


def test_points_used_when_no_percent():
    from src.risk import RiskConfig, RiskManager
    cfg = RiskConfig(max_open_positions=3, max_stake_per_point=200, stop_loss_points=4.0,
                     take_profit_points=8.0, max_daily_loss=300, min_minutes_between_trades=60)
    assert RiskManager(cfg).bracket_distances(400.0) == (4.0, 8.0)


def test_raising_the_cap_lifts_the_halt_but_a_restart_alone_does_not():
    from src.risk import RiskConfig, RiskManager

    class Store:
        def __init__(self): self.kv = {}
        def get_kv(self, k): return self.kv.get(k)
        def set_kv(self, k, v): self.kv[k] = v

    def cfg(cap):
        return RiskConfig(max_open_positions=3, max_stake_per_point=100.0, stop_loss_points=1.0,
                          take_profit_points=2.0, max_daily_loss=cap, min_minutes_between_trades=0)

    store = Store()
    rm = RiskManager(cfg(600), store=store, timezone="America/New_York")
    rm.record_trade_result(-631.0)
    assert rm.state.trading_halted
    again = RiskManager(cfg(600), store=store, timezone="America/New_York")       # restart, same cap
    assert again.state.trading_halted
    assert "cap -600.00" in again.can_trade("US.AAPL", None, 0)[1]
    raised = RiskManager(cfg(1000), store=store, timezone="America/New_York")     # cap raised, restart
    assert not raised.state.trading_halted and raised.state.daily_pnl == -631.0   # the loss still counts
    assert raised.can_trade("US.AAPL", None, 0)[0]
    raised.record_trade_result(-400.0)                                             # -1031: over the new cap
    assert raised.state.trading_halted
    lowered = RiskManager(cfg(100), store=Store(), timezone="America/New_York")
    lowered.state.daily_pnl = -150.0
    lowered._recheck_halt()
    assert lowered.state.trading_halted                                            # a lower cap applies at once
