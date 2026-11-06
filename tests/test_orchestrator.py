"""
End-to-end checks of the entry/exit lifecycle against a fake broker,
covering the failure modes found in the live audit log: duplicate
entries after restart, exit orders resting before the entry filled,
double-selling on exit, shorts, and acting on a still-forming candle.
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest

import src.orchestrator as orch
from src.audit_log import AuditLog
from src.orchestrator import SymbolWorker
from src.risk import RiskConfig, RiskManager
from src.moomoo_client import MoomooClient

SYMBOL = "D05.SI"


class FakeBroker:
    """Records every order call; order states are set by the test."""

    def __init__(self):
        self.market = "SG"
        self.next_id = 100
        self.orders = {}         # id -> state dict
        self.placed = []         # (kind, args)
        self.cancelled = []
        self.held = 0
        self.price = 71.7
        self.bars = None
        self.fail_close = False

    def _new(self, kind, qty, **info):
        self.next_id += 1
        self.orders[self.next_id] = {"status": "HELD", "active": True, "quantity": qty,
                                     "filled": 0, "avg_fill_price": None}
        self.placed.append((kind, qty, info))
        return self.next_id

    def fill(self, order_id, qty=None, price=None):
        o = self.orders[order_id]
        o.update(filled=qty if qty is not None else o["quantity"], avg_fill_price=price,
                 active=False, status="FILLED")

    # --- MoomooClient surface used by the orchestrator ---
    round_price = staticmethod(lambda p: round(p, 2))

    def get_historical_bars(self, symbol, timeframe, n):
        return self.bars

    def has_exposure(self, symbol):
        return self.held != 0

    def create_entry(self, symbol, direction, quantity, entry_limit_price=None):
        return self._new("entry", quantity, direction=direction, limit=entry_limit_price)

    def get_order_state(self, order_id):
        return dict(self.orders[int(order_id)])

    def place_exit_orders(self, symbol, closing_direction, quantity, stop_price, target_price):
        return (self._new("stop", quantity, direction=closing_direction, price=stop_price),
                self._new("target", quantity, direction=closing_direction, price=target_price))

    def place_stop_order(self, symbol, closing_direction, quantity, stop_price):
        return self._new("stop", quantity, direction=closing_direction, price=stop_price)

    def place_target_order(self, symbol, closing_direction, quantity, limit_price):
        return self._new("target", quantity, direction=closing_direction, price=limit_price)

    def cancel_order(self, order_id):
        o = self.orders[int(order_id)]
        self.cancelled.append(int(order_id))
        if o["active"]:
            o.update(active=False, status="CANCELLED")

    def get_position_quantity(self, symbol):
        return self.held

    def close_position(self, symbol, direction, quantity):
        if self.fail_close:
            raise RuntimeError("market closed")
        oid = self._new("close", quantity, direction=direction)
        self.fill(oid, price=self.price)
        self.held -= quantity
        return oid

    def get_current_price(self, symbol):
        return self.price

    def modify_stop_order(self, order_id, price):
        pass

    def kinds(self):
        return [k for k, _, _ in self.placed]


def engulfing_bars(last_bar_time=None):
    """Daily bars ending in a bullish engulfing candle (closed, days ago)."""
    end = last_bar_time or pd.Timestamp("2026-10-30")
    times = pd.date_range(end=end, periods=30, freq="D")
    rows = [dict(open=71.0, high=71.2, low=70.8, close=71.0, volume=1000)] * 28
    rows.append(dict(open=71.5, high=71.55, low=70.9, close=71.0, volume=1000))   # bearish
    rows.append(dict(open=70.95, high=71.75, low=70.9, close=71.7, volume=3000))  # engulfs it
    df = pd.DataFrame(rows)
    df["time"] = (times.astype("int64") // 10**6).to_numpy()
    df["symbol"] = SYMBOL
    return df


@pytest.fixture
def market_open(monkeypatch):
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)


def make_worker(tmp_path, broker, **cfg_overrides):
    db = str(tmp_path / "audit.db")
    audit = AuditLog(db)
    risk = RiskManager(RiskConfig(
        max_open_positions=3, max_stake_per_point=1000, stop_loss_points=0.5,
        take_profit_points=1.0, max_daily_loss=300.0, min_minutes_between_trades=240,
        lot_size=100,
    ), store=audit)
    kronos = MagicMock()
    kronos.confirm_direction.return_value = (True, 0.9)
    cfg = {"timeframe": "DAY", "lookback_bars": 30, "forecast_horizon": 6,
           "kronos_confidence_threshold": 0.62, "order_type": "LIMIT"}
    cfg.update(cfg_overrides)
    worker = SymbolWorker({"symbol": SYMBOL, "quantity": 100}, cfg, broker, kronos,
                          risk, audit, MagicMock())
    return worker, audit, risk


def open_position(tmp_path, broker):
    """Drive a worker through entry + fill; returns (worker, audit, risk, stop_id, target_id)."""
    broker.bars = engulfing_bars()
    worker, audit, risk = make_worker(tmp_path, broker)
    worker.poll_once()
    entry_id = int(audit.get_active_position(SYMBOL)["entry_order_id"])
    broker.fill(entry_id, price=71.7)
    broker.held = 100
    worker.monitor_position()
    pos = audit.get_active_position(SYMBOL)
    return worker, audit, risk, int(pos["stop_order_id"]), int(pos["target_order_id"])


def test_entry_submitted_without_exit_orders(tmp_path, market_open):
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, _ = make_worker(tmp_path, broker)

    worker.poll_once()

    assert broker.kinds() == ["entry"]
    assert audit.get_active_position(SYMBOL)["status"] == "pending_entry"


def test_exit_orders_wait_for_fill_and_use_filled_quantity(tmp_path, market_open):
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    entry_id = int(audit.get_active_position(SYMBOL)["entry_order_id"])

    worker.monitor_position()  # still working
    assert broker.kinds() == ["entry"]

    broker.fill(entry_id, price=71.68)
    worker.monitor_position()

    assert broker.kinds() == ["entry", "stop", "target"]
    pos = audit.get_active_position(SYMBOL)
    assert pos["status"] == "open"
    assert pos["entry_price"] == 71.68


def test_partial_fill_cancels_remainder_then_protects_filled_shares(tmp_path, market_open):
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    entry_id = int(audit.get_active_position(SYMBOL)["entry_order_id"])
    broker.orders[entry_id].update(filled=60, avg_fill_price=71.7)  # partial, still active

    worker.monitor_position()
    assert entry_id in broker.cancelled

    worker.monitor_position()
    _, stop_qty, _ = broker.placed[1]
    assert stop_qty == 60
    assert audit.get_active_position(SYMBOL)["quantity"] == 60


def test_unfilled_entry_is_cancelled(tmp_path, market_open):
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    entry_id = int(audit.get_active_position(SYMBOL)["entry_order_id"])
    broker.orders[entry_id].update(active=False, status="EXPIRED")

    worker.monitor_position()

    assert audit.get_active_position(SYMBOL) is None
    assert broker.kinds() == ["entry"]


def test_restart_does_not_reenter_same_symbol(tmp_path, market_open):
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, _, _ = make_worker(tmp_path, broker)
    worker.poll_once()

    # Fresh process: new worker + risk manager on the same database, and
    # a new engulfing bar has closed since.
    broker.bars = engulfing_bars(pd.Timestamp("2026-10-31"))
    restarted, _, _ = make_worker(tmp_path, broker)
    restarted.poll_once()

    assert broker.kinds() == ["entry"]


def test_restart_does_not_reprocess_same_bar(tmp_path, market_open):
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, _, _ = make_worker(tmp_path, broker)
    worker.poll_once()

    restarted, _, _ = make_worker(tmp_path, broker)
    restarted.kronos_agent.confirm_direction.reset_mock()
    restarted.poll_once()
    restarted.kronos_agent.confirm_direction.assert_not_called()


def test_untracked_broker_holding_blocks_entry(tmp_path, market_open):
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    broker.held = 10  # e.g. left over from the old agent version
    worker, _, _ = make_worker(tmp_path, broker)

    worker.poll_once()

    assert broker.kinds() == []


def test_forming_bar_is_ignored(tmp_path, market_open):
    broker = FakeBroker()
    # The engulfing candle is today's still-open daily bar.
    broker.bars = engulfing_bars(pd.Timestamp.now(tz="UTC").tz_localize(None).normalize())
    worker, _, _ = make_worker(tmp_path, broker)

    worker.poll_once()

    assert broker.kinds() == []
    worker.kronos_agent.confirm_direction.assert_not_called()


def test_no_entries_while_market_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "is_market_open", lambda market: False)
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, _, _ = make_worker(tmp_path, broker)

    worker.poll_once()

    assert broker.kinds() == []


def test_broker_stop_fill_cancels_target_without_selling_again(tmp_path, market_open):
    broker = FakeBroker()
    worker, audit, risk, stop_id, target_id = open_position(tmp_path, broker)
    broker.fill(stop_id, price=71.2)
    broker.held = 0

    worker.monitor_position()

    assert "close" not in broker.kinds()
    assert target_id in broker.cancelled
    assert audit.get_active_position(SYMBOL) is None
    assert risk.state.daily_pnl == pytest.approx((71.2 - 71.7) * 100)


def test_software_exit_respects_fill_that_raced_the_cancel(tmp_path, market_open, monkeypatch):
    broker = FakeBroker()
    worker, audit, _, stop_id, _ = open_position(tmp_path, broker)
    broker.price = 71.1  # below the 71.2 stop

    # The broker stop fills at the moment we try to cancel it.
    real_cancel = broker.cancel_order

    def cancel_races_fill(order_id):
        if int(order_id) == stop_id:
            broker.fill(stop_id, price=71.15)
            broker.held = 0
        real_cancel(order_id)

    monkeypatch.setattr(broker, "cancel_order", cancel_races_fill)
    worker.monitor_position()

    assert "close" not in broker.kinds()
    assert audit.get_active_position(SYMBOL) is None


def test_software_exit_closes_only_held_quantity(tmp_path, market_open):
    broker = FakeBroker()
    worker, audit, _, _, _ = open_position(tmp_path, broker)
    broker.price = 71.1
    broker.held = 40  # shares partly sold elsewhere

    worker.monitor_position()

    closes = [qty for kind, qty, _ in broker.placed if kind == "close"]
    assert closes == [40]
    assert audit.get_active_position(SYMBOL) is None


def test_failed_close_rearms_protection(tmp_path, market_open):
    broker = FakeBroker()
    worker, audit, _, _, _ = open_position(tmp_path, broker)
    broker.price = 71.1
    broker.fail_close = True

    worker.monitor_position()

    pos = audit.get_active_position(SYMBOL)
    assert pos["status"] == "open"
    assert broker.kinds()[-2:] == ["stop", "target"]
    assert int(pos["stop_order_id"]) == broker.next_id - 1


def interrupted_exit(tmp_path, broker):
    """Live-crash scenario: a software exit cancelled both broker exit
    orders, then died before closing. Price later moved back in range."""
    worker, audit, risk, stop_id, target_id = open_position(tmp_path, broker)
    broker.cancel_order(stop_id)
    broker.cancel_order(target_id)
    broker.placed.clear()
    return worker, audit, stop_id, target_id


def test_cancelled_exit_orders_are_replaced(tmp_path, market_open):
    broker = FakeBroker()
    worker, audit, old_stop, old_target = interrupted_exit(tmp_path, broker)
    broker.price = 71.5  # between stop 71.2 and target 72.7: no exit due

    worker.monitor_position()

    pos = audit.get_active_position(SYMBOL)
    assert pos["status"] == "open"
    assert [(k, q, i["price"]) for k, q, i in broker.placed] == [("stop", 100, 71.2), ("target", 100, 72.7)]
    assert {int(pos["stop_order_id"]), int(pos["target_order_id"])}.isdisjoint({old_stop, old_target})
    assert "re-placed" in worker.reporter.send.call_args.args[0]

    worker.monitor_position()  # healthy now - nothing more is placed
    assert len(broker.placed) == 2


def test_cancelled_exit_orders_not_replaced_when_exit_due(tmp_path, market_open):
    broker = FakeBroker()
    worker, audit, _, _ = interrupted_exit(tmp_path, broker)
    broker.price = 71.1  # through the stop

    worker.monitor_position()

    assert broker.kinds() == ["close"]  # straight to the close, no place-then-cancel churn
    assert audit.get_active_position(SYMBOL) is None


def test_cancelled_exit_orders_replaced_while_market_closed(tmp_path, monkeypatch):
    broker = FakeBroker()
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    worker, audit, _, _ = interrupted_exit(tmp_path, broker)
    monkeypatch.setattr(orch, "is_market_open", lambda market: False)

    worker.monitor_position()

    assert broker.kinds() == ["stop", "target"]


def test_only_the_dead_leg_is_replaced(tmp_path, market_open):
    broker = FakeBroker()
    worker, audit, _, _, target_id = open_position(tmp_path, broker)
    stop_id = int(audit.get_active_position(SYMBOL)["stop_order_id"])
    broker.cancel_order(stop_id)
    broker.placed.clear()
    broker.price = 71.5

    worker.monitor_position()

    assert broker.kinds() == ["stop"]
    assert int(audit.get_active_position(SYMBOL)["target_order_id"]) == target_id


def test_daily_loss_cap_halts_after_losing_exit(tmp_path, market_open):
    broker = FakeBroker()
    worker, _, risk, stop_id, _ = open_position(tmp_path, broker)
    broker.fill(stop_id, price=68.0)  # -370 on 100 shares
    broker.held = 0

    worker.monitor_position()

    assert risk.state.trading_halted is True


def test_sell_signal_rejected_on_cash_account(tmp_path, market_open):
    broker = FakeBroker()
    bars = engulfing_bars()
    # Flip the last two candles into a bearish engulfing.
    bars.loc[28, ["open", "high", "low", "close"]] = [70.95, 71.6, 70.9, 71.5]
    bars.loc[29, ["open", "high", "low", "close"]] = [71.55, 71.6, 70.8, 70.9]
    broker.bars = bars
    worker, _, _ = make_worker(tmp_path, broker)
    worker.kronos_agent.confirm_direction.return_value = (True, 0.9)

    worker.poll_once()

    assert broker.kinds() == []
    _, pattern, reason = worker.reporter.report_rejected.call_args.args
    assert pattern == "bearish_engulfing"
    assert "short selling disabled" in reason


def test_base_symbol_matching():
    assert MoomooClient._base_symbol("SG.D05") == "D05"
    assert MoomooClient._base_symbol("us.aapl") == "AAPL"
    assert MoomooClient._base_symbol("AAPL") == "AAPL"
    assert MoomooClient._base_symbol("HK.00700") == "00700"


def test_orchestrator_publishes_active_symbols(tmp_path):
    from src.orchestrator import Orchestrator

    o = Orchestrator.__new__(Orchestrator)
    o.config = {"timeframe": "DAY", "lookback_bars": 30, "forecast_horizon": 6,
                "kronos_confidence_threshold": 0.62}
    o.audit_log = AuditLog(str(tmp_path / "audit.db"))
    o.broker, o.kronos_agent, o.risk_manager, o.reporter = FakeBroker(), MagicMock(), MagicMock(), MagicMock()
    o.screener = MagicMock()
    o.workers = {}

    o._set_workers([{"symbol": "O39.SI"}, {"symbol": "S68.SI", "name": "SGX"}])

    published = o.audit_log.get_kv("active_symbols")
    assert published["source"] == "screener"
    assert [s["symbol"] for s in published["symbols"]] == ["O39.SI", "S68.SI"]
    assert published["symbols"][1]["name"] == "SGX"


def test_stale_entry_cancelled_after_timeout(tmp_path, market_open):
    import time as _t
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, _ = make_worker(tmp_path, broker, entry_timeout_minutes=15)
    worker.poll_once()
    pos = audit.get_active_position(SYMBOL)
    worker.monitor_position()                      # fresh: left working
    assert broker.cancelled == []
    audit.conn.execute("UPDATE positions SET opened_ts=? WHERE symbol=?", (_t.time() - 16 * 60, SYMBOL))
    audit.conn.commit()
    worker.monitor_position()                      # 16 min old: cancelled
    assert int(pos["entry_order_id"]) in broker.cancelled


def test_percent_bracket_from_fill(tmp_path, market_open):
    from src.risk import RiskConfig, RiskManager
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.risk_manager = RiskManager(RiskConfig(
        max_open_positions=3, max_stake_per_point=1000, stop_loss_points=0.5, take_profit_points=1.0,
        max_daily_loss=300.0, min_minutes_between_trades=240, lot_size=100,
        stop_loss_pct=1.0, take_profit_pct=2.0), store=audit)
    worker.poll_once()
    entry_id = int(audit.get_active_position(SYMBOL)["entry_order_id"])
    broker.fill(entry_id, price=70.0)
    broker.held = 100
    worker.monitor_position()
    pos = audit.get_active_position(SYMBOL)
    assert pos["current_stop"] == 69.3 and pos["target"] == 71.4
