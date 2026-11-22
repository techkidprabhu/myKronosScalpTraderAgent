"""
Market-calendar sleep, OpenD reconnect, paper-account DAY exits and the
unmanaged-position alert - against the same FakeBroker as the lifecycle
tests, so no OpenD is needed.
"""
from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

import src.orchestrator as orch
from src.audit_log import AuditLog
from src.orchestrator import Orchestrator
from tests.test_orchestrator import SYMBOL, FakeBroker, open_position

ET = ZoneInfo("America/New_York")


class DayBroker(FakeBroker):
    """A moomoo paper account: exit orders are DAY and expire at the close."""
    exit_time_in_force = "DAY"

    def expire(self, order_id):
        self.orders[int(order_id)].update(active=False, status="CANCELLED_ALL")


# ---------------------------------------------------------------- DAY exits

def test_day_exits_expire_overnight_and_rearm_at_open(tmp_path, monkeypatch):
    broker = DayBroker()
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    worker, audit, _, stop_id, target_id = open_position(tmp_path, broker)

    # Session ends: the DAY orders expire. Nothing is placed while closed.
    monkeypatch.setattr(orch, "is_market_open", lambda market: False)
    broker.expire(stop_id)
    broker.expire(target_id)
    broker.placed.clear()
    worker.monitor_position()
    pos = audit.get_active_position(SYMBOL)
    assert broker.kinds() == []
    assert pos["stop_order_id"] is None and pos["target_order_id"] is None

    # Next session opens: both legs are re-armed at the current levels.
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    broker.price = 71.5
    worker.monitor_position()
    pos = audit.get_active_position(SYMBOL)
    assert [(k, q, i["price"]) for k, q, i in broker.placed] == [("stop", 100, 71.2), ("target", 100, 72.7)]
    assert pos["stop_order_id"] and pos["target_order_id"]
    assert "Session protection armed" in worker.reporter.send.call_args.args[0]

    worker.monitor_position()  # healthy - nothing more placed
    assert len(broker.placed) == 2


def test_gtc_exits_still_replaced_while_closed(tmp_path, monkeypatch):
    broker = FakeBroker()  # live account: GTC
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    worker, audit, _, stop_id, target_id = open_position(tmp_path, broker)
    monkeypatch.setattr(orch, "is_market_open", lambda market: False)
    broker.cancel_order(stop_id)
    broker.placed.clear()
    worker.monitor_position()
    assert broker.kinds() == ["stop"]


def test_rejected_leg_is_rearmed_rate_limited(tmp_path, monkeypatch):
    broker = DayBroker()
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    real_place_exits = broker.place_exit_orders
    broker.place_exit_orders = lambda *a: (None, real_place_exits(*a)[1])  # stop rejected at fill
    worker, audit, _, _, _ = _open_with_rejected_stop(tmp_path, broker)
    broker.placed.clear()
    broker.price = 71.5

    worker.monitor_position()  # first retry is immediate
    assert broker.kinds() == ["stop"]

    pos = audit.get_active_position(SYMBOL)
    broker.cancel_order(int(pos["stop_order_id"]))
    audit.set_exit_order_ids(SYMBOL, None, pos["target_order_id"])
    broker.placed.clear()
    worker.monitor_position()  # within 15 min - no new attempt
    assert broker.kinds() == []


def _open_with_rejected_stop(tmp_path, broker):
    from tests.test_orchestrator import engulfing_bars, make_worker
    broker.bars = engulfing_bars()
    worker, audit, risk = make_worker(tmp_path, broker)
    worker.poll_once()
    entry_id = int(audit.get_active_position(SYMBOL)["entry_order_id"])
    broker.fill(entry_id, price=71.7)
    broker.held = 100
    worker.monitor_position()
    pos = audit.get_active_position(SYMBOL)
    assert pos["stop_order_id"] is None
    return worker, audit, risk, None, pos["target_order_id"]


def test_no_rearm_when_exit_is_due(tmp_path, monkeypatch):
    broker = DayBroker()
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    worker, audit, _, stop_id, target_id = open_position(tmp_path, broker)
    broker.expire(stop_id)
    broker.expire(target_id)
    audit.set_exit_order_ids(SYMBOL, None, None)
    broker.placed.clear()
    broker.price = 71.0  # opened below the stop
    worker.monitor_position()
    assert broker.kinds() == ["close"]
    assert audit.get_active_position(SYMBOL) is None


# ---------------------------------------------------------------- orchestrator

def make_orch(tmp_path, broker, **cfg):
    o = Orchestrator.__new__(Orchestrator)
    o.config = {"timeframe": "DAY", "lookback_bars": 30, "forecast_horizon": 6,
                "kronos_confidence_threshold": 0.62, **cfg}
    o.audit_log = AuditLog(str(tmp_path / "audit.db"))
    o.broker, o.kronos_agent, o.risk_manager, o.reporter = broker, MagicMock(), MagicMock(), MagicMock()
    o.screener = None
    o.workers = {}
    o.phase = None
    o.follow_market, o.warmup_minutes, o.notify_open_close = True, 15, True
    o.sleep_monitor_seconds = 300
    o.connect_timeout, o.ping_interval, o.reconnect_after = 1, 60, 3
    o.unmanaged_mode, o.unmanaged_interval, o.reconcile_missing = "alert", 300, True
    o.broker_ok, o._ping_failures, o._last_ping, o._broker_detail = True, 0, 0.0, ""
    o._last_unmanaged_check, o._last_screener_run, o.screener_refresh_seconds = 0.0, 0.0, 0
    return o


class USBroker(FakeBroker):
    def __init__(self):
        super().__init__()
        self.market = "US"
        self.positions = []
        self.open_orders = []
        self.ping_result = (True, "connected")

    def get_open_positions(self):
        return self.positions

    def get_open_orders(self):
        return self.open_orders

    def ping(self):
        return self.ping_result

    def close(self):
        self.closed = True


@pytest.mark.parametrize("when,phase", [
    (datetime(2026, 11, 2, 10, 0, tzinfo=ET), "active"),
    (datetime(2026, 11, 2, 9, 20, tzinfo=ET), "warmup"),     # 10 min before the open
    (datetime(2026, 11, 2, 7, 0, tzinfo=ET), "sleeping"),
    (datetime(2026, 11, 26, 10, 0, tzinfo=ET), "sleeping"),  # Thanksgiving
    (datetime(2026, 11, 27, 14, 0, tzinfo=ET), "sleeping"),  # after the 1 pm early close
    (datetime(2026, 11, 7, 10, 0, tzinfo=ET), "sleeping"),   # Saturday
])
def test_phase_follows_nasdaq_calendar(tmp_path, when, phase):
    o = make_orch(tmp_path, USBroker())
    assert o.compute_phase(when)[0] == phase


def test_phase_always_when_not_following_market(tmp_path):
    o = make_orch(tmp_path, USBroker())
    o.follow_market = False
    assert o.compute_phase(datetime(2026, 11, 26, 10, 0, tzinfo=ET))[0] == "always"


def test_sleeping_skips_scans_and_slows_monitor(tmp_path, monkeypatch):
    o = make_orch(tmp_path, USBroker())
    worker = MagicMock()
    o.workers = {"AAPL": worker}
    monkeypatch.setattr(o, "update_phase", lambda: "sleeping")
    state = {}
    o.run_cycle(state, now=1000.0)
    o.run_cycle(state, now=1100.0)
    worker.poll_once.assert_not_called()
    assert worker.monitor_position.call_count == 1  # second cycle within 300 s
    o.run_cycle(state, now=1400.0)
    assert worker.monitor_position.call_count == 2


def test_active_phase_scans(tmp_path, monkeypatch):
    o = make_orch(tmp_path, USBroker())
    worker = MagicMock()
    o.workers = {"AAPL": worker}
    monkeypatch.setattr(o, "update_phase", lambda: "active")
    o.run_cycle({}, now=1000.0)
    worker.prepare_signal.assert_called_once()        # the scan (Kronos is asked by the orchestrator)
    worker.finish_signal.assert_called_once()
    worker.monitor_position.assert_called_once()


def test_open_and_close_notifications_and_published_phase(tmp_path, monkeypatch):
    o = make_orch(tmp_path, USBroker())
    phases = iter([("sleeping", 1.0), ("active", None), ("sleeping", 2e9)])
    monkeypatch.setattr(o, "compute_phase", lambda now=None: next(phases))
    o.update_phase()          # startup: no message
    assert not o.reporter.send.called
    o.update_phase()
    assert "NASDAQ open" in o.reporter.send.call_args.args[0]
    o.update_phase()
    assert "NASDAQ closed" in o.reporter.send.call_args.args[0]
    assert o.audit_log.get_kv("agent_phase")["phase"] == "sleeping"


def test_broker_reconnects_after_repeated_failures(tmp_path, monkeypatch):
    old = USBroker()
    o = make_orch(tmp_path, old)
    worker = MagicMock()
    o.workers = {"AAPL": worker}
    new = USBroker()
    monkeypatch.setattr(orch, "connect_with_timeout", lambda timeout: new)
    old.ping_result = (False, "OpenD is running but its moomoo account is not logged in")

    assert o.check_broker(now=100) is False
    assert "Trading paused" in o.reporter.send.call_args.args[0]
    health = o.audit_log.get_kv("broker_health")
    assert health["ok"] is False and "not logged in" in health["detail"]
    worker.poll_once.assert_not_called()
    o.run_cycle({}, now=130)  # within ping interval: still paused, no worker calls
    worker.monitor_position.assert_not_called()

    o.check_broker(now=200)
    assert o.check_broker(now=300) is True     # third failure -> rebuilt
    assert o.broker is new and worker.broker is new and old.closed
    assert "restored" in o.reporter.send.call_args.args[0]
    assert o.audit_log.get_kv("broker_health")["ok"] is True


def test_unmanaged_position_alerted_once(tmp_path):
    broker = USBroker()
    broker.positions = [{"symbol": "US.TSLA", "quantity": -25, "average_cost": 403.0,
                         "market_price": 399.0, "unrealized_pnl": 100.0}]
    o = make_orch(tmp_path, broker)

    o._check_unmanaged(now=1000)
    assert "US.TSLA SHORT 25" in o.reporter.send.call_args.args[0]
    assert o.audit_log.get_kv("unmanaged")["positions"][0]["symbol"] == "US.TSLA"

    o.reporter.send.reset_mock()
    o._check_unmanaged(now=1300)
    assert not o.reporter.send.called  # already alerted

    broker.positions = []
    o._check_unmanaged(now=1600)
    assert o.audit_log.get_kv("unmanaged")["positions"] == []
    assert o.audit_log.get_kv("unmanaged_alerted") == []


def test_just_closed_symbol_is_not_unmanaged(tmp_path):
    broker = USBroker()
    broker.positions = [{"symbol": "US.TSLA", "quantity": -25}]
    o = make_orch(tmp_path, broker)
    o.audit_log.set_kv("recent_closes", {"TSLA": 1000.0})
    o._check_unmanaged(now=1060)  # moomoo's position view still lags
    assert not o.reporter.send.called


def test_tracked_position_and_its_orders_are_not_unmanaged(tmp_path, monkeypatch):
    broker = USBroker()
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    worker, audit, _, stop_id, target_id = open_position(tmp_path, broker)
    o = make_orch(tmp_path, broker)
    o.audit_log = audit
    broker.positions = [{"symbol": f"US.{SYMBOL}", "quantity": 100}]
    broker.open_orders = [{"id": str(stop_id), "symbol": f"US.{SYMBOL}"},
                          {"id": str(target_id), "symbol": f"US.{SYMBOL}"}]
    o._check_unmanaged(now=1000)
    assert not o.reporter.send.called


def test_position_closed_outside_agent_is_reconciled(tmp_path, monkeypatch):
    broker = USBroker()
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    worker, audit, _, stop_id, target_id = open_position(tmp_path, broker)
    o = make_orch(tmp_path, broker)
    o.audit_log = audit
    o.workers = {SYMBOL: worker}
    monkeypatch.setattr(orch, "is_market_open", lambda market: False)  # overnight

    broker.held = 0              # closed by hand / another program
    broker.positions = []
    o._check_unmanaged(now=1000)
    assert audit.get_active_position(SYMBOL) is not None   # not yet - could be lag
    o._check_unmanaged(now=1300)
    assert audit.get_active_position(SYMBOL) is None
    assert stop_id in broker.cancelled and target_id in broker.cancelled
    assert "closed outside the agent" in worker.reporter.send.call_args.args[0]


def test_database_position_gets_monitor_only_worker(tmp_path):
    broker = USBroker()
    o = make_orch(tmp_path, broker)
    o.audit_log.record_pending_entry("TSLA", "SELL", 25, 403.0, 405.9, 399.84, entry_order_id="9")
    o._set_workers([{"symbol": "AAPL"}])
    assert set(o.workers) == {"AAPL", "TSLA"}
    assert o.workers["TSLA"].monitor_only and not o.workers["AAPL"].monitor_only
    o.workers["TSLA"].poll_once()  # never scans
    assert broker.kinds() == []


def test_software_exit_on_externally_closed_position_books_no_pnl(tmp_path, monkeypatch):
    # The live case: the DB still had a TSLA short open after another
    # program closed it at the broker. A stop "trigger" must not invent P&L.
    broker = FakeBroker()
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    worker, audit, risk, _, _ = open_position(tmp_path, broker)
    broker.held = 0
    broker.price = 71.0  # through the stop
    worker.monitor_position()
    assert audit.get_active_position(SYMBOL) is None
    assert "close" not in broker.kinds()
    assert risk.state.daily_pnl == 0.0


def test_stale_row_with_unknown_orders_is_still_reconciled(tmp_path, monkeypatch):
    broker = USBroker()
    o = make_orch(tmp_path, broker)
    o.audit_log.record_pending_entry("TSLA", "SELL", 25, 403.0, 405.9, 399.84, entry_order_id="9")
    o.audit_log.mark_position_open("TSLA", 25, 403.0, "old-stop", "old-target",
                                   stop_price=405.9, target_price=399.84)
    o._set_workers([])
    monkeypatch.setattr(orch, "is_market_open", lambda market: False)
    broker.get_order_state = MagicMock(side_effect=RuntimeError("Order old-stop not found"))
    o._check_unmanaged(now=1000)
    o._check_unmanaged(now=1300)
    assert o.audit_log.get_active_position("TSLA") is None
