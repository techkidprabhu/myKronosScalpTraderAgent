"""Multi-broker layer: config resolution, Tiger and IG adapters (fake SDK /
HTTP - no accounts needed), bounded connect, and the multi-broker runner."""
import threading
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import src.multi_broker as mb
import src.orchestrator as orch
from src import brokers
from src.broker_config import ROOT, broker_specs, enabled_specs, spec_for
from src.brokers.base import BrokerClient, BrokerError, base_symbol

SHARED = {
    "timeframe": "MIN_5", "default_quantity": 20,
    "risk": {"max_open_positions": 5, "lot_size": 1, "max_daily_loss": 600.0},
    "screener": {"enabled": True, "candidate_symbols": ["US.AAPL"]},
    "kronos": {"model_name": "NeoQuasar/Kronos-base"},
}


# ---------------------------------------------------------------- config

def test_no_brokers_section_means_moomoo_only_on_shared_settings():
    specs = broker_specs(SHARED)
    assert [s.name for s in specs] == ["moomoo"]
    s = specs[0]
    assert s.enabled and s.adapter == "moomoo" and s.db_path == ROOT / "data" / "audit.db"
    assert s.config == SHARED


def test_overrides_deep_merge_over_shared_settings():
    cfg = {**SHARED, "brokers": {
        "moomoo": {"enabled": True},
        "tiger": {"enabled": True, "market": "sg", "timeframe": "HOUR_1",
                  "risk": {"lot_size": 100}, "screener": {"candidate_symbols": ["SG.D05"]}},
    }}
    moomoo, tiger = broker_specs(cfg)
    assert moomoo.config["timeframe"] == "MIN_5"
    assert tiger.market == "SG" and tiger.config["timeframe"] == "HOUR_1"
    assert tiger.config["risk"] == {"max_open_positions": 5, "lot_size": 100, "max_daily_loss": 600.0}
    assert tiger.config["screener"] == {"enabled": True, "candidate_symbols": ["SG.D05"]}
    assert "brokers" not in tiger.config and "market" not in tiger.config
    assert tiger.db_path == ROOT / "data" / "audit_tiger.db"
    assert sorted(tiger.overrides) == ["risk", "screener", "timeframe"]
    assert SHARED["risk"]["lot_size"] == 1          # shared settings untouched


def test_non_moomoo_brokers_default_to_disabled():
    cfg = {**SHARED, "brokers": {"moomoo": {}, "tiger": {}}}
    assert [s.name for s in enabled_specs(cfg)] == ["moomoo"]
    assert spec_for(cfg, None).name == "moomoo"
    assert spec_for(cfg, "tiger").enabled is False


def test_bad_broker_config_is_rejected():
    with pytest.raises(ValueError, match="unknown adapter"):
        broker_specs({"brokers": {"x": {"adapter": "nope"}}})
    with pytest.raises(ValueError, match="own db_path"):
        broker_specs({"brokers": {"a": {"adapter": "moomoo", "db_path": "data/x.db"},
                                  "b": {"adapter": "tiger", "db_path": "data/x.db"}}})
    with pytest.raises(ValueError, match="no broker is enabled"):
        enabled_specs({"brokers": {"moomoo": {"enabled": False}}})


def test_two_accounts_of_one_adapter_get_separate_dbs():
    cfg = {"brokers": {"moomoo": {}, "moomoo_hk": {"adapter": "moomoo", "enabled": True, "market": "HK"}}}
    a, b = broker_specs(cfg)
    assert b.adapter == "moomoo" and b.db_path.name == "audit_moomoo_hk.db" and a.db_path != b.db_path


# ---------------------------------------------------------------- registry

def test_registry_lists_capabilities_and_planned_slots():
    caps = brokers.capabilities()
    assert caps["moomoo"]["status"] == "ready" and caps["tiger"]["status"] == "ready"
    assert caps["ig"]["status"] == "experimental"
    assert caps["zerodha"]["status"] == "experimental" and caps["angel"]["status"] == "experimental"
    assert caps["zerodha"]["short_selling"] is False and caps["zerodha"]["markets"] == ["NSE", "BSE"]


def test_generic_connect_never_blocks(monkeypatch):
    class Hanging:
        label = "Slow"
        def __init__(self, **kw):
            time.sleep(5)
    monkeypatch.setitem(brokers.REGISTRY, "slow", ("tests.test_brokers", "Hanging"))
    monkeypatch.setattr(brokers, "broker_class", lambda a: Hanging)
    t0 = time.time()
    with pytest.raises(ConnectionError, match="Slow gave no response"):
        brokers.connect_with_timeout("slow", 0.2)
    assert time.time() - t0 < 2


def test_base_symbol_matches_old_moomoo_rule():
    from src.moomoo_client import MoomooClient
    for s in ["US.AAPL", "SG.D05", "HK.00700", "D05.SI", "AAPL", "us.msft"]:
        assert base_symbol(s) == MoomooClient._base_symbol(s)


# ---------------------------------------------------------------- Tiger

class FakeTigerOrder:
    def __init__(self, oid, symbol, action="BUY", qty=10, filled=0, status="NEW", limit=None, aux=None,
                 avg=0):
        self.id, self.action, self.quantity, self.filled = oid, action, qty, filled
        self.contract = SimpleNamespace(symbol=symbol)
        self.order_type, self.limit_price, self.aux_price = "LMT", limit, aux
        self.status = SimpleNamespace(name=status)
        self.active = status in ("NEW", "PARTIALLY_FILLED", "PENDING_NEW", "HELD")
        self.avg_fill_price, self.time_in_force, self.order_time = avg, "DAY", None


class FakeTigerTrade:
    def __init__(self):
        self.placed, self.cancelled, self.modified = [], [], []
        self.orders = {}
        self.positions = []

    def place_order(self, order):
        oid = 1000 + len(self.placed)
        self.placed.append(order)
        return oid

    def get_order(self, account=None, id=None):
        return self.orders.get(id)

    def get_open_orders(self, account=None, market=None):
        return [o for o in self.orders.values() if o.active]

    def get_positions(self, account=None, market=None):
        return self.positions

    def cancel_order(self, account=None, id=None):
        self.cancelled.append(id)
        return id

    def modify_order(self, order, aux_price=None):
        self.modified.append((order.id, aux_price))
        return order.id

    def get_prime_assets(self, account=None):
        raise RuntimeError("not a prime account")

    def get_assets(self, account=None):
        return [SimpleNamespace(summary=SimpleNamespace(net_liquidation=100000.0, cash=40000.0,
                                                        gross_position_value=60000.0, buying_power=80000.0))]


class FakeTigerQuote:
    def get_bars(self, symbols, period=None, limit=None):
        times = pd.date_range("2026-11-04 01:00", periods=limit, freq="5min")
        return pd.DataFrame({"symbol": symbols[0], "time": (times.astype("int64") // 10**6),
                             "open": 30.0, "high": 30.2, "low": 29.9, "close": 30.1, "volume": 1000})

    def get_stock_briefs(self, symbols):
        return pd.DataFrame({"symbol": symbols, "latest_price": [30.12] * len(symbols), "volume": [5e5] * len(symbols)})

    def get_market_status(self, market):
        return []

    def get_symbols(self, market=None):
        return ["D05.SI", "O39.SI"]


@pytest.fixture
def tiger(monkeypatch):
    from src.brokers.tiger import TigerClient
    monkeypatch.setenv("TIGER_ACCOUNT", "U1234567")
    return TigerClient(market="SG", quote_client=FakeTigerQuote(), trade_client=FakeTigerTrade())


def test_tiger_symbols_are_canonical_both_ways(tiger):
    assert tiger._normalize_symbol("SG.D05") == "D05.SI"
    assert tiger._normalize_symbol("D05") == "D05.SI"
    assert tiger._normalize_symbol("D05.SI") == "D05.SI"
    assert tiger._normalize_symbol("US.AAPL") == "AAPL"
    assert tiger._canonical("D05.SI") == "SG.D05"
    assert tiger.list_market_symbols() == ["SG.D05", "SG.O39"]


def test_tiger_bars_and_prices(tiger):
    bars = tiger.get_historical_bars("SG.D05", "MIN_5", 12)
    assert list(bars.columns[:6]) == ["time", "open", "high", "low", "close", "volume"]
    assert len(bars) == 12 and bars["symbol"].iloc[0] == "SG.D05"
    assert tiger.get_current_price("SG.D05") == 30.12
    snaps = tiger.get_snapshots(["SG.D05", "SG.O39"])
    assert snaps["code"].tolist() == ["SG.D05", "SG.O39"] and snaps["volume"].iloc[0] == 5e5


def test_tiger_orders_use_sgx_ticks_and_exit_tif(tiger, monkeypatch):
    oid = tiger.create_entry("SG.D05", "BUY", 100, entry_limit_price=30.1234)
    o = tiger.trade_client.placed[-1]
    assert oid == "1000" and o.limit_price == 30.12 and o.contract.symbol == "D05.SI" and o.contract.currency == "SGD"
    tiger.place_stop_order("SG.D05", "SELL", 100, 29.876)
    stop = tiger.trade_client.placed[-1]
    assert stop.aux_price == 29.88 and stop.time_in_force == "GTC"
    monkeypatch.setenv("TIGER_EXIT_TIF", "DAY")
    from src.brokers.tiger import TigerClient
    day = TigerClient(market="SG", quote_client=FakeTigerQuote(), trade_client=FakeTigerTrade())
    assert day.exit_time_in_force == "DAY"


def test_tiger_order_state_positions_and_account(tiger):
    tiger.trade_client.orders[7] = FakeTigerOrder(7, "D05.SI", qty=100, filled=100, status="FILLED", avg=30.05)
    assert tiger.get_order_state(7) == {"status": "FILLED", "active": False, "quantity": 100,
                                        "filled": 100, "avg_fill_price": 30.05}
    tiger.trade_client.orders[8] = FakeTigerOrder(8, "D05.SI", action="SELL", qty=100, status="NEW", aux=29.5)
    assert tiger.has_open_orders("SG.D05") and tiger.get_open_orders()[0]["symbol"] == "SG.D05"
    tiger.trade_client.positions = [SimpleNamespace(contract=SimpleNamespace(symbol="D05.SI", currency="SGD"),
                                                    quantity=100, average_cost=30.0, market_price=30.1,
                                                    unrealized_pnl=10.0)]
    assert tiger.get_position_quantity("SG.D05") == 100 and tiger.has_exposure("D05")
    acct = tiger.get_account_summary()
    assert acct == {"currency": "SGD", "total_assets": 100000.0, "cash": 40000.0,
                    "market_val": 60000.0, "power": 80000.0}
    tiger.modify_stop_order(8, 29.712)
    assert tiger.trade_client.modified == [(8, 29.71)]


# ---------------------------------------------------------------- IG

class FakeResp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.content = b"x" if body is not None else b""
        self.text = str(body)

    def json(self):
        return self._body


class FakeIGHttp:
    def __init__(self):
        self.calls = []
        self.expire_next = False
        self.working = []
        self.positions = []
        self.activity = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("LOGIN", url, json))
        return FakeResp(200, {"currentAccountId": "ABC12"}, {"CST": "cst-token", "X-SECURITY-TOKEN": "xst-token"})

    def request(self, method, url, headers=None, json=None, params=None, timeout=None):
        path = url.split("/gateway/deal", 1)[1]
        self.calls.append((method, path, json, headers))
        if self.expire_next:
            self.expire_next = False
            return FakeResp(401, {"errorCode": "error.security.client-token-invalid"})
        if path.startswith("/prices/"):
            return FakeResp(200, {"prices": [
                {"snapshotTimeUTC": "2026-11-04T14:30:00", "openPrice": {"bid": 99, "ask": 101},
                 "highPrice": {"bid": 101, "ask": 103}, "lowPrice": {"bid": 98, "ask": 100},
                 "closePrice": {"bid": 100, "ask": 102}, "lastTradedVolume": 500},
                {"snapshotTimeUTC": "2026-11-04T14:35:00", "openPrice": {"bid": 100, "ask": 102},
                 "highPrice": {"bid": 102, "ask": 104}, "lowPrice": {"bid": 99, "ask": 101},
                 "closePrice": {"bid": 101, "ask": 103}, "lastTradedVolume": 700}]})
        if path.startswith("/markets/"):
            return FakeResp(200, {"snapshot": {"bid": 101.0, "offer": 101.4}})
        if path == "/positions":
            return FakeResp(200, {"positions": self.positions})
        if path == "/workingorders":
            return FakeResp(200, {"workingOrders": self.working})
        if path == "/positions/otc" and method == "POST":
            return FakeResp(200, {"dealReference": "REF1"})
        if path == "/workingorders/otc" and method == "POST":
            return FakeResp(200, {"dealReference": "REF2"})
        if path.startswith("/confirms/"):
            ref = path.rsplit("/", 1)[1]
            return FakeResp(200, {"dealStatus": "ACCEPTED", "dealId": f"DEAL-{ref}", "level": 101.2})
        if path == "/history/activity":
            return FakeResp(200, {"activities": self.activity})
        if path == "/accounts":
            return FakeResp(200, {"accounts": [{"accountId": "ABC12", "currency": "USD",
                                                "balance": {"balance": 10000, "available": 8000, "deposit": 2000}}]})
        return FakeResp(200, {})


@pytest.fixture
def ig(monkeypatch):
    from src.brokers.ig import IGClient
    for k, v in {"IG_API_KEY": "key", "IG_USERNAME": "user", "IG_PASSWORD": "pw"}.items():
        monkeypatch.setenv(k, v)
    return IGClient(market="US", session=FakeIGHttp())


def test_ig_login_and_bars(ig):
    assert ig.account == "ABC12" and ig.sandbox
    bars = ig.get_historical_bars("UA.D.AAPL.CASH.IP", "MIN_5", 2)
    assert bars["close"].tolist() == [101.0, 102.0] and bars["volume"].tolist() == [500.0, 700.0]
    assert bars["time"].iloc[1] - bars["time"].iloc[0] == 5 * 60 * 1000
    method, path, _, headers = ig.http.calls[-1]
    assert path.startswith("/prices/UA.D.AAPL.CASH.IP") and headers["CST"] == "cst-token" and headers["Version"] == "3"
    assert ig.get_current_price("UA.D.AAPL.CASH.IP") == pytest.approx(101.2)


def test_ig_entries_exits_and_netting(ig):
    assert ig.create_entry("UA.D.AAPL.CASH.IP", "BUY", 2) == "DEAL-REF1"
    body = [c for c in ig.http.calls if c[1] == "/positions/otc"][-1][2]
    assert body["orderType"] == "MARKET" and body["forceOpen"] is False and body["size"] == 2
    assert ig.create_entry("UA.D.AAPL.CASH.IP", "BUY", 2, entry_limit_price=100.123456) == "DEAL-REF2"
    w = [c for c in ig.http.calls if c[1] == "/workingorders/otc"][-1][2]
    assert w["type"] == "LIMIT" and w["level"] == 100.1235
    ig.place_stop_order("UA.D.AAPL.CASH.IP", "SELL", 2, 99.5)
    stop = [c for c in ig.http.calls if c[1] == "/workingorders/otc"][-1][2]
    assert stop["type"] == "STOP" and stop["direction"] == "SELL" and stop["forceOpen"] is False


def test_ig_order_state_from_working_orders_positions_and_history(ig):
    ig.http.working = [{"workingOrderData": {"dealId": "W1", "epic": "E", "direction": "SELL",
                                             "orderSize": 2, "orderLevel": 99.5, "orderType": "STOP"}}]
    assert ig.get_order_state("W1")["active"] is True
    assert ig.get_open_orders()[0]["stop_price"] == 99.5
    ig.http.positions = [{"position": {"dealId": "P1", "size": 2, "direction": "SELL", "level": 100.0},
                          "market": {"epic": "E", "bid": 99.0, "offer": 99.2}}]
    assert ig.get_open_positions()[0]["quantity"] == -2
    assert ig.get_order_state("P1") == {"status": "FILLED", "active": False, "quantity": 2, "filled": 2,
                                        "avg_fill_price": 100.0}
    ig.http.activity = [{"status": "ACCEPTED", "details": {"level": 99.4, "size": 2,
                                                           "actions": [{"actionType": "STOP_ORDER_FILLED"}]}}]
    assert ig.get_order_state("GONE")["filled"] == 2
    ig.http.activity = [{"status": "ACCEPTED", "details": {"actions": [{"actionType": "WORKING_ORDER_DELETED"}]}}]
    assert ig.get_order_state("GONE") == {"status": "CANCELLED", "active": False, "quantity": 0,
                                          "filled": 0, "avg_fill_price": None}


def test_ig_logs_in_again_when_session_expires(ig):
    logins = sum(1 for c in ig.http.calls if c[0] == "LOGIN")
    ig.http.expire_next = True
    assert ig.get_current_price("E") == pytest.approx(101.2)
    assert sum(1 for c in ig.http.calls if c[0] == "LOGIN") == logins + 1


def test_ig_needs_credentials(monkeypatch):
    from src.brokers.ig import IGClient
    for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(BrokerError, match="IG credentials missing"):
        IGClient(session=FakeIGHttp())


# ---------------------------------------------------------------- orchestrator + runner

def test_orchestrator_connects_through_the_registry_for_other_brokers(monkeypatch):
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    o.adapter, o.market_override, o.connect_timeout, o.broker_label = "tiger", "SG", 5, "Tiger"
    seen = {}
    monkeypatch.setattr(orch, "connect_broker", lambda adapter, timeout, **kw: seen.update(a=adapter, t=timeout, **kw) or "client")
    assert o._connect() == "client" and seen == {"a": "tiger", "t": 5, "market": "SG"}


def test_shared_kronos_never_runs_two_forecasts_at_once():
    active, peak = [0], [0]

    class Slow:
        def confirm_direction(self, *a, **k):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            active[0] -= 1
            return True, 0.8
    shared = mb.SharedKronos(Slow())
    threads = [threading.Thread(target=shared.confirm_direction) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


def test_runner_builds_one_orchestrator_per_enabled_broker(monkeypatch):
    monkeypatch.setattr(mb, "KronosAgent", lambda **kw: MagicMock())
    built = []

    def fake_build(spec, kronos, multi):
        o = MagicMock()
        o.run_forever.side_effect = lambda: (built.append((spec.name, multi, threading.current_thread().name)),
                                             time.sleep(5))
        return o
    monkeypatch.setattr(mb, "build_orchestrator", fake_build)
    cfg = {**SHARED, "brokers": {"moomoo": {}, "tiger": {"enabled": True}, "ig": {"enabled": False}}}
    runner = mb.MultiBrokerRunner(cfg)
    assert [s.name for s in runner.specs] == ["moomoo", "tiger"]
    t = threading.Thread(target=runner.run_forever, daemon=True)
    t.start()
    for _ in range(50):
        if len(built) == 2:
            break
        time.sleep(0.02)
    assert sorted(built) == [("moomoo", True, "broker-moomoo"), ("tiger", True, "broker-tiger")]


def test_telegram_prefix_tags_the_broker(monkeypatch):
    from src import telegram_bot
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    sent = {}
    monkeypatch.setattr(telegram_bot.requests, "post",
                        lambda url, data=None, timeout=None: sent.update(data) or MagicMock())
    telegram_bot.TelegramReporter(prefix="[Tiger] ").send("Entry filled")
    assert sent["text"] == "[Tiger] Entry filled"


def test_real_orchestrators_per_broker_share_kronos_but_not_state(tmp_path, monkeypatch):
    from tests.test_resilience import USBroker
    monkeypatch.setattr(mb, "KronosAgent", lambda **kw: MagicMock(name="kronos"))
    made = []

    def fake_connect(adapter, timeout, **kw):
        b = USBroker()
        b.market = kw.get("market", "US")
        made.append((adapter, b.market))
        return b
    monkeypatch.setattr(orch, "connect_broker", fake_connect)
    monkeypatch.setattr(orch, "connect_with_timeout", lambda timeout, **kw: fake_connect("moomoo", timeout, **kw))
    cfg = {"timeframe": "MIN_5", "lookback_bars": 30, "forecast_horizon": 5, "kronos_confidence_threshold": 0.6,
           "risk": {"max_open_positions": 3, "max_stake_per_point": 200, "stop_loss_points": 1.0,
                    "take_profit_points": 2.0, "max_daily_loss": 300.0, "min_minutes_between_trades": 15},
           "symbols": [{"symbol": "US.AAPL", "quantity": 10}],
           "brokers": {"moomoo": {"db_path": str(tmp_path / "m.db")},
                       "tiger": {"enabled": True, "market": "SG", "db_path": str(tmp_path / "t.db"),
                                 "symbols": [{"symbol": "SG.D05", "quantity": 100}],
                                 "risk": {"lot_size": 100}}}}
    runner = mb.MultiBrokerRunner(cfg)
    o_m = mb.build_orchestrator(runner.specs[0], runner.kronos, multi=True)
    o_t = mb.build_orchestrator(runner.specs[1], runner.kronos, multi=True)
    assert made == [("moomoo", "US"), ("tiger", "SG")]
    assert o_m.kronos_agent is o_t.kronos_agent is runner.kronos
    assert o_m.workers["US.AAPL"].kronos_agent is runner.kronos
    assert o_m.audit_log is not o_t.audit_log
    assert Path(o_t.audit_log.db_path if hasattr(o_t.audit_log, "db_path") else tmp_path / "t.db").exists()
    assert list(o_t.workers) == ["SG.D05"] and o_t.risk_manager.config.lot_size == 100
    assert o_m.risk_manager.config.lot_size == 1
    assert o_t.reporter.prefix == "[Tiger] " and o_m.reporter.prefix == "[moomoo OpenD] "
    assert o_t.broker_label == "Tiger" and "Tiger" in o_t._label
