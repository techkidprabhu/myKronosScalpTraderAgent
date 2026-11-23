"""IG: the dashboard reads what the agent already fetched instead of spending
the request allowance they share."""
import json
import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

import src.brokers.ig as igmod
from src.audit_log import AuditLog
from src.brokers.base import BrokerError
from src.brokers.ig import IGClient
from src.orchestrator import BROKER_MIRROR_KEY
from tests.test_dashboard_multibroker import AUTH, client  # noqa: F401 - fixture
from tests.test_ig_allowance import CountingHttp
from tests.test_resilience import USBroker, make_orch


class IGLikeBroker(USBroker):
    def __init__(self):
        super().__init__()
        self.positions = [{"symbol": "CS.D.EURUSD.CSD.IP", "quantity": 0.2, "avg_cost": 1.1, "market_value": None,
                           "unrealized_pnl": None, "current_price": 1.101}]
        self.calls = 0

    def get_open_positions(self):
        self.calls += 1
        return super().get_open_positions()

    def get_account_summary(self):
        return {"cash": 9000.0, "total_assets": 10000.0, "currency": "USD"}

    def request_guard_status(self):
        return {"per_min": 20, "used_last_min": 3}


# ---------------------------------------------------------------- agent side

def test_agent_publishes_positions_orders_and_account_every_30s(tmp_path):
    o = make_orch(tmp_path, IGLikeBroker())
    o._publish_broker_mirror(1000.0)
    m = o.audit_log.get_kv(BROKER_MIRROR_KEY)
    assert m["positions"][0]["quantity"] == 0.2 and m["account"]["cash"] == 9000.0 and m["ts"] == 1000.0
    o._publish_broker_mirror(1010.0)                 # too soon: nothing fetched
    assert o.broker.calls == 1
    o._publish_broker_mirror(1031.0)
    assert o.broker.calls == 2


def test_other_brokers_publish_nothing(tmp_path):
    o = make_orch(tmp_path, USBroker())
    o._publish_broker_mirror(1000.0)
    assert o.audit_log.get_kv(BROKER_MIRROR_KEY) is None


# ---------------------------------------------------------------- dashboard side

def ig_dashboard(client, monkeypatch, mirror_age=5):
    import dataclasses
    import dashboard.app as dash
    real = dash.current_spec
    monkeypatch.setattr(dash, "current_spec", lambda raw=None: dataclasses.replace(real(raw), adapter="ig"))
    broker = MagicMock()
    broker.get_open_positions.side_effect = AssertionError("asked IG for positions")
    broker.get_open_orders.side_effect = AssertionError("asked IG for orders")
    broker.get_account_summary.side_effect = AssertionError("asked IG for the account")
    broker.get_snapshots.side_effect = AssertionError("asked IG for prices")
    monkeypatch.setattr(dash, "get_broker", lambda: broker)
    for c in (dash._broker_cache, dash._quote_cache, dash._funds_cache):
        c.clear()
    a = AuditLog(str(dash._db_path()))
    a.set_kv(BROKER_MIRROR_KEY, {"ts": time.time() - mirror_age, "account_ts": time.time() - mirror_age,
                                 "positions": [{"symbol": "CS.D.EURUSD.CSD.IP", "quantity": 0.2, "avg_cost": 1.1,
                                                "market_value": None, "unrealized_pnl": None,
                                                "current_price": 1.101}],
                                 "orders": [], "account": {"cash": 9000.0, "total_assets": 10000.0},
                                 "prices": {"CS.D.EURUSD.CSD.IP": 1.1015}})
    a.close()
    return dash, broker


def test_dashboard_uses_the_agents_figures_without_asking_ig(client, monkeypatch):  # noqa: F811
    dash, broker = ig_dashboard(client, monkeypatch)
    positions, orders = dash._broker_snapshot(broker)
    assert positions[0]["quantity"] == 0.2 and orders == []
    assert dash._account_summary(broker)["cash"] == 9000.0
    assert dash._live_prices(broker, ["CS.D.EURUSD.CSD.IP"]) == {"CS.D.EURUSD.CSD.IP": 1.1015}


def test_dashboard_asks_ig_itself_when_the_agent_is_not_running(client, monkeypatch):  # noqa: F811
    dash, broker = ig_dashboard(client, monkeypatch, mirror_age=600)          # stale
    broker.get_open_positions.side_effect = None
    broker.get_open_positions.return_value = []
    broker.get_open_orders.side_effect = None
    broker.get_open_orders.return_value = []
    assert dash._broker_snapshot(broker) == ([], [])
    broker.get_open_positions.assert_called_once()


def test_symbol_list_prices_come_in_one_request(client, monkeypatch):  # noqa: F811
    import dashboard.app as dash
    broker = MagicMock()
    broker.get_current_price.side_effect = AssertionError("one request per symbol")
    broker.get_snapshots.return_value = pd.DataFrame([{"code": "US.AAPL", "last_price": 200.0},
                                                      {"code": "US.MSFT", "last_price": 400.0}])
    monkeypatch.setattr(dash, "get_broker", lambda: broker)
    monkeypatch.setattr(dash, "active_symbols", lambda cfg: {"symbols": [{"symbol": "US.AAPL", "name": "Apple"},
                                                                         {"symbol": "US.MSFT", "name": "MSFT"}]})
    dash._quote_cache.clear()
    data = client.get("/api/symbols", headers=AUTH).get_json()["symbols"]
    assert broker.get_snapshots.call_count == 1
    assert {d["symbol"]: d["latest_price"] for d in data} == {"US.AAPL": 200.0, "US.MSFT": 400.0}


# ---------------------------------------------------------------- IG client

@pytest.fixture
def ig(monkeypatch):
    for k, v in {"IG_API_KEY": "key", "IG_USERNAME": "u", "IG_PASSWORD": "p"}.items():
        monkeypatch.setenv(k, v)
    return IGClient(market="US", session=CountingHttp())


def test_the_same_candles_within_a_minute_come_from_the_cache(ig, monkeypatch):
    fetched = []
    monkeypatch.setattr(ig, "_rest_bars_fetch", lambda s, r, n: fetched.append(s) or pd.DataFrame({"close": [1.0]}))
    for _ in range(6):                                       # the hero card's 10-second refreshes
        ig._rest_bars("CS.D.EURUSD.CSD.IP", "MIN_3", 90)
    assert fetched == ["CS.D.EURUSD.CSD.IP"]


def test_a_full_budget_says_what_used_it(ig, monkeypatch):
    monkeypatch.setenv("IG_REQUESTS_PER_MIN", "4")
    for _ in range(3):
        ig._request("GET", "/markets/CS.D.EURUSD.CSD.IP", 3)
    with pytest.raises(BrokerError, match=r"used by GET /markets/… x3"):
        ig._request("GET", "/positions", 2)
