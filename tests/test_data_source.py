"""brokers.<name>.data_source: prices from one broker, orders with another
(Zerodha's free API plan has no market data)."""
import pandas as pd
import pytest

from src.broker_config import broker_specs
from src.brokers.data_source import DataRoutedBroker
from src.brokers.zerodha import ZerodhaClient


class Trade:
    label, market, name, account = "Zerodha", "NSE", "zerodha", "AB1234"
    price_feed = None

    def __init__(self):
        self.orders, self.closed = [], False

    def _normalize_symbol(self, s):
        return "NSE:" + s.split(".")[-1]

    def get_historical_bars(self, *a):
        raise AssertionError("trading broker must not be asked for candles")

    def get_current_price(self, s):
        raise AssertionError("trading broker must not be asked for prices")

    def get_snapshots(self, s):
        raise AssertionError("trading broker must not be asked for snapshots")

    def list_market_symbols(self):
        return ["NSE.RELIANCE", "NSE.TCS"]

    def create_entry(self, symbol, direction, quantity, **kw):
        self.orders.append((symbol, direction, quantity))
        return "order-1"

    def get_open_positions(self):
        return [{"symbol": "NSE.TCS"}]

    def ping(self):
        return True, "connected"

    def close(self):
        self.closed = True


class Data:
    label, market = "Angel One", "NSE"

    def __init__(self):
        self.asked, self.ok, self.closed = [], True, False

    def get_historical_bars(self, symbol, resolution, n):
        self.asked.append(("bars", symbol))
        return pd.DataFrame({"time": [1], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]})

    def get_current_price(self, symbol):
        self.asked.append(("price", symbol))
        return 2950.5

    def get_snapshots(self, symbols):
        self.asked.append(("snap", tuple(symbols)))
        return pd.DataFrame([{"code": s + "-EQ", "last_price": 10.0, "volume": 5, "prev_close_price": 9.0}
                             for s in symbols] + [{"code": "NSE.OTHER", "last_price": 1.0, "volume": 1,
                                                   "prev_close_price": 1.0}])

    def ping(self):
        return (True, "connected") if self.ok else (False, "session expired")

    def close(self):
        self.closed = True


def routed():
    t, d = Trade(), Data()
    return DataRoutedBroker(t, d), t, d


def test_prices_come_from_the_data_broker_in_canonical_codes():
    b, t, d = routed()
    assert b.get_current_price("NSE:RELIANCE") == 2950.5
    assert len(b.get_historical_bars("NSE.RELIANCE", "DAY", 10)) == 1
    snaps = b.get_snapshots(["NSE:RELIANCE", "NSE.TCS"])
    assert list(snaps["code"]) == ["NSE.RELIANCE", "NSE.TCS"]         # -EQ stripped, extras dropped
    assert d.asked[0] == ("price", "NSE.RELIANCE") and d.asked[1] == ("bars", "NSE.RELIANCE")


def test_orders_account_and_attributes_stay_with_the_trading_broker():
    b, t, d = routed()
    assert b.create_entry("NSE.TCS", "BUY", 5) == "order-1" and t.orders == [("NSE.TCS", "BUY", 5)]
    assert b.get_open_positions() == [{"symbol": "NSE.TCS"}]
    assert (b.label, b.market, b.account) == ("Zerodha", "NSE", "AB1234")
    assert b.list_market_symbols() == ["NSE.RELIANCE", "NSE.TCS"]


def test_trading_adapter_gets_the_price_feed():
    b, t, d = routed()
    assert t.price_feed("NSE.TCS") == 2950.5


def test_ping_needs_both_and_close_closes_both():
    b, t, d = routed()
    assert b.ping() == (True, "connected · prices from Angel One")
    d.ok = False
    ok, detail = b.ping()
    assert not ok and "Angel One" in detail and "session expired" in detail
    b.close()
    assert t.closed and d.closed


def test_zerodha_uses_the_price_feed_instead_of_kite():
    z = ZerodhaClient.__new__(ZerodhaClient)        # no Kite session needed
    z.price_feed = lambda s: 123.45
    assert z.get_current_price("NSE.INFY") == 123.45


def test_config_resolves_the_data_source():
    cfg = {"timeframe": "DAY", "brokers": {
        "zerodha": {"enabled": True, "market": "NSE", "data_source": "angel"},
        "angel": {"enabled": False},
        "moomoo": {"enabled": True, "data_source": "moomoo"}}}
    specs = {s.name: s for s in broker_specs(cfg)}
    assert specs["zerodha"].data_source == "angel"
    assert specs["moomoo"].data_source is None            # itself = no routing
    assert "data_source" not in specs["zerodha"].config   # not a trading setting
    with pytest.raises(ValueError, match="data_source"):
        broker_specs({"brokers": {"zerodha": {"data_source": "tradingview"}}})


def test_connect_wraps_and_reports_a_failed_data_source(monkeypatch):
    import src.brokers as brokers
    made = {}

    def fake(adapter, timeout=35.0, **kw):
        if adapter == "angel" and made.get("fail"):
            raise ConnectionError("TOTP rejected")
        made[adapter] = Trade() if adapter == "zerodha" else Data()
        return made[adapter]

    monkeypatch.setattr(brokers, "connect_with_timeout", fake)
    b = brokers.connect("zerodha", 5, data_source="angel", market="NSE")
    assert isinstance(b, DataRoutedBroker) and b.get_current_price("NSE.TCS") == 2950.5
    assert brokers.connect("zerodha", 5) is made["zerodha"]           # no data source = plain broker
    made["fail"] = True
    with pytest.raises(brokers.BrokerError, match="Price data source angel"):
        brokers.connect("zerodha", 5, data_source="angel")
    assert made["zerodha"].closed                                    # trading connection not left open
