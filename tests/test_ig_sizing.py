"""IG sizes trades in contracts, not units: one forex contract is 100,000 of
the base currency. Sizing, P&L and order currency must use IG's own figures."""
import pytest

import src.brokers.ig as igmod
import src.orchestrator as orch
from src.brokers.base import BrokerError
from src.brokers.ig import IGClient
from src.risk import RiskConfig, RiskManager, open_risk_at_stops
from tests.test_ig_allowance import CountingHttp, FakeResp
from tests.test_orchestrator import SYMBOL, FakeBroker, engulfing_bars, make_worker

FX = {
    "CS.D.USDCAD.CSD.IP": {"valueOfOnePip": "10.00", "onePipMeans": "0.0001 CAD/USD", "contractSize": "100000",
                           "ccy": "CAD", "min": 0.05},
    "CS.D.EURUSD.CSD.IP": {"valueOfOnePip": "10.00", "onePipMeans": "0.0001 USD/EUR", "contractSize": "100000",
                           "ccy": "USD", "min": 0.05},
    "CS.D.GBPJPY.CSD.IP": {"valueOfOnePip": "1000", "onePipMeans": "0.01 JPY/GBP", "contractSize": "100000",
                           "ccy": "JPY", "min": 0.05},
    "IX.D.SPTRD.IGT.IP": {"valueOfOnePip": "1.00", "onePipMeans": "1 Index Point", "contractSize": "1",
                          "ccy": "USD", "min": 0.1, "type": "INDICES"},
}
QUOTES = {"CS.D.USDJPY.CSD.IP": (150.0, 150.02)}


class IGMarketsHttp(CountingHttp):
    def request(self, method, url, headers=None, json=None, params=None, timeout=None):
        path = url.split("/gateway/deal", 1)[1]
        if path == "/accounts":
            return FakeResp(200, {"accounts": [{"accountId": "ACC", "currency": "USD",
                                                "balance": {"balance": 10000, "available": 9000}}]})
        if path.startswith("/markets/"):
            epic = path.split("/markets/", 1)[1]
            self.calls.append((path, params))
            m = FX.get(epic)
            if not m:
                return FakeResp(404, {"errorCode": "error.service.marketdata.instrument.epic.unavailable"})
            return FakeResp(200, {"instrument": {
                "epic": epic, "type": m.get("type", "CURRENCIES"), "valueOfOnePip": m["valueOfOnePip"],
                "onePipMeans": m["onePipMeans"], "contractSize": m["contractSize"], "unit": "CONTRACTS",
                "currencies": [{"code": m["ccy"], "isDefault": True}]},
                "dealingRules": {"minDealSize": {"unit": "POINTS", "value": m["min"]}}, "snapshot": {}})
        if path == "/markets" and params and "epics" in params:
            self.calls.append((path, params))
            return FakeResp(200, {"marketDetails": [
                {"instrument": {"epic": e}, "snapshot": {"bid": QUOTES[e][0], "offer": QUOTES[e][1]}}
                for e in params["epics"].split(",") if e in QUOTES]})
        return super().request(method, url, headers, json, params, timeout)


@pytest.fixture
def ig(monkeypatch):
    for k, v in {"IG_API_KEY": "key", "IG_USERNAME": "u", "IG_PASSWORD": "p", "IG_CURRENCY": "USD"}.items():
        monkeypatch.setenv(k, v)
    return IGClient(market="IG24", session=IGMarketsHttp())


# ---------------------------------------------------------------- what one contract is worth

def test_usdcad_contract_is_worth_100000_cad_per_point_in_usd(ig):
    s = ig.instrument_spec("CS.D.USDCAD.CSD.IP", price=1.4335)
    assert s["per_point_instrument"] == pytest.approx(100000)
    assert s["value_per_point"] == pytest.approx(100000 / 1.4335)          # CAD -> USD at the pair itself
    assert s["min_size"] == 0.05 and s["size_step"] == 0.01 and s["currency_code"] == "CAD"


def test_usd_quoted_pair_needs_no_conversion(ig):
    assert ig.instrument_spec("CS.D.EURUSD.CSD.IP")["value_per_point"] == pytest.approx(100000)


def test_cross_pairs_convert_with_igs_own_fx_price(ig):
    s = ig.instrument_spec("CS.D.GBPJPY.CSD.IP", price=190.0)
    assert s["value_per_point"] == pytest.approx(100000 / 150.01, rel=1e-4)   # JPY -> USD via USD/JPY


def test_index_points(ig):
    assert ig.instrument_spec("IX.D.SPTRD.IGT.IP")["value_per_point"] == pytest.approx(1.0)


def test_unknown_market_or_conversion_is_an_error_not_a_guess(ig, monkeypatch):
    with pytest.raises(BrokerError):
        ig.instrument_spec("CS.D.NOPE.CSD.IP")
    QUOTES.pop("CS.D.USDJPY.CSD.IP")
    try:
        ig._fx.clear()
        with pytest.raises(BrokerError, match="convert JPY"):
            ig.instrument_spec("CS.D.GBPJPY.CSD.IP", price=190.0)
    finally:
        QUOTES["CS.D.USDJPY.CSD.IP"] = (150.0, 150.02)


def test_details_are_fetched_once(ig):
    for _ in range(3):
        ig.instrument_spec("CS.D.EURUSD.CSD.IP")
    assert sum(1 for p, _ in ig.http.calls if p == "/markets/CS.D.EURUSD.CSD.IP") == 1


def test_orders_use_the_markets_own_currency_code(ig, monkeypatch):
    bodies = []
    orig = ig.http.request

    def spy(method, url, headers=None, json=None, params=None, timeout=None):
        if "/workingorders/otc" in url:
            bodies.append(json)
        return orig(method, url, headers, json, params, timeout)
    monkeypatch.setattr(ig.http, "request", spy)
    ig._working("CS.D.USDCAD.CSD.IP", "SELL", 0.28, 1.4335, "LIMIT")
    assert bodies[0]["currencyCode"] == "CAD" and bodies[0]["size"] == 0.28


def test_rejections_say_what_was_sent(ig, monkeypatch):
    def rejected(method, url, headers=None, json=None, params=None, timeout=None):
        path = url.split("/gateway/deal", 1)[1]
        if path.startswith("/confirms/"):
            return FakeResp(200, {"dealStatus": "REJECTED", "reason": "UNKNOWN"})
        return IGMarketsHttp.request(ig.http, method, url, headers, json, params, timeout)
    monkeypatch.setattr(ig.http, "request", rejected)
    with pytest.raises(BrokerError, match=r"UNKNOWN \(sent SELL size 2790 of CS.D.USDCAD.CSD.IP at 1.4335, currency CAD\)"):
        ig._working("CS.D.USDCAD.CSD.IP", "SELL", 2790, 1.4335, "LIMIT")


# ---------------------------------------------------------------- sizing

def spec(vpp=100000 / 1.4335, min_size=0.05):
    return {"value_per_point": vpp, "min_size": min_size, "size_step": 0.01, "account_currency": "USD",
            "unit": "CONTRACTS"}


def risk(**kw):
    return RiskManager(RiskConfig(max_open_positions=3, stop_loss_points=0.5, take_profit_points=1.0,
                                  max_daily_loss=300.0, min_minutes_between_trades=0,
                                  max_loss_per_trade=20, stop_loss_pct=0.5, take_profit_pct=1.0,
                                  max_stake_per_point=10000, **kw))


def test_the_trade_that_failed_is_now_refused_with_the_reason():
    plan = risk().plan_entry(1.433525, 1, spec=spec())                     # 0.5% stop = ~72 pips
    assert not plan.allowed and "under the minimum 0.05" in plan.reason
    assert "would risk 25.0" in plan.reason                                  # 0.05 x 100,000 x 0.00717 / 1.4335


def test_a_pip_sized_stop_gets_a_fractional_contract_size():
    plan = risk().plan_entry(1.4335, 1, spec=spec(), stop_distance=0.0010, target_distance=0.0015)
    assert plan.allowed and plan.quantity == 0.28                            # 20 / (10 pips x ~69.76 USD)
    assert plan.max_loss == pytest.approx(0.28 * 0.0010 * 100000 / 1.4335) and plan.max_loss <= 20


def test_shares_are_sized_exactly_as_before():
    plan = risk().plan_entry(100.0, 1)
    assert plan.allowed and plan.quantity == 40 and plan.max_loss == pytest.approx(20)   # 20 / 0.5


def test_open_risk_counts_contract_value():
    pos = [{"symbol": "CS.D.EURUSD.CSD.IP", "direction": "BUY", "entry_price": 1.10, "current_stop": 1.099,
            "quantity": 0.2}]
    assert open_risk_at_stops(pos) == pytest.approx(0.0002)                  # the old, per-unit figure
    assert open_risk_at_stops(pos, value_per_point=lambda s: 100000) == pytest.approx(20)


# ---------------------------------------------------------------- the agent end to end

class ContractBroker(FakeBroker):
    """Like IG: size in contracts worth 100 a point."""
    VPP = 100.0

    def instrument_spec(self, symbol, price=None):
        return {"value_per_point": self.VPP, "min_size": 0.05, "size_step": 0.01, "account_currency": "USD",
                "unit": "CONTRACTS"}


@pytest.fixture
def market_open(monkeypatch):
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)


def test_agent_sends_a_contract_size_and_records_pnl_in_money(tmp_path, market_open):
    broker = ContractBroker()
    broker.bars = engulfing_bars()
    worker, audit, rm = make_worker(tmp_path, broker)
    rm.config.max_loss_per_trade = 20                    # stop 0.5 points x 100 a point = 50 per contract
    worker.poll_once()
    assert broker.kinds() == ["entry"] and broker.placed[0][1] == 0.4      # 20 / 50, not 40 "units"
    pos = audit.get_active_position(SYMBOL)
    assert pos["quantity"] == 0.4
    entry_id = int(pos["entry_order_id"])
    broker.fill(entry_id, qty=0.4, price=71.7)
    broker.held = 0.4
    worker.monitor_position()
    pos = audit.get_active_position(SYMBOL)
    worker._finish_exit(pos, "test", [(0.4, pos["entry_price"] - 0.5)])     # stopped out
    assert rm.state.daily_pnl == pytest.approx(-20)                         # money, not -0.2


def test_no_contract_details_means_no_trade(tmp_path, market_open):
    class Broken(ContractBroker):
        def instrument_spec(self, symbol, price=None):
            raise BrokerError("IG gave no contract size")
    broker = Broken()
    broker.bars = engulfing_bars()
    worker, audit, rm = make_worker(tmp_path, broker)
    rm.config.max_loss_per_trade = 20
    worker.poll_once()
    assert broker.kinds() == []
    row = audit.conn.execute("SELECT risk_reason FROM decisions ORDER BY ts DESC").fetchone()
    assert "contract details unavailable" in row[0]


# ---------------------------------------------------------------- dashboard

from tests.test_dashboard_multibroker import AUTH, client  # noqa: E402,F401 - fixture


def test_manual_orders_take_fractions_only_on_ig(client, monkeypatch):  # noqa: F811
    import dataclasses
    import dashboard.app as dash
    real = dash.current_spec

    def as_adapter(name):
        return lambda raw=None: dataclasses.replace(real(raw), adapter=name)
    monkeypatch.setattr(dash, "current_spec", as_adapter("moomoo"))
    r = client.post("/api/manual_order", json={"symbol": "US.AAPL", "action": "BUY", "quantity": 0.5}, headers=AUTH)
    assert r.status_code == 400 and "whole number" in r.get_json()["error"]
    monkeypatch.setattr(dash, "current_spec", as_adapter("ig"))
    r = client.post("/api/manual_order", json={"symbol": "CS.D.EURUSD.CSD.IP", "action": "BUY", "quantity": 0.05},
                    headers=AUTH)
    assert r.status_code == 200, r.get_json()
