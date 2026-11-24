"""IG fixes from the 24 Nov audit: skill entries priced at the market go as
market orders (IG refused limit orders sitting on the price for indices and
oil: ATTACHED_ORDER_LEVEL_ERROR), currencies without a direct IG pair convert
through USD (CHF into SGD), and Kronos rejections say what they mean."""
import pytest

import src.orchestrator as orch
from src import skill_strategy, trading_mode
from src.brokers.ig import IGClient
from src.skill_strategy import Setup
from tests.test_ig_sizing import QUOTES, IGMarketsHttp
from tests.test_orchestrator import SYMBOL, FakeBroker, make_worker
from tests.test_skill_strategy import SETUP, broker_frame, skill_on  # noqa: F401 - fixture


# ---------------------------------------------------------------- wording

def test_kronos_disagreement_is_said_plainly():
    r = orch.kronos_reason(False, 0.0, "BUY", 0.6)
    assert r.startswith("kronos disagrees with BUY: 0% of its forecast moves up")
    assert "needs 0.6" in r
    assert "moves down" in orch.kronos_reason(False, 0.2, "SELL", 0.6)
    assert orch.kronos_reason(True, 0.55, "BUY", 0.6).startswith("kronos agrees with BUY but only weakly")


def test_at_market_uses_the_instruments_rounding():
    assert orch.at_market(8121.700000000001, 8121.7, lambda p: round(p, 1))
    assert not orch.at_market(104, 108.3, lambda p: round(p, 2))
    assert not orch.at_market(None, 1.0)


# ---------------------------------------------------------------- skill entries at the market

class IGLikeBroker(FakeBroker):
    MARKET_FOR_AT_PRICE_ENTRIES = True


def _at_market_setup(df, settings):
    last = float(df["close"].iloc[-1])
    return Setup("ready", "VWAP pullback long", entry=last, stop=last - 2, target=last + 4, quality=0,
                 direction="BUY", details={"setup": "VWAP pullback", "breakout": str(df.index[-1])})


def test_ig_skill_entry_at_the_price_is_a_market_order(tmp_path, skill_on, monkeypatch):
    monkeypatch.setattr(skill_strategy, "find_setup", _at_market_setup)
    broker = IGLikeBroker()
    broker.bars = broker_frame(SETUP)
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    assert broker.kinds() == ["entry"] and broker.placed[0][2]["limit"] is None     # market
    pos = audit.get_active_position(SYMBOL)
    assert pos["initial_stop"] == pytest.approx(106.3) and pos["target"] == pytest.approx(112.3)


def test_other_brokers_keep_the_limit_at_the_price(tmp_path, skill_on, monkeypatch):
    monkeypatch.setattr(skill_strategy, "find_setup", _at_market_setup)
    broker = FakeBroker()
    broker.bars = broker_frame(SETUP)
    worker, _, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    assert broker.placed[0][2]["limit"] == 108.3


def test_a_skill_price_away_from_the_market_stays_a_limit_order_on_ig(tmp_path, skill_on):
    broker = IGLikeBroker()
    broker.bars = broker_frame(SETUP)
    worker, _, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    assert broker.kinds() == ["entry"] and broker.placed[0][2]["limit"] == 104      # the Praga retest price


def test_ig_client_asks_for_market_entries_at_the_price():
    assert IGClient.MARKET_FOR_AT_PRICE_ENTRIES is True


# ---------------------------------------------------------------- CHF into SGD

@pytest.fixture
def ig_sgd(monkeypatch):
    for k, v in {"IG_API_KEY": "key", "IG_USERNAME": "u", "IG_PASSWORD": "p", "IG_CURRENCY": "SGD"}.items():
        monkeypatch.setenv(k, v)
    extra = {"CS.D.USDCHF.CSD.IP": (0.80, 0.80), "CS.D.USDSGD.CSD.IP": (1.30, 1.30)}
    QUOTES.update(extra)
    yield IGClient(market="IG24", session=IGMarketsHttp())
    for k in extra:
        QUOTES.pop(k, None)


def test_chf_converts_into_sgd_through_usd(ig_sgd):
    # 1 CHF = 1/0.80 USD = 1.25 USD = 1.625 SGD
    assert ig_sgd.fx_rate("CHF", "SGD") == pytest.approx(1.25 * 1.30)
    assert ig_sgd.fx_rate("SGD", "CHF") == pytest.approx(1 / (1.25 * 1.30))


def test_no_route_at_all_is_still_an_error(ig_sgd):
    from src.brokers.base import BrokerError
    with pytest.raises(BrokerError, match="convert XYZ"):
        ig_sgd.fx_rate("XYZ", "SGD")
