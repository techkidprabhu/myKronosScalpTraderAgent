"""moomoo "Market Movers" lists as the screener's first stage (US/HK)."""
import numpy as np
import pandas as pd
import pytest

from src.moomoo_client import MoomooClient, RET_OK
from src.screener import Screener
from tests.test_moomoo_client import make_client


# ---------------------------------------------------------------- client

def ranking(codes):
    return RET_OK, (len(codes), pd.DataFrame({"security": codes, "name": codes}))


def test_each_list_calls_the_right_ranking():
    import moomoo as mm
    c = make_client("US")
    c.quote_ctx.get_top_movers_rank.return_value = ranking(["US.NVDA", "US.AMD"])
    c.quote_ctx.get_hot_list.return_value = ranking(["US.TSLA"])
    c.quote_ctx.get_period_change_rank.return_value = ranking(["US.PLTR"])
    c.quote_ctx.get_short_selling_rank.return_value = ranking(["US.GME"])
    assert c.get_market_list("top_gainers", 20) == ["US.NVDA", "US.AMD"]
    kw = c.quote_ctx.get_top_movers_rank.call_args
    assert kw.args == ("US",) and kw.kwargs["count"] == 20 and kw.kwargs["sort_dir"] == mm.RankSortDir.DESCENDING
    c.get_market_list("top_losers", 20)
    assert c.quote_ctx.get_top_movers_rank.call_args.kwargs["sort_dir"] == mm.RankSortDir.ASCENDING
    assert c.get_market_list("hot_list") == ["US.TSLA"]
    assert c.get_market_list("movers_5min") == ["US.PLTR"]
    assert c.quote_ctx.get_period_change_rank.call_args.kwargs["period_type"] == mm.RankPeriodType.FIVE_MIN
    assert c.get_market_list("short_selling", 100) == ["US.GME"]
    assert c.quote_ctx.get_short_selling_rank.call_args.kwargs["count"] == 35       # moomoo's cap


def test_list_errors_are_clear():
    c = make_client("SG")
    with pytest.raises(RuntimeError, match="US and HK only"):
        c.get_market_list("top_gainers")
    c = make_client("US")
    with pytest.raises(ValueError, match="unknown market list"):
        c.get_market_list("tradingview")
    c.quote_ctx.get_hot_list.return_value = (-1, "no quote right")
    with pytest.raises(RuntimeError, match="no quote right"):
        c.get_market_list("hot_list")
    del c.quote_ctx.get_hot_list                                   # older moomoo-api / OpenD
    with pytest.raises(RuntimeError, match="newer moomoo-api"):
        c.get_market_list("hot_list")


# ---------------------------------------------------------------- screener

def bars(last, n=60, step=0.0):
    c = np.linspace(last - step * n, last, n)
    return pd.DataFrame({"time": np.arange(n) * 300_000, "open": c, "high": c * 1.02, "low": c * 0.98,
                         "close": c, "volume": np.full(n, 1e6)})


class ListBroker:
    market, label = "US", "moomoo OpenD"

    def __init__(self, lists, fail=()):
        self.lists, self.fail, self.asked = lists, set(fail), []

    def _normalize_symbol(self, s):
        return s.upper()

    def get_market_list(self, source, count):
        self.asked.append((source, count))
        if source in self.fail:
            raise RuntimeError("no quote right")
        return self.lists.get(source, [])[:count]

    def get_snapshots(self, symbols):
        return pd.DataFrame([{"code": s, "last_price": 50.0, "volume": 5e6} for s in symbols])

    def get_historical_bars(self, symbol, timeframe, n):
        return bars(50.0)

    def list_market_symbols(self):
        raise AssertionError("must not fall back to the whole market")


def screener(broker, **scr):
    cfg = {"timeframe": "MIN_5", "lookback_bars": 60,
           "screener": {"top_n": 10, "min_avg_volume": 0, "min_price": 1, "max_price": 500, **scr}}
    return Screener(broker, cfg)


def test_lists_are_merged_with_the_fixed_list_and_tagged():
    b = ListBroker({"top_gainers": ["US.NVDA", "US.AMD"], "movers_5min": ["US.AMD", "US.PLTR"]})
    s = screener(b, sources=["top_gainers", "movers_5min"], per_source=5, candidate_symbols=["US.AAPL"])
    picked = {c["symbol"]: c for c in s.run()}
    assert set(picked) == {"US.NVDA", "US.AMD", "US.PLTR", "US.AAPL"}
    assert picked["US.AMD"]["sources"] == ["top_gainers", "movers_5min"]
    assert picked["US.AAPL"]["sources"] == ["candidate_symbols"]
    assert b.asked == [("top_gainers", 5), ("movers_5min", 5)]
    assert s.last_report["sources"] == ["top_gainers", "movers_5min"]


def test_a_failing_list_is_skipped():
    b = ListBroker({"top_gainers": ["US.NVDA"]}, fail={"hot_list"})
    s = screener(b, sources=["hot_list", "top_gainers"])
    assert [c["symbol"] for c in s.run()] == ["US.NVDA"]


def test_all_lists_failing_with_no_fixed_list_keeps_current_symbols():
    b = ListBroker({}, fail={"top_gainers"})
    s = screener(b, sources=["top_gainers"])
    assert s.run() == []          # the agent keeps its current symbols on an empty screen


def test_off_by_default_and_ignored_for_brokers_without_lists():
    b = ListBroker({"top_gainers": ["US.NVDA"]})
    s = screener(b, candidate_symbols=["US.AAPL"])
    assert [c["symbol"] for c in s.run()] == ["US.AAPL"] and b.asked == []
    assert "sources" not in s.run()[0]

    class NoLists(ListBroker):
        get_market_list = None
    n = NoLists({})
    n.get_market_list = None
    del n.get_market_list
    s = screener(NoLists({}), sources=["top_gainers"], candidate_symbols=["US.AAPL"])
    assert [c["symbol"] for c in s.run()] == ["US.AAPL"]


def test_settings_validation():
    from dashboard.schema import cross_check, coerce
    assert coerce("screener.sources", "top_gainers, movers_5min") == ["top_gainers", "movers_5min"]
    assert any("Unknown market list" in e for e in cross_check({"screener": {"enabled": True, "sources": ["tradingview"]}}))
    assert any("dynamic screener" in e for e in cross_check({"screener": {"enabled": False, "sources": ["hot_list"]}}))
    assert not cross_check({"screener": {"enabled": True, "sources": ["hot_list", "top_gainers"]}})
