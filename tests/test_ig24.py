"""Around-the-clock IG: market hours, streamed candles, discovery, the 24-hour screener, agent wiring."""
import time
from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src import ig_hours, ig_screener
from src.brokers import ig_stream
from tests.test_brokers import FakeResp

ET = ZoneInfo("America/New_York")


def et(s):
    return datetime.fromisoformat(s).replace(tzinfo=ET)


# ------------------------------------------------------------------ hours

def test_hours_weekend_and_daily_break():
    fri_before, fri_after = et("2026-11-20 16:50"), et("2026-11-20 17:10")
    assert ig_hours.session_minutes("forex", fri_before)[1] == 10
    assert not ig_hours.is_open("forex", fri_after) and not ig_hours.is_open("indices", fri_after)
    assert ig_hours.is_open("crypto", fri_after)
    sun = et("2026-11-22 17:30")
    assert ig_hours.is_open("forex", sun) and not ig_hours.is_open("indices", sun)
    assert ig_hours.next_open("indices", sun) == et("2026-11-22 18:00")
    tue_break = et("2026-11-24 17:30")
    assert ig_hours.is_open("forex", tue_break) and not ig_hours.is_open("commodities", tue_break)
    assert ig_hours.session_minutes("indices", et("2026-11-24 12:00"))[1] == 300
    assert ig_hours.session_minutes("crypto", fri_after) == (float("inf"), float("inf"))


def test_classify():
    assert ig_hours.classify("CURRENCIES", "CS.D.EURUSD.CFD.IP") == "forex"
    assert ig_hours.classify("INDICES", "IX.D.DAX.IFD.IP") == "indices"
    assert ig_hours.classify("CURRENCIES", "CS.D.BITCOIN.CFD.IP") == "crypto"
    assert ig_hours.classify("SHARES", "X", hint="commodities") == "commodities"


# ------------------------------------------------------------------ candle store + stream

MIN = 60_000


def test_store_builds_3_minute_candles_and_keeps_stream_buckets(tmp_path):
    st = ig_stream.CandleStore("MIN_3", tmp_path)
    b = 1_795_000_020_000 - 1_795_000_020_000 % (3 * MIN)
    st.on_minute("E", b, 10, 11, 9.5, 10.5, 3)
    st.on_minute("E", b + MIN, 10.5, 12, 10.4, 11.8, 4)
    st.on_minute("E", b + 2 * MIN, 11.8, 11.9, 10.9, 11.0, 2)
    st.on_minute("E", b + 2 * MIN, 11.8, 12.5, 10.9, 12.2, 5)          # same minute, updated
    row = st.frame("E", 5).iloc[-1]
    assert (row.open, row.high, row.low, row.close, row.volume) == (10, 12.5, 9.5, 12.2, 12)
    hist = pd.DataFrame([[b - 3 * MIN, 9, 9.5, 8.5, 9.2, 1], [b, 1, 1, 1, 1, 1]],
                        columns=["time", "open", "high", "low", "close", "volume"])
    st.merge_history("E", hist)
    f = st.frame("E", 5)
    assert f["time"].tolist() == [b - 3 * MIN, b] and f.iloc[-1].close == 12.2     # streamed bucket kept
    st.save(force=True)
    again = ig_stream.CandleStore("MIN_3", tmp_path)
    assert again.load("E") == 2
    assert ig_stream.stored_frame("E", "MIN_3", 10, root=tmp_path)["close"].tolist() == [9.2, 12.2]


class FakeSub:
    def __init__(self, mode, items, fields):
        self.items, self.listeners = items, []

    def addListener(self, listener):
        self.listeners.append(listener)


class FakeLS:
    """Stands in for lightstreamer.client.LightstreamerClient."""
    subscription = FakeSub

    def __init__(self, endpoint, adapter):
        self.endpoint, self.adapter, self.subs, self.listeners = endpoint, adapter, [], []
        self.connectionDetails = MagicMock()

    def addListener(self, listener):
        self.listeners.append(listener)

    def connect(self):
        for listener in self.listeners:
            listener.onStatusChange("CONNECTED:WS-STREAMING")

    def disconnect(self):
        pass

    def subscribe(self, sub):
        self.subs.append(sub)

    def unsubscribe(self, sub):
        self.subs.remove(sub)


class Update:
    def __init__(self, **kw):
        self.kw = kw

    def getValue(self, field):
        return self.kw.get(field)


def make_stream(tmp_path, made):
    def factory(endpoint, adapter):
        made.append(FakeLS(endpoint, adapter))
        return made[-1]
    factory.subscription = FakeSub
    st = ig_stream.CandleStore("MIN_3", tmp_path)
    return ig_stream.IGStream("https://ls", "ACC", "cst", "xst", st, client_factory=factory, max_subscriptions=2), st


def test_stream_subscribes_caps_and_builds_candles(tmp_path):
    made = []
    s, st = make_stream(tmp_path, made)
    s.start()
    made[0].connectionDetails.setPassword.assert_called_once_with("CST-cst|XST-xst")
    made[0].connectionDetails.setUser.assert_called_once_with("ACC")
    assert s.connected and made[0].adapter == "DEFAULT"
    assert s.set_epics(["A", "B", "C"]) == ["A", "B"]
    assert [x.items for x in made[0].subs] == [["CHART:A:1MINUTE"], ["CHART:B:1MINUTE"]]
    s.set_epics(["B"])
    assert [x.items for x in made[0].subs] == [["CHART:B:1MINUTE"]]
    utm = 1_795_000_020_000 + 25_000
    made[0].subs[0].listeners[0].onItemUpdate(Update(
        BID_OPEN="1.0990", BID_HIGH="1.1010", BID_LOW="1.0985", BID_CLOSE="1.1000",
        OFR_OPEN="1.0992", OFR_HIGH="1.1012", OFR_LOW="1.0987", OFR_CLOSE="1.1002",
        CONS_TICK_COUNT="17", CONS_END="0", UTM=str(utm)))
    row = st.frame("B", 1).iloc[-1]
    assert round(row.close, 4) == 1.1001 and row.volume == 17
    assert row.time == (utm - utm % MIN) - (utm - utm % MIN) % (3 * MIN)
    assert round(s.price("B"), 4) == 1.1001 and s.price("A") is None


# ------------------------------------------------------------------ IG adapter

NAV = {
    "/marketnavigation": {"nodes": [{"id": "1", "name": "Forex"}, {"id": "2", "name": "Indices"},
                                    {"id": "3", "name": "Shares - US"}, {"id": "4", "name": "Weekend Markets"},
                                    {"id": "5", "name": "Cryptocurrency"}]},
    "/marketnavigation/1": {"nodes": [{"id": "11", "name": "Majors"}, {"id": "12", "name": "Options on FX"}], "markets": []},
    "/marketnavigation/11": {"markets": [
        {"epic": "CS.D.EURUSD.CFD.IP", "instrumentName": "EUR/USD", "instrumentType": "CURRENCIES", "expiry": "-"},
        {"epic": "CS.D.EURUSD.TODAY.IP", "instrumentName": "EUR/USD DFB", "instrumentType": "CURRENCIES", "expiry": "DFB"},
        {"epic": "CS.D.EURUSD.BIN.IP", "instrumentName": "Binary", "instrumentType": "BINARY", "expiry": "-"}]},
    "/marketnavigation/2": {"markets": [
        {"epic": "IX.D.DAX.IFD.IP", "instrumentName": "Germany 40", "instrumentType": "INDICES", "expiry": "-"},
        {"epic": "IX.D.DAX.MAR.IP", "instrumentName": "Germany 40 Mar", "instrumentType": "INDICES", "expiry": "MAR-27"}]},
    "/marketnavigation/5": {"markets": [
        {"epic": "CS.D.BITCOIN.CFD.IP", "instrumentName": "Bitcoin", "instrumentType": "CURRENCIES", "expiry": "-"}]},
}

SNAPS = {
    "CS.D.EURUSD.CFD.IP": {"marketStatus": "TRADEABLE", "bid": 1.1000, "offer": 1.10008, "high": 1.1050, "low": 1.0950, "percentageChange": 0.3},
    "IX.D.DAX.IFD.IP": {"marketStatus": "TRADEABLE", "bid": 20000, "offer": 20002, "high": 20150, "low": 19900, "percentageChange": -0.8},
    "CS.D.BITCOIN.CFD.IP": {"marketStatus": "TRADEABLE", "bid": 90000, "offer": 90900, "high": 92000, "low": 88000, "percentageChange": 2.0},
    "CS.D.GBPUSD.CFD.IP": {"marketStatus": "EDITS_ONLY", "bid": 1.30, "offer": 1.30008, "high": 1.31, "low": 1.29, "percentageChange": 0.1},
}


class FakeHttp:
    def __init__(self):
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        return FakeResp(200, {"currentAccountId": "ACC", "lightstreamerEndpoint": "https://ls.example"},
                        {"CST": "c", "X-SECURITY-TOKEN": "x"})

    def request(self, method, url, headers=None, json=None, params=None, timeout=None):
        path = url.split("/gateway/deal", 1)[1]
        self.calls.append((path, params))
        if path in NAV:
            return FakeResp(200, NAV[path])
        if path == "/markets":
            epics = params["epics"].split(",")
            return FakeResp(200, {"marketDetails": [{"instrument": {"epic": e}, "snapshot": SNAPS[e]}
                                                    for e in epics if e in SNAPS]})
        if path.startswith("/prices/"):
            n = int(params["max"])
            now = int(time.time() * 1000) - int(time.time() * 1000) % (3 * MIN)
            rows = [{"snapshotTimeUTC": pd.Timestamp(now - (n - i) * 3 * MIN, unit="ms").isoformat(),
                     **{k: {"bid": 1.0 + i, "ask": 1.0 + i} for k in ("openPrice", "highPrice", "lowPrice", "closePrice")},
                     "lastTradedVolume": 5} for i in range(n)]
            return FakeResp(200, {"prices": rows, "metadata": {"allowance": {
                "remainingAllowance": 9000, "totalAllowance": 10000, "allowanceExpiry": 3600}}})
        return FakeResp(200, {})


@pytest.fixture
def ig(monkeypatch, tmp_path):
    from src.brokers.ig import IGClient
    for k, v in {"IG_API_KEY": "key", "IG_USERNAME": "u", "IG_PASSWORD": "p"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(ig_stream, "CANDLE_DIR", tmp_path / "candles")
    client = IGClient(market="IG24", session=FakeHttp())
    assert client.ls_endpoint == "https://ls.example"
    return client


def test_discovery_keeps_cfds_of_chosen_classes(ig):
    found = ig.discover_markets(["forex", "indices", "crypto"], delay_s=0)
    assert {m["epic"]: m["class"] for m in found} == {
        "CS.D.EURUSD.CFD.IP": "forex", "IX.D.DAX.IFD.IP": "indices", "CS.D.BITCOIN.CFD.IP": "crypto"}
    paths = [c[0] for c in ig.http.calls]
    assert "/marketnavigation/3" not in paths and "/marketnavigation/4" not in paths    # shares, weekend
    assert "/marketnavigation/12" not in paths                                          # options


def test_watch_seeds_once_then_serves_candles_without_history(ig):
    made = []

    def factory(endpoint, adapter):
        made.append(FakeLS(endpoint, adapter))
        return made[-1]
    factory.subscription = FakeSub
    ig.enable_around_the_clock({"seed_bars": 20}, "MIN_3", client_factory=factory)
    assert ig.stream.connected
    ig.watch(["CS.D.EURUSD.CFD.IP"])
    prices = [c for c in ig.http.calls if c[0].startswith("/prices/")]
    assert len(prices) == 1 and prices[0][1]["max"] == 20 and prices[0][1]["resolution"] == "MINUTE_3"
    assert ig.allowance["remaining"] == 9000
    df = ig.get_historical_bars("CS.D.EURUSD.CFD.IP", "MIN_3", 10)
    assert len(df) == 10
    assert len([c for c in ig.http.calls if c[0].startswith("/prices/")]) == 1        # no new history call
    ig.watch(["CS.D.EURUSD.CFD.IP"])                                                     # already watched
    assert len([c for c in ig.http.calls if c[0].startswith("/prices/")]) == 1
    assert [s.items for s in made[0].subs] == [["CHART:CS.D.EURUSD.CFD.IP:1MINUTE"]]


def test_symbol_open_uses_hours_and_live_status(ig):
    ig.enable_around_the_clock({}, "MIN_3", stream=False)
    ig.classes["CS.D.GBPUSD.CFD.IP"] = "forex"
    ig.market_snapshots(["CS.D.GBPUSD.CFD.IP", "CS.D.EURUSD.CFD.IP"])
    weekday = et("2026-11-24 12:00")
    assert not ig.symbol_open("CS.D.GBPUSD.CFD.IP", weekday)       # EDITS_ONLY
    assert ig.symbol_open("CS.D.EURUSD.CFD.IP", weekday)
    assert not ig.symbol_open("CS.D.EURUSD.CFD.IP", et("2026-11-21 12:00"))   # Saturday


# ------------------------------------------------------------------ screener

def test_screener_ranks_filters_and_streams(ig, tmp_path):
    ig.enable_around_the_clock({}, "MIN_3", stream=False)
    cfg = {"timeframe": "MIN_3", "screener": {"enabled": True, "around_the_clock": {
        "markets": ["forex", "indices", "crypto"], "max_watch": 2, "max_spread_pct": 0.2,
        "crypto_max_spread_pct": 0.5}}}
    scr = ig_screener.IG24Screener(ig, cfg, universe_file=tmp_path / "u.json", background=False)
    scr.universe = [{"epic": e, "name": e, "class": ig_hours.classify(None, e)} for e in SNAPS]
    scr.universe_updated = time.time()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ig_hours, "is_open", lambda cls, now=None: True)
        picked = scr.run()
    rep = {c["symbol"]: c for c in scr.last_report["candidates"]}
    assert rep["CS.D.BITCOIN.CFD.IP"]["excluded"] == "spread too wide"           # 1% spread
    assert rep["CS.D.GBPUSD.CFD.IP"]["excluded"] == "not tradeable now"
    assert {p["symbol"] for p in picked} == {"CS.D.EURUSD.CFD.IP", "IX.D.DAX.IFD.IP"}
    assert ig.watched == [p["symbol"] for p in picked]
    assert scr.last_report["mode"] == "ig24" and rep["CS.D.EURUSD.CFD.IP"]["picked"]


def test_screener_discovers_and_caches_universe(ig, tmp_path):
    ig.enable_around_the_clock({}, "MIN_3", stream=False)
    cfg = {"timeframe": "MIN_3", "screener": {"around_the_clock": {"markets": ["forex", "indices"], "max_nav_nodes": 10}}}
    import src.brokers.ig as igmod
    orig = igmod.IGClient.discover_markets
    ig.discover_markets = lambda classes, max_nodes, progress=None: orig(ig, classes, max_nodes, delay_s=0)
    scr = ig_screener.IG24Screener(ig, cfg, universe_file=tmp_path / "u.json", background=False)
    scr.run()
    assert {m["epic"] for m in scr.universe} == {"CS.D.EURUSD.CFD.IP", "IX.D.DAX.IFD.IP"}
    again = ig_screener.IG24Screener(ig, cfg, universe_file=tmp_path / "u.json", background=False)
    assert len(again.universe) == 2                                                  # read from the cache


def test_settings_cap_at_40():
    s = ig_screener.settings({"screener": {"around_the_clock": {"max_watch": 99, "markets": ["forex", "shares"]}}})
    assert s["max_watch"] == 40 and s["markets"] == ["forex"]


# ------------------------------------------------------------------ agent wiring

def test_phase_and_flat_before_each_close(monkeypatch):
    from src.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o.follow_market, o.around_the_clock, o.warmup_minutes = True, True, 15
    o.broker = MagicMock()
    o.broker.any_open.return_value = False
    o.broker.next_open.return_value = et("2026-11-22 17:00")
    phase, nxt = o.compute_phase(et("2026-11-22 16:50"))
    assert phase == "warmup" and nxt == et("2026-11-22 17:00").timestamp()
    o.broker.any_open.return_value = True
    assert o.compute_phase(et("2026-11-23 10:00"))[0] == "active"

    o.risk_manager = MagicMock()
    o.risk_manager.config.flat_by_minutes_before_close = 10
    o.audit_log = MagicMock()
    o.audit_log.get_active_positions.return_value = [
        {"symbol": "IDX", "status": "open", "direction": "BUY", "quantity": 1},
        {"symbol": "FX", "status": "open", "direction": "BUY", "quantity": 1}]
    o.broker.session_minutes_for.side_effect = lambda s: (300.0, 8.0) if s == "IDX" else (300.0, 4000.0)
    o.broker.get_current_price.return_value = 100.0
    w = MagicMock()
    o._worker_for = lambda s: w
    o.flat_by_close()
    assert w._software_exit.call_count == 1 and w._software_exit.call_args[0][0]["symbol"] == "IDX"


def test_worker_uses_per_market_hours(tmp_path, monkeypatch):
    from tests.test_orchestrator import FakeBroker, make_worker
    b = FakeBroker()
    b.market = "IG24"
    b.symbol_open = lambda s: False
    b.session_minutes_for = lambda s: (30.0, 600.0)
    worker, _, _ = make_worker(tmp_path, b)
    assert worker._market_open() is False and worker._session_minutes() == (30.0, 600.0)


def test_crypto_has_its_own_spread_limit(ig, tmp_path):
    ig.enable_around_the_clock({}, "MIN_3", stream=False)
    cfg = {"timeframe": "MIN_3", "screener": {"around_the_clock": {"markets": ["crypto", "forex"], "max_watch": 5}}}
    scr = ig_screener.IG24Screener(ig, cfg, universe_file=tmp_path / "u.json", background=False)
    scr.universe = [{"epic": "CS.D.BITCOIN.CFD.IP", "name": "Bitcoin", "class": "crypto"}]
    scr.universe_updated = time.time()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ig_hours, "is_open", lambda cls, now=None: True)
        picked = scr.run()
    assert [p["symbol"] for p in picked] == ["CS.D.BITCOIN.CFD.IP"]       # 1% spread, under crypto's 1.5%


def test_history_requests_are_capped_around_the_clock(ig):
    ig.enable_around_the_clock({"seed_bars": 40}, "MIN_3", stream=False)
    ig.get_historical_bars("CS.D.AUDUSD.CFD.IP", "MIN_3", 1500)
    assert [c[1]["max"] for c in ig.http.calls if c[0].startswith("/prices/")] == [40]


class NoNavHttp(FakeHttp):
    """An account where IG answers 404 for /marketnavigation (seen live)."""
    SEARCH = {
        "EUR/USD": [{"epic": "CS.D.EURUSD.MINI.IP", "instrumentName": "EUR/USD Mini", "instrumentType": "CURRENCIES", "expiry": "-"},
                    {"epic": "CS.D.EURUSD.CFD.IP", "instrumentName": "EUR/USD", "instrumentType": "CURRENCIES", "expiry": "-"},
                    {"epic": "CS.D.EURUSD.TODAY.IP", "instrumentName": "EUR/USD", "instrumentType": "CURRENCIES", "expiry": "DFB"}],
        "Germany 40": [{"epic": "IX.D.DAX.IFD.IP", "instrumentName": "Germany 40 Cash (€1)", "instrumentType": "INDICES", "expiry": "-"},
                       {"epic": "IX.D.DAX.BIN.IP", "instrumentName": "Germany 40 Binary", "instrumentType": "BINARY", "expiry": "-"}],
    }

    def request(self, method, url, headers=None, json=None, params=None, timeout=None):
        path = url.split("/gateway/deal", 1)[1]
        if path.startswith("/marketnavigation"):
            self.calls.append((path, params))
            return FakeResp(404, "<!doctype html><title>HTTP Status 404 – Not Found</title>")
        if path == "/markets" and params and "searchTerm" in params:
            self.calls.append((path, params))
            return FakeResp(200, {"markets": self.SEARCH.get(params["searchTerm"], [])})
        return super().request(method, url, headers, json, params, timeout)


def test_discovery_falls_back_to_search_when_navigation_404s(monkeypatch, tmp_path):
    from src.brokers.ig import IGClient
    for k, v in {"IG_API_KEY": "key", "IG_USERNAME": "u", "IG_PASSWORD": "p"}.items():
        monkeypatch.setenv(k, v)
    client = IGClient(market="IG24", session=NoNavHttp())
    found = client.discover_markets(["forex", "indices"], delay_s=0)
    assert {m["epic"]: m["class"] for m in found} == {"CS.D.EURUSD.CFD.IP": "forex", "IX.D.DAX.IFD.IP": "indices"}
    terms = [c[1]["searchTerm"] for c in client.http.calls if c[1] and "searchTerm" in c[1]]
    assert terms == client.SEARCH_TERMS["forex"] + client.SEARCH_TERMS["indices"]


def test_failed_discovery_waits_10_minutes(ig, tmp_path):
    ig.enable_around_the_clock({}, "MIN_3", stream=False)
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("down")
    ig.discover_markets = boom
    scr = ig_screener.IG24Screener(ig, {"timeframe": "MIN_3", "screener": {}}, universe_file=tmp_path / "u.json",
                                   background=False)
    scr._run()
    scr._run()
    assert len(calls) == 1 and scr.last_report["universe"] == 0


def test_empty_screener_result_is_still_published(tmp_path):
    from src.orchestrator import Orchestrator, SCREENER_REPORT_KEY
    o = Orchestrator.__new__(Orchestrator)
    o.screener = MagicMock()
    o.screener.run.return_value = []
    o.screener.last_report = {"mode": "ig24", "candidates": [], "discovering": {"nodes": 3, "markets": 1}}
    o.audit_log = MagicMock()
    o._set_workers = MagicMock()
    o._refresh_symbols_from_screener()
    o.audit_log.set_kv.assert_called_once_with(SCREENER_REPORT_KEY, o.screener.last_report)
    o._set_workers.assert_not_called()


def test_closed_markets_still_get_a_report(monkeypatch):
    from src.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o.config = {"polling_interval_seconds": 15}
    o.around_the_clock, o.screener, o.screener_refresh_seconds = True, MagicMock(), 900
    o._last_screener_run, o._stream_published = 0.0, 1e18
    o.check_broker = lambda now: True
    o.update_phase = lambda: "sleeping"
    o.broker, o.workers, o._last_unmanaged_check, o.unmanaged_interval = MagicMock(), {}, 1e18, 300
    o.sleep_monitor_seconds = 300
    o.process_manual_orders = o.flat_by_close = lambda: None
    o.audit_log = MagicMock()
    o._refresh_symbols_from_screener = MagicMock()
    o.run_cycle({}, now=1000.0)
    o.run_cycle({}, now=1100.0)              # within refresh_minutes: not again
    o.run_cycle({}, now=2000.0)
    assert o._refresh_symbols_from_screener.call_count == 2


def test_search_pauses_and_retries_when_ig_limit_is_hit(monkeypatch):
    from src.brokers.ig import IGClient
    for k, v in {"IG_API_KEY": "key", "IG_USERNAME": "u", "IG_PASSWORD": "p"}.items():
        monkeypatch.setenv(k, v)
    http = NoNavHttp()
    client = IGClient(market="IG24", session=http)
    sleeps = []
    import src.brokers.ig as igmod
    monkeypatch.setattr(igmod.time, "sleep", lambda s: sleeps.append(s))
    state = {"n": 0}
    orig = http.request

    def limited(method, url, headers=None, json=None, params=None, timeout=None):
        if params and params.get("searchTerm") == "EUR/USD" and state["n"] == 0:
            state["n"] += 1
            return FakeResp(403, {"errorCode": "error.public-api.exceeded-api-key-allowance"})
        return orig(method, url, headers, json, params, timeout)
    http.request = limited
    client.SEARCH_TERMS = {"forex": ["EUR/USD"]}
    found = client.search_markets(["forex"], delay_s=1)
    assert [m["epic"] for m in found] == ["CS.D.EURUSD.CFD.IP"] and 60 in sleeps
