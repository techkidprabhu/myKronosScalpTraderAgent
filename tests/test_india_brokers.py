"""Zerodha Kite and Angel One adapters against fake SDK clients - no
accounts needed. Also NSE ticks/sessions and the no-short guard."""
from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import src.multi_broker as mb
from src.brokers.base import BrokerError
from src.brokers.india import round_to_instrument, to_frame
from src.market import is_market_open, tick_size

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------- exchange rules

@pytest.mark.parametrize("price, tick", [(120.0, 0.01), (249.99, 0.01), (250.0, 0.05), (1000.0, 0.05),
                                         (1000.5, 0.10), (5000.0, 0.10), (7500.0, 0.50), (15000.0, 1.0),
                                         (25000.0, 5.0)])
def test_nse_tick_bands(price, tick):
    assert tick_size(price, "NSE") == tick


def test_nse_session_and_holidays():
    assert is_market_open("NSE", datetime(2026, 11, 5, 9, 15, tzinfo=IST))
    assert not is_market_open("NSE", datetime(2026, 11, 5, 15, 30, tzinfo=IST))
    assert not is_market_open("NSE", datetime(2026, 11, 7, 11, 0, tzinfo=IST))     # Saturday
    assert not is_market_open("NSE", datetime(2026, 11, 10, 11, 0, tzinfo=IST))    # Diwali-Balipratipada


def test_instrument_tick_wins_over_band_table():
    assert round_to_instrument(1234.567, 0.05, "NSE") == 1234.55    # instrument says 0.05
    assert round_to_instrument(1234.567, None, "NSE") == 1234.6     # band table says 0.10


def test_ist_bars_become_utc_start_times_and_group_into_4h():
    days = []
    for d in ("2026-11-03", "2026-11-04"):
        for h in range(7):     # 09:15 ... 15:15 IST hourly bars
            t = pd.Timestamp(f"{d} 09:15", tz=IST) + pd.Timedelta(hours=h)
            days.append((t, 100 + h, 101 + h, 99 + h, 100.5 + h, 1000))
    one = to_frame(days, "HOUR_1", 100, "NSE.X")
    assert pd.Timestamp(one["time"].iloc[0], unit="ms") == pd.Timestamp("2026-11-03 03:45")   # 09:15 IST
    four = to_frame(days, "HOUR_4", 100, "NSE.X")
    assert len(four) == 4                                  # 4 + 3 bars per day
    first = four.iloc[0]
    assert first["open"] == 100 and first["close"] == 103.5 and first["high"] == 104 and first["volume"] == 4000


# ---------------------------------------------------------------- Zerodha

class FakeKite:
    def __init__(self):
        self.placed, self.gtts, self.cancelled, self.deleted, self.modified = [], {}, [], [], []
        self.history = {}
        self.instr = [{"tradingsymbol": "RELIANCE", "instrument_token": 738561, "tick_size": 0.1,
                       "instrument_type": "EQ", "segment": "NSE"},
                      {"tradingsymbol": "ITC", "instrument_token": 424961, "tick_size": 0.05,
                       "instrument_type": "EQ", "segment": "NSE"},
                      {"tradingsymbol": "NIFTY26NOVFUT", "instrument_token": 1, "tick_size": 0.05,
                       "instrument_type": "FUT", "segment": "NFO-FUT"}]
        self.hist_calls = []

    def profile(self):
        return {"user_id": "AB1234"}

    def instruments(self, exchange):
        return self.instr

    def historical_data(self, token, start, end, interval):
        self.hist_calls.append((token, interval, start, end))
        out, t = [], pd.Timestamp(end).floor("D").tz_localize(IST) + pd.Timedelta(hours=9, minutes=15)
        for i in range(6):
            out.append({"date": t + pd.Timedelta(minutes=5 * i), "open": 1400.0, "high": 1401.0,
                        "low": 1399.0, "close": 1400.5, "volume": 500})
        return out

    def ltp(self, keys):
        return {k: {"last_price": 1402.3} for k in keys}

    def quote(self, keys):
        return {k: {"last_price": 1402.3, "volume": 123456} for k in keys}

    def positions(self):
        return {"net": [{"exchange": "NSE", "tradingsymbol": "RELIANCE", "quantity": 5, "average_price": 1400.0,
                         "last_price": 1402.3, "pnl": 11.5, "product": "CNC"}]}

    def holdings(self):
        return [{"exchange": "NSE", "tradingsymbol": "ITC", "quantity": 0, "t1_quantity": 10,
                 "average_price": 410.0, "last_price": 412.0, "pnl": 20.0}]

    def margins(self, segment):
        return {"net": 50000.0, "available": {"cash": 45000.0}, "utilised": {"debits": 5000.0}}

    def place_order(self, **kw):
        self.placed.append(kw)
        return f"2611{len(self.placed):08d}"

    def orders(self):
        return [{"order_id": "1", "exchange": "NSE", "tradingsymbol": "RELIANCE", "transaction_type": "SELL",
                 "order_type": "SL-M", "quantity": 5, "filled_quantity": 0, "price": 0, "trigger_price": 1390.0,
                 "status": "TRIGGER PENDING", "validity": "DAY"},
                {"order_id": "2", "exchange": "NSE", "tradingsymbol": "ITC", "transaction_type": "BUY",
                 "order_type": "LIMIT", "quantity": 10, "status": "COMPLETE"}]

    def order_history(self, order_id):
        return self.history.get(order_id, [])

    def place_gtt(self, **kw):
        tid = 9000 + len(self.gtts)
        self.gtts[tid] = {"id": tid, "status": "active", "condition": {
            "exchange": kw["exchange"], "tradingsymbol": kw["tradingsymbol"], "trigger_values": kw["trigger_values"]},
            "orders": kw["orders"], "_kw": kw}
        return {"trigger_id": tid}

    def get_gtt(self, tid):
        return self.gtts[tid]

    def get_gtts(self):
        return list(self.gtts.values())

    def modify_gtt(self, **kw):
        self.modified.append(kw)

    def delete_gtt(self, tid):
        self.deleted.append(tid)
        self.gtts[tid]["status"] = "deleted"

    def cancel_order(self, variety, order_id):
        self.cancelled.append(order_id)

    def modify_order(self, variety, order_id, **kw):
        self.modified.append((order_id, kw))


@pytest.fixture
def kite():
    from src.brokers.zerodha import ZerodhaClient
    return ZerodhaClient(market="NSE", kite=FakeKite())


def test_kite_bars_prices_and_snapshots(kite):
    bars = kite.get_historical_bars("NSE.RELIANCE", "MIN_5", 6)
    assert len(bars) == 6 and bars["symbol"].iloc[0] == "NSE.RELIANCE"
    token, interval, start, end = kite.kite.hist_calls[0]
    assert token == 738561 and interval == "5minute"
    assert kite.get_current_price("RELIANCE") == 1402.3
    snaps = kite.get_snapshots(["NSE.RELIANCE", "NSE.ITC"])
    assert snaps["code"].tolist() == ["NSE.RELIANCE", "NSE.ITC"] and snaps["volume"].iloc[0] == 123456
    assert kite.list_market_symbols() == ["NSE.RELIANCE", "NSE.ITC"]
    assert kite.account == "AB1234"


def test_kite_positions_holdings_orders_and_account(kite):
    pos = {p["symbol"]: p["quantity"] for p in kite.get_open_positions()}
    assert pos == {"NSE.RELIANCE": 5, "NSE.ITC": 10}          # T1 holdings count
    assert kite.has_exposure("NSE.ITC") and kite.get_position_quantity("RELIANCE") == 5
    orders = kite.get_open_orders()
    assert [o["id"] for o in orders] == ["1"] and orders[0]["stop_price"] == 1390.0
    assert kite.get_account_summary() == {"currency": "INR", "total_assets": 50000.0, "cash": 45000.0,
                                          "market_val": 5000.0, "power": 50000.0}


def test_kite_entry_is_cnc_day_with_instrument_tick(kite):
    kite.create_entry("NSE.RELIANCE", "BUY", 5, entry_limit_price=1402.37)
    o = kite.kite.placed[-1]
    assert o["product"] == "CNC" and o["validity"] == "DAY" and o["order_type"] == "LIMIT"
    assert o["price"] == 1402.4 and o["tradingsymbol"] == "RELIANCE" and o["exchange"] == "NSE"
    assert "market_protection" not in o                  # LIMIT orders don't carry it
    kite.create_entry("NSE.ITC", "BUY", 10)
    assert kite.kite.placed[-1]["order_type"] == "MARKET"
    assert kite.kite.placed[-1]["market_protection"] == -1   # mandatory since 1 Apr 2026


def test_kite_exits_are_gtts_with_buffered_stop_limit(kite):
    assert kite.exit_time_in_force == "GTC"
    stop_id, target_id = kite.place_exit_orders("NSE.RELIANCE", "SELL", 5, 1380.03, 1430.08)
    assert stop_id.startswith("gtt:") and target_id.startswith("gtt:")
    stop = kite.kite.gtts[int(stop_id[4:])]["_kw"]
    assert stop["trigger_type"] == "single" and stop["trigger_values"] == [1380.0]
    assert stop["orders"][0]["price"] == 1373.1 and stop["orders"][0]["transaction_type"] == "SELL"
    target = kite.kite.gtts[int(target_id[4:])]["_kw"]
    assert target["trigger_values"] == [1430.1] and target["orders"][0]["price"] == 1430.1
    assert {o["id"] for o in kite.get_open_orders()} >= {stop_id, target_id}


def test_kite_gtt_state_follows_the_order_it_places(kite):
    stop_id, _ = kite.place_exit_orders("NSE.RELIANCE", "SELL", 5, 1380.0, 1430.0)
    tid = int(stop_id[4:])
    assert kite.get_order_state(stop_id) == {"status": "GTT_ACTIVE", "active": True, "quantity": 5,
                                             "filled": 0, "avg_fill_price": None}
    g = kite.kite.gtts[tid]
    g["status"] = "triggered"
    g["orders"][0]["result"] = {"order_result": {"order_id": "777", "status": "success"}}
    kite.kite.history["777"] = [{"status": "OPEN", "quantity": 5, "filled_quantity": 0}]
    assert kite.get_order_state(stop_id)["active"] is True
    kite.cancel_order(stop_id)                         # triggered: cancels the working child order
    assert kite.kite.cancelled == ["777"]
    kite.kite.history["777"] = [{"status": "COMPLETE", "quantity": 5, "filled_quantity": 5, "average_price": 1379.5}]
    assert kite.get_order_state(stop_id) == {"status": "COMPLETE", "active": False, "quantity": 5,
                                             "filled": 5, "avg_fill_price": 1379.5}


def test_kite_cancel_and_modify_gtt(kite):
    stop_id, target_id = kite.place_exit_orders("NSE.RELIANCE", "SELL", 5, 1380.0, 1430.0)
    kite.modify_stop_order(stop_id, 1390.04)
    mod = kite.kite.modified[-1]
    assert mod["trigger_values"] == [1390.0] and mod["orders"][0]["price"] == 1383.0   # 1383.05 rounded away from the trigger
    kite.cancel_order(target_id)
    assert kite.kite.deleted == [int(target_id[4:])]
    assert kite.get_order_state(target_id)["active"] is False


def test_kite_old_day_order_counts_as_expired(kite):
    assert kite.get_order_state("123") == {"status": "EXPIRED", "active": False, "quantity": 0,
                                           "filled": 0, "avg_fill_price": None}


def test_kite_order_mode_uses_slm_day_orders(monkeypatch):
    from src.brokers.zerodha import ZerodhaClient
    monkeypatch.setenv("KITE_EXIT_MODE", "order")
    k = ZerodhaClient(market="NSE", kite=FakeKite())
    assert k.exit_time_in_force == "DAY"
    k.place_stop_order("NSE.RELIANCE", "SELL", 5, 1380.03)
    assert k.kite.placed[-1]["order_type"] == "SL-M" and k.kite.placed[-1]["trigger_price"] == 1380.0
    assert k.kite.placed[-1]["market_protection"] == -1


def test_kite_expired_token_says_how_to_fix():
    from src.brokers.zerodha import ZerodhaClient
    bad = FakeKite()
    bad.profile = MagicMock(side_effect=RuntimeError("Incorrect `api_key` or `access_token`."))
    with pytest.raises(BrokerError, match="kite_login.py"):
        ZerodhaClient(market="NSE", kite=bad)


def test_kite_login_parses_redirect_url():
    import kite_login
    assert kite_login.request_token_from(
        "https://127.0.0.1/?action=login&type=login&status=success&request_token=abc123XYZ") == "abc123XYZ"
    assert kite_login.request_token_from("  abc123XYZ ") == "abc123XYZ"


# ---------------------------------------------------------------- Angel

SCRIP = [{"token": "2885", "symbol": "RELIANCE-EQ", "exch_seg": "NSE", "tick_size": "10.000000"},
         {"token": "1660", "symbol": "ITC-EQ", "exch_seg": "NSE", "tick_size": "5.000000"},
         {"token": "99926000", "symbol": "Nifty 50", "exch_seg": "NSE", "tick_size": "5.000000"}]


class FakeSmart:
    def __init__(self):
        self.placed, self.cancelled, self.modified, self.candle_calls = [], [], [], []
        self.book = []

    def rmsLimit(self):
        return {"status": True, "data": {"net": "50000", "availablecash": "45000", "utiliseddebits": "5000"}}

    def getCandleData(self, params):
        self.candle_calls.append(params)
        day = params["todate"][:10]
        return {"status": True, "data": [[f"{day}T09:{15 + 5 * i:02d}:00+05:30", 1400, 1401, 1399, 1400.5, 700]
                                         for i in range(4)]}

    def ltpData(self, ex, ts, token):
        return {"status": True, "data": {"ltp": 1402.3}}

    def getMarketData(self, mode, tokens):
        (ex, toks), = tokens.items()
        return {"status": True, "data": {"fetched": [{"symbolToken": t, "ltp": 1402.3, "tradeVolume": 9999}
                                                     for t in toks]}}

    def position(self):
        return {"status": True, "data": [{"exchange": "NSE", "tradingsymbol": "RELIANCE-EQ", "netqty": "5",
                                          "avgnetprice": "1400", "ltp": "1402.3", "unrealised": "11.5"}]}

    def holding(self):
        return {"status": True, "data": {"holdings": [{"exchange": "NSE", "tradingsymbol": "ITC-EQ",
                                                       "quantity": "0", "t1quantity": "10", "averageprice": "410",
                                                       "ltp": "412", "profitandloss": "20"}]}}

    def orderBook(self):
        return {"status": True, "data": self.book}

    def placeOrder(self, params):
        self.placed.append(params)
        oid = f"26110500{len(self.placed):04d}"
        self.book.append({"orderid": oid, "exchange": params["exchange"], "tradingsymbol": params["tradingsymbol"],
                          "transactiontype": params["transactiontype"], "ordertype": params["ordertype"],
                          "quantity": params["quantity"], "filledshares": "0", "price": params.get("price"),
                          "triggerprice": params.get("triggerprice", "0"), "status": "open",
                          "duration": "DAY", "variety": params["variety"], "averageprice": "0"})
        return oid

    def modifyOrder(self, params):
        self.modified.append(params)
        return {"status": True, "data": {}}

    def cancelOrder(self, order_id, variety):
        self.cancelled.append((order_id, variety))
        return {"status": True, "data": {}}


@pytest.fixture
def angel(monkeypatch):
    from src.brokers.angel import AngelClient
    monkeypatch.setenv("ANGEL_CLIENT_CODE", "A12345")
    return AngelClient(market="NSE", api=FakeSmart(), scrip=SCRIP)


def test_angel_symbols_ticks_bars_and_prices(angel):
    assert angel._ids("NSE.RELIANCE") == ("NSE", "RELIANCE-EQ", "2885", 0.10)
    assert angel._ids("ITC")[3] == 0.05
    with pytest.raises(BrokerError, match="Unknown Angel instrument"):
        angel._ids("NSE.NOPE")
    bars = angel.get_historical_bars("NSE.RELIANCE", "MIN_5", 4)
    assert len(bars) == 4 and bars["symbol"].iloc[0] == "NSE.RELIANCE"
    assert angel.api.candle_calls[0]["interval"] == "FIVE_MINUTE" and angel.api.candle_calls[0]["symboltoken"] == "2885"
    assert angel.get_current_price("NSE.RELIANCE") == 1402.3
    snaps = angel.get_snapshots(["NSE.RELIANCE", "NSE.ITC"])
    assert sorted(snaps["code"]) == ["NSE.ITC", "NSE.RELIANCE"] and snaps["volume"].iloc[0] == 9999
    assert sorted(angel.list_market_symbols()) == ["NSE.ITC", "NSE.RELIANCE"]


def test_angel_positions_holdings_account(angel):
    pos = {p["symbol"]: p["quantity"] for p in angel.get_open_positions()}
    assert pos == {"NSE.RELIANCE": 5, "NSE.ITC": 10}
    acct = angel.get_account_summary()
    assert acct["currency"] == "INR" and acct["total_assets"] == 50000.0 and acct["cash"] == 45000.0


def test_angel_orders_stop_target_modify_cancel(angel):
    entry = angel.create_entry("NSE.RELIANCE", "BUY", 5, entry_limit_price=1402.37)
    p = angel.api.placed[-1]
    assert p["producttype"] == "DELIVERY" and p["duration"] == "DAY" and p["price"] == "1402.4"
    assert angel.get_order_state(entry)["active"] is True
    stop_id, target_id = angel.place_exit_orders("NSE.RELIANCE", "SELL", 5, 1380.03, 1430.08)
    stop = angel.api.placed[-2]
    assert stop["variety"] == "STOPLOSS" and stop["ordertype"] == "STOPLOSS_LIMIT" and stop["triggerprice"] == "1380.0"
    assert stop["price"] == "1373.1"                     # 0.5% below the trigger, rounded down
    assert angel.api.placed[-1]["ordertype"] == "LIMIT" and angel.api.placed[-1]["price"] == "1430.1"
    assert angel.exit_time_in_force == "DAY"
    angel.modify_stop_order(stop_id, 1390.04)
    mod = angel.api.modified[-1]
    assert mod["triggerprice"] == "1390.0" and mod["quantity"] == "5" and mod["ordertype"] == "STOPLOSS_LIMIT"
    assert mod["price"] == "1383.0"
    angel.cancel_order(stop_id)
    assert angel.api.cancelled[-1] == (stop_id, "STOPLOSS")
    assert {o["id"] for o in angel.get_open_orders()} == {entry, stop_id, target_id}


def test_angel_unknown_order_is_an_expired_day_order(angel):
    assert angel.get_order_state("999")["active"] is False


def test_angel_api_error_is_raised():
    from src.brokers.angel import AngelClient
    api = FakeSmart()
    api.rmsLimit = lambda: {"status": False, "message": "Invalid Token", "data": None}
    a = AngelClient(market="NSE", api=api, scrip=SCRIP)
    ok, detail = a.ping()
    assert ok is False and "Invalid Token" in detail


# ---------------------------------------------------------------- guard

def test_runner_turns_off_shorting_for_brokers_that_cant(monkeypatch):
    monkeypatch.setattr(mb, "KronosAgent", lambda **kw: MagicMock())
    cfg = {"risk": {"allow_short": True, "lot_size": 1},
           "brokers": {"moomoo": {}, "zerodha": {"enabled": True, "market": "NSE"}}}
    runner = mb.MultiBrokerRunner(cfg)
    by = {s.name: s for s in runner.specs}
    assert by["moomoo"].config["risk"]["allow_short"] is True
    assert by["zerodha"].config["risk"]["allow_short"] is False


def test_angel_never_sends_market_orders(angel):
    """Algo MARKET/IOC orders are prohibited at Angel since 1 Apr 2026."""
    angel.create_entry("NSE.RELIANCE", "BUY", 5)            # LTP 1402.3
    buy = angel.api.placed[-1]
    assert buy["ordertype"] == "LIMIT" and buy["price"] == "1409.4"   # +0.5%, rounded up to 0.10
    angel.close_position("NSE.RELIANCE", "SELL", 5)
    sell = angel.api.placed[-1]
    assert sell["ordertype"] == "LIMIT" and sell["price"] == "1395.2"  # -0.5%, rounded down
    assert all(p["ordertype"] not in ("MARKET", "STOPLOSS_MARKET") for p in angel.api.placed)
