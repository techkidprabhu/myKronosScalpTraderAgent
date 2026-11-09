"""
MoomooClient against fake OpenD contexts - no gateway needed. The fakes
return the same (ret, DataFrame) tuples the moomoo SDK does.
"""
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from moomoo import RET_ERROR, RET_OK, KLType, ModifyOrderOp, OrderType, TimeInForce, TrdEnv, TrdSide

import src.moomoo_client as mc
from src.market import drop_unclosed_bar
from src.moomoo_client import MoomooClient
from src.orchestrator import bars_to_df


def make_client(market="US"):
    client = MoomooClient.__new__(MoomooClient)
    client.market = market
    client.trd_env = TrdEnv.SIMULATE
    client.acc_id = 123
    client.account = "123"
    client.quote_ctx = MagicMock()
    client.trade_ctx = MagicMock()
    return client


def order_row(oid="1", status="SUBMITTED", qty=10, dealt=0, avg=0.0, aux=None, code="US.AAPL"):
    return {"order_id": oid, "code": code, "trd_side": "BUY", "order_type": "NORMAL",
            "order_status": status, "qty": qty, "dealt_qty": dealt, "dealt_avg_price": avg,
            "price": 100.0, "aux_price": aux, "time_in_force": "DAY",
            "create_time": "2026-11-02 10:00:00.000"}


# ---- symbols ----

def test_normalize_symbol():
    us, hk, sg = make_client("US"), make_client("HK"), make_client("SG")
    assert us._normalize_symbol("AAPL") == "US.AAPL"
    assert us._normalize_symbol("us.aapl") == "US.AAPL"
    assert us._normalize_symbol("HK.00700") == "HK.00700"
    assert hk._normalize_symbol("00700") == "HK.00700"
    assert sg._normalize_symbol("D05") == "SG.D05"
    assert make_client("CN")._normalize_symbol("600519") == "SH.600519"


def test_round_price_uses_market_ticks():
    assert make_client("US").round_price(187.3449) == 187.34
    assert make_client("HK").round_price(381.33) == 381.4  # HK 200-500 band ticks 0.2
    assert make_client("SG").round_price(0.537) == 0.535


# ---- historical bars ----

def kline_df(time_keys, base=100.0):
    n = len(time_keys)
    return pd.DataFrame({
        "code": ["US.AAPL"] * n, "time_key": time_keys,
        "open": [base + i for i in range(n)], "high": [base + i + 2 for i in range(n)],
        "low": [base + i - 1 for i in range(n)], "close": [base + i + 1 for i in range(n)],
        "volume": [1000] * n,
    })


def test_daily_bars_shape_and_utc_start_time():
    client = make_client("US")
    client.quote_ctx.request_history_kline.return_value = (
        RET_OK, kline_df(["2026-10-29 00:00:00", "2026-10-30 00:00:00"]), None)

    df = bars_to_df(client.get_historical_bars("AAPL", "DAY", 10))

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    # 00:00 New York (EDT) = 04:00 UTC
    assert df.index[0] == pd.Timestamp("2026-10-29 04:00")
    args, kwargs = client.quote_ctx.request_history_kline.call_args
    assert args[0] == "US.AAPL" and kwargs["ktype"] == KLType.K_DAY


def test_intraday_bars_are_restamped_to_start_time():
    client = make_client("US")
    # moomoo stamps the 09:30-10:30 bar as 10:30
    client.quote_ctx.request_history_kline.return_value = (
        RET_OK, kline_df(["2026-10-30 10:30:00", "2026-10-30 11:30:00"]), None)

    df = bars_to_df(client.get_historical_bars("AAPL", "HOUR_1", 10))

    assert df.index[0] == pd.Timestamp("2026-10-30 13:30")  # 09:30 EDT in UTC


def test_hour4_aggregates_60m_bars_per_day():
    client = make_client("US")
    day1 = [f"2026-10-29 {h}" for h in ("10:30:00", "11:30:00", "12:30:00", "13:30:00",
                                        "14:30:00", "15:30:00", "16:00:00")]
    day2 = ["2026-10-30 10:30:00", "2026-10-30 11:30:00"]
    client.quote_ctx.request_history_kline.return_value = (RET_OK, kline_df(day1 + day2), None)

    df = bars_to_df(client.get_historical_bars("AAPL", "HOUR_4", 10))

    assert len(df) == 3  # day1: 4 + 3 bars, day2: 2 bars - groups never span days
    first = df.iloc[0]
    assert first["open"] == 100 and first["close"] == 104 and first["high"] == 105 and first["low"] == 99
    assert first["volume"] == 4000
    assert df.index[1] == pd.Timestamp("2026-10-29 17:30")  # 13:30 EDT start of 2nd group
    # The in-progress day2 group is not treated as closed
    assert len(drop_unclosed_bar(df, "HOUR_4", now=pd.Timestamp("2026-10-30 16:00"))) == 2


def test_history_paging_is_followed():
    client = make_client("US")
    client.quote_ctx.request_history_kline.side_effect = [
        (RET_OK, kline_df(["2026-10-28 00:00:00"]), "page2"),
        (RET_OK, kline_df(["2026-10-29 00:00:00"]), None),
    ]
    assert len(client.get_historical_bars("AAPL", "DAY", 10)) == 2
    assert client.quote_ctx.request_history_kline.call_args.kwargs["page_req_key"] == "page2"


def test_sdk_error_raises():
    client = make_client("US")
    client.quote_ctx.get_market_snapshot.return_value = (RET_ERROR, "no quote right")
    with pytest.raises(RuntimeError, match="no quote right"):
        client.get_current_price("AAPL")


# ---- account state ----

def test_positions_are_signed_and_normalised():
    client = make_client("US")
    client.trade_ctx.position_list_query.return_value = (RET_OK, pd.DataFrame([
        {"code": "US.AAPL", "stock_name": "Apple", "qty": 10, "position_side": "LONG",
         "average_cost": 180.0, "cost_price": 179.0, "nominal_price": 185.0,
         "unrealized_pl": 50.0, "pl_val": 60.0, "currency": "USD"},
        {"code": "US.TSLA", "stock_name": "Tesla", "qty": 5, "position_side": "SHORT",
         "average_cost": 0.0, "cost_price": 250.0, "nominal_price": 240.0,
         "unrealized_pl": float("nan"), "pl_val": 50.0, "currency": "USD"},
    ]))

    pos = {p["symbol"]: p for p in client.get_open_positions()}

    assert pos["US.AAPL"]["quantity"] == 10 and pos["US.AAPL"]["average_cost"] == 180.0
    assert pos["US.TSLA"]["quantity"] == -5
    assert pos["US.TSLA"]["average_cost"] == 250.0  # falls back to cost_price
    assert pos["US.TSLA"]["unrealized_pnl"] == 50.0  # falls back to pl_val
    assert client.get_position_quantity("TSLA") == -5
    assert client.trade_ctx.position_list_query.call_args.kwargs == {"trd_env": TrdEnv.SIMULATE, "acc_id": 123}


def test_order_state_active_and_filled():
    client = make_client("US")
    client.trade_ctx.order_list_query.return_value = (
        RET_OK, pd.DataFrame([order_row("7", "FILLED_ALL", qty=10, dealt=10, avg=101.5)]))

    state = client.get_order_state("7")

    assert state == {"status": "FILLED_ALL", "active": False, "quantity": 10,
                     "filled": 10, "avg_fill_price": 101.5}


def test_order_state_partial_is_active():
    client = make_client("US")
    client.trade_ctx.order_list_query.return_value = (
        RET_OK, pd.DataFrame([order_row("7", "FILLED_PART", dealt=4, avg=100.0)]))
    state = client.get_order_state("7")
    assert state["active"] is True and state["filled"] == 4


def test_order_state_falls_back_to_history():
    client = make_client("US")
    client.trade_ctx.order_list_query.return_value = (RET_OK, pd.DataFrame(columns=list(order_row())))
    client.trade_ctx.history_order_list_query.return_value = (
        RET_OK, pd.DataFrame([order_row("8", "CANCELLED_ALL"), order_row("9", "FILLED_ALL", dealt=10, avg=99.0)]))

    state = client.get_order_state("9")

    assert state["filled"] == 10 and state["avg_fill_price"] == 99.0


def test_unknown_order_raises():
    client = make_client("US")
    empty = (RET_OK, pd.DataFrame(columns=list(order_row())))
    client.trade_ctx.order_list_query.return_value = empty
    client.trade_ctx.history_order_list_query.return_value = empty
    with pytest.raises(RuntimeError, match="not found"):
        client.get_order_state("404")


# ---- orders ----

def placed(order_id="555"):
    return (RET_OK, pd.DataFrame([{"order_id": order_id}]))


def test_limit_entry():
    client = make_client("US")
    client.trade_ctx.place_order.return_value = placed("555")

    oid = client.create_entry("AAPL", "BUY", 10, entry_limit_price=187.3449)

    assert oid == "555"
    kw = client.trade_ctx.place_order.call_args.kwargs
    assert kw["code"] == "US.AAPL" and kw["price"] == 187.34 and kw["qty"] == 10
    assert kw["trd_side"] == TrdSide.BUY and kw["order_type"] == OrderType.NORMAL
    assert kw["trd_env"] == TrdEnv.SIMULATE and kw["acc_id"] == 123


def test_market_entry_and_close():
    client = make_client("US")
    client.trade_ctx.place_order.return_value = placed()
    client.create_entry("AAPL", "BUY", 10)
    assert client.trade_ctx.place_order.call_args.kwargs["order_type"] == OrderType.MARKET
    client.close_position("AAPL", "SELL", 10)
    kw = client.trade_ctx.place_order.call_args.kwargs
    assert kw["order_type"] == OrderType.MARKET and kw["trd_side"] == TrdSide.SELL


def test_exit_orders_are_gtc_stop_and_limit_on_live_accounts(monkeypatch):
    monkeypatch.delenv("MOOMOO_EXIT_TIF", raising=False)
    client = make_client("US")
    client.trd_env = TrdEnv.REAL
    client.trade_ctx.place_order.side_effect = [placed("s1"), placed("t1")]

    stop_id, target_id = client.place_exit_orders("AAPL", "SELL", 10, 180.004, 200.0)

    assert (stop_id, target_id) == ("s1", "t1")
    stop_kw, target_kw = [c.kwargs for c in client.trade_ctx.place_order.call_args_list]
    assert stop_kw["order_type"] == OrderType.STOP and stop_kw["aux_price"] == 180.0
    assert stop_kw["time_in_force"] == TimeInForce.GTC
    assert target_kw["order_type"] == OrderType.NORMAL and target_kw["price"] == 200.0
    assert target_kw["time_in_force"] == TimeInForce.GTC


def test_paper_account_exit_orders_are_day(monkeypatch):
    # moomoo paper rejects GTC: "Paper trading does not support GTC orders"
    monkeypatch.delenv("MOOMOO_EXIT_TIF", raising=False)
    client = make_client("US")  # SIMULATE
    client.trade_ctx.place_order.side_effect = [placed("s1"), placed("t1")]

    client.place_exit_orders("AAPL", "SELL", 10, 180.0, 200.0)

    assert client.exit_time_in_force == "DAY" and not client.supports_gtc
    assert [c.kwargs["time_in_force"] for c in client.trade_ctx.place_order.call_args_list] == \
        [TimeInForce.DAY, TimeInForce.DAY]


def test_exit_time_in_force_env_override(monkeypatch):
    client = make_client("US")
    monkeypatch.setenv("MOOMOO_EXIT_TIF", "gtc")
    assert client.exit_time_in_force == "GTC"
    client.trd_env = TrdEnv.REAL
    monkeypatch.setenv("MOOMOO_EXIT_TIF", "DAY")
    assert client.exit_time_in_force == "DAY"


def test_place_exit_orders_resilience(monkeypatch):
    client = make_client("US")

    def stop_fail(*args, **kwargs):
        raise RuntimeError("Stop orders are not supported for this security.")

    monkeypatch.setattr(client, "place_stop_order", stop_fail)
    monkeypatch.setattr(client, "place_target_order", lambda *a, **k: "67890")

    assert client.place_exit_orders("AAPL", "SELL", 10, 180.0, 200.0) == (None, "67890")


def test_modify_stop_keeps_quantity():
    client = make_client("US")
    client.trade_ctx.order_list_query.return_value = (
        RET_OK, pd.DataFrame([order_row("s1", "SUBMITTED", qty=10, aux=180.0)]))
    client.trade_ctx.modify_order.return_value = (RET_OK, pd.DataFrame())

    client.modify_stop_order("s1", 182.506)

    args, kw = client.trade_ctx.modify_order.call_args
    assert args == (ModifyOrderOp.NORMAL, "s1", 10, 182.51)
    assert kw["aux_price"] == 182.51


def test_cancel_order():
    client = make_client("US")
    client.trade_ctx.modify_order.return_value = (RET_OK, pd.DataFrame())
    client.cancel_order(42)
    args, _ = client.trade_ctx.modify_order.call_args
    assert args[:2] == (ModifyOrderOp.CANCEL, "42")


# ---- connection ----

def test_unreachable_opend_fails_fast(monkeypatch):
    monkeypatch.setenv("MOOMOO_OPEND_PORT", "1")  # nothing listens here
    monkeypatch.setattr(mc, "OpenQuoteContext", MagicMock(side_effect=AssertionError("must not connect")))
    start = datetime.now()
    with pytest.raises(ConnectionError, match="OpenD is not reachable"):
        MoomooClient()
    assert (datetime.now() - start).total_seconds() < 5


class FakeQuoteCtx:
    def __init__(self, states):
        self.states = list(states)
        self.closed = False

    def get_global_state(self):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def close(self):
        self.closed = True


def connect_env(monkeypatch, quote, trade=None):
    import socket
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    monkeypatch.setenv("MOOMOO_OPEND_PORT", str(server.getsockname()[1]))
    monkeypatch.setenv("MOOMOO_ACC_ID", "123")
    monkeypatch.setenv("MOOMOO_TRADING_ENV", "SIMULATE")
    monkeypatch.setattr(mc, "OpenQuoteContext", MagicMock(return_value=quote))
    monkeypatch.setattr(mc, "OpenSecTradeContext", MagicMock(return_value=trade or MagicMock()))
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    return server


def test_connect_waits_for_login(monkeypatch):
    quote = FakeQuoteCtx([(RET_ERROR, "connecting"),
                          (RET_OK, {"qot_logined": False, "trd_logined": False}),
                          (RET_OK, {"qot_logined": True, "trd_logined": True})])
    server = connect_env(monkeypatch, quote)
    try:
        client = MoomooClient(connect_timeout=5)
    finally:
        server.close()
    assert client.quote_ctx is quote
    assert mc.OpenQuoteContext.call_args.kwargs["is_async_connect"] is True
    assert client.ping() == (True, "connected")


def test_connect_not_logged_in_times_out_and_closes(monkeypatch):
    quote = FakeQuoteCtx([(RET_OK, {"qot_logined": False, "trd_logined": False})])
    server = connect_env(monkeypatch, quote)
    clock = iter(range(0, 1000, 5))
    monkeypatch.setattr(mc.time, "monotonic", lambda: next(clock))
    try:
        with pytest.raises(ConnectionError, match="not logged in"):
            MoomooClient(connect_timeout=20)
    finally:
        server.close()
    assert quote.closed  # no SDK thread left reconnecting inside OpenD
    mc.OpenSecTradeContext.assert_not_called()


def test_connect_with_timeout_never_blocks(monkeypatch):
    import threading
    release = threading.Event()
    created = []

    class Hanging:
        def __init__(self, **kw):
            release.wait(5)
            created.append(self)
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(mc, "MoomooClient", Hanging)
    start = datetime.now()
    with pytest.raises(ConnectionError, match="no response"):
        mc.connect_with_timeout(0.2)
    assert (datetime.now() - start).total_seconds() < 2
    release.set()
    for _ in range(50):
        if created:
            break
        __import__("time").sleep(0.05)
    __import__("time").sleep(0.05)
    assert created and created[0].closed  # the late connection is not leaked


def test_ping_reports_logged_out():
    client = make_client("US")
    client.quote_ctx.get_global_state.return_value = (RET_OK, {"qot_logined": True, "trd_logined": False})
    ok, detail = client.ping()
    assert not ok and "not logged in" in detail
    client.quote_ctx.get_global_state.side_effect = RuntimeError("socket closed")
    assert client.ping()[0] is False


def test_symbol_worker_order_type(tmp_path):
    from src.audit_log import AuditLog
    from src.orchestrator import SymbolWorker

    worker = SymbolWorker(
        symbol_cfg={"symbol": "US.AAPL", "name": "Apple", "quantity": 10},
        global_cfg={"timeframe": "DAY", "lookback_bars": 200, "forecast_horizon": 6,
                    "kronos_confidence_threshold": 0.62, "order_type": "LIMIT"},
        broker=MagicMock(),
        kronos_agent=MagicMock(),
        risk_manager=MagicMock(),
        audit_log=AuditLog(str(tmp_path / "audit.db")),
        reporter=MagicMock(),
    )
    assert worker.order_type == "LIMIT"


# ---- 'N/A' fields (paper accounts) - regression for a live crash ----

def test_positions_tolerate_na_strings():
    client = make_client("US")
    client.trade_ctx.position_list_query.return_value = (RET_OK, pd.DataFrame([
        {"code": "US.AAPL", "stock_name": "Apple", "qty": 10.0, "position_side": "LONG",
         "average_cost": "N/A", "cost_price": 180.0, "nominal_price": 185.0,
         "unrealized_pl": "N/A", "pl_val": 50.0, "currency": "USD"},
        {"code": "US.MSFT", "stock_name": "Microsoft", "qty": "N/A", "position_side": "LONG",
         "average_cost": "N/A", "cost_price": "N/A", "nominal_price": "N/A",
         "unrealized_pl": "N/A", "pl_val": "N/A", "currency": "USD"},
    ]))

    pos = {p["symbol"]: p for p in client.get_open_positions()}

    assert pos["US.AAPL"] == {"symbol": "US.AAPL", "name": "Apple", "quantity": 10,
                              "average_cost": 180.0, "market_price": 185.0,
                              "unrealized_pnl": 50.0, "today_pnl": None, "currency": "USD"}
    assert pos["US.MSFT"]["quantity"] == 0 and pos["US.MSFT"]["unrealized_pnl"] is None
    assert client.get_position_quantity("AAPL") == 10


def test_order_and_account_tolerate_na_strings():
    client = make_client("US")
    client.trade_ctx.order_list_query.return_value = (RET_OK, pd.DataFrame([
        order_row("7", "SUBMITTED", qty=10, dealt="N/A", avg="N/A", aux="N/A")]))
    assert client.get_order_state("7") == {"status": "SUBMITTED", "active": True, "quantity": 10,
                                           "filled": 0, "avg_fill_price": None}

    client.trade_ctx.accinfo_query.return_value = (RET_OK, pd.DataFrame([
        {"total_assets": 100000.0, "cash": "N/A", "market_val": 1850.0, "power": ""}]))
    assert client.get_account_summary() == {"currency": "USD", "total_assets": 100000.0,
                                            "cash": None, "market_val": 1850.0, "power": None}


def test_current_price_na_raises_cleanly():
    client = make_client("US")
    client.quote_ctx.get_market_snapshot.return_value = (RET_OK, pd.DataFrame([{"last_price": "N/A"}]))
    with pytest.raises(RuntimeError, match="No valid last price"):
        client.get_current_price("AAPL")
