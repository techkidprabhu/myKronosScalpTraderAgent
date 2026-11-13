"""
Thin wrapper around the `moomoo-api` SDK (moomoo OpenAPI) for
historical K-lines, quotes and order placement.

Unlike Tiger/IG, moomoo talks to a local gateway process - **OpenD** -
over TCP (default 127.0.0.1:11111). OpenD must be running and logged in
to your moomoo account before this client can connect.

SDK:  pip install moomoo-api
Docs: https://openapi.moomoo.com/moomoo-api-doc/en/intro/intro.html

The public surface mirrors the Tiger client so the orchestrator,
screener and dashboard stay broker-agnostic. Positions and orders are
returned as plain dicts (see _position_record / _order_record).

Rate limits worth knowing (per moomoo docs):
  - request_history_kline: 60 requests / 30s, plus a 7-day rolling
    quota on how many distinct symbols' history you may pull.
  - order/position queries: 10 / 30s per account, but only when
    refresh_cache=True. We use OpenD's cache (refresh_cache=False),
    which OpenD keeps current from its own push feed.
  - place/modify order: 15 / 30s per account.
"""
import logging
import os
import socket
import threading
import time
from datetime import datetime, timedelta

import pandas as pd

from moomoo import (
    RET_OK,
    KLType,
    ModifyOrderOp,
    OpenQuoteContext,
    OpenSecTradeContext,
    OrderStatus,
    OrderType,
    PositionSide,
    SecurityFirm,
    SysConfig,
    TimeInForce,
    TrdEnv,
    TrdMarket,
    TrdSide,
)

from src.brokers.base import BrokerClient, Capabilities
from src.market import market_timezone, round_to_tick

logger = logging.getLogger(__name__)

# Agent timeframe -> (moomoo KLType used for the fetch, bars to aggregate).
# moomoo's historical K-line API has no 4-hour bar, so HOUR_4 is built
# from 60-minute bars, grouped 4 at a time within each trading day.
RESOLUTION_MAP = {
    "MIN_1": (KLType.K_1M, 1),
    "MIN_5": (KLType.K_5M, 1),
    "MIN_15": (KLType.K_15M, 1),
    "HOUR_1": (KLType.K_60M, 1),
    "HOUR_4": (KLType.K_60M, 4),
    "DAY": (KLType.K_DAY, 1),
}

INTRADAY_MINUTES = {KLType.K_1M: 1, KLType.K_5M: 5, KLType.K_15M: 15, KLType.K_60M: 60}

# Code prefixes moomoo uses, e.g. "US.AAPL", "HK.00700", "SG.D05", "SH.600519"
CODE_PREFIXES = {"US", "HK", "SG", "SH", "SZ", "JP", "MY", "AU", "CA"}

# Orders that can still fill (or are still being worked on by the exchange)
ACTIVE_STATUSES = {
    OrderStatus.UNSUBMITTED,
    OrderStatus.WAITING_SUBMIT,
    OrderStatus.SUBMITTING,
    OrderStatus.SUBMITTED,
    OrderStatus.FILLED_PART,
    OrderStatus.CANCELLING_PART,
    OrderStatus.CANCELLING_ALL,
}

CURRENCY = {"US": "USD", "HK": "HKD", "SG": "SGD", "CN": "CNH", "JP": "JPY", "MY": "MYR"}

# How far back to look for an order that's no longer in today's list
# (e.g. a GTC stop that filled while the agent was down).
HISTORY_LOOKBACK_DAYS = 30


# How long to wait for OpenD to report its moomoo account logged in. A
# desktop OpenD sitting at "Please enter account" accepts the TCP
# connection but never answers - the SDK would wait forever.
DEFAULT_CONNECT_TIMEOUT = 20.0


def _opend_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _num(value) -> float | None:
    """moomoo fills fields it can't compute with the string 'N/A' (common
    on paper accounts: average_cost, unrealized_pl, ...), and sometimes
    '' or NaN. Return a float, or None for anything non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _qty(value) -> int:
    """Share counts: non-numeric -> 0."""
    f = _num(value)
    return int(f) if f is not None else 0


def _positive(value) -> float | None:
    f = _num(value)
    return f if f is not None and f > 0 else None


def _check(ret, data, what: str):
    if ret != RET_OK:
        raise RuntimeError(f"{what} failed: {data}")
    return data


class MoomooClient(BrokerClient):
    name = "moomoo"
    label = "moomoo OpenD"
    capabilities = Capabilities(
        paper_trading=True, stop_orders=True, gtc_orders=True, short_selling=True,
        market_scan=True, markets=("US", "HK", "SG", "CN"),
        notes="Needs the OpenD gateway running and logged in. Paper accounts reject GTC, "
              "so exits there are DAY orders re-placed each session.",
    )

    def __init__(self, connect_timeout: float | None = None, market: str | None = None):
        self.host = os.environ.get("MOOMOO_OPEND_HOST", "127.0.0.1")
        self.port = int(os.environ.get("MOOMOO_OPEND_PORT", "11111"))
        env = os.environ.get("MOOMOO_TRADING_ENV", "SIMULATE").upper()
        self.trd_env = TrdEnv.REAL if env == "REAL" else TrdEnv.SIMULATE
        self.sandbox = self.trd_env == TrdEnv.SIMULATE  # kept for dashboard parity

        # brokers.<name>.market in config.yaml wins over MOOMOO_TRADE_MARKET.
        self.market = (market or os.environ.get("MOOMOO_TRADE_MARKET", "US")).upper()
        if self.market not in CURRENCY:
            raise ValueError(f"moomoo market must be one of {sorted(CURRENCY)}, got {self.market!r}")
        firm = os.environ.get("MOOMOO_SECURITY_FIRM", "FUTUSG").upper()
        self.security_firm = getattr(SecurityFirm, firm)
        self.acc_id = int(os.environ.get("MOOMOO_ACC_ID", "0") or 0)
        self.account = str(self.acc_id) if self.acc_id else ""

        # OpenD on another machine needs an encrypted connection.
        rsa_path = os.environ.get("MOOMOO_RSA_KEY_PATH")
        if rsa_path:
            SysConfig.set_init_rsa_file(rsa_path)

        # The SDK's context constructors retry forever if OpenD is down,
        # which would hang the agent/dashboard - fail fast instead.
        if not _opend_reachable(self.host, self.port):
            raise ConnectionError(
                f"moomoo OpenD is not reachable at {self.host}:{self.port}. "
                "Start OpenD and log in first (https://openapi.moomoo.com)."
            )

        if connect_timeout is None:
            connect_timeout = float(os.environ.get("MOOMOO_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT))
        enc = bool(rsa_path) or None
        self.quote_ctx = self.trade_ctx = None
        try:
            # Async connect returns at once, so a gateway that never finishes
            # the handshake can't freeze us: we poll its login state with a
            # deadline and close the context ourselves on failure (otherwise
            # the SDK keeps reconnecting in the background and leaks
            # connections inside OpenD).
            try:
                self.quote_ctx = OpenQuoteContext(host=self.host, port=self.port,
                                                  is_encrypt=enc, is_async_connect=True)
            except TypeError:  # older SDK without is_async_connect
                self.quote_ctx = OpenQuoteContext(host=self.host, port=self.port, is_encrypt=enc)
            self._wait_logged_in(connect_timeout)
            self.trade_ctx = OpenSecTradeContext(
                filter_trdmarket=getattr(TrdMarket, self.market),
                host=self.host, port=self.port,
                is_encrypt=enc,
                security_firm=self.security_firm,
            )

            if self.trd_env == TrdEnv.REAL:
                pwd = os.environ.get("MOOMOO_TRADING_PASSWORD")
                if not pwd:
                    raise RuntimeError("MOOMOO_TRADING_PASSWORD is required to unlock REAL trading")
                _check(*self.trade_ctx.unlock_trade(pwd), "unlock_trade")
                logger.info("moomoo REAL trading unlocked")
        except Exception:
            self.close()
            raise

        if not self.acc_id:
            self.acc_id = self._discover_account()
            self.account = str(self.acc_id) if self.acc_id else ""
        logger.info("moomoo connected: env=%s market=%s acc_id=%s exits=%s", self.trd_env, self.market,
                    self.acc_id or "(OpenD default)", self.exit_time_in_force)

    def _wait_logged_in(self, timeout: float):
        """Poll OpenD until it reports its moomoo account logged in for
        both quotes and trading, or raise ConnectionError with the reason."""
        deadline, last = time.monotonic() + timeout, "no answer from OpenD"
        while True:
            try:
                ret, data = self.quote_ctx.get_global_state()
            except Exception as exc:  # connection still being set up
                ret, data = None, str(exc)
            if ret == RET_OK and isinstance(data, dict):
                if data.get("qot_logined") and data.get("trd_logined"):
                    return
                last = "OpenD is running but its moomoo account is not logged in"
            elif data:
                last = str(data)[:160]
            if time.monotonic() >= deadline:
                break
            time.sleep(1)
        raise ConnectionError(
            f"{last} (waited {timeout:.0f}s). Desktop OpenD: log in in its window. "
            "Command-line OpenD: check login_account/login_pwd_md5 in OpenD.xml, or phone verification."
        )

    def ping(self) -> tuple[bool, str]:
        """Cheap health check: (ok, detail). Never raises."""
        try:
            ret, data = self.quote_ctx.get_global_state()
        except Exception as exc:
            return False, f"OpenD did not answer: {exc}"
        if ret != RET_OK or not isinstance(data, dict):
            return False, f"OpenD error: {str(data)[:160]}"
        if not (data.get("qot_logined") and data.get("trd_logined")):
            return False, "OpenD is running but its moomoo account is not logged in"
        return True, "connected"

    @property
    def supports_gtc(self) -> bool:
        """moomoo paper (SIMULATE) accounts reject GTC orders ("Paper
        trading does not support GTC orders"), so exits there are DAY
        orders that the orchestrator re-places every session."""
        return self.trd_env == TrdEnv.REAL

    @property
    def exit_time_in_force(self) -> str:
        """'GTC' or 'DAY' for resting stop/target orders. MOOMOO_EXIT_TIF
        (AUTO / GTC / DAY) overrides the account-type default."""
        forced = os.environ.get("MOOMOO_EXIT_TIF", "AUTO").upper()
        if forced in ("GTC", "DAY"):
            return forced
        return "GTC" if self.supports_gtc else "DAY"

    @property
    def _exit_tif(self):
        return TimeInForce.GTC if self.exit_time_in_force == "GTC" else TimeInForce.DAY

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discover_account(self) -> int:
        """First account for this trading env that is authorised for the
        configured market, preferring stock (not options/futures) paper
        accounts. 0 = let OpenD pick (acc_index 0)."""
        try:
            accs = _check(*self.trade_ctx.get_acc_list(), "get_acc_list")
        except Exception as exc:
            logger.warning("Could not list moomoo accounts: %s", exc)
            return 0
        accs = accs[accs["trd_env"] == self.trd_env]
        if "trdmarket_auth" in accs.columns:
            authed = accs[accs["trdmarket_auth"].apply(lambda m: self.market in (m or []))]
            accs = authed if not authed.empty else accs
        if self.trd_env == TrdEnv.SIMULATE and "sim_acc_type" in accs.columns:
            stock = accs[accs["sim_acc_type"] == "STOCK"]
            accs = stock if not stock.empty else accs
        return int(accs.iloc[0]["acc_id"]) if not accs.empty else 0

    @property
    def _acc(self) -> dict:
        return {"trd_env": self.trd_env, "acc_id": self.acc_id}

    def _normalize_symbol(self, symbol: str) -> str:
        """'AAPL' -> 'US.AAPL', '00700' -> 'HK.00700', 'D05' -> 'SG.D05'.
        Already-prefixed codes pass through (upper-cased)."""
        s = symbol.strip().upper()
        if s.split(".", 1)[0] in CODE_PREFIXES and "." in s:
            return s
        prefix = {"CN": "SH"}.get(self.market, self.market)
        return f"{prefix}.{s}"

    @staticmethod
    def _base_symbol(symbol: str) -> str:
        """'US.AAPL' -> 'AAPL', for matching records that may or may not
        carry the market prefix."""
        s = str(symbol).strip().upper()
        head, _, rest = s.partition(".")
        return rest if rest and head in CODE_PREFIXES else s

    def round_price(self, price: float) -> float:
        return round_to_tick(price, self.market)

    @staticmethod
    def _side(direction: str):
        return TrdSide.BUY if direction.upper() == "BUY" else TrdSide.SELL

    @staticmethod
    def _order_id(place_result) -> str:
        return str(place_result.iloc[0]["order_id"])

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_historical_bars(self, symbol: str, resolution: str, num_points: int) -> pd.DataFrame:
        """Last `num_points` bars for `symbol`, oldest first, as a DataFrame
        with columns time (epoch ms, bar START, UTC), open, high, low,
        close, volume - the shape orchestrator.bars_to_df expects.

        moomoo stamps intraday bars with their END time in exchange-local
        time; we convert to start time so drop_unclosed_bar is exact."""
        ktype, group = RESOLUTION_MAP[resolution]
        code = self._normalize_symbol(symbol)
        want = num_points * group + group  # extra so a partial group can be discarded

        # Calendar span generous enough to cover `want` bars incl. weekends/holidays
        minutes = INTRADAY_MINUTES.get(ktype)
        days = (want * 1.6 + 10) if minutes is None else (want * minutes / 330 * 1.6 + 7)
        start = (datetime.now() - timedelta(days=int(days) + 1)).strftime("%Y-%m-%d")

        frames, page_key = [], None
        while True:
            ret, data, page_key = self.quote_ctx.request_history_kline(
                code, start=start, ktype=ktype, max_count=1000, page_req_key=page_key,
            )
            frames.append(_check(ret, data, f"request_history_kline({code})"))
            if page_key is None:
                break
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        tz = market_timezone(self.market)
        ts = pd.to_datetime(df["time_key"]).dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        if minutes is not None:
            ts = ts - pd.Timedelta(minutes=minutes)
        df["ts"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.dropna(subset=["ts"]).sort_values("ts")

        if group > 1:
            df = self._aggregate(df, group, tz)

        df = df.tail(num_points)
        out = pd.DataFrame({
            "time": (df["ts"].astype("datetime64[ns]").astype("int64") // 10**6).to_numpy(),
            "open": df["open"].astype(float).to_numpy(),
            "high": df["high"].astype(float).to_numpy(),
            "low": df["low"].astype(float).to_numpy(),
            "close": df["close"].astype(float).to_numpy(),
            "volume": df["volume"].astype(float).to_numpy(),
        })
        out["symbol"] = code
        return out

    @staticmethod
    def _aggregate(df: pd.DataFrame, group: int, tz) -> pd.DataFrame:
        """Combine every `group` consecutive bars of the same trading day
        into one bar stamped with the first bar's start time. A day's last
        group may hold fewer bars; drop_unclosed_bar treats it as closed
        only once start + full duration has passed (late, never early)."""
        local_day = df["ts"].dt.tz_localize("UTC").dt.tz_convert(tz).dt.date
        idx = df.groupby(local_day).cumcount() // group
        agg = df.groupby([local_day, idx], sort=False).agg(
            ts=("ts", "first"), open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        )
        return agg.reset_index(drop=True).sort_values("ts")

    def get_current_price(self, symbol: str) -> float:
        code = self._normalize_symbol(symbol)
        data = _check(*self.quote_ctx.get_market_snapshot([code]), f"get_market_snapshot({code})")
        price = _positive(data.iloc[0]["last_price"])
        if price is None:
            raise RuntimeError(f"No valid last price for {code}")
        return price

    def get_snapshots(self, symbols: list[str]) -> pd.DataFrame:
        """Snapshot for up to 400 codes per call (code, last_price, volume, ...)."""
        codes = [self._normalize_symbol(s) for s in symbols]
        return _check(*self.quote_ctx.get_market_snapshot(codes), "get_market_snapshot")

    # moomoo app "Market Movers" lists the screener can start from.
    # name -> (OpenQuoteContext method, max per call, extra kwargs builder)
    MARKET_LISTS = {
        "top_gainers": ("get_top_movers_rank", 200, lambda mm: {"sort_dir": mm.RankSortDir.DESCENDING}),
        "top_losers": ("get_top_movers_rank", 200, lambda mm: {"sort_dir": mm.RankSortDir.ASCENDING}),
        "hot_list": ("get_hot_list", 200, lambda mm: {}),                      # combined heat (default)
        "movers_5min": ("get_period_change_rank", 200,
                        lambda mm: {"period_type": mm.RankPeriodType.FIVE_MIN, "sort_dir": mm.RankSortDir.DESCENDING}),
        "short_selling": ("get_short_selling_rank", 35, lambda mm: {}),       # daily short-volume change
    }

    def get_market_list(self, source: str, count: int = 30) -> list[str]:
        """Codes from one of moomoo's ranking lists (US and HK only).
        Raises if the market, the list name or the installed moomoo-api
        doesn't support it - the screener logs it and carries on."""
        if source not in self.MARKET_LISTS:
            raise ValueError(f"unknown market list {source!r} (known: {', '.join(self.MARKET_LISTS)})")
        if self.market not in ("US", "HK"):
            raise RuntimeError(f"moomoo's market lists cover US and HK only, not {self.market}")
        import moomoo as mm
        method, cap, extra = self.MARKET_LISTS[source]
        fn = getattr(self.quote_ctx, method, None)
        if fn is None:
            raise RuntimeError(f"{method} needs a newer moomoo-api and OpenD "
                               "(pip install -U moomoo-api, and update OpenD)")
        data = _check(*fn(self.market, count=max(1, min(int(count), cap)), **extra(mm)), method)
        df = data[1] if isinstance(data, tuple) else data
        if df is None or len(df) == 0 or "security" not in df:
            return []
        return [str(c) for c in df["security"].tolist() if c]

    def list_market_symbols(self) -> list[str]:
        """Every listed stock code in the configured market (thousands)."""
        market = {"CN": "SH"}.get(self.market, self.market)
        data = _check(*self.quote_ctx.get_stock_basicinfo(market, "STOCK"), "get_stock_basicinfo")
        return data["code"].tolist()

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    @staticmethod
    def _position_record(row) -> dict:
        qty = abs(_qty(row.get("qty")))
        if row.get("position_side") == PositionSide.SHORT:
            qty = -qty
        # average_cost is 'N/A' on securities paper accounts -> use cost_price
        cost = _positive(row.get("average_cost")) or _positive(row.get("cost_price"))
        pnl = _num(row.get("unrealized_pl"))
        if pnl is None:
            pnl = _num(row.get("pl_val"))
        return {
            "symbol": row["code"],
            "name": row.get("stock_name"),
            "quantity": qty,
            "average_cost": cost,
            "market_price": _positive(row.get("nominal_price")),
            "unrealized_pnl": pnl,
            "today_pnl": _num(row.get("today_pl_val")),
            "currency": row.get("currency"),
        }

    @staticmethod
    def _order_record(row) -> dict:
        return {
            "id": str(row["order_id"]),
            "symbol": row["code"],
            "action": row.get("trd_side"),
            "order_type": row.get("order_type"),
            "quantity": _qty(row.get("qty")),
            "filled": _qty(row.get("dealt_qty")),
            "limit_price": _positive(row.get("price")),
            "stop_price": _positive(row.get("aux_price")),
            "status": row.get("order_status"),
            "time_in_force": row.get("time_in_force"),
            "order_time": row.get("create_time"),
        }

    def get_open_positions(self) -> list[dict]:
        data = _check(*self.trade_ctx.position_list_query(**self._acc), "position_list_query")
        return [self._position_record(r) for _, r in data.iterrows()]

    def get_open_orders(self) -> list[dict]:
        data = _check(*self.trade_ctx.order_list_query(
            status_filter_list=list(ACTIVE_STATUSES), **self._acc), "order_list_query")
        return [self._order_record(r) for _, r in data.iterrows()]

    def get_account_summary(self) -> dict:
        currency = CURRENCY.get(self.market, "USD")
        data = _check(*self.trade_ctx.accinfo_query(currency=currency, **self._acc), "accinfo_query")
        row = data.iloc[0]
        return {"currency": currency, **{k: _num(row.get(k)) for k in
                                         ("total_assets", "cash", "market_val", "power")}}

    def get_position_quantity(self, symbol: str) -> int:
        """Signed share count currently held (negative = short)."""
        base = self._base_symbol(symbol)
        return sum(p["quantity"] for p in self.get_open_positions()
                   if self._base_symbol(p["symbol"]) == base)

    def has_open_orders(self, symbol: str) -> bool:
        base = self._base_symbol(symbol)
        return any(self._base_symbol(o["symbol"]) == base for o in self.get_open_orders())

    def has_exposure(self, symbol: str) -> bool:
        """True if the broker shows any holding or working order in symbol,
        including ones this agent didn't place. Raises if the broker can't
        be queried; callers should treat that as 'don't trade'."""
        return self.get_position_quantity(symbol) != 0 or self.has_open_orders(symbol)

    def _find_order(self, order_id) -> dict:
        oid = str(order_id)
        data = _check(*self.trade_ctx.order_list_query(order_id=oid, **self._acc), "order_list_query")
        data = data[data["order_id"].astype(str) == oid] if not data.empty else data
        if data.empty:
            start = (datetime.now() - timedelta(days=HISTORY_LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
            hist = _check(*self.trade_ctx.history_order_list_query(start=start, **self._acc),
                          "history_order_list_query")
            data = hist[hist["order_id"].astype(str) == oid] if not hist.empty else hist
        if data.empty:
            raise RuntimeError(f"Order {oid} not found")
        return self._order_record(data.iloc[0]) | {
            "avg_fill_price": data.iloc[0].get("dealt_avg_price"),
        }

    def get_order_state(self, order_id) -> dict:
        """{status, active, quantity, filled, avg_fill_price}. `active` is
        False once the order can no longer fill."""
        o = self._find_order(order_id)
        return {
            "status": o["status"],
            "active": o["status"] in ACTIVE_STATUSES,
            "quantity": o["quantity"],
            "filled": o["filled"],
            "avg_fill_price": _positive(o["avg_fill_price"]),
        }

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def _place(self, code, direction, quantity, order_type, price=0.0, aux_price=None,
               time_in_force=TimeInForce.DAY, what="place_order") -> str:
        kwargs = dict(price=price, qty=quantity, code=code, trd_side=self._side(direction),
                      order_type=order_type, time_in_force=time_in_force, **self._acc)
        if aux_price is not None:
            kwargs["aux_price"] = aux_price
        data = _check(*self.trade_ctx.place_order(**kwargs), f"{what}({code})")
        return self._order_id(data)

    def create_entry(self, symbol: str, direction: str, quantity: int,
                     entry_limit_price: float | None = None, currency: str | None = None) -> str:
        """Entry order only (LIMIT if entry_limit_price is given, else
        MARKET), time-in-force DAY. Protective orders are placed with
        place_exit_orders once the entry has filled."""
        code = self._normalize_symbol(symbol)
        logger.info("Placing %s entry: code=%s qty=%s limit=%s", direction, code, quantity, entry_limit_price)
        if entry_limit_price is not None:
            return self._place(code, direction, quantity, OrderType.NORMAL,
                               price=self.round_price(entry_limit_price), what="entry")
        return self._place(code, direction, quantity, OrderType.MARKET, what="entry")

    def place_stop_order(self, symbol: str, closing_direction: str, quantity: int,
                         stop_price: float, currency: str | None = None) -> str:
        """Resting STOP (triggers a market order once price crosses
        stop_price). GTC on live accounts so it protects across sessions
        even if the agent is down; DAY on paper accounts, which reject GTC
        - the orchestrator re-places those at each session open."""
        code = self._normalize_symbol(symbol)
        stop_price = self.round_price(stop_price)
        logger.info("Placing stop order: code=%s qty=%s stop=%s", code, quantity, stop_price)
        return self._place(code, closing_direction, quantity, OrderType.STOP, price=stop_price,
                           aux_price=stop_price, time_in_force=self._exit_tif, what="stop")

    def place_target_order(self, symbol: str, closing_direction: str, quantity: int,
                           limit_price: float, currency: str | None = None) -> str:
        code = self._normalize_symbol(symbol)
        limit_price = self.round_price(limit_price)
        logger.info("Placing target order: code=%s qty=%s limit=%s", code, quantity, limit_price)
        return self._place(code, closing_direction, quantity, OrderType.NORMAL, price=limit_price,
                           time_in_force=self._exit_tif, what="target")

    def modify_stop_order(self, order_id, new_stop_price: float):
        """Move a resting stop's trigger price (quantity unchanged)."""
        o = self._find_order(order_id)
        new_stop = self.round_price(new_stop_price)
        logger.info("Modifying stop order %s -> %s", order_id, new_stop)
        return _check(*self.trade_ctx.modify_order(
            ModifyOrderOp.NORMAL, str(order_id), o["quantity"], new_stop,
            aux_price=new_stop, **self._acc), f"modify_order({order_id})")

    def cancel_order(self, order_id):
        return _check(*self.trade_ctx.modify_order(
            ModifyOrderOp.CANCEL, str(order_id), 0, 0, **self._acc), f"cancel_order({order_id})")

    def place_exit_orders(self, symbol: str, closing_direction: str, quantity: int,
                          stop_price: float, target_price: float) -> tuple:
        """Resting stop + target for a FILLED entry. Returns (stop_id,
        target_id); either is None if rejected, in which case the
        orchestrator's software monitor enforces that level. Not linked
        (no OCO) - the orchestrator cancels the survivor on a fill."""
        stop_id = target_id = None
        try:
            stop_id = self.place_stop_order(symbol, closing_direction, quantity, stop_price)
        except Exception as exc:
            logger.warning("Could not place broker-side stop for %s (%s) - software monitor will enforce it", symbol, exc)
        try:
            target_id = self.place_target_order(symbol, closing_direction, quantity, target_price)
        except Exception as exc:
            logger.warning("Could not place broker-side target for %s (%s) - software monitor will enforce it", symbol, exc)
        return stop_id, target_id

    def close_position(self, symbol: str, direction: str, quantity: int, currency: str | None = None) -> str:
        """Market order in the closing direction. Returns the order id."""
        code = self._normalize_symbol(symbol)
        logger.info("Placing %s market close: code=%s qty=%s", direction, code, quantity)
        return self._place(code, direction, quantity, OrderType.MARKET, what="close")

    def close(self):
        for ctx in (getattr(self, "quote_ctx", None), getattr(self, "trade_ctx", None)):
            if ctx is not None:
                try:
                    ctx.close()
                except Exception:
                    logger.debug("Error closing moomoo context", exc_info=True)


def connect_with_timeout(timeout: float = 35.0, **kwargs) -> MoomooClient:
    """MoomooClient() with a hard wall-clock limit. Runs the connect in a
    daemon thread so nothing inside the SDK can freeze the caller; if the
    connect finishes after we gave up, that late client is closed."""
    result: dict = {}
    lock = threading.Lock()
    gave_up = False

    def target():
        try:
            client = MoomooClient(**kwargs)
        except BaseException as exc:  # reported to the caller
            result["error"] = exc
            return
        with lock:
            late = gave_up
            if not late:
                result["client"] = client
        if late:
            client.close()

    worker = threading.Thread(target=target, name="moomoo-connect", daemon=True)
    worker.start()
    worker.join(timeout)
    with lock:
        if "client" not in result and "error" not in result:
            gave_up = True
    if gave_up:
        raise ConnectionError(
            f"moomoo OpenD gave no response within {timeout:.0f}s - it is reachable but not ready "
            "(not logged in, or waiting for phone verification)."
        )
    if "error" in result:
        raise result["error"]
    return result["client"]
