"""
Angel One SmartAPI adapter (pip install smartapi-python pyotp) - EXPERIMENTAL.

Ported from moomoo-ensemble-agent/src/brokers/angel.py onto this agent's
BrokerClient interface, with NSE tick bands and 5-minute candles. Tested
against a fake SmartAPI client only - Angel has no paper account, so start
with a tiny quantity and watch the first trades.

.env:
    ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_MPIN, ANGEL_TOTP_SECRET
    ANGEL_PRODUCT=INTRADAY   INTRADAY = scalping (squared off by Angel One before the close). DELIVERY = holding.
    ANGEL_LIMIT_PCT=0.5      buffer for "marketable" limit orders - see below

Exchange rules for API orders (SEBI retail-algo framework, since 1 Apr 2026):
  * orders must come from the static IP registered in the SmartAPI dashboard
    (a home connection with a changing IP will have its orders rejected);
  * MARKET and IOC orders are not allowed for algos. So this adapter never
    sends one: a market entry or close becomes a LIMIT priced ANGEL_LIMIT_PCT
    beyond the last price (fills like a market order in normal conditions),
    and the stop is STOPLOSS_LIMIT with its limit the same % past the trigger.

Login is automatic: a TOTP code is generated from ANGEL_TOTP_SECRET each
time the agent connects, and the agent reconnects by itself after Angel ends
the session (around midnight).

Symbol tokens come from Angel's public scrip master, downloaded once a day
to data/angel_scrip_master.json (~40 MB). Equity symbols carry "-EQ"
(NSE.RELIANCE -> RELIANCE-EQ); the instrument's own tick is taken from it.

Exits: stop = STOPLOSS_LIMIT, target = LIMIT, both DAY orders re-armed by
the orchestrator at every session open. If Angel rejects the second sell
order for the same delivery shares, the software monitor enforces the
target instead. Order status is only available for today's orders.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.brokers.base import BrokerClient, BrokerError, Capabilities, split_symbol
from src.brokers.india import fetch_chunked, history_window, round_to_instrument, to_frame
from src.market import round_to_tick

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIP_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
SCRIP_FILE = ROOT / "data" / "angel_scrip_master.json"
INTERVAL = {"MIN_1": "ONE_MINUTE", "MIN_3": "THREE_MINUTE", "MIN_5": "FIVE_MINUTE", "MIN_15": "FIFTEEN_MINUTE",
            "HOUR_1": "ONE_HOUR", "HOUR_4": "ONE_HOUR", "DAY": "ONE_DAY"}
MAX_DAYS = {"ONE_MINUTE": 30, "THREE_MINUTE": 60, "FIVE_MINUTE": 100, "FIFTEEN_MINUTE": 200, "ONE_HOUR": 400, "ONE_DAY": 2000}
ACTIVE = {"open", "trigger pending", "open pending", "modify pending", "validation pending",
          "after market order req received", "modify validation pending"}


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _data(resp) -> object:
    """SmartAPI wraps results as {"status", "message", "data"}."""
    if isinstance(resp, dict):
        if resp.get("status") is False:
            raise BrokerError(f"Angel: {resp.get('message') or resp}")
        return resp.get("data")
    return resp


# Angel's candle API allows about 3 requests a second per account. Over that it
# answers with an empty body (the SDK then fails to parse it) or an "access
# rate" message, so candle requests are spaced out and retried.
CANDLE_MIN_INTERVAL = 0.4      # seconds between candle requests
CANDLE_RETRIES = 3             # extra attempts after a rate-limit style failure
CANDLE_BACKOFF = 1.0           # first wait; doubles each retry


def _rate_limited(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (isinstance(exc, ValueError)                    # JSONDecodeError: empty / non-JSON reply
            or "access rate" in text or "too many" in text or "rate limit" in text
            or "couldn't parse the json" in text or "expecting value" in text)


class AngelClient(BrokerClient):
    name = "angel"
    label = "Angel One"
    capabilities = Capabilities(
        paper_trading=False, stop_orders=True, gtc_orders=False, short_selling=False,
        market_scan=True, markets=("NSE", "BSE"), status="experimental",
        notes="No paper account - start with tiny size. TOTP login is automatic. "
              "Exits are DAY orders re-armed each session.",
    )

    def __init__(self, market: str | None = None, connect_timeout: float | None = None, api=None,
                 scrip: list | None = None):
        self.market = (market or os.environ.get("ANGEL_MARKET", "NSE")).upper()
        if self.market not in ("NSE", "BSE"):
            raise BrokerError(f"Angel market must be NSE or BSE, got {self.market!r}")
        self.sandbox = False
        self.product = os.environ.get("ANGEL_PRODUCT", "INTRADAY").upper()
        self.limit_pct = float(os.environ.get("ANGEL_LIMIT_PCT", "0.5"))
        self.account = os.environ.get("ANGEL_CLIENT_CODE", "")
        self._scrip: dict[str, dict] = {}
        self._scrip_day = None
        if scrip is not None:
            self._index_scrip(scrip)
        if api is None:
            try:
                import pyotp
                from SmartApi import SmartConnect
            except ImportError as exc:
                raise BrokerError("Angel needs: pip install smartapi-python pyotp") from exc
            need = ["ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_MPIN", "ANGEL_TOTP_SECRET"]
            missing = [k for k in need if not os.environ.get(k)]
            if missing:
                raise BrokerError(f"Angel credentials missing in .env: {', '.join(missing)}")
            api = SmartConnect(api_key=os.environ["ANGEL_API_KEY"], timeout=int(connect_timeout or 20))
            totp = pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now()
            resp = api.generateSession(os.environ["ANGEL_CLIENT_CODE"], os.environ["ANGEL_MPIN"], totp)
            if not resp or not resp.get("status"):
                raise BrokerError(f"Angel login failed: {(resp or {}).get('message') or resp}")
        self.api = api
        self._candle_lock = threading.Lock()
        self._last_candle_at = 0.0

    # ---- symbols ----------------------------------------------------
    def _index_scrip(self, rows: list):
        self._scrip = {}
        for r in rows:
            self._scrip[f"{r.get('exch_seg')}.{r.get('symbol')}"] = r
        self._scrip_day = datetime.now().date()

    def _load_scrip(self):
        if self._scrip and self._scrip_day == datetime.now().date():
            return
        if not SCRIP_FILE.exists() or time.time() - SCRIP_FILE.stat().st_mtime > 86400:
            import requests
            SCRIP_FILE.parent.mkdir(parents=True, exist_ok=True)
            resp = requests.get(SCRIP_URL, timeout=120)
            resp.raise_for_status()
            SCRIP_FILE.write_text(resp.text)
        self._index_scrip(json.loads(SCRIP_FILE.read_text()))

    def _ids(self, symbol: str) -> tuple[str, str, str, float | None]:
        """canonical -> (exchange, tradingsymbol, token, tick)."""
        market, ticker = split_symbol(symbol, self.market)
        ex = market if market in ("NSE", "BSE") else self.market
        ts = ticker if (ticker.endswith("-EQ") or ex == "BSE") else f"{ticker}-EQ"
        self._load_scrip()
        row = self._scrip.get(f"{ex}.{ts}") or self._scrip.get(f"{ex}.{ticker}")
        if not row:
            raise BrokerError(f"Unknown Angel instrument {symbol}")
        tick = _num(row.get("tick_size"))
        return ex, row["symbol"], str(row["token"]), (tick / 100 if tick else None)   # paise -> rupees

    @staticmethod
    def _canonical(ex: str, ts: str) -> str:
        return f"{ex}.{ts.removesuffix('-EQ')}"

    def _normalize_symbol(self, symbol: str) -> str:
        return self._ids(symbol)[1]

    def _round(self, price: float, symbol: str, mode: str = "nearest") -> float:
        try:
            tick = self._ids(symbol)[3]
        except Exception:  # noqa: BLE001
            tick = None
        return round_to_instrument(price, tick, self.market, mode)

    def _beyond(self, side: str, price: float, symbol: str) -> float:
        """A limit ANGEL_LIMIT_PCT past `price` in the trade's direction,
        rounded further that way - fills like a market order would."""
        k = self.limit_pct / 100
        if side == "BUY":
            return self._round(price * (1 + k), symbol, "up")
        return self._round(price * (1 - k), symbol, "down")

    def round_price(self, price: float) -> float:
        return round_to_tick(price, self.market)

    @property
    def exit_time_in_force(self) -> str:
        return "DAY"

    def ping(self) -> tuple[bool, str]:
        try:
            _data(self.api.rmsLimit())
            return True, "connected"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:200]

    # ---- market data ------------------------------------------------
    def get_historical_bars(self, symbol: str, resolution: str, num_points: int) -> pd.DataFrame:
        ex, ts, token, _ = self._ids(symbol)
        interval = INTERVAL[resolution]
        end = datetime.now()
        start = end - history_window(resolution, num_points)

        def fetch(a, b):
            return self._candles({
                "exchange": ex, "symboltoken": token, "interval": interval,
                "fromdate": a.strftime("%Y-%m-%d %H:%M"), "todate": b.strftime("%Y-%m-%d %H:%M"),
            })
        rows = fetch_chunked(fetch, start, end, MAX_DAYS[interval], pause=0)   # _candles paces itself
        return to_frame([tuple(r[:6]) for r in rows], resolution, num_points, self._canonical(ex, ts))

    def _candles(self, params: dict) -> list:
        """One candle request, spaced from the previous one (all threads
        share the account's limit) and retried when Angel throttles it."""
        wait = CANDLE_BACKOFF
        for attempt in range(CANDLE_RETRIES + 1):
            with self._candle_lock:
                gap = CANDLE_MIN_INTERVAL - (time.monotonic() - self._last_candle_at)
                if gap > 0:
                    time.sleep(gap)
                try:
                    return _data(self.api.getCandleData(params)) or []
                except Exception as exc:  # noqa: BLE001
                    if attempt == CANDLE_RETRIES or not _rate_limited(exc):
                        raise
                    logger.info("Angel throttled a candle request (%s) - retry %d/%d in %.0fs",
                                str(exc)[:80], attempt + 1, CANDLE_RETRIES, wait)
                finally:
                    self._last_candle_at = time.monotonic()
            time.sleep(wait)
            wait *= 2
        return []

    def get_current_price(self, symbol: str) -> float:
        ex, ts, token, _ = self._ids(symbol)
        price = _num((_data(self.api.ltpData(ex, ts, token)) or {}).get("ltp"))
        if not price:
            raise BrokerError(f"No Angel price for {symbol}")
        return price

    def get_snapshots(self, symbols: list[str]) -> pd.DataFrame:
        ids = {}
        for s in symbols:
            try:
                ex, ts, token, _ = self._ids(s)
                ids.setdefault(ex, {})[token] = self._canonical(ex, ts)
            except BrokerError:
                continue
        rows = []
        for ex, tokens in ids.items():
            toks = list(tokens)
            for i in range(0, len(toks), 50):            # getMarketData: 50 tokens per call
                data = _data(self.api.getMarketData("FULL", {ex: toks[i:i + 50]})) or {}
                for q in data.get("fetched", []) or []:
                    rows.append({"code": tokens.get(str(q.get("symbolToken")), q.get("tradingSymbol")),
                                 "last_price": _num(q.get("ltp")), "volume": _num(q.get("tradeVolume")),
                                 "prev_close_price": _num(q.get("close"))})
                time.sleep(0.35)
        return pd.DataFrame(rows, columns=["code", "last_price", "volume", "prev_close_price"])

    def list_market_symbols(self) -> list[str]:
        self._load_scrip()
        return [self._canonical(r["exch_seg"], r["symbol"]) for r in self._scrip.values()
                if r.get("exch_seg") == self.market and str(r.get("symbol", "")).endswith("-EQ")]

    # ---- account ----------------------------------------------------
    def get_open_positions(self) -> list[dict]:
        out = []
        for p in _data(self.api.position()) or []:
            q = int(float(p.get("netqty") or 0))
            if q:
                out.append({"symbol": self._canonical(p["exchange"], p["tradingsymbol"]),
                            "name": p.get("symbolname") or p.get("tradingsymbol"), "quantity": q,
                            "average_cost": _num(p.get("avgnetprice") or p.get("netprice")),
                            "market_price": _num(p.get("ltp")),
                            "unrealized_pnl": _num(p.get("unrealised") or p.get("pnl")), "currency": "INR"})
        holding = _data(self.api.holding()) or []
        if isinstance(holding, dict):          # newer API: {"holdings": [...], "totalholding": {...}}
            holding = holding.get("holdings") or []
        for h in holding:
            q = int(float(h.get("quantity") or 0)) + int(float(h.get("t1quantity") or 0))
            if q:
                out.append({"symbol": self._canonical(h["exchange"], h["tradingsymbol"]),
                            "name": h.get("tradingsymbol"), "quantity": q,
                            "average_cost": _num(h.get("averageprice")), "market_price": _num(h.get("ltp")),
                            "unrealized_pnl": _num(h.get("profitandloss")), "currency": "INR"})
        return out

    def _order_record(self, o: dict) -> dict:
        st = str(o.get("status", "")).lower()
        return {"id": str(o["orderid"]), "symbol": self._canonical(o["exchange"], o["tradingsymbol"]),
                "action": o.get("transactiontype"), "order_type": o.get("ordertype"),
                "quantity": int(float(o.get("quantity") or 0)), "filled": int(float(o.get("filledshares") or 0)),
                "limit_price": _num(o.get("price")) or None, "stop_price": _num(o.get("triggerprice")) or None,
                "status": st, "time_in_force": o.get("duration"), "order_time": o.get("updatetime"),
                "variety": o.get("variety"), "avg_fill_price": _num(o.get("averageprice")) or None}

    def _book(self) -> list[dict]:
        return [self._order_record(o) for o in (_data(self.api.orderBook()) or [])]

    def get_open_orders(self) -> list[dict]:
        return [o for o in self._book() if o["status"] in ACTIVE]

    def get_account_summary(self) -> dict:
        d = _data(self.api.rmsLimit()) or {}
        return {"currency": "INR", "total_assets": _num(d.get("net")), "cash": _num(d.get("availablecash")),
                "market_val": _num(d.get("utiliseddebits")), "power": _num(d.get("net"))}

    def get_order_state(self, order_id) -> dict:
        for o in self._book():
            if o["id"] == str(order_id):
                return {"status": o["status"], "active": o["status"] in ACTIVE, "quantity": o["quantity"],
                        "filled": o["filled"], "avg_fill_price": o["avg_fill_price"]}
        # The order book holds today's orders only: an older DAY order has expired.
        return {"status": "expired", "active": False, "quantity": 0, "filled": 0, "avg_fill_price": None}

    # ---- orders -----------------------------------------------------
    def _place(self, params: dict) -> str:
        resp = self.api.placeOrder(params)
        oid = resp if isinstance(resp, str) else ((resp or {}).get("data") or {}).get("orderid")
        if not oid:
            raise BrokerError(f"Angel placeOrder failed: {resp}")
        return str(oid)

    def _base(self, symbol, side, quantity) -> dict:
        ex, ts, token, _ = self._ids(symbol)
        return {"variety": "NORMAL", "tradingsymbol": ts, "symboltoken": token, "transactiontype": side,
                "exchange": ex, "producttype": self.product, "duration": "DAY", "quantity": str(int(quantity))}

    def create_entry(self, symbol, direction, quantity, entry_limit_price=None, currency=None) -> str:
        logger.info("Angel %s entry: %s qty=%s limit=%s", direction, symbol, quantity, entry_limit_price)
        p = self._base(symbol, direction, quantity)
        if entry_limit_price is not None:
            price = self._round(entry_limit_price, symbol)
        else:   # no MARKET orders for algos: marketable limit instead
            price = self._beyond(direction, self.get_current_price(symbol), symbol)
        p.update(ordertype="LIMIT", price=str(price))
        return self._place(p)

    def place_stop_order(self, symbol, closing_direction, quantity, stop_price, currency=None) -> str:
        p = self._base(symbol, closing_direction, quantity)
        trigger = self._round(stop_price, symbol)
        p.update(variety="STOPLOSS", ordertype="STOPLOSS_LIMIT", triggerprice=str(trigger),
                 price=str(self._beyond(closing_direction, trigger, symbol)))
        return self._place(p)

    def place_target_order(self, symbol, closing_direction, quantity, limit_price, currency=None) -> str:
        p = self._base(symbol, closing_direction, quantity)
        p.update(ordertype="LIMIT", price=str(self._round(limit_price, symbol)))
        return self._place(p)

    def _find(self, order_id) -> dict:
        for o in self._book():
            if o["id"] == str(order_id):
                return o
        raise BrokerError(f"Angel order {order_id} not found")

    def modify_stop_order(self, order_id, new_stop_price: float):
        o = self._find(order_id)
        ex, ts, token, _ = self._ids(o["symbol"])
        trigger = self._round(new_stop_price, o["symbol"])
        _data(self.api.modifyOrder({
            "variety": "STOPLOSS", "orderid": str(order_id), "ordertype": "STOPLOSS_LIMIT",
            "producttype": self.product, "duration": "DAY",
            "price": str(self._beyond(o["action"], trigger, o["symbol"])),
            "triggerprice": str(trigger), "quantity": str(o["quantity"]),
            "tradingsymbol": ts, "symboltoken": token, "exchange": ex}))

    def cancel_order(self, order_id):
        o = self._find(order_id)
        return _data(self.api.cancelOrder(str(order_id), o.get("variety") or
                                          ("STOPLOSS" if o.get("stop_price") else "NORMAL")))

    def close_position(self, symbol, direction, quantity, currency=None) -> str:
        logger.info("Angel %s close (marketable limit): %s qty=%s", direction, symbol, quantity)
        p = self._base(symbol, direction, quantity)
        p.update(ordertype="LIMIT", price=str(self._beyond(direction, self.get_current_price(symbol), symbol)))
        return self._place(p)
