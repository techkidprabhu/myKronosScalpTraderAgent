"""
Zerodha Kite Connect adapter (pip install kiteconnect) - EXPERIMENTAL.

Ported from moomoo-ensemble-agent/src/brokers/zerodha.py onto this agent's
BrokerClient interface, and extended with GTT exits and NSE tick bands.
Tested against a fake Kite client only - Zerodha has no paper account, so
start with a tiny quantity and watch the first trades.

.env:
    KITE_API_KEY, KITE_API_SECRET
    KITE_ACCESS_TOKEN        optional - normally written daily by kite_login.py
    KITE_PRODUCT=CNC         CNC = delivery (swing). MIS would be squared off intraday.
    KITE_EXIT_MODE=gtt       gtt (default) or order - see below
    KITE_STOP_LIMIT_PCT=0.5  GTT stop: limit price this % beyond the trigger
    KITE_MARKET_PROTECTION=-1  sent with every MARKET / SL-M order (-1 = Zerodha's
                             automatic protection, or 1-100 %). Mandatory since
                             1 Apr 2026 - market orders without it are rejected.

Exchange rules for API orders (SEBI retail-algo framework, since 1 Apr 2026):
order requests must come from the static IP registered on the Kite
developer dashboard (developers.kite.trade -> profile -> IP whitelist). A
home connection with a changing IP will have its orders rejected; data
calls still work from anywhere.

Login: Kite access tokens expire every morning (~06:00 IST). Run
`python kite_login.py` once a day; it saves the token to
data/kite_access_token, which this adapter reads. Until then the agent waits,
retrying, and says so on Telegram.

Exits:
  * gtt (default): stop and target are single-leg GTT triggers. They live up
    to a year (no daily re-placing), survive the agent being down, and don't
    block the shares the way two pending sell orders would. When a GTT fires
    Kite places a LIMIT order - the stop's limit sits KITE_STOP_LIMIT_PCT
    beyond the trigger so it fills in a normal move; in a gap the agent's
    software monitor closes at market instead.
  * order: SL-M stop + LIMIT target as DAY orders, re-armed every session by
    the orchestrator. Kite may reject the second sell order for the same
    CNC shares; the software monitor then enforces that level.

Symbols: canonical "NSE.RELIANCE" / "BSE.500325"-style tradingsymbols.
Short selling: not with CNC - keep risk.allow_short false for this broker.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.brokers.base import BrokerClient, BrokerError, Capabilities, split_symbol
from src.brokers.india import fetch_chunked, history_window, round_to_instrument, to_frame
from src.market import round_to_tick

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
TOKEN_FILE = ROOT / "data" / "kite_access_token"

INTERVAL = {"MIN_1": "minute", "MIN_5": "5minute", "MIN_15": "15minute",
            "HOUR_1": "60minute", "HOUR_4": "60minute", "DAY": "day"}
MAX_DAYS = {"minute": 60, "5minute": 100, "15minute": 200, "60minute": 400, "day": 2000}
ACTIVE = {"OPEN", "TRIGGER PENDING", "OPEN PENDING", "VALIDATION PENDING", "PUT ORDER REQ RECEIVED",
          "MODIFY PENDING", "MODIFY VALIDATION PENDING", "AMO REQ RECEIVED"}
GTT_PREFIX = "gtt:"


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def read_access_token() -> str:
    """Token saved by kite_login.py today, else KITE_ACCESS_TOKEN."""
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    return os.environ.get("KITE_ACCESS_TOKEN", "")


class ZerodhaClient(BrokerClient):
    name = "zerodha"
    label = "Zerodha Kite"
    capabilities = Capabilities(
        paper_trading=False, stop_orders=True, gtc_orders=True, short_selling=False,
        market_scan=True, markets=("NSE", "BSE"), status="experimental",
        notes="No paper account - start with tiny size. Access token expires daily: run kite_login.py. "
              "Exits are GTT triggers (CNC delivery).",
    )

    def __init__(self, market: str | None = None, connect_timeout: float | None = None, kite=None):
        self.market = (market or os.environ.get("KITE_MARKET", "NSE")).upper()
        if self.market not in ("NSE", "BSE"):
            raise BrokerError(f"Zerodha market must be NSE or BSE, got {self.market!r}")
        self.sandbox = False
        self.product = os.environ.get("KITE_PRODUCT", "CNC").upper()
        self.exit_mode = os.environ.get("KITE_EXIT_MODE", "gtt").lower()
        self.stop_limit_pct = float(os.environ.get("KITE_STOP_LIMIT_PCT", "0.5"))
        self.market_protection = float(os.environ.get("KITE_MARKET_PROTECTION", "-1"))
        self._instruments: dict[str, dict] = {}
        self._instruments_day = None
        if kite is None:
            try:
                from kiteconnect import KiteConnect
            except ImportError as exc:
                raise BrokerError("Zerodha needs the Kite SDK: pip install kiteconnect") from exc
            key, token = os.environ.get("KITE_API_KEY"), read_access_token()
            if not key:
                raise BrokerError("KITE_API_KEY missing in .env")
            if not token:
                raise BrokerError("No Kite access token - run `python kite_login.py` (tokens expire daily)")
            kite = KiteConnect(api_key=key, timeout=int(connect_timeout or 20))
            kite.set_access_token(token)
        self.kite = kite
        try:
            profile = self.kite.profile()       # fail fast on an expired token
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ in ("TokenException", "PermissionException") or "access_token" in str(exc):
                raise BrokerError(f"Kite rejected the access token ({exc}) - it expires every morning: "
                                  "run `python kite_login.py`") from exc
            raise BrokerError(f"Kite unreachable: {exc}") from exc
        self.account = str((profile or {}).get("user_id", ""))

    # ---- symbols ----------------------------------------------------
    def _parts(self, symbol: str) -> tuple[str, str]:
        market, ticker = split_symbol(symbol, self.market)
        return (market if market in ("NSE", "BSE") else self.market), ticker

    def _normalize_symbol(self, symbol: str) -> str:
        ex, ts = self._parts(symbol)
        return f"{ex}:{ts}"

    def _instrument(self, symbol: str) -> dict:
        ex, ts = self._parts(symbol)
        today = datetime.now().date()
        if self._instruments_day != today:
            self._instruments, self._instruments_day = {}, today
        key = f"{ex}.{ts}"
        if key not in self._instruments:
            for ins in self.kite.instruments(ex):
                self._instruments[f"{ex}.{ins['tradingsymbol']}"] = ins
        if key not in self._instruments:
            raise BrokerError(f"Unknown Kite instrument {symbol}")
        return self._instruments[key]

    def _round(self, price: float, symbol: str, mode: str = "nearest") -> float:
        try:
            tick = _num(self._instrument(symbol).get("tick_size"))
        except Exception:  # noqa: BLE001
            tick = None
        return round_to_instrument(price, tick, self.market, mode)

    def round_price(self, price: float) -> float:
        return round_to_tick(price, self.market)

    @property
    def exit_time_in_force(self) -> str:
        return "GTC" if self.exit_mode == "gtt" else "DAY"

    def ping(self) -> tuple[bool, str]:
        try:
            self.kite.profile()
            return True, "connected"
        except Exception as exc:  # noqa: BLE001
            return False, f"{exc} (expired token? run kite_login.py)"[:200]

    # ---- market data ------------------------------------------------
    def get_historical_bars(self, symbol: str, resolution: str, num_points: int) -> pd.DataFrame:
        interval = INTERVAL[resolution]
        token = self._instrument(symbol)["instrument_token"]
        end = datetime.now()
        start = end - history_window(resolution, num_points)
        candles = fetch_chunked(lambda a, b: self.kite.historical_data(token, a, b, interval),
                                start, end, MAX_DAYS[interval])
        rows = [(c["date"], c["open"], c["high"], c["low"], c["close"], c.get("volume", 0)) for c in candles]
        ex, ts = self._parts(symbol)
        return to_frame(rows, resolution, num_points, f"{ex}.{ts}")

    def get_current_price(self, symbol: str) -> float:
        key = self._normalize_symbol(symbol)
        price = _num(self.kite.ltp([key])[key]["last_price"])
        if not price:
            raise BrokerError(f"No Kite price for {symbol}")
        return price

    def get_snapshots(self, symbols: list[str]) -> pd.DataFrame:
        rows = []
        keys = [self._normalize_symbol(s) for s in symbols]
        for i in range(0, len(keys), 500):           # Kite quote() takes up to 500
            for key, q in (self.kite.quote(keys[i:i + 500]) or {}).items():
                rows.append({"code": key.replace(":", "."), "last_price": _num(q.get("last_price")),
                             "volume": _num(q.get("volume"))})
        return pd.DataFrame(rows, columns=["code", "last_price", "volume"])

    def list_market_symbols(self) -> list[str]:
        return [f"{self.market}.{i['tradingsymbol']}" for i in self.kite.instruments(self.market)
                if i.get("instrument_type") == "EQ" and i.get("segment") == self.market]

    # ---- account ----------------------------------------------------
    def get_open_positions(self) -> list[dict]:
        out = []
        net = (self.kite.positions() or {}).get("net", []) or []
        holdings = [dict(h, quantity=(h.get("quantity") or 0) + (h.get("t1_quantity") or 0))
                    for h in (self.kite.holdings() or [])]
        for p in net + holdings:
            qty = int(p.get("quantity") or 0)
            if qty:
                out.append({"symbol": f"{p['exchange']}.{p['tradingsymbol']}", "name": p.get("tradingsymbol"),
                            "quantity": qty, "average_cost": _num(p.get("average_price")),
                            "market_price": _num(p.get("last_price")), "unrealized_pnl": _num(p.get("pnl")),
                            "currency": "INR"})
        return out

    def _order_record(self, o: dict) -> dict:
        return {"id": str(o["order_id"]), "symbol": f"{o['exchange']}.{o['tradingsymbol']}",
                "action": o.get("transaction_type"), "order_type": o.get("order_type"),
                "quantity": int(o.get("quantity") or 0), "filled": int(o.get("filled_quantity") or 0),
                "limit_price": _num(o.get("price")) or None, "stop_price": _num(o.get("trigger_price")) or None,
                "status": o.get("status"), "time_in_force": o.get("validity"),
                "order_time": str(o.get("order_timestamp") or "")}

    def _gtt_record(self, g: dict) -> dict:
        leg = (g.get("orders") or [{}])[0]
        cond = g.get("condition") or {}
        trigger = (cond.get("trigger_values") or [None])[0]
        return {"id": f"{GTT_PREFIX}{g['id']}", "symbol": f"{cond.get('exchange', self.market)}.{cond.get('tradingsymbol', '')}",
                "action": leg.get("transaction_type"), "order_type": "GTT",
                "quantity": int(leg.get("quantity") or 0), "filled": 0,
                "limit_price": _num(leg.get("price")), "stop_price": _num(trigger),
                "status": str(g.get("status", "")).upper(), "time_in_force": "GTT",
                "order_time": str(g.get("created_at") or "")}

    def get_open_orders(self) -> list[dict]:
        orders = [self._order_record(o) for o in (self.kite.orders() or []) if o.get("status") in ACTIVE]
        try:
            orders += [self._gtt_record(g) for g in (self.kite.get_gtts() or []) if g.get("status") == "active"]
        except Exception:  # noqa: BLE001
            logger.debug("get_gtts failed", exc_info=True)
        return orders

    def get_account_summary(self) -> dict:
        m = self.kite.margins("equity") or {}
        avail = m.get("available", {}) or {}
        used = m.get("utilised", {}) or {}
        return {"currency": "INR", "total_assets": _num(m.get("net")), "cash": _num(avail.get("cash")),
                "market_val": _num(used.get("debits")), "power": _num(m.get("net"))}

    def _regular_state(self, order_id: str) -> dict:
        try:
            hist = self.kite.order_history(order_id)
        except Exception as exc:  # noqa: BLE001
            hist = None
            logger.debug("order_history(%s) failed: %s", order_id, exc)
        if not hist:
            # Kite only serves today's orders: an older DAY order has expired.
            return {"status": "EXPIRED", "active": False, "quantity": 0, "filled": 0, "avg_fill_price": None}
        o = hist[-1]
        return {"status": o.get("status"), "active": o.get("status") in ACTIVE,
                "quantity": int(o.get("quantity") or 0), "filled": int(o.get("filled_quantity") or 0),
                "avg_fill_price": _num(o.get("average_price")) or None}

    def _gtt_child(self, g: dict) -> str | None:
        for leg in g.get("orders") or []:
            res = ((leg.get("result") or {}).get("order_result") or {})
            if res.get("order_id"):
                return str(res["order_id"])
        return None

    def get_order_state(self, order_id) -> dict:
        oid = str(order_id)
        if not oid.startswith(GTT_PREFIX):
            return self._regular_state(oid)
        g = self.kite.get_gtt(int(oid[len(GTT_PREFIX):]))
        qty = int(((g.get("orders") or [{}])[0]).get("quantity") or 0)
        status = str(g.get("status", "")).lower()
        if status == "active":
            return {"status": "GTT_ACTIVE", "active": True, "quantity": qty, "filled": 0, "avg_fill_price": None}
        if status == "triggered":
            child = self._gtt_child(g)
            if child:
                return self._regular_state(child)
            return {"status": "GTT_TRIGGER_FAILED", "active": False, "quantity": qty, "filled": 0,
                    "avg_fill_price": None}
        return {"status": f"GTT_{status.upper()}", "active": False, "quantity": qty, "filled": 0,
                "avg_fill_price": None}

    # ---- orders -----------------------------------------------------
    def _regular(self, symbol, side, quantity, order_type, price=None, trigger=None) -> str:
        ex, ts = self._parts(symbol)
        kw = dict(variety="regular", exchange=ex, tradingsymbol=ts, transaction_type=side,
                  quantity=int(quantity), product=self.product, order_type=order_type, validity="DAY")
        if price is not None:
            kw["price"] = self._round(price, symbol)
        if trigger is not None:
            kw["trigger_price"] = self._round(trigger, symbol)
        if order_type in ("MARKET", "SL-M"):
            mp = self.market_protection
            kw["market_protection"] = int(mp) if mp == int(mp) else mp
        try:
            return str(self.kite.place_order(**kw))
        except TypeError as exc:
            if "market_protection" in str(exc):
                raise BrokerError("Your kiteconnect SDK is too old for market_protection "
                                  "(required since 1 Apr 2026): pip install -U kiteconnect") from exc
            raise

    def _gtt(self, symbol, side, quantity, trigger, limit, limit_mode="nearest") -> str:
        ex, ts = self._parts(symbol)
        trigger, limit = self._round(trigger, symbol), self._round(limit, symbol, limit_mode)
        resp = self.kite.place_gtt(
            trigger_type="single", tradingsymbol=ts, exchange=ex, trigger_values=[trigger],
            last_price=self.get_current_price(symbol),
            orders=[{"exchange": ex, "tradingsymbol": ts, "transaction_type": side, "quantity": int(quantity),
                     "order_type": "LIMIT", "product": self.product, "price": limit}])
        tid = (resp or {}).get("trigger_id")
        if tid is None:
            raise BrokerError(f"Kite GTT not created: {resp}")
        return f"{GTT_PREFIX}{tid}"

    def _stop_limit(self, side: str, stop: float) -> float:
        k = self.stop_limit_pct / 100
        return stop * (1 - k) if side == "SELL" else stop * (1 + k)

    @staticmethod
    def _away(side: str) -> str:
        """Round a stop's limit away from the trigger, so the buffer never shrinks."""
        return "down" if side == "SELL" else "up"

    def create_entry(self, symbol, direction, quantity, entry_limit_price=None, currency=None) -> str:
        logger.info("Kite %s entry: %s qty=%s limit=%s", direction, symbol, quantity, entry_limit_price)
        if entry_limit_price is not None:
            return self._regular(symbol, direction, quantity, "LIMIT", price=entry_limit_price)
        return self._regular(symbol, direction, quantity, "MARKET")

    def place_stop_order(self, symbol, closing_direction, quantity, stop_price, currency=None) -> str:
        if self.exit_mode == "gtt":
            return self._gtt(symbol, closing_direction, quantity, stop_price,
                             self._stop_limit(closing_direction, stop_price), self._away(closing_direction))
        return self._regular(symbol, closing_direction, quantity, "SL-M", trigger=stop_price)

    def place_target_order(self, symbol, closing_direction, quantity, limit_price, currency=None) -> str:
        if self.exit_mode == "gtt":
            return self._gtt(symbol, closing_direction, quantity, limit_price, limit_price)
        return self._regular(symbol, closing_direction, quantity, "LIMIT", price=limit_price)

    def modify_stop_order(self, order_id, new_stop_price: float):
        oid = str(order_id)
        if not oid.startswith(GTT_PREFIX):
            return self.kite.modify_order("regular", oid, trigger_price=self.round_price(new_stop_price))
        g = self.kite.get_gtt(int(oid[len(GTT_PREFIX):]))
        cond = g.get("condition") or {}
        leg = dict((g.get("orders") or [{}])[0])
        symbol = f"{cond.get('exchange', self.market)}.{cond.get('tradingsymbol')}"
        side = leg.get("transaction_type")
        trigger = self._round(new_stop_price, symbol)
        leg = {k: leg[k] for k in ("exchange", "tradingsymbol", "transaction_type", "quantity",
                                   "order_type", "product") if k in leg}
        leg["price"] = self._round(self._stop_limit(side, trigger), symbol, self._away(side))
        self.kite.modify_gtt(trigger_id=g["id"], trigger_type="single", tradingsymbol=cond.get("tradingsymbol"),
                             exchange=cond.get("exchange"), trigger_values=[trigger],
                             last_price=self.get_current_price(symbol), orders=[leg])

    def cancel_order(self, order_id):
        oid = str(order_id)
        if not oid.startswith(GTT_PREFIX):
            return self.kite.cancel_order("regular", oid)
        g = self.kite.get_gtt(int(oid[len(GTT_PREFIX):]))
        if str(g.get("status")).lower() == "triggered":
            child = self._gtt_child(g)          # cancel the LIMIT order it placed, if still working
            if child and self._regular_state(child)["active"]:
                self.kite.cancel_order("regular", child)
            return None
        if str(g.get("status")).lower() == "active":
            return self.kite.delete_gtt(g["id"])
        return None

    def close_position(self, symbol, direction, quantity, currency=None) -> str:
        logger.info("Kite %s market close: %s qty=%s", direction, symbol, quantity)
        return self._regular(symbol, direction, quantity, "MARKET")
