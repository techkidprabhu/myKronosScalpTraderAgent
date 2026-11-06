"""
IG adapter (IG REST Trading API) - EXPERIMENTAL.

Written against IG's public REST API documentation
(https://labs.ig.com/rest-trading-api-reference) and tested only against
a fake HTTP session here - not against a real IG account. Run it on an IG
DEMO account and watch the first trades before trusting it.

Credentials come from .env (never config.yaml):
    IG_API_KEY, IG_USERNAME, IG_PASSWORD
    IG_ACC_TYPE=DEMO        DEMO (default) or LIVE
    IG_ACC_ID=              optional - switch to this account after login
    IG_CURRENCY=USD         currency code sent with orders
    IG_EXPIRY=-             "-" for CFDs (default), "DFB" for spread bets

How it maps onto the agent:
  * symbols are IG epics (e.g. "UA.D.AAPL.CASH.IP"), used as-is.
  * quantity = deal size (contracts / stake). Keep it a whole number.
  * entries: MARKET -> an OTC market deal; LIMIT -> a LIMIT working order
    (IG has no resting DAY limit on positions) - the agent's
    entry_timeout_minutes cancels it if it doesn't fill.
  * stop / target: STOP and LIMIT working orders in the closing direction
    with forceOpen=false, so a fill nets the position off. Your IG account
    must not be set to "force open" for this to close rather than hedge.
  * order state: open working orders and positions first, then the
    activity history (POSITION_OPENED, STOP_ORDER_FILLED, ...).
  * IG limits historical price points per week (10,000 on most accounts):
    keep lookback_bars and the number of symbols modest.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.brokers.base import BrokerClient, BrokerError, Capabilities

logger = logging.getLogger(__name__)

BASE_URLS = {"DEMO": "https://demo-api.ig.com/gateway/deal", "LIVE": "https://api.ig.com/gateway/deal"}
RESOLUTIONS = {"MIN_1": "MINUTE", "MIN_5": "MINUTE_5", "MIN_15": "MINUTE_15",
               "HOUR_1": "HOUR", "HOUR_4": "HOUR_4", "DAY": "DAY"}

FILLED_ACTIONS = {"POSITION_OPENED", "POSITION_CLOSED", "POSITION_PARTIALLY_CLOSED",
                  "LIMIT_ORDER_FILLED", "STOP_ORDER_FILLED"}
DEAD_ACTIONS = {"WORKING_ORDER_DELETED", "LIMIT_ORDER_DELETED", "STOP_ORDER_DELETED", "POSITION_DELETED"}


def _mid(price: dict | None) -> float | None:
    if not price:
        return None
    bid, ask = price.get("bid"), price.get("ask") if "ask" in price else price.get("offer")
    vals = [v for v in (bid, ask) if v is not None]
    if vals:
        return sum(vals) / len(vals)
    return price.get("lastTraded")


class IGClient(BrokerClient):
    name = "ig"
    label = "IG"
    capabilities = Capabilities(
        paper_trading=True, stop_orders=True, gtc_orders=True, short_selling=True,
        market_scan=False, markets=("US", "HK", "SG"), status="experimental",
        notes="Written from IG's REST docs, not tested on a real account. Symbols are IG epics; "
              "set screener.candidate_symbols (IG can't list a whole market).",
    )

    def __init__(self, market: str | None = None, connect_timeout: float | None = None, session=None):
        self.market = (market or os.environ.get("IG_MARKET", "US")).upper()
        acc_type = os.environ.get("IG_ACC_TYPE", "DEMO").upper()
        if acc_type not in BASE_URLS:
            raise BrokerError("IG_ACC_TYPE must be DEMO or LIVE")
        self.sandbox = acc_type == "DEMO"
        self.base = BASE_URLS[acc_type]
        self.timeout = float(connect_timeout or 20)
        self.currency = os.environ.get("IG_CURRENCY", "USD").upper()
        self.expiry = os.environ.get("IG_EXPIRY", "-")
        self.account = os.environ.get("IG_ACC_ID", "")
        self._tokens: dict = {}
        if session is None:
            import requests
            session = requests.Session()
        self.http = session
        self._login()

    # ---- HTTP -------------------------------------------------------
    def _headers(self, version: int) -> dict:
        h = {"X-IG-API-KEY": os.environ.get("IG_API_KEY", ""), "Version": str(version),
             "Content-Type": "application/json; charset=UTF-8", "Accept": "application/json; charset=UTF-8"}
        h.update(self._tokens)
        return h

    def _login(self):
        missing = [k for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD") if not os.environ.get(k)]
        if missing:
            raise BrokerError(f"IG credentials missing in .env: {', '.join(missing)}")
        self._tokens = {}
        resp = self.http.post(f"{self.base}/session", headers=self._headers(2), timeout=self.timeout,
                              json={"identifier": os.environ["IG_USERNAME"], "password": os.environ["IG_PASSWORD"]})
        if resp.status_code != 200:
            raise BrokerError(f"IG login failed ({resp.status_code}): {resp.text[:200]}")
        self._tokens = {"CST": resp.headers.get("CST", ""),
                        "X-SECURITY-TOKEN": resp.headers.get("X-SECURITY-TOKEN", "")}
        body = resp.json() if resp.content else {}
        current = body.get("currentAccountId", "")
        if self.account and self.account != current:
            self._request("PUT", "/session", 1, {"accountId": self.account, "defaultAccount": False})
        self.account = self.account or current

    def _request(self, method: str, path: str, version: int = 1, body: dict | None = None,
                 params: dict | None = None, retry: bool = True) -> dict:
        headers = self._headers(version)
        if method == "DELETE" and body is not None:      # IG wants DELETE-with-body as POST + _method
            method, headers["_method"] = "POST", "DELETE"
        resp = self.http.request(method, f"{self.base}{path}", headers=headers, json=body,
                                 params=params, timeout=self.timeout)
        if resp.status_code == 401 and retry:            # session expired - log in again once
            self._login()
            return self._request(method if "_method" not in headers else "DELETE", path, version,
                                 body, params, retry=False)
        if resp.status_code >= 400:
            raise BrokerError(f"IG {method} {path} failed ({resp.status_code}): {resp.text[:300]}")
        return resp.json() if resp.content else {}

    def _confirm(self, deal_reference: str) -> dict:
        """Deal confirmation - IG accepts the request first and decides a
        moment later; poll briefly."""
        last = None
        for _ in range(10):
            try:
                c = self._request("GET", f"/confirms/{deal_reference}", 1)
                if c.get("dealStatus"):
                    if c["dealStatus"] != "ACCEPTED":
                        raise BrokerError(f"IG rejected the deal: {c.get('reason') or c['dealStatus']}")
                    return c
            except BrokerError as exc:
                if "rejected" in str(exc):
                    raise
                last = exc
            time.sleep(0.5)
        raise BrokerError(f"IG gave no confirmation for {deal_reference}: {last}")

    # ---- connection -------------------------------------------------
    def ping(self) -> tuple[bool, str]:
        try:
            self._request("GET", "/session", 1)
            return True, "connected"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:200]

    def close(self):
        try:
            self._request("DELETE", "/session", 1, retry=False)
        except Exception:  # noqa: BLE001
            pass

    # ---- market data ------------------------------------------------
    def round_price(self, price: float) -> float:
        return round(float(price), 4)

    def get_historical_bars(self, symbol: str, resolution: str, num_points: int) -> pd.DataFrame:
        data = self._request("GET", f"/prices/{symbol}", 3,
                             params={"resolution": RESOLUTIONS[resolution], "max": int(num_points), "pageSize": 0})
        rows = []
        for p in data.get("prices", []):
            ts = pd.Timestamp(p.get("snapshotTimeUTC") or p["snapshotTime"])
            rows.append({
                "time": int(ts.tz_localize(None).value // 10**6) if ts.tzinfo else int(ts.value // 10**6),
                "open": _mid(p.get("openPrice")), "high": _mid(p.get("highPrice")),
                "low": _mid(p.get("lowPrice")), "close": _mid(p.get("closePrice")),
                "volume": float(p.get("lastTradedVolume") or 0),
            })
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df = df.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)
        df["symbol"] = symbol
        return df

    def _snapshot(self, symbol: str) -> dict:
        return self._request("GET", f"/markets/{symbol}", 3).get("snapshot", {})

    def get_current_price(self, symbol: str) -> float:
        snap = self._snapshot(symbol)
        price = _mid({"bid": snap.get("bid"), "offer": snap.get("offer")})
        if not price:
            raise BrokerError(f"No IG price for {symbol}")
        return float(price)

    # ---- account ----------------------------------------------------
    def get_open_positions(self) -> list[dict]:
        out = []
        for item in self._request("GET", "/positions", 2).get("positions", []):
            pos, mkt = item.get("position", {}), item.get("market", {})
            size = float(pos.get("size") or 0)
            qty = int(round(size)) * (-1 if pos.get("direction") == "SELL" else 1)
            mid = _mid({"bid": mkt.get("bid"), "offer": mkt.get("offer")})
            level = pos.get("level")
            pnl = ((mid - level) * qty) if (mid is not None and level is not None) else None
            out.append({"symbol": mkt.get("epic"), "name": mkt.get("instrumentName"), "quantity": qty,
                        "average_cost": level, "market_price": mid, "unrealized_pnl": pnl,
                        "currency": pos.get("currency"), "deal_id": pos.get("dealId")})
        return out

    def get_open_orders(self) -> list[dict]:
        out = []
        for item in self._request("GET", "/workingorders", 2).get("workingOrders", []):
            w = item.get("workingOrderData", {})
            typ = w.get("orderType")
            out.append({
                "id": w.get("dealId"), "symbol": w.get("epic"), "action": w.get("direction"),
                "order_type": typ, "quantity": int(round(float(w.get("orderSize") or 0))), "filled": 0,
                "limit_price": w.get("orderLevel") if typ == "LIMIT" else None,
                "stop_price": w.get("orderLevel") if typ == "STOP" else None,
                "status": "WORKING", "time_in_force": w.get("timeInForce"),
                "order_time": w.get("createdDateUTC") or w.get("createdDate"),
            })
        return out

    def get_account_summary(self) -> dict:
        accounts = self._request("GET", "/accounts", 1).get("accounts", [])
        acc = next((a for a in accounts if a.get("accountId") == self.account), accounts[0] if accounts else {})
        bal = acc.get("balance", {}) or {}
        return {"currency": acc.get("currency", self.currency), "total_assets": bal.get("balance"),
                "cash": bal.get("available"), "market_val": bal.get("deposit"), "power": bal.get("available")}

    def get_order_state(self, order_id) -> dict:
        oid = str(order_id)
        for o in self.get_open_orders():
            if o["id"] == oid:
                return {"status": "WORKING", "active": True, "quantity": o["quantity"], "filled": 0,
                        "avg_fill_price": None}
        for p in self.get_open_positions():
            if p.get("deal_id") == oid:
                return {"status": "FILLED", "active": False, "quantity": abs(p["quantity"]),
                        "filled": abs(p["quantity"]), "avg_fill_price": p["average_cost"]}
        frm = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
        acts = self._request("GET", "/history/activity", 3,
                             params={"from": frm, "detailed": "true", "dealId": oid, "pageSize": 50}).get("activities", [])
        filled = dead = None
        for a in acts:
            det = a.get("details", {}) or {}
            types = {x.get("actionType") for x in det.get("actions", [])}
            if a.get("status") == "REJECTED" or types & DEAD_ACTIONS:
                dead = a
            if types & FILLED_ACTIONS:
                filled = a
        if filled is not None:
            det = filled.get("details", {}) or {}
            size = int(round(float(det.get("size") or 0)))
            return {"status": "FILLED", "active": False, "quantity": size, "filled": size,
                    "avg_fill_price": det.get("level") or filled.get("level")}
        if dead is not None:
            return {"status": "CANCELLED", "active": False, "quantity": 0, "filled": 0, "avg_fill_price": None}
        raise BrokerError(f"IG order {oid} not found")

    # ---- orders -----------------------------------------------------
    def _deal(self, body: dict) -> str:
        ref = self._request("POST", "/positions/otc", 2, body).get("dealReference")
        return self._confirm(ref)["dealId"]

    def _working(self, symbol, direction, quantity, level, typ, tif="GOOD_TILL_CANCELLED") -> str:
        body = {"epic": symbol, "expiry": self.expiry, "direction": direction, "size": quantity,
                "level": self.round_price(level), "type": typ, "timeInForce": tif,
                "guaranteedStop": False, "forceOpen": False, "currencyCode": self.currency}
        ref = self._request("POST", "/workingorders/otc", 2, body).get("dealReference")
        return self._confirm(ref)["dealId"]

    def create_entry(self, symbol, direction, quantity, entry_limit_price=None, currency=None) -> str:
        logger.info("IG %s entry: epic=%s size=%s limit=%s", direction, symbol, quantity, entry_limit_price)
        if entry_limit_price is not None:
            return self._working(symbol, direction, quantity, entry_limit_price, "LIMIT")
        return self._deal({"epic": symbol, "expiry": self.expiry, "direction": direction, "size": quantity,
                           "orderType": "MARKET", "guaranteedStop": False, "forceOpen": False,
                           "currencyCode": currency or self.currency})

    def place_stop_order(self, symbol, closing_direction, quantity, stop_price, currency=None) -> str:
        return self._working(symbol, closing_direction, quantity, stop_price, "STOP")

    def place_target_order(self, symbol, closing_direction, quantity, limit_price, currency=None) -> str:
        return self._working(symbol, closing_direction, quantity, limit_price, "LIMIT")

    def modify_stop_order(self, order_id, new_stop_price: float):
        self._request("PUT", f"/workingorders/otc/{order_id}", 2,
                      {"level": self.round_price(new_stop_price), "type": "STOP",
                       "timeInForce": "GOOD_TILL_CANCELLED", "guaranteedStop": False})

    def cancel_order(self, order_id):
        self._request("DELETE", f"/workingorders/otc/{order_id}", 2)

    def close_position(self, symbol, direction, quantity, currency=None) -> str:
        """Opposite market deal with forceOpen=false, which IG nets against
        the open position (closing it)."""
        logger.info("IG %s market close: epic=%s size=%s", direction, symbol, quantity)
        return self._deal({"epic": symbol, "expiry": self.expiry, "direction": direction, "size": quantity,
                           "orderType": "MARKET", "guaranteedStop": False, "forceOpen": False,
                           "currencyCode": currency or self.currency})
