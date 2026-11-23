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
    keep lookback_bars and the number of symbols modest - or use the
    around-the-clock mode below, which streams candles instead.

Around the clock (broker market "IG24", screener.around_the_clock):
  forex, index, commodity and crypto CFDs are found by browsing IG's market
  navigation (cached for a day in data/ig_universe.json), ranked from cheap
  price snapshots, and the picked markets are streamed (src/brokers/
  ig_stream.py) - candles come from the stream and data/ig_candles, not from
  the history allowance. Each market follows its own hours (src/ig_hours.py)
  and IG's live status (TRADEABLE).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import json
from pathlib import Path

import pandas as pd

from src import ig_hours
from src.brokers import ig_stream
from src.brokers.base import BrokerClient, BrokerError, Capabilities, size_num

logger = logging.getLogger(__name__)

BASE_URLS = {"DEMO": "https://demo-api.ig.com/gateway/deal", "LIVE": "https://api.ig.com/gateway/deal"}
RESOLUTIONS = {"MIN_1": "MINUTE", "MIN_3": "MINUTE_3", "MIN_5": "MINUTE_5", "MIN_15": "MINUTE_15",
               "HOUR_1": "HOUR", "HOUR_4": "HOUR_4", "DAY": "DAY"}

def encrypt_password(password: str, encryption_key_b64: str, time_stamp) -> str:
    """The password as IG's encrypted login wants it (see IGClient._encrypted_password)."""
    import base64
    message = base64.b64encode(f"{password}|{int(time_stamp)}".encode())
    der = base64.b64decode(encryption_key_b64)
    try:
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives.serialization import load_der_public_key
        encrypted = load_der_public_key(der).encrypt(message, padding.PKCS1v15())
    except ImportError:
        try:
            from Crypto.Cipher import PKCS1_v1_5
            from Crypto.PublicKey import RSA
        except ImportError as exc:
            raise BrokerError("IG needs an encrypted password: run  pip install cryptography") from exc
        encrypted = PKCS1_v1_5.new(RSA.import_key(der)).encrypt(message)
    return base64.b64encode(encrypted).decode()


def _f(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


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


class RequestGuard:
    """Keeps this process inside IG's request allowances.

    IG allows each API key 60 non-trading requests a minute, and each
    account 30, shared by everything that uses them - here the agent and the
    dashboard. Going over returns error.public-api.exceeded-*-allowance, and
    a key that keeps knocking while over can stay blocked for hours. So:

    - non-trading requests come out of a per-process budget
      (IG_REQUESTS_PER_MIN; the agent defaults to 20, the dashboard sets 8),
      waiting up to IG_REQUEST_MAX_WAIT seconds for room;
    - when IG says an allowance is used up, every non-trading request from
      this process stops for a while (1, 2, 4 ... up to 15 minutes) instead
      of retrying, so the allowance can reset;
    - orders (create/amend/close positions and working orders) are never held
      back: they have their own, larger allowance and must always go out.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.sent: list[float] = []
        self.paths: list[tuple[float, str]] = []     # (time, request) for the budget message
        self.paused_until = 0.0
        self.strikes = 0
        self.reason = ""

    @staticmethod
    def per_min() -> int:
        try:
            return max(1, int(os.environ.get("IG_REQUESTS_PER_MIN", "20")))
        except ValueError:
            return 20

    @staticmethod
    def max_wait() -> float:
        try:
            return max(0.0, float(os.environ.get("IG_REQUEST_MAX_WAIT", "20")))
        except ValueError:
            return 20.0

    def paused_message(self) -> str:
        until = time.strftime("%H:%M:%S", time.localtime(self.paused_until))
        return (f"IG request allowance used up ({self.reason}) - pausing IG requests until {until} so it can "
                f"reset. The agent and the dashboard share the API key's allowance.")

    def busiest(self) -> str:
        """The requests that used this minute's budget, most first."""
        import collections
        now = time.time()
        recent = [p for t, p in self.paths if now - t < 60.0]
        top = collections.Counter(recent).most_common(3)
        return ", ".join(f"{p} x{n}" for p, n in top)

    def acquire(self, path: str = ""):
        """Wait for room in the budget; raise while paused or if the wait
        would be longer than IG_REQUEST_MAX_WAIT."""
        deadline = time.time() + self.max_wait()
        while True:
            with self.lock:
                now = time.time()
                if now < self.paused_until:
                    raise BrokerError(self.paused_message())
                self.sent = [t for t in self.sent if now - t < 60.0]
                if len(self.sent) < self.per_min():
                    self.sent.append(now)
                    self.paths = [(t, p) for t, p in self.paths if now - t < 60.0] + [(now, path)]
                    return
                wait = 60.0 - (now - self.sent[0]) + 0.05
            if time.time() + wait > deadline:
                used = self.busiest()
                raise BrokerError(f"IG request budget ({self.per_min()} a minute for this process) is full - "
                                  "try again shortly" + (f" (used by {used})" if used else ""))
            time.sleep(min(wait, 1.0))

    def exceeded(self, error_code: str):
        with self.lock:
            self.strikes += 1
            pause = min(900.0, 60.0 * 2 ** (self.strikes - 1))
            self.paused_until = max(self.paused_until, time.time() + pause)
            self.reason = error_code.replace("error.public-api.", "")
        logger.warning("IG says %s - pausing IG requests for %.0f s", error_code, pause)

    def ok(self):
        if self.strikes:
            with self.lock:
                self.strikes = 0

    def status(self) -> dict:
        now = time.time()
        with self.lock:
            used = len([t for t in self.sent if now - t < 60.0])
        return {"per_min": self.per_min(), "used_last_min": used,
                "paused_until": self.paused_until if self.paused_until > now else None, "reason": self.reason}


GUARD = RequestGuard()           # one per process: every IGClient here shares it
ALLOWANCE_CODES = ("error.public-api.exceeded-api-key-allowance", "error.public-api.exceeded-account-allowance")
ACCOUNT_CACHE_SECONDS = 4.0      # positions / working orders / accounts, reused within one agent cycle


SPEC_CACHE_SECONDS = 6 * 3600     # contract size / pip value / deal rules hardly ever change
FX_CACHE_SECONDS = 600            # currency conversion for P&L and sizing


def _num(text) -> float | None:
    """The leading number of IG's text fields ("100000", "0.0001 CAD/USD")."""
    import re
    m = re.match(r"\s*([0-9]*\.?[0-9]+)", str(text or ""))
    return float(m.group(1)) if m else None


def _is_trading(method: str, path: str) -> bool:
    return method != "GET" and (path.startswith("/positions") or path.startswith("/workingorders"))


class IGClient(BrokerClient):
    name = "ig"
    label = "IG"
    capabilities = Capabilities(
        paper_trading=True, stop_orders=True, gtc_orders=True, short_selling=True,
        market_scan=False, markets=("US", "HK", "SG", "IG24"), status="experimental",
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
        self._cache: dict = {}            # GET path -> (time, data) for the account endpoints
        self._specs: dict = {}            # epic -> (time, instrument_spec)
        self._fx: dict = {}               # (from, to) -> (time, rate)
        self._acct_ccy: str | None = None
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
        self._cache = {}
        resp = None
        if not IGClient._needs_encryption:
            GUARD.acquire()
            body = {"identifier": os.environ["IG_USERNAME"], "password": os.environ["IG_PASSWORD"]}
            resp = self.http.post(f"{self.base}/session", headers=self._headers(2), timeout=self.timeout, json=body)
            if resp.status_code != 200 and "encryption.required" in (resp.text or ""):
                IGClient._needs_encryption = True     # remember: later logins skip the doomed first try
        if IGClient._needs_encryption:
            # Some IG accounts only accept an RSA-encrypted password.
            password = self._encrypted_password()
            GUARD.acquire()
            body = {"identifier": os.environ["IG_USERNAME"], "password": password, "encryptedPassword": True}
            resp = self.http.post(f"{self.base}/session", headers=self._headers(2), timeout=self.timeout, json=body)
        if resp.status_code != 200:
            self._check_allowance(resp)
            raise BrokerError(f"IG login failed ({resp.status_code}): {resp.text[:200]}")
        GUARD.ok()
        self._tokens = {"CST": resp.headers.get("CST", ""),
                        "X-SECURITY-TOKEN": resp.headers.get("X-SECURITY-TOKEN", "")}
        body = resp.json() if resp.content else {}
        self.ls_endpoint = body.get("lightstreamerEndpoint") or getattr(self, "ls_endpoint", None)
        current = body.get("currentAccountId", "")
        if self.account and self.account != current:
            self._request("PUT", "/session", 1, {"accountId": self.account, "defaultAccount": False})
        self.account = self.account or current

    def _encrypted_password(self) -> str:
        """IG's encrypted login: GET /session/encryptionKey gives an RSA public
        key (base64 DER) and a timestamp; the password is
        base64(RSA-PKCS1v1.5(base64("password|timestamp")))."""
        GUARD.acquire()
        resp = self.http.get(f"{self.base}/session/encryptionKey", headers=self._headers(1), timeout=self.timeout)
        if resp.status_code != 200:
            self._check_allowance(resp)
            raise BrokerError(f"IG encryption key request failed ({resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        return encrypt_password(os.environ["IG_PASSWORD"], data["encryptionKey"], data["timeStamp"])

    _needs_encryption = False          # learnt at the first login, for the life of the process

    @staticmethod
    def _check_allowance(resp):
        """Pause this process's IG requests when IG says an allowance is used up."""
        if resp is not None and resp.status_code == 403:
            text = resp.text or ""
            for code in ALLOWANCE_CODES:
                if code in text:
                    GUARD.exceeded(code)
                    return

    def request_guard_status(self) -> dict:
        return GUARD.status()

    def _request(self, method: str, path: str, version: int = 1, body: dict | None = None,
                 params: dict | None = None, retry: bool = True) -> dict:
        cacheable = method == "GET" and not params and path in ("/positions", "/workingorders", "/accounts")
        if cacheable:
            hit = self._cache.get(path)
            if hit and time.time() - hit[0] < ACCOUNT_CACHE_SECONDS:
                return hit[1]
        trading = _is_trading(method, path)
        if not trading:
            label = path.split("?")[0]
            for pre in ("/markets/", "/prices/", "/confirms/"):
                if label.startswith(pre):
                    label = pre + "…"
            GUARD.acquire(f"{method} {label}")
        else:
            self._cache = {}                              # positions / orders are about to change
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
            self._check_allowance(resp)
            raise BrokerError(f"IG {method} {path} failed ({resp.status_code}): {resp.text[:300]}")
        GUARD.ok()
        data = resp.json() if resp.content else {}
        if cacheable:
            self._cache[path] = (time.time(), data)
        return data

    def _confirm(self, deal_reference: str, sent: dict | None = None) -> dict:
        """Deal confirmation - IG accepts the request first and decides a
        moment later; poll briefly."""
        last = None
        for _ in range(10):
            try:
                c = self._request("GET", f"/confirms/{deal_reference}", 1)
                if c.get("dealStatus"):
                    if c["dealStatus"] != "ACCEPTED":
                        what = ""
                        if sent:
                            what = (f" (sent {sent.get('direction')} size {sent.get('size')} of {sent.get('epic')}"
                                    + (f" at {sent['level']}" if sent.get("level") is not None else "")
                                    + f", currency {sent.get('currencyCode')})")
                        raise BrokerError(f"IG rejected the deal: {c.get('reason') or c['dealStatus']}{what}")
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
        store = getattr(self, "store", None)
        if store is not None and resolution == store.tf and store.has(symbol):
            return store.frame(symbol, num_points)         # streamed - no history allowance used
        if store is None and resolution in ig_stream.TF_MINUTES:
            saved = ig_stream.stored_frame(symbol, resolution, num_points)
            if saved is not None and len(saved) >= min(int(num_points), 20):
                return saved                                # the agent's candles (e.g. for the dashboard)
        if store is not None:
            # around the clock: protect the weekly history allowance from bulk
            # requests (e.g. self-learning collecting 1,500 candles per market)
            num_points = min(int(num_points), int(self.atc.get("seed_bars") or 150))
        return self._rest_bars(symbol, resolution, num_points)

    def _rest_bars(self, symbol: str, resolution: str, num_points: int) -> pd.DataFrame:
        # The same candles asked for again within a minute (the dashboard's
        # 10-second refreshes) come from here, not IG: each request uses
        # the per-minute allowance and the weekly price-history allowance.
        key = (symbol, resolution, int(num_points))
        hit = getattr(self, "_bars_cache", {}).get(key)
        if hit and time.time() - hit[0] < 60:
            return hit[1].copy()
        df = self._rest_bars_fetch(symbol, resolution, num_points)
        if not hasattr(self, "_bars_cache"):
            self._bars_cache = {}
        self._bars_cache[key] = (time.time(), df)
        return df.copy()

    def _rest_bars_fetch(self, symbol: str, resolution: str, num_points: int) -> pd.DataFrame:
        data = self._request("GET", f"/prices/{symbol}", 3,
                             params={"resolution": RESOLUTIONS[resolution], "max": int(num_points), "pageSize": 0})
        allowance = ((data.get("metadata") or {}).get("allowance") or {})
        if allowance:
            self.allowance = {"remaining": allowance.get("remainingAllowance"),
                              "total": allowance.get("totalAllowance"),
                              "resets_in_s": allowance.get("allowanceExpiry"), "ts": time.time()}
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
        stream = getattr(self, "stream", None)
        live = stream.price(symbol) if stream is not None else None
        if live:
            return float(live)
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
            qty = size_num(size * (-1 if pos.get("direction") == "SELL" else 1))
            mid = _mid({"bid": mkt.get("bid"), "offer": mkt.get("offer")})
            level = pos.get("level")
            per = _num(pos.get("contractSize")) or 1.0         # in the market's own currency
            pnl = ((mid - level) * qty * per) if (mid is not None and level is not None) else None
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
                "order_type": typ, "quantity": size_num(float(w.get("orderSize") or 0)), "filled": 0,
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
            size = size_num(float(det.get("size") or 0))
            return {"status": "FILLED", "active": False, "quantity": size, "filled": size,
                    "avg_fill_price": det.get("level") or filled.get("level")}
        if dead is not None:
            return {"status": "CANCELLED", "active": False, "quantity": 0, "filled": 0, "avg_fill_price": None}
        raise BrokerError(f"IG order {oid} not found")

    # ---- around the clock ------------------------------------------
    def enable_around_the_clock(self, settings: dict, timeframe: str, stream: bool = True, client_factory=None):
        """Streamed candles + per-market hours (broker market IG24)."""
        self.atc = dict(settings or {})
        self.store = ig_stream.CandleStore(timeframe)
        self.classes: dict[str, str] = {}      # epic -> forex / indices / commodities / crypto
        self.names: dict[str, str] = {}
        self.status: dict[str, dict] = {}      # epic -> {"status", "bid", "offer", "ts"}
        self.watched: list[str] = []
        self._status_at = 0.0
        self.stream = None
        if stream:
            try:
                if client_factory is None:
                    import lightstreamer.client  # noqa: F401
                if not self.ls_endpoint:
                    raise BrokerError("IG gave no streaming endpoint at login")
                self.stream = ig_stream.IGStream(self.ls_endpoint, self.account, self._tokens.get("CST", ""),
                                                 self._tokens.get("X-SECURITY-TOKEN", ""), self.store,
                                                 client_factory=client_factory)
                self.stream.start()
            except ImportError:
                logger.warning("IG streaming needs  pip install lightstreamer-client-lib  - using history "
                               "candles (limited by IG's weekly allowance)")
            except Exception:
                logger.exception("IG stream could not start - using history candles")
                self.stream = None

    def watch(self, epics: list[str]) -> list[str]:
        """Stream these markets: load their saved candles, fetch history only
        for what is missing, and subscribe. Returns the markets watched."""
        store = getattr(self, "store", None)
        if store is None:
            return list(epics)
        limit = ig_stream.MAX_SUBSCRIPTIONS if self.stream is None else self.stream.max
        epics = list(dict.fromkeys(epics))[:limit]
        seed = int(self.atc.get("seed_bars") or 150)
        for epic in epics:
            if epic in self.watched:
                continue
            store.load(epic)
            last = store.last_time(epic)
            missing = seed if last is None else int((time.time() * 1000 - last) // store.ms) + 2
            missing = min(seed, missing)
            if missing > 2 or last is None:
                try:
                    store.merge_history(epic, self._rest_bars(epic, store.tf, missing))
                except Exception as exc:
                    logger.warning("IG history for %s unavailable: %s", epic, exc)
        if self.stream is not None:
            try:
                self.stream.set_epics(epics)
            except Exception:
                logger.exception("IG stream subscription failed")
        self.watched = epics
        store.save(force=True)
        return epics

    def keep_up(self):
        """Every cycle: save streamed candles, refresh watched markets' live
        status once a minute, restart a stream that has been down 2 minutes."""
        if getattr(self, "store", None) is None:
            return
        self.store.save()
        if self.watched and time.time() - self._status_at >= 60:
            self._status_at = time.time()
            try:
                self.market_snapshots(self.watched)
            except Exception as exc:
                logger.warning("IG market status refresh failed: %s", exc)
        s = self.stream
        if s is not None and not s.connected and time.time() - s.status_since > 120:
            logger.warning("IG stream down for 2 min - reconnecting with a fresh login")
            try:
                s.stop()
                self._login()
                s.password = f"CST-{self._tokens.get('CST', '')}|XST-{self._tokens.get('X-SECURITY-TOKEN', '')}"
                s.endpoint = self.ls_endpoint or s.endpoint
                s.start()
                s.set_epics(self.watched)
            except Exception:
                logger.exception("IG stream reconnect failed")
            s.status_since = time.time()
        elif s is None and self.watched:
            # REST fallback: fetch only the candles that closed since the last call
            for epic in self.watched:
                last = self.store.last_time(epic)
                if last is not None and time.time() * 1000 - last >= 2 * self.store.ms:
                    n = min(int((time.time() * 1000 - last) // self.store.ms) + 1, 50)
                    try:
                        self.store.merge_history(epic, self._rest_bars(epic, self.store.tf, n))
                    except Exception as exc:
                        logger.debug("IG catch-up for %s failed: %s", epic, exc)

    def get_snapshots(self, symbols: list[str]) -> pd.DataFrame:
        """Prices for many markets in one request (instead of one per market),
        streamed prices first when the agent's stream has them."""
        rows, need = [], []
        stream = getattr(self, "stream", None)
        for s in symbols:
            live = stream.price(s) if stream is not None else None
            if live:
                rows.append({"code": s, "last_price": float(live), "volume": None})
            else:
                need.append(s)
        if need:
            for epic, snap in self.market_snapshots(need).items():
                price = _mid({"bid": snap.get("bid"), "offer": snap.get("offer")})
                if price:
                    rows.append({"code": epic, "last_price": float(price), "volume": None})
        return pd.DataFrame(rows, columns=["code", "last_price", "volume"])

    # ---- contract size and sizing --------------------------------------
    def account_currency(self) -> str:
        if not self._acct_ccy:
            try:
                self._acct_ccy = str(self.get_account_summary().get("currency") or self.currency).upper()
            except Exception:  # noqa: BLE001
                return self.currency
        return self._acct_ccy

    def instrument_spec(self, symbol: str, price: float | None = None) -> dict:
        """What one unit of IG "size" is worth, so trades are sized in IG's
        units: on IG, size is a number of contracts, and one forex contract is
        100,000 of the base currency - not one unit like a share.

        Returns value_per_point (account currency gained or lost per 1.0 move
        in the price, per 1 size), min_size and size_step (IG's dealing rules),
        currency_code (the code IG wants on orders for this market) and how it
        was worked out. Raises BrokerError when IG's figures or the currency
        conversion are unavailable - the agent then doesn't trade it."""
        spec = self._market_spec(symbol)
        acct = self.account_currency()
        rate = self.fx_rate(spec["pnl_currency"], acct, symbol=symbol, own=spec.get("pair"), price=price)
        spec.update(value_per_point=spec["per_point_instrument"] * rate, account_currency=acct, fx_rate=rate)
        return spec

    def _market_spec(self, symbol: str) -> dict:
        """IG's own figures for one market (cached): contract value per point
        in the market's currency, deal-size rules and the order currency code."""
        hit = self._specs.get(symbol)
        if hit and time.time() - hit[0] < SPEC_CACHE_SECONDS:
            return dict(hit[1])
        data = self._request("GET", f"/markets/{symbol}", 3)
        ins, rules = data.get("instrument") or {}, data.get("dealingRules") or {}
        ccys = ins.get("currencies") or []
        order_ccy = next((c.get("code") for c in ccys if c.get("isDefault")), None) or \
            (ccys[0].get("code") if ccys else self.currency)
        pip_value, pip_means = _num(ins.get("valueOfOnePip")), _num(ins.get("onePipMeans"))
        contract = _num(ins.get("contractSize"))
        if pip_value and pip_means:
            per_point = pip_value / pip_means         # e.g. 10 / 0.0001 = 100,000 per 1.0 move
        elif contract:
            per_point = contract
        else:
            raise BrokerError(f"IG gave no contract size or pip value for {symbol}")
        pair = symbol.split(".")[2] if symbol.count(".") >= 3 else ""
        is_fx = str(ins.get("type") or "").upper() == "CURRENCIES" and len(pair) == 6 and pair.isalpha()
        pnl_ccy = pair[3:] if is_fx else str(order_ccy or self.currency)
        min_size = float(((rules.get("minDealSize") or {}).get("value")) or 0) or None
        unit = str(ins.get("unit") or "CONTRACTS").upper()
        spec = {"per_point_instrument": per_point, "pnl_currency": pnl_ccy.upper(),
                "currency_code": order_ccy, "min_size": min_size,
                "size_step": 1.0 if unit == "SHARES" else 0.01, "unit": unit,
                "contract_size": contract, "pair": pair if is_fx else None}
        self._specs[symbol] = (time.time(), dict(spec))
        return spec

    def fx_rate(self, frm: str, to: str, symbol: str = "", own: str | None = None, price: float | None = None) -> float:
        """How many `to` one `frm` is worth, from IG's own forex prices."""
        frm, to = frm.upper(), to.upper()
        if frm == to:
            return 1.0
        if own and price and own == to + frm:          # e.g. USDCAD with a USD account: 1 CAD = 1/price USD
            return 1.0 / float(price)
        if own and price and own == frm + to:
            return float(price)
        hit = self._fx.get((frm, to))
        if hit and time.time() - hit[0] < FX_CACHE_SECONDS:
            return hit[1]
        style = ".".join(symbol.split(".")[3:]) if symbol.count(".") >= 3 else "CSD.IP"
        cands = {}
        for pair, invert in ((frm + to, False), (to + frm, True)):
            for st in dict.fromkeys((style, "CSD.IP", "CFD.IP", "MINI.IP")):
                cands[f"CS.D.{pair}.{st}"] = invert
        snaps = self.market_snapshots(list(cands))
        for epic, invert in cands.items():
            mid = _mid({"bid": (snaps.get(epic) or {}).get("bid"), "offer": (snaps.get(epic) or {}).get("offer")})
            if mid:
                rate = 1.0 / mid if invert else mid
                self._fx[(frm, to)] = (time.time(), rate)
                return rate
        raise BrokerError(f"No IG price to convert {frm} into the account's {to}")

    def _order_currency(self, symbol: str) -> str:
        try:
            return self._market_spec(symbol)["currency_code"] or self.currency
        except Exception:  # noqa: BLE001
            return self.currency

    def market_snapshots(self, epics: list[str]) -> dict[str, dict]:
        """Live status and prices, 50 markets per request (no history
        allowance used)."""
        out = {}
        for i in range(0, len(epics), 50):
            chunk = epics[i:i + 50]
            data = self._request("GET", "/markets", 2, params={"epics": ",".join(chunk), "filter": "SNAPSHOT_ONLY"})
            for pos, md in enumerate(data.get("marketDetails") or []):
                epic = ((md.get("instrument") or {}).get("epic")) or (chunk[pos] if pos < len(chunk) else None)
                snap = md.get("snapshot") or {}
                if not epic:
                    continue
                out[epic] = {"status": snap.get("marketStatus"), "bid": _f(snap.get("bid")),
                             "offer": _f(snap.get("offer")), "high": _f(snap.get("high")),
                             "low": _f(snap.get("low")), "change_pct": _f(snap.get("percentageChange")),
                             "ts": time.time()}
        if hasattr(self, "status"):
            self.status.update(out)
        return out

    NAV_KEYWORDS = {"forex": ("forex", "currenc", " fx"), "indices": ("indices", "index"),
                    "commodities": ("commodit", "metal", "energ"), "crypto": ("crypto", "bitcoin")}
    NAV_SKIP = ("weekend", "option", "binary", "sprint", "knock", "barrier", "bungee", "future", "forward")

    # Well-known 24-hour markets, looked up by name when the navigation menus
    # aren't available to this account (IG answers 404 for some).
    SEARCH_TERMS = {
        "forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD", "EUR/GBP",
                  "EUR/JPY", "GBP/JPY", "EUR/CHF", "AUD/JPY", "EUR/AUD", "GBP/CHF", "USD/SGD"],
        "indices": ["Wall Street", "US 500", "US Tech 100", "Germany 40", "FTSE 100", "Japan 225",
                    "Hong Kong HS50", "Australia 200", "France 40", "EU Stocks 50"],
        "commodities": ["Spot Gold", "Spot Silver", "US Crude", "Brent Crude", "Natural Gas", "Copper"],
        "crypto": ["Bitcoin", "Ether", "Litecoin", "Ripple", "Bitcoin Cash"],
    }

    def _tradeable_cfd(self, m: dict) -> bool:
        itype = str(m.get("instrumentType") or "")
        return bool(m.get("epic")) and str(m.get("expiry") or "-") == self.expiry and \
            not itype.startswith(("BINARY", "BUNGEE", "KNOCKOUTS", "OPT_", "SPRINT"))

    def search_markets(self, classes: list[str], delay_s: float = 3.0, progress=None) -> list[dict]:
        """Find the well-known markets of these classes by name
        (GET /markets?searchTerm=...): one request per name."""
        found: dict[str, dict] = {}
        done = 0
        for cls in classes:
            for term in self.SEARCH_TERMS.get(cls, []):
                if done and delay_s:
                    time.sleep(delay_s)
                done += 1
                res = None
                for attempt in (1, 2):
                    try:
                        res = self._request("GET", "/markets", 1, params={"searchTerm": term}).get("markets") or []
                        break
                    except BrokerError as exc:
                        if attempt == 1 and ("exceeded" in str(exc) or "allowance used up" in str(exc)) and delay_s:
                            wait = max(60.0, GUARD.paused_until - time.time())   # IG's limits are per minute
                            logger.info("IG request limit reached - pausing the market search for %.0f s", wait)
                            time.sleep(wait)
                            continue
                        logger.warning("IG market search %r: %s", term, exc)
                if res is None:
                    continue
                hits = [m for m in res if self._tradeable_cfd(m)]
                name = term.lower()
                exact = [m for m in hits if str(m.get("instrumentName") or "").lower() == name]
                starts = [m for m in hits if str(m.get("instrumentName") or "").lower().startswith(name)
                          and "mini" not in str(m.get("instrumentName") or "").lower()]
                pick = (exact or starts or hits)[:1]
                for m in pick:
                    if m["epic"] not in found:
                        found[m["epic"]] = {"epic": m["epic"], "name": m.get("instrumentName") or m["epic"],
                                            "class": ig_hours.classify(m.get("instrumentType"), m["epic"], cls),
                                            "instrument_type": m.get("instrumentType")}
                if progress:
                    progress(done, len(found))
        return list(found.values())

    def discover_markets(self, classes: list[str], max_nodes: int = 120, delay_s: float = 2.1,
                         progress=None) -> list[dict]:
        """Browse IG's market navigation for these asset classes; if this
        account can't (IG answers 404 for some) or it finds nothing, look the
        well-known markets up by name instead. Keeps the markets tradeable on
        this account type (expiry "-" for CFDs). Slow on purpose: IG allows 30
        such requests a minute per account."""
        try:
            found = self._navigate(classes, max_nodes, delay_s, progress)
        except BrokerError as exc:
            logger.warning("IG market navigation unavailable (%s) - searching markets by name instead",
                           str(exc).split(":")[0])
            found = []
        if not found:
            found = self.search_markets(classes, delay_s, progress)
        return found

    def _navigate(self, classes: list[str], max_nodes: int, delay_s: float, progress) -> list[dict]:
        found: dict[str, dict] = {}
        top = self._request("GET", "/marketnavigation", 1).get("nodes") or []
        queue = []
        for node in top:
            name = " " + str(node.get("name") or "").lower()
            if any(s in name for s in self.NAV_SKIP):
                continue
            # most specific first: "Cryptocurrency" must not count as forex ("currenc")
            cls = next((c for c in ("crypto", "forex", "indices", "commodities")
                        if any(k in name for k in self.NAV_KEYWORDS[c])), None)
            if cls in classes:
                queue.append((node.get("id"), cls, 0))
        visited = 0
        while queue and visited < max_nodes:
            node_id, cls, depth = queue.pop(0)
            if delay_s:
                time.sleep(delay_s)
            try:
                data = self._request("GET", f"/marketnavigation/{node_id}", 1)
            except BrokerError as exc:
                logger.warning("IG navigation node %s: %s", node_id, exc)
                continue
            visited += 1
            for m in data.get("markets") or []:
                epic = m.get("epic")
                itype = str(m.get("instrumentType") or "")
                if epic in found or not self._tradeable_cfd(m):
                    continue
                found[epic] = {"epic": epic, "name": m.get("instrumentName") or epic,
                               "class": ig_hours.classify(itype, epic, cls), "instrument_type": itype}
            if depth < 3:
                for child in data.get("nodes") or []:
                    name = str(child.get("name") or "").lower()
                    if not any(s in name for s in self.NAV_SKIP):
                        queue.append((child.get("id"), cls, depth + 1))
            if progress:
                progress(visited, len(found))
        return list(found.values())

    def _class(self, symbol: str) -> str:
        cls = getattr(self, "classes", {}).get(symbol) or ig_hours.classify(None, symbol)
        return cls or "indices"

    def symbol_open(self, symbol: str, now=None) -> bool:
        """Open for new entries: its market's hours, and IG's live status
        when known (TRADEABLE)."""
        if not ig_hours.is_open(self._class(symbol), now):
            return False
        st = getattr(self, "status", {}).get(symbol)
        if st and st.get("status") and time.time() - st.get("ts", 0) < 600:
            return st["status"] == "TRADEABLE"
        return True

    def session_minutes_for(self, symbol: str, now=None):
        return ig_hours.session_minutes(self._class(symbol), now)

    def any_open(self, now=None) -> bool:
        classes = (getattr(self, "atc", {}) or {}).get("markets") or list(ig_hours.CLASSES)
        return any(ig_hours.is_open(c, now) for c in classes)

    def next_open(self, now=None):
        classes = (getattr(self, "atc", {}) or {}).get("markets") or list(ig_hours.CLASSES)
        opens = [ig_hours.next_open(c, now) for c in classes]
        opens = [o for o in opens if o is not None]
        return min(opens) if opens else None

    def stream_summary(self) -> dict:
        s = getattr(self, "stream", None)
        return {"streaming": s.summary() if s is not None else None,
                "watched": len(getattr(self, "watched", []) or []),
                "allowance": getattr(self, "allowance", None),
                "candles": "stream" if s is not None else "history"}

    # ---- orders -----------------------------------------------------
    def _deal(self, body: dict) -> str:
        ref = self._request("POST", "/positions/otc", 2, body).get("dealReference")
        return self._confirm(ref, body)["dealId"]

    def _working(self, symbol, direction, quantity, level, typ, tif="GOOD_TILL_CANCELLED") -> str:
        body = {"epic": symbol, "expiry": self.expiry, "direction": direction, "size": quantity,
                "level": self.round_price(level), "type": typ, "timeInForce": tif,
                "guaranteedStop": False, "forceOpen": False, "currencyCode": self._order_currency(symbol)}
        ref = self._request("POST", "/workingorders/otc", 2, body).get("dealReference")
        return self._confirm(ref, body)["dealId"]

    def create_entry(self, symbol, direction, quantity, entry_limit_price=None, currency=None) -> str:
        logger.info("IG %s entry: epic=%s size=%s limit=%s", direction, symbol, quantity, entry_limit_price)
        if entry_limit_price is not None:
            return self._working(symbol, direction, quantity, entry_limit_price, "LIMIT")
        return self._deal({"epic": symbol, "expiry": self.expiry, "direction": direction, "size": quantity,
                           "orderType": "MARKET", "guaranteedStop": False, "forceOpen": False,
                           "currencyCode": self._order_currency(symbol)})

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
                           "currencyCode": self._order_currency(symbol)})
