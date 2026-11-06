"""
Tiger Brokers adapter (tigeropen SDK), ported from tiger-swing-agent's
tiger_client.py and fitted to the common BrokerClient interface.

SDK:  pip install tigeropen
Docs: https://quant.itigerup.com/openapi/en/python/quickStart/prepare.html

Credentials come from .env (never config.yaml):
    TIGER_ID, TIGER_ACCOUNT, TIGER_PRIVATE_KEY_PATH
    TIGER_SANDBOX=true      paper/sandbox (default) - false = live
    TIGER_EXIT_TIF=GTC      GTC (default) or DAY for resting stop/target

Symbols are canonical codes: "SG.D05" -> Tiger "D05.SI", "US.AAPL" -> "AAPL".
Positions and orders are reported back in canonical form, so the
orchestrator's duplicate/exposure checks match across brokers.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from src.brokers.base import BrokerClient, BrokerError, Capabilities, base_symbol, split_symbol
from src.market import round_to_tick

logger = logging.getLogger(__name__)

CURRENCY = {"US": "USD", "SG": "SGD", "HK": "HKD"}


def _sdk():
    """Import tigeropen lazily so the agent runs without it when Tiger is off."""
    try:
        from tigeropen.common.consts import BarPeriod, Market
        from tigeropen.common.util.contract_utils import stock_contract
        from tigeropen.common.util.order_utils import limit_order, market_order, stop_order
        from tigeropen.common.util.signature_utils import read_private_key
        from tigeropen.quote.quote_client import QuoteClient
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise BrokerError("Tiger needs the tigeropen SDK: pip install tigeropen") from exc
    return {
        "BarPeriod": BarPeriod, "Market": Market, "stock_contract": stock_contract,
        "limit_order": limit_order, "market_order": market_order, "stop_order": stop_order,
        "read_private_key": read_private_key, "QuoteClient": QuoteClient,
        "TigerOpenClientConfig": TigerOpenClientConfig, "TradeClient": TradeClient,
    }


def _num(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


class TigerClient(BrokerClient):
    name = "tiger"
    label = "Tiger"
    capabilities = Capabilities(
        paper_trading=True, stop_orders=True, gtc_orders=True, short_selling=True,
        market_scan=True, markets=("SG", "US", "HK"),
        notes="Ported from tiger-swing-agent. Uses .env TIGER_* credentials and a private key file.",
    )

    RESOLUTIONS = {
        "MIN_1": "ONE_MINUTE", "MIN_5": "FIVE_MINUTES", "MIN_15": "FIFTEEN_MINUTES",
        "HOUR_1": "ONE_HOUR", "HOUR_4": "FOUR_HOURS", "DAY": "DAY",
    }

    def __init__(self, market: str | None = None, connect_timeout: float | None = None,
                 quote_client=None, trade_client=None):
        self.sdk = _sdk()
        self.market = (market or os.environ.get("TIGER_MARKET", "SG")).upper()
        if self.market not in CURRENCY:
            raise BrokerError(f"Tiger market must be one of {sorted(CURRENCY)}, got {self.market!r}")
        self.sandbox = os.environ.get("TIGER_SANDBOX", "true").lower() == "true"
        self.account = os.environ.get("TIGER_ACCOUNT", "")
        self._exit_tif = os.environ.get("TIGER_EXIT_TIF", "GTC").upper()

        if quote_client is not None and trade_client is not None:   # tests / injection
            self.quote_client, self.trade_client = quote_client, trade_client
            return
        missing = [k for k in ("TIGER_ID", "TIGER_ACCOUNT", "TIGER_PRIVATE_KEY_PATH") if not os.environ.get(k)]
        if missing:
            raise BrokerError(f"Tiger credentials missing in .env: {', '.join(missing)}")
        cfg = self.sdk["TigerOpenClientConfig"](sandbox_debug=self.sandbox)
        cfg.private_key = self.sdk["read_private_key"](os.environ["TIGER_PRIVATE_KEY_PATH"])
        cfg.tiger_id = os.environ["TIGER_ID"]
        cfg.account = self.account
        if connect_timeout:
            cfg.timeout = int(connect_timeout)
        self.quote_client = self.sdk["QuoteClient"](cfg)
        self.trade_client = self.sdk["TradeClient"](cfg)

    # ---- helpers ----------------------------------------------------
    def _tiger_market(self):
        return getattr(self.sdk["Market"], self.market)

    def _normalize_symbol(self, symbol: str) -> str:
        """Canonical -> Tiger: 'SG.D05'/'D05' -> 'D05.SI', 'US.AAPL' -> 'AAPL',
        'HK.00700' -> '00700'."""
        s = str(symbol).strip().upper()
        if s.endswith(".SI"):
            return s
        market, ticker = split_symbol(s, self.market)
        return f"{ticker}.SI" if market == "SG" else ticker

    def _canonical(self, tiger_symbol: str) -> str:
        s = str(tiger_symbol).strip().upper()
        if s.endswith(".SI"):
            return f"SG.{s[:-3]}"
        return f"{self.market}.{s}" if self.market != "SG" else f"US.{s}"

    def _currency(self, tiger_symbol: str) -> str:
        if tiger_symbol.endswith(".SI"):
            return "SGD"
        return CURRENCY.get(self.market, "USD") if self.market != "SG" else "USD"

    def _contract(self, tiger_symbol: str, currency: str | None = None):
        return self.sdk["stock_contract"](tiger_symbol, currency or self._currency(tiger_symbol))

    def round_price(self, price: float) -> float:
        return round_to_tick(price, self.market)

    @property
    def exit_time_in_force(self) -> str:
        return "DAY" if self._exit_tif == "DAY" else "GTC"

    def ping(self) -> tuple[bool, str]:
        try:
            self.quote_client.get_market_status(self._tiger_market())
            return True, "connected"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:200]

    # ---- market data ------------------------------------------------
    def get_historical_bars(self, symbol: str, resolution: str, num_points: int) -> pd.DataFrame:
        sym = self._normalize_symbol(symbol)
        period = getattr(self.sdk["BarPeriod"], self.RESOLUTIONS[resolution])
        bars = self.quote_client.get_bars([sym], period=period, limit=num_points)
        cols = ["time", "open", "high", "low", "close", "volume"]
        if bars is None or len(bars) == 0:
            return pd.DataFrame(columns=cols)
        out = bars[cols].copy().sort_values("time")
        for c in cols[1:]:
            out[c] = out[c].astype(float)
        out["time"] = out["time"].astype("int64")
        out["symbol"] = self._canonical(sym)
        return out.reset_index(drop=True)

    def get_current_price(self, symbol: str) -> float:
        """Brief quote first; falls back to the latest 1-min / daily bar when
        real-time quote permission is missing (as in the original agent)."""
        sym = self._normalize_symbol(symbol)
        try:
            briefs = self.quote_client.get_stock_briefs([sym])
            if briefs is not None and len(briefs) and "latest_price" in briefs.columns:
                price = _num(briefs.iloc[0]["latest_price"])
                if price and price > 0:
                    return price
        except Exception:  # noqa: BLE001
            logger.debug("get_stock_briefs failed for %s, falling back to recent bar", sym)
        for res in ("MIN_1", "DAY"):
            try:
                bars = self.get_historical_bars(symbol, res, 1)
                if len(bars):
                    return float(bars.iloc[-1]["close"])
            except Exception:  # noqa: BLE001
                continue
        raise BrokerError(f"Could not retrieve current price for {symbol}")

    def get_snapshots(self, symbols: list[str]) -> pd.DataFrame:
        tiger_syms = [self._normalize_symbol(s) for s in symbols]
        briefs = self.quote_client.get_stock_briefs(tiger_syms)
        rows = []
        for _, r in (briefs if briefs is not None else pd.DataFrame()).iterrows():
            rows.append({"code": self._canonical(r["symbol"]), "last_price": _num(r.get("latest_price")),
                         "volume": _num(r.get("volume"))})
        return pd.DataFrame(rows, columns=["code", "last_price", "volume"])

    def list_market_symbols(self) -> list[str]:
        return [self._canonical(s) for s in self.quote_client.get_symbols(market=self._tiger_market())]

    # ---- account ----------------------------------------------------
    def _status_name(self, order) -> str:
        st = getattr(order, "status", None)
        return getattr(st, "name", str(st) if st is not None else "")

    def _order_record(self, o) -> dict:
        return {
            "id": str(o.id),
            "symbol": self._canonical(o.contract.symbol),
            "action": o.action,
            "order_type": o.order_type,
            "quantity": int(o.quantity or 0),
            "filled": int(o.filled or 0),
            "limit_price": _num(o.limit_price),
            "stop_price": _num(o.aux_price),
            "status": self._status_name(o),
            "time_in_force": getattr(o, "time_in_force", None),
            "order_time": getattr(o, "order_time", None),
        }

    def get_open_positions(self) -> list[dict]:
        out = []
        for p in self.trade_client.get_positions(account=self.account, market=self._tiger_market()) or []:
            out.append({
                "symbol": self._canonical(p.contract.symbol),
                "name": getattr(p.contract, "name", None),
                "quantity": int(p.quantity or 0),
                "average_cost": _num(p.average_cost),
                "market_price": _num(p.market_price),
                "unrealized_pnl": _num(p.unrealized_pnl),
                "currency": getattr(p.contract, "currency", None),
            })
        return out

    def get_open_orders(self) -> list[dict]:
        orders = self.trade_client.get_open_orders(account=self.account, market=self._tiger_market())
        return [self._order_record(o) for o in orders or []]

    def get_account_summary(self) -> dict:
        currency = CURRENCY.get(self.market, "USD")
        summary = None
        try:
            prime = self.trade_client.get_prime_assets(account=self.account)
            seg = (getattr(prime, "segments", {}) or {}).get("S") if prime is not None else None
            if seg is not None:
                summary = {"total_assets": _num(seg.net_liquidation), "cash": _num(seg.cash_balance),
                           "market_val": _num(getattr(seg, "gross_position_value", None)),
                           "power": _num(seg.buying_power)}
        except Exception:  # noqa: BLE001 - not a prime account
            pass
        if summary is None:
            assets = self.trade_client.get_assets(account=self.account) or []
            s = assets[0].summary if assets else None
            summary = {"total_assets": _num(getattr(s, "net_liquidation", None)),
                       "cash": _num(getattr(s, "cash", None)),
                       "market_val": _num(getattr(s, "gross_position_value", None)),
                       "power": _num(getattr(s, "buying_power", None))}
        return {"currency": currency, **summary}

    def get_order_state(self, order_id) -> dict:
        order = self.trade_client.get_order(account=self.account, id=int(order_id))
        if order is None:
            raise BrokerError(f"Order {order_id} not found")
        return {
            "status": self._status_name(order),
            "active": bool(order.active),
            "quantity": int(order.quantity or 0),
            "filled": int(order.filled or 0),
            "avg_fill_price": _num(order.avg_fill_price) or None,
        }

    def get_position_quantity(self, symbol: str) -> int:
        base = base_symbol(symbol)
        return sum(p["quantity"] for p in self.get_open_positions() if base_symbol(p["symbol"]) == base)

    # ---- orders -----------------------------------------------------
    def _place(self, order) -> str:
        oid = self.trade_client.place_order(order)
        if oid is None:
            raise BrokerError("Tiger returned no order id")
        return str(oid)

    def create_entry(self, symbol, direction, quantity, entry_limit_price=None, currency=None) -> str:
        sym = self._normalize_symbol(symbol)
        contract = self._contract(sym, currency)
        if entry_limit_price is not None:
            order = self.sdk["limit_order"](self.account, contract, direction, quantity,
                                            self.round_price(entry_limit_price))
        else:
            order = self.sdk["market_order"](self.account, contract, direction, quantity)
        logger.info("Tiger %s entry: symbol=%s qty=%s limit=%s", direction, sym, quantity, entry_limit_price)
        return self._place(order)

    def place_stop_order(self, symbol, closing_direction, quantity, stop_price, currency=None) -> str:
        sym = self._normalize_symbol(symbol)
        stop_price = self.round_price(stop_price)
        order = self.sdk["stop_order"](self.account, self._contract(sym, currency), closing_direction,
                                       quantity, stop_price, time_in_force=self.exit_time_in_force)
        logger.info("Tiger stop order: symbol=%s qty=%s stop=%s", sym, quantity, stop_price)
        return self._place(order)

    def place_target_order(self, symbol, closing_direction, quantity, limit_price, currency=None) -> str:
        sym = self._normalize_symbol(symbol)
        limit_price = self.round_price(limit_price)
        order = self.sdk["limit_order"](self.account, self._contract(sym, currency), closing_direction,
                                        quantity, limit_price, time_in_force=self.exit_time_in_force)
        logger.info("Tiger target order: symbol=%s qty=%s limit=%s", sym, quantity, limit_price)
        return self._place(order)

    def modify_stop_order(self, order_id, new_stop_price: float):
        order = self.trade_client.get_order(account=self.account, id=int(order_id))
        if order is None:
            raise BrokerError(f"Order {order_id} not found for modification")
        logger.info("Tiger modify stop %s -> %s", order_id, new_stop_price)
        return self.trade_client.modify_order(order=order, aux_price=self.round_price(new_stop_price))

    def cancel_order(self, order_id):
        return self.trade_client.cancel_order(account=self.account, id=int(order_id))

    def close_position(self, symbol, direction, quantity, currency=None) -> str:
        sym = self._normalize_symbol(symbol)
        order = self.sdk["market_order"](self.account, self._contract(sym, currency), direction, quantity)
        logger.info("Tiger %s market close: symbol=%s qty=%s", direction, sym, quantity)
        return self._place(order)
