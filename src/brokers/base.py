"""
The broker interface every adapter implements.

This is the exact set of calls the orchestrator, screener and dashboard
already make on the broker - written down so a new broker only has to
provide them. Nothing in the Kronos pipeline (patterns -> indicator gate
-> Kronos -> risk) knows which broker it is talking to.

Conventions (all adapters):
  * symbols are canonical "<MARKET>.<TICKER>" codes ("US.AAPL", "SG.D05",
    "HK.00700"); an adapter converts to its own format internally and
    reports positions/orders back in canonical form. A broker whose
    instruments don't fit that scheme (IG epics) uses its native code.
  * get_historical_bars returns a DataFrame with columns time (epoch ms,
    bar START, UTC), open, high, low, close, volume - oldest first.
  * positions / orders are plain dicts (see the docstrings below).
  * direction is "BUY" or "SELL"; quantities are positive ints, position
    quantities are signed (negative = short).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import pandas as pd

# Market prefixes used in canonical symbol codes.
CODE_PREFIXES = {"US", "HK", "SG", "SH", "SZ", "JP", "MY", "AU", "CA", "NSE", "BSE", "UK"}


def base_symbol(symbol: str) -> str:
    """'US.AAPL' -> 'AAPL', 'AAPL' -> 'AAPL'. Used to match broker records
    that may or may not carry the market prefix. Same rule MoomooClient
    always used; adapters report canonical codes, so no suffix handling."""
    s = str(symbol).strip().upper()
    head, _, rest = s.partition(".")
    return rest if rest and head in CODE_PREFIXES else s


def split_symbol(symbol: str, default_market: str) -> tuple[str, str]:
    """'SG.D05' -> ('SG', 'D05'); 'D05' -> (default_market, 'D05')."""
    s = str(symbol).strip().upper()
    head, _, rest = s.partition(".")
    if rest and head in CODE_PREFIXES:
        return head, rest
    return default_market.upper(), s


@dataclass
class Capabilities:
    paper_trading: bool = False
    stop_orders: bool = False
    gtc_orders: bool = False
    short_selling: bool = False
    market_scan: bool = False          # can list a whole market for the screener
    markets: tuple = ()
    status: str = "ready"              # ready | experimental | planned
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["markets"] = list(self.markets)
        return d


class BrokerError(RuntimeError):
    pass


class BrokerClient(ABC):
    """Base class for broker adapters. Subclasses set `name`, `label`,
    `capabilities`, and the attributes `market` (e.g. "US"), `account`
    (display id), `sandbox` (True = paper/demo)."""

    name: str = "base"
    label: str = "Broker"              # used in logs and Telegram ("moomoo OpenD", "Tiger", ...)
    capabilities = Capabilities()

    market: str = "US"
    account: str = ""
    sandbox: bool = True
    # Set when another broker supplies this one's prices (data_source): a
    # callable(symbol) -> last price. Adapters that need a price while
    # placing orders (Zerodha GTTs) must go through get_current_price so
    # they honour it.
    price_feed = None

    # ---- connection -------------------------------------------------
    def ping(self) -> tuple[bool, str]:
        """(ok, detail). Default: a cheap account call."""
        try:
            self.get_account_summary()
            return True, "connected"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:200]

    def close(self) -> None:
        pass

    @property
    def exit_time_in_force(self) -> str:
        """'GTC' or 'DAY' for resting stop/target orders. With DAY the
        orchestrator re-places exits at each session open."""
        return "GTC"

    # ---- symbols ----------------------------------------------------
    def _normalize_symbol(self, symbol: str) -> str:
        """Canonical code -> the broker's own instrument code."""
        return symbol

    @staticmethod
    def _base_symbol(symbol: str) -> str:
        return base_symbol(symbol)

    def round_price(self, price: float) -> float:
        from src.market import round_to_tick
        return round_to_tick(price, self.market)

    # ---- market data ------------------------------------------------
    @abstractmethod
    def get_historical_bars(self, symbol: str, resolution: str, num_points: int) -> pd.DataFrame: ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> float: ...

    def get_snapshots(self, symbols: list[str]) -> pd.DataFrame:
        """Columns code, last_price, volume (volume may be None when the
        broker has none - the screener then judges volume on bars)."""
        rows = []
        for s in symbols:
            try:
                rows.append({"code": s, "last_price": self.get_current_price(s), "volume": None})
            except Exception:  # noqa: BLE001
                continue
        return pd.DataFrame(rows, columns=["code", "last_price", "volume"])

    def list_market_symbols(self) -> list[str]:
        raise BrokerError(f"{self.label} can't list a whole market - set screener.candidate_symbols")

    # ---- account ----------------------------------------------------
    @abstractmethod
    def get_open_positions(self) -> list[dict]:
        """[{symbol, name, quantity (signed), average_cost, market_price,
        unrealized_pnl, currency}]"""

    @abstractmethod
    def get_open_orders(self) -> list[dict]:
        """Working orders: [{id, symbol, action, order_type, quantity, filled,
        limit_price, stop_price, status, time_in_force, order_time}]"""

    @abstractmethod
    def get_account_summary(self) -> dict:
        """{currency, total_assets, cash, market_val, power}"""

    @abstractmethod
    def get_order_state(self, order_id) -> dict:
        """{status, active, quantity, filled, avg_fill_price}. `active` is
        False once the order can no longer fill."""

    def get_position_quantity(self, symbol: str) -> int:
        base = base_symbol(symbol)
        return sum(p["quantity"] for p in self.get_open_positions() if base_symbol(p["symbol"]) == base)

    def has_open_orders(self, symbol: str) -> bool:
        base = base_symbol(symbol)
        return any(base_symbol(o["symbol"]) == base for o in self.get_open_orders())

    def has_exposure(self, symbol: str) -> bool:
        """Any holding or working order in symbol, including ones this agent
        didn't place. Raises if the broker can't be queried - callers treat
        that as 'don't trade'."""
        return self.get_position_quantity(symbol) != 0 or self.has_open_orders(symbol)

    # ---- orders -----------------------------------------------------
    @abstractmethod
    def create_entry(self, symbol: str, direction: str, quantity: int,
                     entry_limit_price: float | None = None, currency: str | None = None) -> str:
        """Entry only (LIMIT if a price is given, else MARKET), DAY."""

    @abstractmethod
    def place_stop_order(self, symbol: str, closing_direction: str, quantity: int,
                         stop_price: float, currency: str | None = None) -> str: ...

    @abstractmethod
    def place_target_order(self, symbol: str, closing_direction: str, quantity: int,
                           limit_price: float, currency: str | None = None) -> str: ...

    @abstractmethod
    def modify_stop_order(self, order_id, new_stop_price: float): ...

    @abstractmethod
    def cancel_order(self, order_id): ...

    @abstractmethod
    def close_position(self, symbol: str, direction: str, quantity: int,
                       currency: str | None = None) -> str:
        """Market order in the closing direction. Returns the order id."""

    def place_exit_orders(self, symbol: str, closing_direction: str, quantity: int,
                          stop_price: float, target_price: float) -> tuple:
        """Resting stop + target for a FILLED entry -> (stop_id, target_id);
        either is None if rejected (the software monitor then enforces it)."""
        import logging
        log = logging.getLogger(type(self).__module__)
        stop_id = target_id = None
        try:
            stop_id = self.place_stop_order(symbol, closing_direction, quantity, stop_price)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not place broker-side stop for %s (%s) - software monitor will enforce it",
                        symbol, exc)
        try:
            target_id = self.place_target_order(symbol, closing_direction, quantity, target_price)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not place broker-side target for %s (%s) - software monitor will enforce it",
                        symbol, exc)
        return stop_id, target_id
