"""
Price data from a different broker than the one that trades.

Some broker APIs take orders but give no market data - Zerodha's free
"Personal" Kite Connect plan is the common case. With

    brokers:
      zerodha:
        data_source: angel

every price the agent reads for that broker (candles for patterns and
Kronos, the screener's snapshots, live prices for stops and the dashboard)
comes from the data broker, while orders, positions, funds and exits stay
with the trading broker. Nothing else changes: this object looks like the
trading broker to the rest of the agent.

Symbols are canonical codes ("NSE.RELIANCE") on both sides, so the two
brokers must cover the same market.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _canon(symbol: str) -> str:
    """'NSE:RELIANCE' / 'nse.reliance-eq' -> 'NSE.RELIANCE'."""
    return str(symbol).strip().upper().replace(":", ".").removesuffix("-EQ")


class DataRoutedBroker:
    """The trading broker, with market-data calls answered by another one."""

    def __init__(self, trade, data):
        self._trade = trade
        self._data = data
        self.data_label = getattr(data, "label", "data broker")
        # Prices the trading adapter needs internally (e.g. the last price a
        # Zerodha GTT must carry) also come from the data broker.
        try:
            trade.price_feed = self.get_current_price
        except Exception:  # noqa: BLE001
            logger.debug("Trading broker does not take a price feed", exc_info=True)

    # everything not defined here (orders, positions, account, attributes)
    # is the trading broker's
    def __getattr__(self, name):
        return getattr(self._trade, name)

    # ---- market data: the data broker ---------------------------------
    def get_historical_bars(self, symbol, resolution, num_points):
        return self._data.get_historical_bars(_canon(symbol), resolution, num_points)

    def get_current_price(self, symbol):
        return self._data.get_current_price(_canon(symbol))

    def get_snapshots(self, symbols):
        wanted = {_canon(s): _canon(s) for s in symbols}
        df = self._data.get_snapshots(list(wanted))
        if df is None or df.empty:
            return df
        df = df.copy()
        df["code"] = [_canon(c) for c in df["code"]]
        return df[df["code"].isin(wanted)].reset_index(drop=True)

    def list_market_symbols(self):
        # Prefer the trading broker's list: every symbol on it can be ordered.
        try:
            return self._trade.list_market_symbols()
        except Exception as exc:  # noqa: BLE001
            logger.info("%s can't list the market (%s) - using %s's list",
                        getattr(self._trade, "label", "broker"), exc, self.data_label)
            return [_canon(s) for s in self._data.list_market_symbols()]

    # ---- connection: both must be up -----------------------------------
    def ping(self):
        ok, detail = self._trade.ping()
        if not ok:
            return ok, detail
        ok, detail = self._data.ping()
        if not ok:
            return False, f"price data ({self.data_label}): {detail}"[:200]
        return True, f"connected · prices from {self.data_label}"

    def close(self):
        for client in (self._data, self._trade):
            try:
                client.close()
            except Exception:  # noqa: BLE001
                logger.debug("Closing broker failed", exc_info=True)
