"""
Dynamic symbol screener - replaces the static `symbols:` config list with
a periodically-refreshed set ranked by liquidity, volatility, and trend,
filtered to a configurable price range.

Universe source: moomoo's get_stock_basicinfo(market) returns every
listed stock in a market (thousands of codes), so screening the entire
market on every refresh is slow and burns K-line quota. Two options,
both supported here:
  1) `candidate_symbols` in config - an explicit list you maintain
     (fastest, recommended for frequent refreshes)
  2) leave `candidate_symbols` empty and the screener pulls the full
     market list - fine for an infrequent (e.g. daily) refresh. The
     price/volume prefilter uses batched snapshots (up to 400 codes per
     request) so only survivors cost a historical K-line request; mind
     the 7-day history quota (see moomoo "Authorities and Quota").
"""
import logging
import time

import pandas as pd

from src.orchestrator import bars_to_df
from src.moomoo_client import MoomooClient, _num

logger = logging.getLogger(__name__)


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range as a percentage of the last close."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    last_close = close.iloc[-1]
    if last_close == 0 or pd.isna(atr):
        return 0.0
    return float(atr / last_close * 100)


def compute_trend_strength(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> float:
    """Simple trend proxy: % gap between fast and slow EMA. Positive = uptrend."""
    if len(df) < slow:
        return 0.0
    ema_fast = df["close"].ewm(span=fast).mean().iloc[-1]
    ema_slow = df["close"].ewm(span=slow).mean().iloc[-1]
    if ema_slow == 0:
        return 0.0
    return float((ema_fast - ema_slow) / ema_slow * 100)


class Screener:
    SNAPSHOT_BATCH = 400  # get_market_snapshot accepts up to 400 codes

    def __init__(self, broker: MoomooClient, config: dict):
        # The screened market is the broker's MOOMOO_TRADE_MARKET - you can
        # only trade what that account/context is set up for.
        self.broker = broker
        cfg = config.get("screener", {})
        self.candidate_symbols = cfg.get("candidate_symbols", [])
        self.min_price = cfg.get("min_price", 0.0)
        self.max_price = cfg.get("max_price", float("inf"))
        self.min_avg_volume = cfg.get("min_avg_volume", 0)
        self.atr_period = cfg.get("atr_period", 14)
        self.min_atr_pct = cfg.get("min_atr_pct", 0.0)
        self.max_atr_pct = cfg.get("max_atr_pct", float("inf"))
        self.top_n = cfg.get("top_n", 10)
        self.timeframe = config["timeframe"]
        self.lookback_bars = max(config["lookback_bars"], self.atr_period + 5)

    def _get_universe(self) -> list[str]:
        if self.candidate_symbols:
            return [self.broker._normalize_symbol(s) for s in self.candidate_symbols]
        symbols = self.broker.list_market_symbols()
        logger.warning(
            "No candidate_symbols configured - screening the full %s "
            "market (%d symbols). Consider setting candidate_symbols "
            "for faster, quota-friendly refreshes.", self.broker.market, len(symbols),
        )
        return list(symbols)

    def _price_volume_prefilter(self, symbols: list[str]) -> list[str]:
        """Stage 1: cheap price/volume filter using batched snapshots
        before spending a historical K-line request on each survivor. If
        snapshots fail (e.g. no quote permission for that market), the
        batch is passed through and judged on historical bars instead."""
        survivors = []
        for i in range(0, len(symbols), self.SNAPSHOT_BATCH):
            batch = symbols[i:i + self.SNAPSHOT_BATCH]
            try:
                snaps = self.broker.get_snapshots(batch)
            except Exception as exc:
                logger.warning(
                    "Snapshot failed for batch starting at %s (%s); "
                    "bypassing prefilter and evaluating via historical bars",
                    batch[0], exc,
                )
                survivors.extend(batch)
                continue
            for _, row in snaps.iterrows():
                price = _num(row.get("last_price"))
                volume = _num(row.get("volume"))   # None: broker has no volume in quotes (IG)
                if price is None or price <= 0:
                    continue
                volume_ok = volume is None or volume >= self.min_avg_volume
                if self.min_price <= price <= self.max_price and volume_ok:
                    survivors.append(row["code"])
            time.sleep(0.5)  # snapshots: 60 requests / 30s
        return survivors

    def run(self) -> list[dict]:
        """Returns a ranked list of dicts: {symbol, price, atr_pct,
        avg_volume, trend_strength}, filtered by price/volume/ATR bounds,
        sorted by ATR% descending, truncated to top_n."""
        universe = self._get_universe()
        if not universe:
            logger.warning("Screener universe is empty")
            return []

        prefiltered = self._price_volume_prefilter(universe)
        logger.info("Price/volume prefilter: %d/%d symbols survived", len(prefiltered), len(universe))

        candidates = []
        for symbol in prefiltered:
            try:
                bars = self.broker.get_historical_bars(
                    symbol, self.timeframe, self.lookback_bars
                )
                df = bars_to_df(bars)
                if df.empty or len(df) < self.atr_period + 1:
                    continue

                last_price = df["close"].iloc[-1]
                avg_volume = df["volume"].tail(20).mean()
                atr_pct = compute_atr_pct(df, self.atr_period)
                trend = compute_trend_strength(df)

                if not (self.min_price <= last_price <= self.max_price):
                    continue
                if avg_volume < self.min_avg_volume:
                    continue
                if not (self.min_atr_pct <= atr_pct <= self.max_atr_pct):
                    continue

                candidates.append({
                    "symbol": symbol,
                    "price": float(last_price),
                    "avg_volume": float(avg_volume),
                    "atr_pct": atr_pct,
                    "trend_strength": trend,
                })
            except Exception:
                logger.exception("Screener failed to evaluate %s - skipping", symbol)
                continue

        candidates.sort(key=lambda c: c["atr_pct"], reverse=True)
        selected = candidates[: self.top_n]
        logger.info(
            "Screener selected %d/%d candidates: %s",
            len(selected), len(universe), [c["symbol"] for c in selected],
        )
        return selected
