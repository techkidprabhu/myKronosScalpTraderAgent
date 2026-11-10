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


SCORE_MODES = ("disabled", "log", "rank")
SCORE_DEFAULTS = {
    "mode": "disabled",          # disabled | log (compute + show, rank by ATR% as before) | rank
    "benchmark": None,           # e.g. "US.SPY"; None = the market's first sentiment benchmark
    "rs_lookback": 60,           # bars for the relative-strength return
    "rvol_recent": 3,            # recent bars averaged for relative volume
    "rvol_period": 20,           # ...against the average of this many bars before them
    "squeeze_period": 20,        # Bollinger period for the squeeze reading
    "weight_relative_strength": 50,
    "weight_relative_volume": 25,
    "weight_squeeze": 25,
    "min_relative_strength": None,  # rank mode only: drop symbols lagging the benchmark by more than this %
}


def compute_return_pct(df: pd.DataFrame, lookback: int) -> float | None:
    """% change of the close over the last `lookback` bars (fewer if the
    history is shorter, but at least 5)."""
    close = df["close"].dropna()
    n = min(lookback, len(close) - 1)
    if n < 5 or close.iloc[-1 - n] <= 0:
        return None
    return float((close.iloc[-1] / close.iloc[-1 - n] - 1) * 100)


def compute_relative_volume(df: pd.DataFrame, recent: int = 3, period: int = 20) -> float | None:
    """Average volume of the last `recent` bars over the average of the
    `period` bars before them. 1.0 = normal, 2.0 = twice the usual."""
    vol = df["volume"].dropna()
    if len(vol) < recent + max(5, period // 2):
        return None
    base = vol.iloc[-(recent + period):-recent].mean()
    if not base or base <= 0:
        return None
    return float(vol.iloc[-recent:].mean() / base)


def compute_squeeze(df: pd.DataFrame, period: int = 20, history: int = 120) -> float | None:
    """How tight the Bollinger bands are now compared with the symbol's own
    recent history: 100 = tightest of the last `history` bars (coiled),
    0 = widest. None with too little data."""
    close = df["close"]
    mid = close.rolling(period).mean()
    width = (4 * close.rolling(period).std() / mid).dropna().tail(history)
    if len(width) < 20:
        return None
    now = width.iloc[-1]
    return float(round((width > now).sum() / (len(width) - 1) * 100, 1))


def _pct_ranks(values: list[float | None]) -> list[float]:
    """0..100 rank of each value among the non-missing ones (missing or a
    lone value = 50, so it neither helps nor hurts)."""
    known = sorted(v for v in values if v is not None)
    if len(known) < 2:
        return [50.0] * len(values)
    out = []
    for v in values:
        if v is None:
            out.append(50.0)
            continue
        below = sum(1 for k in known if k < v)
        equal = sum(1 for k in known if k == v)
        out.append((below + (equal - 1) / 2) / (len(known) - 1) * 100)
    return out


def score_candidates(candidates: list[dict], cfg: dict) -> None:
    """Adds `score` (0..100) to each candidate in place: a weighted blend of
    relative strength and relative volume (each ranked across the
    candidates) and the squeeze reading."""
    weights = [max(0.0, float(cfg.get(k) or 0)) for k in
               ("weight_relative_strength", "weight_relative_volume", "weight_squeeze")]
    total = sum(weights) or 1.0
    rs = _pct_ranks([c.get("return_pct") for c in candidates])
    rv = _pct_ranks([c.get("relative_volume") for c in candidates])
    for c, a, b in zip(candidates, rs, rv):
        sq = c.get("squeeze")
        parts = (a, b, 50.0 if sq is None else sq)
        c["score"] = round(sum(w * x for w, x in zip(weights, parts)) / total, 1)


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
        self.score_cfg = {**SCORE_DEFAULTS, **(cfg.get("score") or {})}
        mode = str(self.score_cfg.get("mode") or "disabled").lower()
        self.score_mode = mode if mode in SCORE_MODES else "disabled"
        self.sentiment_cfg = config.get("sentiment") or {}
        # What the last run saw, for the dashboard: every candidate that
        # passed the filters, with its readings, score and whether it was picked.
        self.last_report: dict | None = None

    def _benchmark_return(self) -> tuple[str | None, float | None]:
        """Return of the benchmark over the same window, or (None, None) if
        it can't be read - symbols are then compared on their own return."""
        symbol = self.score_cfg.get("benchmark")
        if not symbol:
            from src.sentiment import benchmarks_for
            defaults = benchmarks_for(self.broker.market, {"benchmarks": self.sentiment_cfg.get("benchmarks")})
            symbol = defaults[0][0] if defaults else None
        if not symbol:
            return None, None
        try:
            df = bars_to_df(self.broker.get_historical_bars(symbol, self.timeframe, self.lookback_bars))
            return symbol, compute_return_pct(df, int(self.score_cfg["rs_lookback"]))
        except Exception as exc:
            logger.warning("Screener benchmark %s unavailable (%s) - using plain returns", symbol, exc)
            return symbol, None

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

                cand = {
                    "symbol": symbol,
                    "price": float(last_price),
                    "avg_volume": float(avg_volume),
                    "atr_pct": atr_pct,
                    "trend_strength": trend,
                }
                if self.score_mode != "disabled":
                    sc = self.score_cfg
                    cand["return_pct"] = compute_return_pct(df, int(sc["rs_lookback"]))
                    cand["relative_volume"] = compute_relative_volume(
                        df, int(sc["rvol_recent"]), int(sc["rvol_period"]))
                    cand["squeeze"] = compute_squeeze(df, int(sc["squeeze_period"]))
                candidates.append(cand)
            except Exception:
                logger.exception("Screener failed to evaluate %s - skipping", symbol)
                continue

        candidates.sort(key=lambda c: c["atr_pct"], reverse=True)
        benchmark = bench_ret = None
        if self.score_mode != "disabled" and candidates:
            benchmark, bench_ret = self._benchmark_return()
            for c in candidates:
                r = c.get("return_pct")
                c["relative_strength"] = None if r is None else round(r - (bench_ret or 0.0), 2)
            score_candidates(candidates, self.score_cfg)
            if self.score_mode == "rank":
                floor = self.score_cfg.get("min_relative_strength")
                if floor is not None and bench_ret is not None:
                    candidates = [c for c in candidates
                                  if c["relative_strength"] is None or c["relative_strength"] >= float(floor)]
                candidates.sort(key=lambda c: c["score"], reverse=True)
        selected = candidates[: self.top_n]
        picked = {c["symbol"] for c in selected}
        self.last_report = {
            "updated": time.time(), "mode": self.score_mode, "timeframe": self.timeframe,
            "benchmark": benchmark, "benchmark_return_pct": bench_ret,
            "rs_lookback": int(self.score_cfg["rs_lookback"]),
            "ranked_by": "score" if self.score_mode == "rank" else "atr_pct",
            "universe": len(universe), "top_n": self.top_n,
            "candidates": [{**c, "picked": c["symbol"] in picked} for c in candidates[:50]],
        }
        logger.info(
            "Screener selected %d/%d candidates: %s",
            len(selected), len(universe), [c["symbol"] for c in selected],
        )
        return selected
