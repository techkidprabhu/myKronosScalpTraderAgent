"""Screener stock score: relative strength, relative volume and squeeze.
Off by default; 'log' shows it without changing picks; 'rank' picks by it."""
import numpy as np
import pandas as pd

from src.screener import (Screener, _pct_ranks, compute_relative_volume, compute_return_pct,
                          compute_squeeze, score_candidates)


def bars(closes, volumes=None):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    vol = np.full(n, 1_000_000.0) if volumes is None else np.asarray(volumes, dtype=float)
    return pd.DataFrame({"time": np.arange(n) * 300_000, "open": closes, "high": closes * 1.01,
                         "low": closes * 0.99, "close": closes, "volume": vol})


class FakeBroker:
    market = "US"

    def __init__(self, data):
        self.data = data

    def _normalize_symbol(self, s):
        return s

    def get_snapshots(self, symbols):
        return pd.DataFrame([{"code": s, "last_price": float(self.data[s]["close"].iloc[-1]),
                              "volume": 5_000_000} for s in symbols])

    def get_historical_bars(self, symbol, timeframe, n):
        if symbol not in self.data:
            raise RuntimeError("no data")
        return self.data[symbol]


def make(mode=None, **score):
    n = 150
    rng = np.random.default_rng(1)
    wiggle = 1 + rng.normal(0, 0.004, n)
    data = {
        # strong riser, volume surge at the end
        "US.STRONG": bars(np.linspace(100, 130, n) * wiggle, np.r_[np.full(n - 3, 1e6), np.full(3, 3e6)]),
        # flat
        "US.FLAT": bars(np.full(n, 100.0) * wiggle),
        # falling, and the most volatile (largest ATR%)
        "US.WEAK": bars(np.linspace(100, 80, n) * (1 + rng.normal(0, 0.02, n))),
        "US.SPY": bars(np.linspace(100, 103, n)),
    }
    cfg = {"timeframe": "DAY", "lookback_bars": 150,
           "screener": {"candidate_symbols": ["US.STRONG", "US.FLAT", "US.WEAK"], "top_n": 1,
                        "min_avg_volume": 0}}
    if mode:
        cfg["screener"]["score"] = {"mode": mode, **score}
    return Screener(FakeBroker(data), cfg)


def test_disabled_keeps_old_ranking_and_fields():
    s = make()
    picked = s.run()
    assert [c["symbol"] for c in picked] == ["US.WEAK"]        # highest ATR%
    assert "score" not in picked[0] and s.last_report["mode"] == "disabled"


def test_log_mode_scores_but_does_not_change_picks():
    s = make("log")
    picked = s.run()
    assert [c["symbol"] for c in picked] == ["US.WEAK"]
    rep = s.last_report
    assert rep["ranked_by"] == "atr_pct" and rep["benchmark"] == "US.SPY"
    assert all("score" in c for c in rep["candidates"])
    assert [c["symbol"] for c in rep["candidates"] if c["picked"]] == ["US.WEAK"]


def test_rank_mode_picks_the_strongest():
    s = make("rank")
    picked = s.run()
    assert [c["symbol"] for c in picked] == ["US.STRONG"]
    strong = picked[0]
    assert strong["relative_strength"] > 0 and strong["relative_volume"] > 2
    assert s.last_report["ranked_by"] == "score"


def test_min_relative_strength_drops_laggards_in_rank_mode():
    s = make("rank", min_relative_strength=0)
    s.top_n = 5
    assert [c["symbol"] for c in s.run()] == ["US.STRONG"]


def test_benchmark_failure_falls_back_to_plain_returns():
    s = make("rank", benchmark="US.MISSING")
    picked = s.run()
    assert picked[0]["symbol"] == "US.STRONG"
    assert s.last_report["benchmark_return_pct"] is None


def test_unknown_mode_is_treated_as_disabled():
    assert make("turbo").score_mode == "disabled"


def test_readings():
    df = bars(np.linspace(100, 110, 61))
    assert round(compute_return_pct(df, 60), 1) == 10.0
    assert compute_return_pct(bars([1, 2, 3]), 60) is None
    vol = np.r_[np.full(40, 100.0), np.full(3, 250.0)]
    assert compute_relative_volume(bars(np.full(43, 10.0), vol)) == 2.5
    # wide swings, then a tight coil -> squeeze near 100
    closes = np.r_[100 + 10 * np.sin(np.arange(120) / 3), 100 + 0.2 * np.sin(np.arange(40) / 3)]
    assert compute_squeeze(bars(closes)) >= 90


def test_pct_ranks_and_weights():
    assert _pct_ranks([1, 2, 3]) == [0.0, 50.0, 100.0]
    assert _pct_ranks([None, 5]) == [50.0, 50.0]
    cands = [{"return_pct": 10, "relative_volume": 1, "squeeze": 0},
             {"return_pct": 0, "relative_volume": 3, "squeeze": 100}]
    score_candidates(cands, {"weight_relative_strength": 100, "weight_relative_volume": 0, "weight_squeeze": 0})
    assert [c["score"] for c in cands] == [100.0, 0.0]
