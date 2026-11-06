"""
Indicator gate: unit rules plus the pattern -> indicators -> Kronos wiring
inside SymbolWorker, using the fake broker from test_orchestrator.
"""
import numpy as np
import pandas as pd
import pytest

from src import indicator_gate as gate
from src.indicator_gate import IndicatorConfig
from tests.test_orchestrator import SYMBOL, FakeBroker, make_worker, market_open  # noqa: F401


def trend_bars(n=120, drift=0.004, seed=3, end="2026-10-30", engulf=True):
    """Daily bars trending by `drift`/bar, ending in a bullish engulfing pair."""
    rng = np.random.default_rng(seed)
    close = 70 * np.exp(np.cumsum(rng.normal(drift, 0.008, n - 2)))
    rows = []
    for i, c in enumerate(close):
        o = close[i - 1] if i else c
        rows.append(dict(open=o, high=max(o, c) * 1.003, low=min(o, c) * 0.997, close=c, volume=1000))
    last = close[-1]
    if engulf:
        rows.append(dict(open=last * 1.006, high=last * 1.007, low=last * 0.998, close=last, volume=1000))
        rows.append(dict(open=last * 0.999, high=last * 1.012, low=last * 0.998, close=last * 1.01, volume=3000))
    df = pd.DataFrame(rows)
    times = pd.date_range(end=end, periods=len(df), freq="D")
    df["time"] = (times.astype("int64") // 10**6).to_numpy()
    df["symbol"] = SYMBOL
    return df


def as_df(raw):
    from src.orchestrator import bars_to_df
    return bars_to_df(raw)


# ---------------------------------------------------------------- unit rules
def test_config_defaults_disabled_and_validates():
    assert IndicatorConfig.from_dict(None).enabled is False
    with pytest.raises(ValueError):
        IndicatorConfig.from_dict({"mode": "block"})


def test_uptrend_passes_buy_rejects_sell():
    df = as_df(trend_bars(drift=0.004))
    cfg = IndicatorConfig(enabled=True, max_rsi_for_buy=None)
    buy = gate.evaluate(df, "BUY", cfg)
    sell = gate.evaluate(df, "SELL", cfg)
    assert buy.passed and buy.alignment > 0
    assert not sell.passed and any("bias" in r for r in sell.reasons)


def test_rsi_overbought_blocks_buy():
    df = as_df(trend_bars(drift=0.01))
    v = gate.evaluate(df, "BUY", IndicatorConfig(enabled=True, max_rsi_for_buy=60))
    assert not v.passed and any("RSI" in r for r in v.reasons)


def test_require_ma_trend():
    df = as_df(trend_bars(drift=-0.004))
    v = gate.evaluate(df, "BUY", IndicatorConfig(enabled=True, min_alignment=-100,
                                                 require_ma_trend=True, max_rsi_for_buy=None))
    assert not v.passed and any("EMA20/EMA50" in r for r in v.reasons)


def test_log_mode_never_blocks():
    df = as_df(trend_bars(drift=-0.004))
    v = gate.evaluate(df, "BUY", IndicatorConfig(enabled=True, mode="log"))
    assert v.passed and v.reasons


def test_too_few_bars_skips():
    df = as_df(trend_bars(n=30))
    v = gate.evaluate(df, "BUY", IndicatorConfig(enabled=True))
    assert v.passed and v.skipped
    assert "skipped" in v.short_text()


def test_intraday_uses_market_time_vwap():
    df = as_df(trend_bars(n=200))
    df.index = pd.date_range("2026-10-27 13:30", periods=len(df), freq="5min")  # naive UTC
    v = gate.evaluate(df, "BUY", IndicatorConfig(enabled=True, max_rsi_for_buy=None),
                      timeframe="MIN_5", tz="America/New_York")
    assert any("vwap" in s for s in v.summary["signals"])


# ---------------------------------------------------------- worker wiring
IND_ON = {"indicators": {"enabled": True, "max_rsi_for_buy": None}}


def test_worker_rejects_counter_trend_before_kronos(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    broker.bars = trend_bars(drift=-0.004)
    worker, audit, _ = make_worker(tmp_path, broker, lookback_bars=120, **IND_ON)
    worker.poll_once()

    assert broker.kinds() == []
    worker.kronos_agent.confirm_direction.assert_not_called()
    worker.reporter.report_rejected.assert_called_once()
    assert "indicators" in worker.reporter.report_rejected.call_args[0][2]
    stages = [r[0] for r in audit.conn.execute("SELECT stage FROM snapshots ORDER BY id")]
    assert "indicator_check" in stages and "kronos_result" not in stages


def test_worker_with_trend_enters_and_reports_indicators(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    broker.bars = trend_bars(drift=0.004)
    worker, audit, _ = make_worker(tmp_path, broker, lookback_bars=120, **IND_ON)
    worker.poll_once()

    assert broker.kinds() == ["entry"]
    worker.kronos_agent.confirm_direction.assert_called_once()
    kwargs = worker.reporter.report_trade_opened.call_args.kwargs
    assert kwargs["indicators"].startswith("bias bullish")


def test_worker_log_mode_still_trades(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    broker.bars = trend_bars(drift=-0.004)
    worker, _, _ = make_worker(tmp_path, broker, lookback_bars=120,
                               indicators={"enabled": True, "mode": "log"})
    worker.poll_once()
    assert broker.kinds() == ["entry"]


def test_worker_disabled_is_unchanged(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    broker.bars = trend_bars(drift=-0.004)
    worker, _, _ = make_worker(tmp_path, broker, lookback_bars=120)
    worker.poll_once()
    assert broker.kinds() == ["entry"]
    assert worker.reporter.report_trade_opened.call_args.kwargs["indicators"] is None


def test_indicator_crash_does_not_stop_trading(tmp_path, market_open, monkeypatch):  # noqa: F811
    broker = FakeBroker()
    broker.bars = trend_bars(drift=-0.004)
    worker, _, _ = make_worker(tmp_path, broker, lookback_bars=120, **IND_ON)
    monkeypatch.setattr(gate, "evaluate", lambda *a, **k: 1 / 0)
    worker.poll_once()
    assert broker.kinds() == ["entry"]
