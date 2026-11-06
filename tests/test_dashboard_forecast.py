"""The chart's Kronos badge (/api/forecast) must judge a signal the way
the agent does: closed bars only, the configured horizon and sampling,
> 50 % and >= threshold - and say 'outlook' when there is no pattern."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import dashboard.app as dash
from src.audit_log import AuditLog
from tests.test_orchestrator import engulfing_bars

ROOT = Path(__file__).resolve().parent.parent
AUTH = {"Authorization": f"Bearer {dash.DASHBOARD_TOKEN}"}


class Broker:
    market = "US"

    def __init__(self, bars):
        self.bars = bars

    def get_historical_bars(self, symbol, timeframe, n):
        return self.bars


class Kronos:
    """Forecast path whose share of bars above the last close is `up`."""

    def __init__(self, up=1.0):
        self.up, self.calls = up, []

    def forecast(self, df, forecast_horizon, sample_count=None, temperature=None, **kw):
        self.calls.append({"last_bar": df.index[-1], "rows": len(df), "horizon": forecast_horizon,
                           "sample_count": sample_count, "temperature": temperature})
        last = float(df["close"].iloc[-1])
        n_up = int(round(self.up * forecast_horizon))
        closes = [last + 1] * n_up + [last - 1] * (forecast_horizon - n_up)
        step = df.index[-1] - df.index[-2]
        idx = [df.index[-1] + step * (i + 1) for i in range(forecast_horizon)]
        return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 0}, index=idx)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    shutil.copy(ROOT / "config" / "config.example.yaml", cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.pop("brokers", None)
    cfg.update(timeframe="DAY", forecast_horizon=5, kronos_confidence_threshold=0.6, lookback_bars=200)
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(dash, "CONFIG_PATH", cfg_path)
    db = tmp_path / "audit.db"
    AuditLog(str(db)).close()
    monkeypatch.setattr(dash, "DB_PATH", db)
    dash.app.config["TESTING"] = True

    def make(bars, up):
        kronos = Kronos(up)
        monkeypatch.setattr(dash, "get_broker", lambda: Broker(bars))
        monkeypatch.setattr(dash, "get_kronos_agent", lambda: kronos)
        return kronos
    with dash.app.test_client() as c:
        yield c, make


def with_forming_bar(bars):
    """Append today's still-forming daily bar (a bearish one)."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    forming = pd.DataFrame([dict(open=72.0, high=72.0, low=69.0, close=69.5, volume=10,
                                 time=int(now.value // 10**6), symbol=bars["symbol"].iloc[0])])
    return pd.concat([bars, forming], ignore_index=True)


def post(c, **body):
    return c.post("/api/forecast", json={"symbol": "US.AAPL", **body}, headers=AUTH)


def test_uses_closed_bars_and_configured_settings(setup):
    c, make = setup
    bars = engulfing_bars()                     # ends in a bullish engulfing, days ago
    kronos = make(with_forming_bar(bars), up=1.0)
    d = post(c).get_json()
    call = kronos.calls[-1]
    assert call["last_bar"] == pd.Timestamp("2026-10-30")       # forming bar dropped
    assert call["horizon"] == 5 and d["horizon"] == 5            # config, not a hard-coded 6
    assert call["sample_count"] is None and call["temperature"] is None   # model's own config
    assert d["mode"] == "confirm" and d["pattern"]["direction"] == "BUY"
    assert d["confirmed"] is True and d["confidence"] == 1.0
    assert d["based_on_bar"] == int(pd.Timestamp("2026-10-30").timestamp())
    assert d["timeframe"] == "DAY" and d["computed_at"] > 0


def test_rejected_below_threshold(setup):
    c, make = setup
    make(engulfing_bars(), up=0.4)              # 2 of 5 bars up: 40 % < 60 %
    d = post(c).get_json()
    assert d["mode"] == "confirm" and d["confirmed"] is False and d["confidence"] == 0.4


def test_no_pattern_is_an_outlook_not_a_confirmation(setup):
    c, make = setup
    flat = engulfing_bars()
    flat.loc[flat.index[-2:], ["open", "high", "low", "close"]] = [71.0, 71.2, 70.8, 71.0]
    make(flat, up=0.0)
    d = post(c).get_json()
    assert d["pattern"] is None and d["mode"] == "outlook"
    assert d["direction"] == "SELL" and d["confidence"] == 1.0
    assert d["confirmed"] is False             # nothing to confirm


def test_explicit_overrides_still_work_for_the_simulator(setup):
    c, make = setup
    kronos = make(engulfing_bars(), up=1.0)
    d = post(c, horizon=8, temperature=0.7).get_json()
    assert d["horizon"] == 8 and kronos.calls[-1]["temperature"] == 0.7
