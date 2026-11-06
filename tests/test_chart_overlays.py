"""Chart indicator overlays returned by /api/bars."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import dashboard.app as dash
from src.audit_log import AuditLog
from src.kronos_indicators import indicators as ind_lib
from src.orchestrator import bars_to_df

ROOT = Path(__file__).resolve().parent.parent
AUTH = {"Authorization": f"Bearer {dash.DASHBOARD_TOKEN}"}


class BarsBroker:
    market = "US"

    def __init__(self):
        self.requested = []

    def get_historical_bars(self, symbol, timeframe, n):
        self.requested.append(n)
        step = pd.Timedelta(minutes=5) if timeframe == "MIN_5" else pd.Timedelta(days=1)
        # Weekday regular-session-ish times so VWAP has sessions to reset on
        end = pd.Timestamp("2026-11-03 20:55")
        times = pd.date_range(end=end, periods=n, freq=step)
        rng = np.random.default_rng(7)
        close = 300 + np.cumsum(rng.normal(0, 0.5, n))
        return pd.DataFrame({
            "time": (times.astype("datetime64[ns]").astype("int64") // 10**6).to_numpy(),
            "open": close - 0.1, "high": close + 0.6, "low": close - 0.6, "close": close,
            "volume": np.full(n, 1000.0),
        })


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    shutil.copy(ROOT / "config" / "config.example.yaml", cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["indicators"] = {"enabled": True, "mode": "filter", "ma_fast": 20, "ma_slow": 50,
                         "ma_kind": "ema", "max_rsi_for_buy": 75, "min_rsi_for_sell": 25,
                         "require_ma_trend": True}
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(dash, "CONFIG_PATH", cfg_path)
    db = tmp_path / "audit.db"
    AuditLog(str(db)).close()
    monkeypatch.setattr(dash, "DB_PATH", db)
    broker = BarsBroker()
    monkeypatch.setattr(dash, "get_broker", lambda: broker)
    monkeypatch.setattr(dash, "get_kronos_agent", lambda: None)
    dash.app.config["TESTING"] = True
    with dash.app.test_client() as c:
        c.broker = broker
        yield c


def _bars(client, **params):
    q = "&".join(f"{k}={v}" for k, v in {"symbol": "US.AAPL", **params}.items())
    return client.get(f"/api/bars?{q}", headers=AUTH).get_json()


def test_overlays_match_gate_settings_and_are_warmed_up(client):
    data = _bars(client, limit=80, timeframe="MIN_5")
    ov = data["overlays"]
    assert len(data["bars"]) == 80
    assert client.broker.requested == [80 + dash.OVERLAY_WARMUP_BARS]   # extra history for warm-up

    by_id = {l["id"]: l for l in ov["lines"]}
    assert by_id["ma_fast"]["label"] == "EMA 20" and by_id["ma_slow"]["label"] == "EMA 50"
    assert set(by_id) == {"ma_fast", "ma_slow", "ma_200", "vwap"}
    bar_times = [b["time"] for b in data["bars"]]
    for line in ov["lines"]:
        # every visible bar has a value - no blank start while the MA warms up
        assert [p["time"] for p in line["data"]] == bar_times, line["id"]
    assert ov["rsi"]["upper"] == 75 and ov["rsi"]["lower"] == 25
    assert [p["time"] for p in ov["rsi"]["data"]] == bar_times
    assert ov["gate"] == {"enabled": True, "mode": "filter", "require_ma_trend": True}


def test_overlay_values_equal_library_on_full_history(client):
    data = _bars(client, limit=50, timeframe="MIN_5")
    full = bars_to_df(client.broker.get_historical_bars("US.AAPL", "MIN_5", 50 + dash.OVERLAY_WARMUP_BARS))
    ema20 = ind_lib.moving_average(full, 20, "ema").tail(50)
    got = [p["value"] for p in next(l for l in data["overlays"]["lines"] if l["id"] == "ma_fast")["data"]]
    assert got == pytest.approx(ema20.round(4).tolist())
    rsi = ind_lib.rsi(full["close"], 14).tail(50)
    assert [p["value"] for p in data["overlays"]["rsi"]["data"]] == pytest.approx(rsi.round(4).tolist())


def test_no_vwap_on_daily_bars(client):
    data = _bars(client, limit=60, timeframe="DAY")
    assert "vwap" not in {l["id"] for l in data["overlays"]["lines"]}


def test_overlays_can_be_turned_off(client):
    data = _bars(client, limit=60, timeframe="MIN_5", overlays=0)
    assert data["overlays"] is None
    assert client.broker.requested == [60]


def test_overlay_failure_keeps_candles(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("indicator bug")
    monkeypatch.setattr(dash, "_chart_overlays", boom)
    data = _bars(client, limit=40, timeframe="MIN_5")
    assert len(data["bars"]) == 40 and data["overlays"] is None
