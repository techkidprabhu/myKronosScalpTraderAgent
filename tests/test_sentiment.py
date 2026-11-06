"""Market Sentiment Meter: regime classification and the /api/sentiment
endpoint (fake broker, no network)."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import dashboard.app as dash
from src import sentiment
from src.audit_log import AuditLog

ROOT = Path(__file__).resolve().parent.parent
AUTH = {"Authorization": f"Bearer {dash.DASHBOARD_TOKEN}"}
N = 250


def frame(close, noise=0.4, seed=1):
    rng = np.random.default_rng(seed)
    c = np.asarray(close, dtype=float)
    o = c + rng.normal(0, noise, len(c))
    idx = pd.date_range("2026-01-01", periods=len(c), freq="B")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + abs(rng.normal(0, noise, len(c))),
                         "low": np.minimum(o, c) - abs(rng.normal(0, noise, len(c))), "close": c,
                         "volume": 1e6}, index=idx)


t = np.arange(N)
_r = np.random.default_rng(0)
SERIES = {
    "up": (100 + 0.4 * t + _r.normal(0, 0.6, N), 0.4),
    "down": (200 - 0.4 * t + _r.normal(0, 0.6, N), 0.4),
    "stable": (np.r_[100 + np.sin(t[:170] / 6) * 6 + _r.normal(0, 1.5, 170), 100 + _r.normal(0, 0.15, 80)], 0.05),
    "choppy": (100 + np.sin(t / 4) * 6 + _r.normal(0, 1.5, N), 0.4),
}


@pytest.mark.parametrize("name, regime", [("up", "Up trend"), ("down", "Down trend"),
                                          ("stable", "Stable"), ("choppy", "Sideways")])
def test_regimes(name, regime):
    close, noise = SERIES[name]
    r = sentiment.classify(frame(close, noise))
    assert r["regime"] == regime, r
    if regime == "Up trend":
        assert r["score"] > 50
    if regime == "Down trend":
        assert r["score"] < -50
    if regime in ("Stable", "Sideways"):
        assert -50 <= r["score"] <= 50          # no trend: needle only leans
    assert 0 <= r["confidence"] <= 1 and r["reason"]


def test_too_few_bars():
    assert "error" in sentiment.classify(frame(SERIES["up"][0][:40]))


def test_overall_majority_and_tie_break():
    up = {"regime": "Up trend", "score": 70}
    side = {"regime": "Sideways", "score": -10}
    assert sentiment.overall([up, up, side]) == {"regime": "Up trend", "score": 43.3, "agreement": 2, "of": 3}
    assert sentiment.overall([up, side])["regime"] == "Up trend"     # tie -> strongest score
    assert sentiment.overall([{"error": "x"}])["regime"] is None


def test_breadth():
    frames = {k: frame(v[0], v[1]) for k, v in SERIES.items()}
    b = sentiment.breadth(frames)
    assert b["of"] == 4 and 0 <= b["above_ema50_pct"] <= 100
    assert sentiment.breadth({"a": frame(SERIES["up"][0])}) == {"above_ema50_pct": 100, "above": 1, "of": 1,
                                                               "label": "broad strength"}


def test_benchmarks_defaults_and_overrides():
    s = sentiment.settings({})
    assert [b[0] for b in sentiment.benchmarks_for("US", s)] == ["US.SPY", "US.QQQ", "US.IWM"]
    assert [b[0] for b in sentiment.benchmarks_for("NSE", s)][0] == "NSE.NIFTYBEES"
    s = sentiment.settings({"sentiment": {"benchmarks": {"US": ["us.dia", {"symbol": "US.SPY", "name": "SPX"}],
                                                        "SG": None}}})
    assert sentiment.benchmarks_for("US", s) == [("US.DIA", "us.dia"), ("US.SPY", "SPX")]
    assert sentiment.benchmarks_for("SG", s)[0][0] == "SG.ES3"           # None -> defaults
    assert sentiment.benchmarks_for("XX", s) == []


# ---------------------------------------------------------------- API

class SentBroker:
    market = "US"
    account, sandbox = "x", True

    def __init__(self):
        self.calls = []

    def ping(self):
        return True, "ok"

    def get_historical_bars(self, symbol, timeframe, n):
        self.calls.append(symbol)
        if symbol == "US.IWM":
            raise RuntimeError("no quote right for US.IWM")
        close, noise = SERIES["up" if symbol == "US.SPY" else "choppy" if symbol == "US.QQQ" else "up"]
        df = frame(close[-n:], noise)
        return pd.DataFrame({"time": (df.index.astype("datetime64[ns]").astype("int64") // 10**6),
                             **{c: df[c].to_numpy() for c in ("open", "high", "low", "close", "volume")}})


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    shutil.copy(ROOT / "config" / "config.example.yaml", cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.pop("brokers", None)
    cfg["symbols"] = [{"symbol": "US.AAPL"}, {"symbol": "US.MSFT"}]
    cfg["screener"] = {"enabled": False}
    cfg["sentiment"] = {"enabled": True, "timeframe": "DAY", "refresh_minutes": 15, "breadth_symbols": 5}
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(dash, "CONFIG_PATH", cfg_path)
    db = tmp_path / "audit.db"
    AuditLog(str(db)).close()
    monkeypatch.setattr(dash, "DB_PATH", db)
    broker = SentBroker()
    monkeypatch.setattr(dash, "get_broker", lambda: broker)
    monkeypatch.setattr(dash, "_sentiment_cache", {})
    dash.app.config["TESTING"] = True
    with dash.app.test_client() as c:
        c.broker, c.cfg_path = broker, cfg_path
        yield c


def test_api_reads_benchmarks_and_breadth(client):
    d = client.get("/api/sentiment", headers=AUTH).get_json()
    assert d["enabled"] and d["market"] == "US" and d["timeframe"] == "DAY" and d["cached"] is False
    by = {b["symbol"]: b for b in d["benchmarks"]}
    assert by["US.SPY"]["regime"] == "Up trend" and by["US.SPY"]["name"] == "S&P 500"
    assert by["US.QQQ"]["regime"] == "Sideways"
    assert "no quote right" in by["US.IWM"]["error"]          # one failure doesn't hide the rest
    assert d["overall"]["of"] == 2 and d["overall"]["regime"] in ("Up trend", "Sideways")
    assert d["breadth"]["of"] == 2 and d["breadth"]["above"] == 2        # US.AAPL, US.MSFT on the 'up' fake


def test_api_caches_and_refreshes_on_request(client):
    client.get("/api/sentiment", headers=AUTH)
    n = len(client.broker.calls)
    again = client.get("/api/sentiment", headers=AUTH).get_json()
    assert again["cached"] is True and len(client.broker.calls) == n
    forced = client.get("/api/sentiment?refresh=1", headers=AUTH).get_json()
    assert forced["cached"] is False and len(client.broker.calls) == 2 * n


def test_api_disabled(client):
    cfg = yaml.safe_load(client.cfg_path.read_text())
    cfg["sentiment"]["enabled"] = False
    client.cfg_path.write_text(yaml.safe_dump(cfg))
    assert client.get("/api/sentiment", headers=AUTH).get_json() == {"enabled": False}


def test_api_broker_offline_keeps_last_reading(client, monkeypatch):
    first = client.get("/api/sentiment", headers=AUTH).get_json()
    monkeypatch.setattr(dash, "get_broker", lambda: None)
    d = client.get("/api/sentiment?refresh=1", headers=AUTH).get_json()
    assert d["stale"] is True and d["overall"] == first["overall"] and d["error"]
    monkeypatch.setattr(dash, "_sentiment_cache", {})
    d = client.get("/api/sentiment", headers=AUTH).get_json()
    assert d["enabled"] is True and "overall" not in d and d["error"]
