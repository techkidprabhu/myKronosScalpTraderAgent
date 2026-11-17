"""Your rules (config/skills.md): parsing, measures, verdicts, the agent and the dashboard."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import rules

EXAMPLE = Path(__file__).resolve().parent.parent / "config" / "skills.example.md"


def test_starter_rules_parse_cleanly():
    parsed, problems = rules.parse(EXAMPLE.read_text())
    assert len(parsed) >= 8 and not problems
    assert all(r.error is None for r in parsed), [r.error for r in parsed if r.error]
    assert {r.applies for r in parsed} == {"buy", "sell", "both"}
    assert any(not r.enabled for r in parsed)                 # an example of "- off"


def test_parse_reports_mistakes_per_rule():
    text = """# notes
- a list before the first rule is just notes

## Good
- applies to: buy
- block if: rsi > 70 and adx >= 25
- block if: trend_vs_trade is against
- why: test

## Typo in the measure
- block if: rsii > 70

## Words need is
- block if: trend > up

## Not a number
- block if: rsi > high

## Nothing to check
- applies to: sell

## Bad side
- applies to: long
- block if: rsi > 1
"""
    parsed, problems = rules.parse(text)
    by = {r.name: r for r in parsed}
    assert by["Good"].error is None and len(by["Good"].block_if) == 2 and len(by["Good"].block_if[0]) == 2
    assert "unknown measure 'rsii'" in by["Typo in the measure"].error and "did you mean" in by["Typo in the measure"].error
    assert "is a word" in by["Words need is"].error
    assert "not a number" in by["Not a number"].error
    assert "no 'block if' or 'only if'" in by["Nothing to check"].error
    assert "buy, sell or both" in by["Bad side"].error
    assert rules.parse("## A\n- block if: rsi > 1\n## a\n- block if: rsi > 2\n")[1]   # duplicate names flagged


def M(**kw):
    base = {k: None for k in rules.METRICS}
    base.update(direction="buy", pattern="hammer")
    base.update(kw)
    return base


def test_verdicts():
    parsed, _ = rules.parse("""
## Chase
- applies to: buy
- block if: price_vs_vwap_pct > 1.5
## Trend
- only if: price_vs_ema200_pct > 0
## Inside only
- patterns: inside_bar
- block if: adx < 20
## Either
- block if: rsi > 80
- block if: volume_ratio < 0.5
## Switched off
- block if: rsi > 0
- off
""")
    v = rules.evaluate(parsed, M(price_vs_vwap_pct=2.0, price_vs_ema200_pct=1.0, rsi=50, volume_ratio=1.0))
    assert not v.passed and [h["rule"] for h in v.hits] == ["Chase"] and "2 > 1.5" in v.hits[0]["detail"]
    assert v.reason().startswith("rule 'Chase'")
    v = rules.evaluate(parsed, M(direction="sell", price_vs_vwap_pct=2.0, price_vs_ema200_pct=-1, rsi=50, volume_ratio=1))
    assert [h["rule"] for h in v.hits] == ["Trend"] and "needs price_vs_ema200_pct > 0" in v.hits[0]["detail"]
    v = rules.evaluate(parsed, M(price_vs_ema200_pct=1, rsi=50, volume_ratio=0.3, adx=10))
    assert [h["rule"] for h in v.hits] == ["Either"]           # inside-bar rule doesn't apply to a hammer
    v = rules.evaluate(parsed, M(pattern="inside_bar", price_vs_ema200_pct=1, rsi=50, volume_ratio=1, adx=10))
    assert [h["rule"] for h in v.hits] == ["Inside only"]
    # can't be checked (daily candles: no VWAP) -> passes, listed as unchecked, never guessed
    v = rules.evaluate(parsed, M(price_vs_ema200_pct=1, rsi=50, volume_ratio=1))
    assert v.passed and "Chase" in v.unchecked
    v = rules.evaluate(parsed, M(price_vs_vwap_pct=5, price_vs_ema200_pct=1, rsi=50, volume_ratio=1), mode="log")
    assert v.passed and v.hits                                  # log mode records, never stops


def bars(n=260, intraday=False, seed=3):
    rng = np.random.default_rng(seed)
    freq = "5min" if intraday else "D"
    end = pd.Timestamp("2026-11-17 20:55") if intraday else pd.Timestamp("2026-11-17")
    idx = pd.date_range(end=end, periods=n, freq=freq)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998, "close": c,
                         "volume": rng.integers(800, 1200, n).astype(float)}, index=idx)


def test_measures_from_candles():
    from zoneinfo import ZoneInfo
    d = rules.compute_metrics(bars(), "BUY", "hammer", 0.6, "DAY", session_minutes=(30, 360))
    for k in ("price_vs_ema20_pct", "price_vs_ema200_pct", "rsi", "adx", "atr_pct", "volume_ratio",
              "candle_range_atr", "body_pct", "gap_pct", "day_change_pct", "room_to_high_pct", "room_to_low_pct"):
        assert d[k] is not None, k
    assert d["price_vs_vwap_pct"] is None                      # no VWAP on daily candles
    assert d["trend"] in ("up", "down", "flat") and d["trend_vs_trade"] in ("with", "against", "flat")
    assert d["minutes_since_open"] == 30 and d["minutes_to_close"] == 360 and d["direction"] == "buy"
    assert 0 <= d["rsi"] <= 100 and d["room_to_high_pct"] >= 0 and d["room_to_low_pct"] >= 0

    df = bars(400, intraday=True)
    di = rules.compute_metrics(df, "SELL", None, None, "MIN_5", tz=ZoneInfo("America/New_York"))
    assert di["price_vs_vwap_pct"] is not None and di["gap_pct"] is not None and di["direction"] == "sell"
    assert di["minutes_to_close"] is None                       # no session given
    assert rules.compute_metrics(pd.DataFrame(), "BUY")["rsi"] is None


def test_rulebook_rereads_the_file_and_falls_back_to_the_example(tmp_path):
    cfg = {"skills": {"enabled": True, "mode": "filter", "file": str(tmp_path / "skills.md")}}
    book = rules.RuleBook(cfg)
    assert book.load() and book.example                         # yours doesn't exist yet: the starter set
    f = tmp_path / "skills.md"
    f.write_text("## Only one\n- block if: rsi > 1\n")
    assert [r.name for r in book.load()] == ["Only one"] and not book.example
    time.sleep(0.01)
    f.write_text("## Changed\n- block if: rsi > 1\n## Second\n- block if: adx > 1\n")
    import os
    os.utime(f, (time.time() + 5, time.time() + 5))
    assert [r.name for r in book.load()] == ["Changed", "Second"]
    assert rules.settings({"skills": {"mode": "weird"}})["mode"] == "log" and rules.settings({})["enabled"] is False


# ---------------------------------------------------------------- in the agent

def test_agent_stops_or_records_with_your_rules(tmp_path, monkeypatch):
    from tests.test_orchestrator import FakeBroker, engulfing_bars, make_worker, SYMBOL
    import src.orchestrator as orch
    from dashboard.analytics import decision_stage
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    f = tmp_path / "skills.md"
    f.write_text("## Volume spike\n- applies to: buy\n- block if: volume_ratio > 2\n- why: test\n")

    broker = FakeBroker()
    broker.bars = engulfing_bars()                                # last candle 3x the usual volume
    worker, audit, _ = make_worker(tmp_path, broker, skills={"enabled": True, "mode": "filter", "file": str(f)})
    worker.poll_once()
    assert broker.kinds() == [] and worker.kronos_agent.confirm_direction.call_count == 0
    cur = audit.conn.execute("SELECT * FROM decisions ORDER BY id DESC")
    row = dict(zip([c[0] for c in cur.description], cur.fetchone()))
    assert row["action"] == "rejected" and row["risk_reason"].startswith("rule 'Volume spike'")
    assert decision_stage(row) == "rules"

    (tmp_path / "log").mkdir()
    broker2 = FakeBroker()
    broker2.bars = engulfing_bars()
    w2, audit2, _ = make_worker(tmp_path / "log", broker2, skills={"enabled": True, "mode": "log", "file": str(f)})
    w2.poll_once()
    assert broker2.kinds() == ["entry"]                         # log mode never stops
    snap = audit2.conn.execute("SELECT details FROM snapshots WHERE stage = 'rules_check'").fetchone()
    assert json.loads(snap[0])["hits"][0]["rule"] == "Volume spike"

    (tmp_path / "bad").mkdir()
    f2 = tmp_path / "bad" / "skills.md"
    f2.write_text("## Broken\n- block if: nonsense > 1\n")
    broker3 = FakeBroker()
    broker3.bars = engulfing_bars()
    w3, _, _ = make_worker(tmp_path / "bad", broker3, skills={"enabled": True, "mode": "filter", "file": str(f2)})
    w3.poll_once()
    assert broker3.kinds() == ["entry"]                         # a broken rule is skipped, never blocks

    (tmp_path / "off").mkdir()
    broker4 = FakeBroker()
    broker4.bars = engulfing_bars()
    w4, audit4, _ = make_worker(tmp_path / "off", broker4)     # rules off by default
    w4.poll_once()
    assert broker4.kinds() == ["entry"] and w4.rulebook is None
    assert audit4.conn.execute("SELECT COUNT(*) FROM snapshots WHERE stage = 'rules_check'").fetchone()[0] == 0


# ---------------------------------------------------------------- dashboard

def test_dashboard_reads_saves_and_tries_rules(tmp_path, monkeypatch):
    import dashboard.app as app_mod
    from tests.test_orchestrator import engulfing_bars
    root = tmp_path
    (root / "config").mkdir()
    monkeypatch.setattr(app_mod, "ROOT_DIR", root)
    monkeypatch.setattr(rules, "ROOT", root)
    monkeypatch.setattr(rules, "EXAMPLE_PATH", EXAMPLE)
    cfg = {"timeframe": "DAY", "lookback_bars": 30, "skills": {"enabled": True, "mode": "log"}}
    monkeypatch.setattr(app_mod, "load_config", lambda: cfg)
    app_mod.app.config["TESTING"] = True
    c = app_mod.app.test_client()
    hdr = {"Authorization": f"Bearer {app_mod.DASHBOARD_TOKEN}"}

    d = c.get("/api/rules", headers=hdr).get_json()
    assert d["using_example"] and d["file"] == "config/skills.md" and len(d["rules"]) >= 8 and d["metrics"]
    r = c.post("/api/rules", headers=hdr, json={"text": "## Hot RSI\n- block if: rsi > 1\n## Bad\n- block if: zzz > 1\n"})
    out = r.get_json()
    assert r.status_code == 200 and "1 has an error and is skipped" in out["message"]
    assert (root / "config" / "skills.md").read_text().startswith("## Hot RSI")
    c.post("/api/rules", headers=hdr, json={"text": "## Hot RSI\n- block if: rsi > 1\n"})
    assert list((root / "config" / "backups").glob("skills-*.md"))           # the previous version is kept
    assert c.post("/api/rules", headers=hdr, json={"nope": 1}).status_code == 400

    class Client:
        market = "US"
        def get_historical_bars(self, symbol, tf, n):
            return engulfing_bars()
    monkeypatch.setattr(app_mod, "get_broker", lambda: Client())
    chk = c.get("/api/rules/check?symbol=US.AAPL", headers=hdr).get_json()
    assert chk["pattern"] == "bullish_engulfing" and chk["buy"]["metrics"]["rsi"] is not None
    assert [h["rule"] for h in chk["buy"]["stopped_by"]] == ["Hot RSI"]

    cfg["skills"]["file"] = "../outside.md"
    assert c.post("/api/rules", headers=hdr, json={"text": "x"}).status_code == 400
