"""Self-learning Kronos: data collection, windows, and a real (tiny) training run."""
import json
import time

import numpy as np
import pandas as pd
import pytest

from src.finetune import data as ftdata
from src.finetune import settings
from src.finetune.trainer import list_runs, original_names, read_status, verdict, window_sizes, write_status

MIN5 = 300_000


def bars(n, start_ms=1_780_000_000_000, seed=0, base=100.0):
    rng = np.random.default_rng(seed)
    close = base * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    return pd.DataFrame({"time": start_ms + np.arange(n) * MIN5, "open": close * 0.999, "high": close * 1.002,
                         "low": close * 0.997, "close": close, "volume": rng.integers(1000, 5000, n).astype(float)})


class Broker:
    def __init__(self, n=300):
        self.n, self.asked = n, []

    def get_historical_bars(self, symbol, timeframe, count):
        self.asked.append(symbol)
        if symbol == "US.BAD":
            raise RuntimeError("no quote right")
        return bars(self.n, seed=len(symbol)).tail(count)


# ---------------------------------------------------------------- collection

def test_collect_merges_without_duplicates(tmp_path):
    b = Broker(300)
    s = ftdata.collect(b, "moomoo", "MIN_5", ["US.AAPL", "US.BAD", "US.AAPL"], bars=200, root=tmp_path)
    assert (s["symbols"], s["failed"], s["bars"]) == (1, 1, 200)
    f = ftdata.bars_path("moomoo", "US.AAPL", "MIN_5", tmp_path)
    b.n = 320                                                    # next session: 20 newer bars
    ftdata.collect(b, "moomoo", "MIN_5", ["US.AAPL"], bars=200, drop_last=True, root=tmp_path)
    df = pd.read_csv(f)
    assert len(df) == 219 and df["time"].is_unique and df["time"].is_monotonic_increasing
    inv = ftdata.inventory(tmp_path)
    assert inv == [{"broker": "moomoo", "timeframe": "MIN_5", "symbols": 1, "bars": 219, "updated": inv[0]["updated"]}]


def test_symbol_names_are_safe_file_names(tmp_path):
    p = ftdata.bars_path("zerodha", "NSE.M&M", "MIN_5", tmp_path)
    assert p.name == "NSE.M&M__MIN_5.csv"
    assert "/" not in ftdata.bars_path("x", "../../etc/passwd", "DAY", tmp_path).name


def test_symbols_and_signals_come_from_the_audit_log(tmp_path):
    from src.audit_log import AuditLog
    a = AuditLog(str(tmp_path / "a.db"))
    a.log_decision("NSE.TCS", "hammer", "BUY", 0.6, True, 0.8, True, "ok", "opened")
    a.log_decision("NSE.ITC", "hammer", "BUY", 0.6, None, None, None, "indicator", "rejected")
    assert set(ftdata.symbols_from_audit(tmp_path / "a.db")) == {"NSE.TCS", "NSE.ITC"}
    sig = ftdata.signal_times(tmp_path / "a.db")
    assert list(sig) == ["NSE.TCS"] and abs(sig["NSE.TCS"][0] - time.time()) < 5
    assert ftdata.symbols_from_audit(tmp_path / "missing.db") == []


# ---------------------------------------------------------------- windows

def make_set(tmp_path, signals=None, n=400):
    for i, sym in enumerate(["US.AAA", "US.BBB"]):
        ftdata.merge_bars(ftdata.bars_path("moomoo", sym, "MIN_5", tmp_path), bars(n, seed=i))
    series = [ftdata.Series(f) for f in sorted(tmp_path.glob("*/*.csv"))]
    return ftdata.WindowSet(series, lookback=40, predict=10, val_fraction=0.2, signals=signals)


def test_split_is_by_date_with_no_overlap(tmp_path):
    ws = make_set(tmp_path)
    assert ws.train and ws.val
    for si, i in ws.train:
        assert ws.series[si].time[i + ws.window - 1] < ws.cutoff_ms          # whole window before the cut-off
    for si, i in ws.val:
        assert ws.series[si].time[i + ws.lookback] >= ws.cutoff_ms           # predicted part after it
    x, st = ws.sample_train(6)
    assert x.shape == (6, 50, 6) and st.shape == (6, 50, 5) and np.abs(x).max() <= 5.0
    assert np.allclose(x[:, :40].mean(axis=1), 0, atol=1e-3)                 # normalised on the lookback only


def test_signal_windows_end_on_the_signal_bar(tmp_path):
    t0 = 1_780_000_000_000
    sig_bar = 100
    ws = make_set(tmp_path, {("moomoo", "US.AAA"): [(t0 + sig_bar * MIN5) / 1000 + 30]})
    assert len(ws.train_signal) == 1
    si, start = ws.train_signal[0]
    assert ws.series[si].symbol == "US.AAA" and start + ws.lookback - 1 == sig_bar
    x, _ = ws.sample_train(10, signal_fraction=0.5)
    assert x.shape[0] == 10
    items = ws.val_items(20)
    assert len(items) == len(set(items)) <= 20


def test_short_series_are_left_out(tmp_path):
    ftdata.merge_bars(ftdata.bars_path("angel", "NSE.X", "DAY", tmp_path), bars(30))
    ws = ftdata.WindowSet([ftdata.Series(next(tmp_path.glob("*/*.csv")))], 40, 10)
    assert ws.series == [] and ws.train == []


# ---------------------------------------------------------------- settings, status, verdict

def test_settings_and_window_sizes():
    assert settings({})["enabled"] is False and settings({"finetune": {"max_steps": 5}})["max_steps"] == 5
    assert window_sizes({"lookback_bars": 200, "forecast_horizon": 5}) == (200, 10)
    assert window_sizes({"lookback_bars": 2000, "forecast_horizon": 30}) == (400, 30)
    assert window_sizes({"lookback_bars": 200, "kronos": {"max_context": 128}}) == (118, 10)


def test_training_always_starts_from_the_original(tmp_path):
    run = tmp_path / "kronos-ft-1"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"base_model": "NeoQuasar/Kronos-base",
                                                 "base_tokenizer": "NeoQuasar/Kronos-Tokenizer-base"}))
    cfg = {"kronos": {"model_name": str(run), "tokenizer_name": "tok"}}
    assert original_names(cfg) == ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base")
    assert original_names({"kronos": {"model_name": "NeoQuasar/Kronos-small"}})[0] == "NeoQuasar/Kronos-small"


def test_status_detects_a_dead_run(tmp_path):
    p = tmp_path / "s.json"
    assert read_status(p) == {"state": "idle"}
    write_status(p, state="running", pid=2 ** 22 + 12345, step=10)
    assert read_status(p)["state"] == "failed"
    import os
    write_status(p, state="running", pid=os.getpid(), step=10)
    assert read_status(p)["state"] == "running"


def test_verdict():
    base = {"val_loss": 5.0, "direction_hit_pct": 50.0}
    assert verdict(base, {"val_loss": 4.8, "direction_hit_pct": 54.0})[0] == "better"
    assert verdict(base, {"val_loss": 4.99, "direction_hit_pct": 51.0})[0] == "no_gain"
    assert verdict(base, {"val_loss": 4.7, "direction_hit_pct": 45.0})[0] == "worse"
    assert verdict(base, {"val_loss": 9.9, "direction_hit_pct": 55.0}, same_tokenizer=False)[0] == "better"


# ---------------------------------------------------------------- a real run with a tiny model

def test_tiny_end_to_end_run(tmp_path):
    torch = pytest.importorskip("torch")
    from src.kronos_agent import _ensure_kronos_importable
    if not _ensure_kronos_importable():
        pytest.skip("Kronos source not available")
    from model import Kronos, KronosTokenizer
    from src.finetune.trainer import run_training

    torch.manual_seed(0)
    tok = KronosTokenizer(d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=1, n_dec_layers=1, ffn_dropout_p=0.0,
                          attn_dropout_p=0.0, resid_dropout_p=0.0, s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0,
                          gamma=1.1, zeta=0.05, group_size=4)
    model = Kronos(s1_bits=4, s2_bits=4, n_layers=1, d_model=32, n_heads=2, ff_dim=64, ffn_dropout_p=0.0,
                   attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True)
    tok.save_pretrained(tmp_path / "tok")
    model.save_pretrained(tmp_path / "base")
    root = tmp_path / "training"
    for i, (broker, sym) in enumerate([("moomoo", "US.AAA"), ("zerodha", "NSE.BBB")]):
        ftdata.merge_bars(ftdata.bars_path(broker, sym, "MIN_5", root), bars(500, seed=i))
    cfg = {"lookback_bars": 30, "forecast_horizon": 5, "timeframe": "MIN_5",
           "kronos": {"model_name": str(tmp_path / "base"), "tokenizer_name": str(tmp_path / "tok"),
                      "sample_count": 1, "max_context": 64},
           "finetune": {"max_steps": 12, "eval_every": 4, "batch_size": 4, "min_train_windows": 50,
                        "val_windows": 12, "eval_forecasts": 4, "learning_rate": 1e-3, "threads": 1}}
    rep = run_training(cfg, models_dir=tmp_path / "models", status_path=tmp_path / "status.json", training_root=root)
    assert rep["steps"] <= 12 and rep["symbols"] == 2 and rep["symbols_by_broker"] == {"moomoo": 1, "zerodha": 1}
    assert rep["verdict"] in ("better", "no_gain", "worse")
    assert rep["original"]["val_loss"] > 0 and rep["fine_tuned"]["val_loss"] <= rep["original"]["val_loss"] + 0.5
    st = read_status(tmp_path / "status.json")
    assert st["state"] == "done" and st["run_id"] == rep["run_id"]
    runs = list_runs(tmp_path / "models")
    assert runs[0]["run_id"] == rep["run_id"] and runs[0]["has_weights"] == (rep["verdict"] == "better")
    if runs[0]["has_weights"]:                                  # the saved folder loads like any Kronos model
        Kronos.from_pretrained(runs[0]["path"], local_files_only=True)

    cfg["finetune"]["min_train_windows"] = 10 ** 9
    with pytest.raises(RuntimeError, match="Not enough candles"):
        run_training(cfg, models_dir=tmp_path / "models", status_path=tmp_path / "status.json", training_root=root)
    assert read_status(tmp_path / "status.json")["state"] == "failed"


# ---------------------------------------------------------------- agent hook, scheduler, dashboard

def test_schedule_due():
    from datetime import datetime
    from src.finetune.scheduler import due
    ft = {"enabled": True, "train_time": "05:30", "train_days": ["sat"]}
    sat = datetime(2026, 11, 14, 6, 0)
    assert due(ft, sat, None)
    assert not due(ft, sat.replace(hour=5, minute=0), None)                 # too early
    assert not due(ft, sat, sat.replace(hour=5, minute=31).timestamp())     # already ran today
    assert due(ft, sat, sat.replace(day=7).timestamp())                     # last week's run
    assert not due(ft, datetime(2026, 11, 13, 6, 0), None)                  # Friday
    assert not due(ft, sat.replace(hour=13), None)                          # far too late
    assert not due({**ft, "enabled": False}, sat, None) and not due({**ft, "train_time": ""}, sat, None)


def test_agent_collects_at_market_close_only_when_enabled(tmp_path, monkeypatch):
    import src.finetune as ft
    from src import orchestrator as orch
    from src.audit_log import AuditLog
    monkeypatch.setattr(ft, "TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(ftdata, "TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(ft, "COLLECT_REQUEST", tmp_path / "collect.request")
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    o.broker, o.broker_name, o.broker_ok = Broker(300), "moomoo", True
    o.workers, o.audit_log = {"US.AAPL": object()}, AuditLog(str(tmp_path / "a.db"))
    o.config = {"timeframe": "MIN_5"}

    def tick(old, phase):
        o._finetune_tick(old, phase)
        t = getattr(o, "_ft_thread", None)
        if t:
            t.join(10)

    tick("active", "sleeping")                                   # switched off: nothing
    assert not (tmp_path / "training").exists()
    (tmp_path / "collect.request").write_text("now")             # dashboard "Collect candles now"
    tick("active", "active")
    f = ftdata.bars_path("moomoo", "US.AAPL", "MIN_5", tmp_path / "training")
    assert len(pd.read_csv(f)) == 299                            # market open: forming bar left out
    o.config = {"timeframe": "MIN_5", "finetune": {"enabled": True}}
    o.broker.n = 310
    tick("active", "active")                                     # nothing new to do mid-session
    assert len(pd.read_csv(f)) == 299
    tick("active", "sleeping")                                   # market just closed
    assert len(pd.read_csv(f)) == 310


def test_dashboard_overview_and_model_switch(tmp_path, monkeypatch):
    import dashboard.app as app_mod
    import src.finetune as ft
    from src.finetune import trainer
    models = tmp_path / "models"
    run = models / "kronos-ft-20261114-0530"
    run.mkdir(parents=True)
    (run / "model.safetensors").write_bytes(b"x")
    (run / "report.json").write_text(json.dumps({
        "run_id": run.name, "created": time.time(), "minutes": 95, "steps": 1500, "symbols": 40,
        "base_model": "NeoQuasar/Kronos-base", "base_tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base", "verdict": "better", "reason": "loss 3% lower",
        "original": {"val_loss": 5.1, "direction_hit_pct": 51.0}, "fine_tuned": {"val_loss": 4.9, "direction_hit_pct": 55.0}}))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("# my config\nkronos:\n  model_name: NeoQuasar/Kronos-base   # original\n"
                        "  tokenizer_name: NeoQuasar/Kronos-Tokenizer-base\ntimeframe: MIN_5\n")
    monkeypatch.setattr(trainer, "MODELS_DIR", models)
    monkeypatch.setattr(trainer, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(ft, "TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(ftdata, "TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(app_mod, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(app_mod, "load_config", lambda: __import__("yaml").safe_load(cfg_path.read_text()))
    app_mod._ft_inventory_cache["ts"] = 0
    app_mod.app.config["TESTING"] = True
    c = app_mod.app.test_client()
    hdr = {"Authorization": f"Bearer {app_mod.DASHBOARD_TOKEN}"}
    assert c.get("/api/finetune").status_code == 401             # behind the dashboard login like everything else
    r = c.get("/api/finetune", headers=hdr)
    assert r.status_code == 200
    d = r.get_json()
    assert d["enabled"] is False and d["in_use"] is None and d["runs"][0]["verdict"] == "better"
    assert c.post("/api/finetune/use", json={"run_id": "nope"}, headers=hdr).status_code == 404
    assert c.post("/api/finetune/use", json={"run_id": run.name}, headers=hdr).get_json()["success"]
    text = cfg_path.read_text()
    assert str(run) in text and "# my config" in text
    assert c.get("/api/finetune", headers=hdr).get_json()["in_use"] == run.name
    assert c.post("/api/finetune/use", json={"run_id": "original"}, headers=hdr).get_json()["success"]
    assert "model_name: NeoQuasar/Kronos-base" in cfg_path.read_text()


# ---------------------------------------------------------------- shadow mode

class FakeAgent:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def confirm_direction(self, df, direction, horizon):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_shadow_records_without_touching_the_verdict(tmp_path):
    from src.finetune.shadow import ShadowRunner, report
    from src.multi_broker import SharedKronos
    t0 = 1_780_000_000_000
    root = tmp_path / "training"
    df = bars(200, start_ms=t0)
    df.loc[52:, "close"] = df.loc[50, "close"] * 1.05              # price is higher 5 bars after bar 50
    ftdata.merge_bars(ftdata.bars_path("moomoo", "US.AAA", "MIN_5", root), df)
    main, shadow_agent = FakeAgent((True, 0.8)), FakeAgent((True, 0.4))
    runner = ShadowRunner(shadow_agent, "kronos-ft-x", tmp_path / "shadow.db")
    k = SharedKronos(main, runner)
    hist = df.iloc[:51]
    assert k.confirm_direction(hist, "BUY", 5) == (True, 0.8)       # the trading verdict is the original's
    kw = dict(broker="moomoo", symbol="US.AAA", timeframe="MIN_5", last_close=float(hist["close"].iloc[-1]), threshold=0.6)
    import unittest.mock as um
    with um.patch("time.time", return_value=(t0 + 50 * MIN5) / 1000 + 20):
        assert k.shadow_submit(hist, "BUY", 5, main_confirmed=True, main_confidence=0.8, **kw)
        assert k.shadow_submit(hist, "SELL", 5, main_confirmed=True, main_confidence=0.2, **kw)
    runner.q.join()
    assert shadow_agent.calls == 2 and main.calls == 1
    rep = report(tmp_path / "shadow.db", root, days=10 ** 5)
    assert rep["signals"] == 2 and rep["agree_pct"] == 50.0 and rep["scored"] == 2
    assert rep["original"] == {"confirmed": 1, "scored": 1, "right_pct": 100.0}     # BUY passed, price rose
    assert rep["shadow"]["confirmed"] == 0                                          # 0.4 is under the threshold
    assert rep["only_original"]["signals"] == 1 and rep["recent_disagreements"][0]["right"] is True


def test_shadow_failures_and_off_state_are_harmless(tmp_path):
    from src.finetune.shadow import ShadowRunner, build, report
    from src.multi_broker import SharedKronos
    assert build({}) is None and build({"finetune": {"shadow_model": "/no/such/model"}, "kronos": {"local_files_only": True}}) is None
    k = SharedKronos(FakeAgent((False, 0.1)))
    assert k.shadow_submit(None, "BUY", 5) is False                  # shadow off
    runner = ShadowRunner(FakeAgent(RuntimeError("boom")), "m", tmp_path / "s.db")
    assert runner.submit(bars(10), "BUY", 5, broker="b", symbol="s", timeframe="t", last_close=1.0, threshold=0.6,
                         main_confirmed=True, main_confidence=0.9)
    runner.q.join()                                                  # the error stays in the shadow thread
    assert report(tmp_path / "s.db")["signals"] == 0
    assert report(tmp_path / "missing.db") == {"signals": 0, "model": None}


def test_shadow_model_switch_keeps_the_config_intact():
    import yaml
    from dashboard.config_writer import set_shadow_model
    text = "# top\nkronos:\n  model_name: NeoQuasar/Kronos-base  # original\nfinetune:\n  enabled: true\n"
    on = set_shadow_model(text, "/m/kronos-ft-1")
    assert "# top" in on and "# original" in on
    cfg = yaml.safe_load(on)
    assert cfg["finetune"] == {"enabled": True, "shadow_model": "/m/kronos-ft-1", "shadow_tokenizer": None}
    assert cfg["kronos"]["model_name"] == "NeoQuasar/Kronos-base"   # the trading model is not touched
    assert yaml.safe_load(set_shadow_model(on, None))["finetune"]["shadow_model"] is None
