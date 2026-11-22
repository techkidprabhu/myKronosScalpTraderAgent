"""Opt-in ways to read Kronos for scalping (kronos_scalp:) and the
skill + Kronos trading mode. Defaults must be the original behaviour."""
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

import src.orchestrator as orch
from src import trading_mode
from src.kronos_agent import DEFAULT_OPTIONS, KronosAgent, is_default, kronos_options
from tests.test_kronos_scalp_timing import FakePredictor, OldPredictor, agent, fake_worker, make_orch, session_bars
from tests.test_orchestrator import SYMBOL, FakeBroker, engulfing_bars, make_worker
from tests.test_skill_strategy import SETUP, broker_frame, skill_on  # noqa: F401 - fixture


# ---------------------------------------------------------------- options

def test_defaults_are_the_original_rule():
    assert kronos_options({}) == DEFAULT_OPTIONS and is_default(kronos_options(None))
    o = kronos_options({"kronos_scalp": {"confidence_method": "PATHS", "paths": 500, "min_move_pct": -1}})
    assert o == {"method": "paths", "paths": 50, "shadow_paths": False, "min_move_pct": 0.0}
    assert kronos_options({"kronos_scalp": {"confidence_method": "magic"}})["method"] == "bars"
    assert not is_default({"method": "bars", "min_move_pct": 0.05})


def test_min_move_ignores_forecasts_too_small_to_pay_for_the_trade():
    pred = [100.02, 100.03, 100.2]                           # +0.02%, +0.03%, +0.2%
    assert KronosAgent.agreement(pred, 100, "BUY") == (True, 1.0)            # original rule: all agree
    ok, conf = KronosAgent.agreement(pred, 100, "BUY", min_move_pct=0.05)
    assert (ok, conf) == (False, pytest.approx(1 / 3))                       # only the +0.2% one counts
    assert KronosAgent.agreement([99.9], 100, "SELL", min_move_pct=0.05) == (True, 1.0)


class PathPredictor(FakePredictor):
    """Each forecast path ends at the next value in `ends` (cycled)."""
    def __init__(self, ends):
        super().__init__()
        self.ends, self.k, self.kw = ends, 0, []

    def _pred(self, df, y_ts):
        v = self.ends[self.k % len(self.ends)]
        self.k += 1
        return pd.DataFrame({"close": [v] * len(y_ts)}, index=y_ts)

    def predict_batch(self, df_list, x_list, y_list, pred_len, **kw):
        self.kw.append(kw)
        return super().predict_batch(df_list, x_list, y_list, pred_len, **kw)


def test_paths_confidence_is_the_share_of_separate_forecasts_that_agree():
    last = session_bars()["close"].iloc[-1]
    p = PathPredictor([last + 1, last + 1, last - 1, last + 1])        # 3 of every 4 paths end higher
    out = agent(p).confirm_many([(session_bars(), "BUY"), (session_bars(), "SELL")], 3, method="paths", paths=8)
    assert out[0] == (True, 0.75) and out[1] == (False, 0.25)
    assert p.batch_calls == 1 and p.kw[0]["sample_count"] == 1       # one pass, nothing averaged away


def test_paths_without_predict_batch_still_work():
    class Old(OldPredictor):
        def __init__(self):
            super().__init__()
            self._p = PathPredictor([0.0, 1e9])                       # alternate down / up
    out = agent(Old()).confirm_many([(session_bars(), "BUY")], 3, method="paths", paths=4)
    assert out == [(False, 0.5)]


# ---------------------------------------------------------------- worker

@pytest.fixture
def market_open(monkeypatch):
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)


def test_default_config_asks_kronos_exactly_as_before(tmp_path, market_open):
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    worker.kronos_agent.confirm_direction.assert_called_once()
    worker.kronos_agent.confirm_many.assert_not_called()
    d = audit.conn.execute("SELECT details FROM decisions").fetchone()[0]
    assert "kronos_method" not in d


def test_paths_method_and_min_move_are_used_and_recorded(tmp_path, market_open):
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, _ = make_worker(tmp_path, broker, kronos_scalp={"confidence_method": "paths", "paths": 12,
                                                                    "min_move_pct": 0.05})
    worker.kronos_agent.confirm_many.return_value = [(True, 0.75)]
    worker.poll_once()
    kw = worker.kronos_agent.confirm_many.call_args.kwargs
    assert (kw["method"], kw["paths"], kw["min_move_pct"]) == ("paths", 12, 0.05)
    worker.kronos_agent.confirm_direction.assert_not_called()
    assert broker.kinds() == ["entry"]
    row = audit.conn.execute("SELECT kronos_confidence, details FROM decisions").fetchone()
    assert row[0] == 0.75
    assert {"kronos_method": "paths", "paths": 12, "min_move_pct": 0.05}.items() <= json.loads(row[1]).items()


def test_shadow_paths_are_recorded_but_never_decide(tmp_path, market_open):
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, _ = make_worker(tmp_path, broker, kronos_scalp={"shadow_paths": True})
    worker.kronos_agent.confirm_direction.return_value = (True, 0.9)
    worker.kronos_agent.confirm_many.return_value = [(False, 0.3)]       # paths would have said no
    worker.poll_once()
    assert broker.kinds() == ["entry"]                                   # the original rule decided
    row = audit.conn.execute("SELECT kronos_confidence, details FROM decisions").fetchone()
    assert row[0] == 0.9 and json.loads(row[1])["paths_confidence"] == 0.3


def test_orchestrator_passes_the_options_even_for_one_symbol(tmp_path):
    o = make_orch(tmp_path)
    o.kronos_agent = MagicMock()
    o.kronos_agent.confirm_many.return_value = [(True, 0.8)]
    w = fake_worker("A")
    w.kronos_opts = kronos_options({"kronos_scalp": {"min_move_pct": 0.1}})
    w.kronos_extra_base.return_value = {"kronos_method": "bars", "min_move_pct": 0.1}
    o.workers = {"A": w}
    o.poll_workers()
    assert o.kronos_agent.confirm_many.call_args.kwargs["min_move_pct"] == 0.1
    ctx = w.finish_signal.call_args.args[0]
    assert ctx["kronos_extra"] == {"kronos_method": "bars", "min_move_pct": 0.1}   # recorded with the decision


# ---------------------------------------------------------------- skill + Kronos

def test_skill_kronos_mode_needs_kronos_to_agree_with_the_setup(tmp_path, skill_on):
    trading_mode.write("skill_kronos", "config/Praga.md")
    broker = FakeBroker()
    broker.bars = broker_frame(SETUP)
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.kronos_agent.confirm_direction.return_value = (True, 0.4)        # below the 0.62 threshold
    worker.poll_once()
    assert worker.kronos_agent.confirm_direction.call_args.args[1] == "BUY"
    assert broker.kinds() == []
    row = audit.conn.execute("SELECT action, kronos_confidence, risk_reason, details FROM decisions").fetchone()
    assert row[0] == "rejected" and row[1] == 0.4 and "kronos confidence" in row[2]
    assert json.loads(row[3])["engine"] == "skill+kronos"
    worker.poll_once()                                                       # same setup isn't asked again
    assert worker.kronos_agent.confirm_direction.call_count == 1


def test_skill_kronos_mode_trades_the_skill_levels_when_kronos_agrees(tmp_path, skill_on):
    trading_mode.write("skill_kronos", "config/Praga.md")
    broker = FakeBroker()
    broker.bars = broker_frame(SETUP)
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.kronos_agent.confirm_direction.return_value = (True, 0.9)
    worker.poll_once()
    assert broker.kinds() == ["entry"] and broker.placed[0][2]["limit"] == 104      # the skill's price
    pos = audit.get_active_position(SYMBOL)
    assert pos["initial_stop"] == 103.75 and pos["target"] == 111.89                # the skill's levels
    row = audit.conn.execute("SELECT action, kronos_confidence FROM decisions").fetchone()
    assert row == ("opened", 0.9)


def test_skill_mode_alone_still_never_calls_kronos(tmp_path, skill_on):
    broker = FakeBroker()
    broker.bars = broker_frame(SETUP)
    worker, _, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    worker.kronos_agent.confirm_direction.assert_not_called()
    worker.kronos_agent.confirm_many.assert_not_called()


# ---------------------------------------------------------------- dashboard

from tests.test_dashboard_multibroker import AUTH, client  # noqa: E402,F401,F811 - fixture


def test_dashboard_switches_to_skill_kronos(client, monkeypatch, tmp_path):  # noqa: F811
    from tests.test_skill_strategy import SKILL_MD
    monkeypatch.setattr(trading_mode, "ROOT", tmp_path)
    monkeypatch.setattr(trading_mode, "SWITCH_PATH", tmp_path / "data" / "trading_mode.json")
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "Praga.md").write_text(SKILL_MD)
    import dashboard.app as dash
    monkeypatch.setattr(dash, "_skill_files", lambda: [])
    r = client.post("/api/trading_mode", json={"mode": "skill_kronos", "file": "config/Praga.md"}, headers=AUTH)
    assert r.status_code == 200 and r.get_json()["mode"] == "skill_kronos" and r.get_json()["threshold"] == 0.6
    assert client.post("/api/trading_mode", json={"mode": "auto"}, headers=AUTH).status_code == 400
    trading_mode._cache.update(mtime=None, value=None)


def test_scorecard_includes_the_recorded_paths_measure(client):  # noqa: F811
    import time
    from src.audit_log import AuditLog
    a = AuditLog(str(client.tmp / "audit.db"))
    now = time.time()
    for i, price in enumerate([100, 100.5, 101, 102, 102, 102, 102]):
        a.conn.execute("INSERT INTO snapshots (ts, symbol, price, stage, details) VALUES (?,?,?,?,?)",
                       (now - 3000 + i * 300 - 1, "US.A", price, "scanning", "{}"))
    a.conn.execute("INSERT INTO decisions (ts, epic, pattern_direction, kronos_confirmed, kronos_confidence, action, details) "
                   "VALUES (?,?,?,?,?,?,?)", (now - 3000, "US.A", "BUY", 0, 0.33, "rejected",
                                              json.dumps({"paths_confidence": 0.8})))
    a.conn.commit()
    d = client.get("/api/kronos_scorecard", headers=AUTH).get_json()
    assert d["groups"][2]["right"] == 1                      # the rule in use said no; price rose
    assert d["paths"]["groups"][1]["right"] == 1 and d["paths"]["paths"] == 10   # paths said yes
