"""Kronos on 3-minute candles: forecast timestamps across gaps, one batched
model pass per candle, polling right after each candle closes, timing."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.audit_log import AuditLog
from src.kronos_agent import KronosAgent
from src.orchestrator import KRONOS_TIMING_KEY, Orchestrator
from tests.test_orchestrator import SYMBOL, FakeBroker, engulfing_bars, make_worker


def session_bars(n_yesterday=20, n_today=1, step="3min"):
    """3-minute bars: the end of yesterday's session, then today's first candle."""
    y = pd.date_range("2026-11-19 20:00", periods=n_yesterday, freq=step)
    t = pd.date_range("2026-11-20 14:30", periods=n_today, freq=step)
    idx = y.append(t)
    close = np.linspace(100, 101, len(idx))
    return pd.DataFrame({"open": close, "high": close + .1, "low": close - .1, "close": close,
                         "volume": 1000.0}, index=idx)


class FakePredictor:
    def __init__(self, rise=0.5):
        self.rise, self.calls, self.batch_calls, self.y = rise, 0, 0, []

    def _pred(self, df, y_ts):
        last = df["close"].iloc[-1]
        return pd.DataFrame({"close": [last + self.rise] * len(y_ts)}, index=y_ts)

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kw):
        self.calls += 1
        self.y.append(list(y_timestamp))
        return self._pred(df, y_timestamp)

    def predict_batch(self, df_list, x_list, y_list, pred_len, **kw):
        self.batch_calls += 1
        self.y.extend(list(y) for y in y_list)
        return [self._pred(d, y) for d, y in zip(df_list, y_list)]


class OldPredictor:
    """An older Kronos copy without predict_batch."""
    def __init__(self):
        self._p, self.calls = FakePredictor(), 0

    def predict(self, *a, **kw):
        self.calls += 1
        return self._p.predict(*a, **kw)


def agent(predictor):
    a = KronosAgent.__new__(KronosAgent)
    a.max_context, a.sample_count, a.temperature, a.top_p, a.verbose = 512, 3, 0.5, 0.9, False
    a.predictor = predictor
    return a


# ---------------------------------------------------------------- timestamps

def test_first_candle_after_the_open_forecasts_the_next_minutes_not_tomorrow():
    df = session_bars()
    p = FakePredictor()
    agent(p).confirm_direction(df, "BUY", 3, bar_duration=pd.Timedelta(minutes=3))
    assert p.y[0] == [pd.Timestamp("2026-11-20 14:33"), pd.Timestamp("2026-11-20 14:36"),
                      pd.Timestamp("2026-11-20 14:39")]


def test_without_a_candle_length_the_usual_gap_is_used_not_the_overnight_one():
    df = session_bars()
    assert df.index[-1] - df.index[-2] > pd.Timedelta(hours=17)        # the old rule's step
    assert KronosAgent.bar_step(df) == pd.Timedelta(minutes=3)
    p = FakePredictor()
    agent(p).forecast(df, 2)
    assert p.y[0][0] == pd.Timestamp("2026-11-20 14:33")


def test_answer_rule_is_unchanged():
    assert KronosAgent.agreement([101, 102, 99], 100, "BUY") == (True, pytest.approx(2 / 3))
    assert KronosAgent.agreement([101, 102, 99], 100, "SELL") == (False, pytest.approx(1 / 3))


# ---------------------------------------------------------------- batching

def test_confirm_many_batches_same_length_series_and_keeps_order():
    p = FakePredictor(rise=0.5)
    a = agent(p)
    d1, d2, d3 = session_bars(), session_bars(), session_bars(n_yesterday=30)   # d3 is longer
    out = a.confirm_many([(d1, "BUY"), (d2, "SELL"), (d3, "BUY")], 3, bar_duration=pd.Timedelta(minutes=3))
    assert out == [(True, 1.0), (False, 0.0), (True, 1.0)]
    assert p.batch_calls == 1 and p.calls == 1                       # d1+d2 together, d3 alone


def test_confirm_many_without_predict_batch_runs_one_at_a_time():
    p = OldPredictor()
    out = agent(p).confirm_many([(session_bars(), "BUY"), (session_bars(), "BUY")], 3)
    assert out == [(True, 1.0), (True, 1.0)] and p.calls == 2


def test_one_bad_series_does_not_sink_the_others():
    out = agent(FakePredictor()).confirm_many([(session_bars().iloc[:1], "BUY"), (session_bars(), "BUY")], 3)
    assert isinstance(out[0], ValueError) and out[1] == (True, 1.0)


# ---------------------------------------------------------------- orchestrator

def make_orch(tmp_path, **cfg):
    o = Orchestrator.__new__(Orchestrator)
    o.config = {"timeframe": "MIN_3", "polling_interval_seconds": 15, **cfg}
    o.audit_log = AuditLog(str(tmp_path / "audit.db"))
    o.workers = {}
    return o


def fake_worker(name):
    w = MagicMock()
    w.symbol, w.forecast_horizon, w.bar_duration = name, 3, pd.Timedelta(minutes=3)
    w.prepare_signal.return_value = {"df": session_bars(), "signal": SimpleNamespace(direction="BUY")}
    return w


def test_symbols_signalling_on_one_candle_share_one_kronos_pass(tmp_path):
    o = make_orch(tmp_path)
    o.kronos_agent = MagicMock()
    o.kronos_agent.confirm_many.return_value = [(True, 0.9), (False, 0.2)]
    a, b, quiet = fake_worker("A"), fake_worker("B"), fake_worker("C")
    quiet.prepare_signal.return_value = None
    o.workers = {"A": a, "B": b, "C": quiet}
    o.poll_workers()
    o.kronos_agent.confirm_many.assert_called_once()
    assert a.finish_signal.call_args.args[1] == (True, 0.9)
    assert b.finish_signal.call_args.args[1] == (False, 0.2)
    quiet.finish_signal.assert_not_called()
    a.kronos_agent.confirm_direction.assert_not_called()
    rec = o.audit_log.get_kv(KRONOS_TIMING_KEY)
    assert rec["calls"] == 1 and rec["symbols"] == 2 and rec["last_symbols"] == 2


def test_a_failed_batch_falls_back_to_one_symbol_at_a_time(tmp_path):
    o = make_orch(tmp_path)
    o.kronos_agent = MagicMock()
    o.kronos_agent.confirm_many.side_effect = RuntimeError("boom")
    a, b = fake_worker("A"), fake_worker("B")
    a.kronos_agent.confirm_direction.return_value = (True, 0.8)
    b.kronos_agent.confirm_direction.side_effect = RuntimeError("model died")
    o.workers = {"A": a, "B": b}
    o.poll_workers()
    assert a.finish_signal.call_args.args[1] == (True, 0.8)
    assert isinstance(b.finish_signal.call_args.args[1], RuntimeError)   # reported as a Kronos error
    assert a.kronos_agent.confirm_direction.call_args.kwargs["bar_duration"] == pd.Timedelta(minutes=3)


def test_polls_once_right_after_each_candle_closes(tmp_path):
    o = make_orch(tmp_path)
    state = {}
    t = 1_763_640_000.0                      # a multiple of 180 s: a 3-minute candle boundary
    assert o._candle_closed_since(state, t + 10) is False          # first look: just remembers
    assert o._candle_closed_since(state, t + 100) is False
    assert o._candle_closed_since(state, t + 182) is False         # closed, but give the broker 3 s
    assert o._candle_closed_since(state, t + 183) is True
    assert o._candle_closed_since(state, t + 190) is False         # once per candle
    assert o.seconds_until_next_candle(t + 190) == pytest.approx(173)


def test_candle_close_polling_can_be_turned_off_and_skips_long_candles(tmp_path):
    assert make_orch(tmp_path, candle_close_poll=False).seconds_until_next_candle(0) is None
    assert make_orch(tmp_path, timeframe="DAY").seconds_until_next_candle(0) is None
    assert make_orch(tmp_path, timeframe="HOUR_1").seconds_until_next_candle(0) is None


def test_worker_gives_kronos_the_candle_length(tmp_path, monkeypatch):
    import src.orchestrator as orch
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    broker = FakeBroker()
    broker.bars = engulfing_bars()
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    assert worker.kronos_agent.confirm_direction.call_args.kwargs["bar_duration"] == pd.Timedelta(days=1)
    assert broker.kinds() == ["entry"]
