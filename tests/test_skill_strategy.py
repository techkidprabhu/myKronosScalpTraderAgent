"""The price-action engine (src/skill_strategy.py) on hand-made charts."""
import pandas as pd

from src import skill_strategy as ss

SETUP = [
    (107, 108, 106, 107.5), (107.5, 109, 107, 108.5),
    (108.5, 112, 108, 111),            # 2: swing high -> target
    (111, 111.5, 109, 110), (110, 111, 107, 108), (108, 109, 105, 106), (106, 107, 103, 104),
    (104, 104.5, 100, 101),            # 7: A red (entry = its open, 104)
    (100.5, 105.5, 100.2, 105),        # 8: A green - bullish engulfing at the swing low
    (105, 106, 104, 105.5), (105.5, 105.8, 103, 103.5),
    (103.5, 104, 102, 102.5),          # 11: higher swing low (turnaround)
    (102.5, 104.5, 102.3, 104.2), (104.2, 105, 103.5, 104.8),
    (104.0, 107.2, 103.8, 107),        # 14: strong breakout above 105.5
    (107, 108, 106.5, 107.8), (107.8, 108.5, 107, 108.3),  # 2 green candles, no retest yet
]


def frame(rows, start="2026-11-02 09:30"):
    idx = pd.date_range(start, periods=len(rows), freq="5min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1000)


def settings(**kw):
    s = ss.StrategySettings()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_full_setup_is_ready_with_entry_stop_target():
    st = ss.find_setup(frame(SETUP), settings())
    assert st.status == "ready", st.reason
    assert st.entry == 104                         # red candle open
    assert round(st.stop, 3) == round(103.8 * 0.9995, 3)  # under the breakout candle low
    assert round(st.target, 3) == round(112 * 0.999, 3)   # under the swing high
    assert st.quality == 2                         # strong breakout + 2 green candles


def test_waits_for_min_retest_candles():
    st = ss.find_setup(frame(SETUP[:16]), settings())   # one candle after the breakout
    assert st.status == "waiting" and "more candle" in st.reason


def test_no_breakout_yet_waits():
    st = ss.find_setup(frame(SETUP[:14]), settings())
    assert st.status == "waiting"


def test_long_upper_wick_moves_entry_to_red_high():
    rows = list(SETUP)
    rows[7] = (104, 107.5, 100, 101)                # wick 3.5 > body 3
    rows[8] = (100.5, 107.6, 100.2, 105)
    rows[14] = (104.8, 108.2, 103.8, 108)            # break above the new 107.6 high
    rows[15] = (108, 109, 107.6, 108.8)
    rows[16] = (108.8, 109.5, 108, 109.3)
    rows[2] = (108.5, 113, 108, 111)
    st = ss.find_setup(frame(rows), settings())
    assert st.status == "ready", st.reason
    assert st.entry == 107.5 and "long upper wick" in st.details["entry_rule"]


def test_quick_retest_is_not_taken():
    rows = list(SETUP)
    rows[15] = (107, 107.5, 103.9, 107.2)            # straight back to the entry
    assert ss.find_setup(frame(rows), settings()).status != "ready"


def test_min_quality_filters_weak_setups():
    rows = list(SETUP)
    rows[16] = (107.8, 108, 107, 107.2)              # red: only one green after the breakout
    assert ss.find_setup(frame(rows), settings(min_quality=2)).status == "invalid"
    assert ss.find_setup(frame(rows), settings(min_quality=1)).status == "ready"


def test_bearish_engulfing_exit_and_big_candle_exit_at():
    opened = frame(SETUP).index[-1].isoformat()
    rows = SETUP + [(108.3, 109.5, 108.2, 109.3), (109.4, 109.5, 108.1, 108.2)]
    sig = ss.exit_signal(frame(rows), settings(), opened)
    assert sig.kind == "close_now" and "bearish engulfing" in sig.reason

    big = SETUP + [(108.3, 109.5, 108.2, 109.3), (109.6, 109.7, 101.0, 101.5)]
    sig = ss.exit_signal(frame(big), settings(), opened)
    assert sig.kind == "exit_at" and sig.price == 108.2


def test_no_exit_before_the_entry():
    opened = frame(SETUP).index[-1].isoformat()
    assert ss.exit_signal(frame(SETUP), settings(), opened) is None


def test_settings_block_parsing():
    s, probs = ss.parse("# x\n```yaml\nstrategy: praga_price_action\nswing_candles: 3\nmin_quality: '1'\nbogus: 2\n```\n")
    assert s.swing_candles == 3 and s.min_quality == 1
    assert any("bogus" in p for p in probs)
    assert ss.parse("no block here")[0] is None
    assert ss.parse("```yaml\nstrategy: other\n```")[0] is None


# ---------------------------------------------------------------- agent wiring

import pytest  # noqa: E402

import src.orchestrator as orch  # noqa: E402
from src import trading_mode  # noqa: E402
from tests.test_orchestrator import FakeBroker, make_worker, SYMBOL  # noqa: E402

SKILL_MD = "# My method\n\nWords for people.\n\n```yaml\nstrategy: praga_price_action\nswing_candles: 2\n```\n"


def broker_frame(rows):
    df = frame(rows, start="2026-10-01 09:30")
    out = df.reset_index(drop=True)
    out["time"] = (df.index.astype("int64") // 10**6).to_numpy()
    out["symbol"] = SYMBOL
    return out


@pytest.fixture
def skill_on(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    monkeypatch.setattr(trading_mode, "ROOT", tmp_path)
    monkeypatch.setattr(trading_mode, "SWITCH_PATH", tmp_path / "data" / "trading_mode.json")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "Praga.md").write_text(SKILL_MD)
    orch._skill_cache.clear()
    trading_mode.write("skill", "config/Praga.md")
    yield
    trading_mode._cache.update(mtime=None, value=None)


def test_switch_defaults_to_kronos(tmp_path):
    assert trading_mode.read(tmp_path / "missing.json")["mode"] == "kronos"
    (tmp_path / "bad.json").write_text("{nope")
    assert trading_mode.read(tmp_path / "bad.json")["mode"] == "kronos"


def test_skill_file_must_live_in_config(tmp_path, monkeypatch):
    monkeypatch.setattr(trading_mode, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        trading_mode.resolve_file("../secrets.md")
    with pytest.raises(ValueError):
        trading_mode.resolve_file("config/x.txt")


def test_skill_mode_trades_the_setup_without_kronos(tmp_path, skill_on):
    broker = FakeBroker()
    broker.bars = broker_frame(SETUP)
    worker, audit, _ = make_worker(tmp_path, broker)

    worker.poll_once()

    worker.kronos_agent.confirm_direction.assert_not_called()
    assert broker.kinds() == ["entry"]
    assert broker.placed[0][2]["limit"] == 104 and broker.placed[0][2]["direction"] == "BUY"
    pos = audit.get_active_position(SYMBOL)
    assert pos["status"] == "pending_entry" and pos["initial_stop"] == 103.75 and pos["target"] == 111.89

    # the retest fills: the bracket uses the setup's own levels
    entry_id = int(pos["entry_order_id"])
    broker.fill(entry_id, price=104)
    broker.held = 100
    worker.monitor_position()
    assert [(k, i["price"]) for k, _, i in broker.placed[1:]] == [("stop", 103.75), ("target", 111.89)]

    # a bearish engulfing closes the trade at the next scan
    broker.price = 108.3
    broker.bars = broker_frame(SETUP + [(108.3, 109.5, 108.2, 109.3), (109.4, 109.5, 108.1, 108.2)])
    worker.poll_once()
    assert broker.kinds()[-1] == "close"
    assert audit.get_active_position(SYMBOL) is None
    assert audit.get_kv("skill_trade:" + SYMBOL, None) is None
    worker.kronos_agent.confirm_direction.assert_not_called()


def test_stale_retest_order_is_cancelled(tmp_path, skill_on):
    broker = FakeBroker()
    broker.bars = broker_frame(SETUP)
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    entry_id = int(audit.get_active_position(SYMBOL)["entry_order_id"])

    broker.bars = broker_frame(SETUP + [(108, 108.2, 103.0, 103.2)])   # close below the stop
    worker.poll_once()
    assert entry_id in broker.cancelled
    worker.monitor_position()
    assert audit.get_active_position(SYMBOL) is None


def test_kronos_mode_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)
    monkeypatch.setattr(trading_mode, "SWITCH_PATH", tmp_path / "none.json")
    broker = FakeBroker()
    broker.bars = broker_frame(SETUP)
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    assert broker.kinds() == []            # last candle is no pattern: Kronos path, nothing to do
    assert audit.get_kv("skill_trade:" + SYMBOL, None) is None


def test_risk_limits_still_apply_to_skill_entries(tmp_path, skill_on):
    broker = FakeBroker()
    broker.bars = broker_frame(SETUP)
    worker, audit, risk = make_worker(tmp_path, broker)
    risk.config.max_open_positions = 0
    worker.poll_once()
    assert broker.kinds() == []
    worker.poll_once()                    # same candle: not retried or re-logged
    assert broker.kinds() == []


def test_big_bearish_engulfing_moves_the_target(tmp_path, skill_on):
    broker = FakeBroker()
    broker.bars = broker_frame(SETUP)
    worker, audit, _ = make_worker(tmp_path, broker)
    worker.poll_once()
    broker.fill(int(audit.get_active_position(SYMBOL)["entry_order_id"]), price=104)
    broker.held = 100
    worker.monitor_position()
    old_target = int(audit.get_active_position(SYMBOL)["target_order_id"])

    broker.bars = broker_frame(SETUP + [(108.3, 109.5, 108.2, 109.3), (109.6, 109.7, 101.0, 101.5)])
    worker.poll_once()

    pos = audit.get_active_position(SYMBOL)
    assert pos is not None and pos["target"] == 108.2
    assert old_target in broker.cancelled and broker.placed[-1][0] == "target"


def test_trend_end_closes_the_trade():
    opened = frame(SETUP).index[-1].isoformat()
    rows = SETUP + [
        (108.3, 109, 107.9, 108.9), (108.9, 109.2, 108.6, 109.1),
        (109.1, 109.3, 108.0, 108.3), (108.0, 109.6, 107.9, 109.4),   # bullish engulfing (low 107.9)
        (109.4, 109.8, 109.0, 109.6), (109.6, 109.9, 109.2, 109.7),
        (109.7, 109.75, 107.0, 107.4),                               # close below both
    ]
    s = settings(exit_on_bearish_engulfing=False)
    sig = ss.exit_signal(frame(rows), s, opened)
    assert sig is not None and "trend ended" in sig.reason


def test_retest_order_ignores_the_global_entry_timeout(tmp_path, skill_on):
    broker = FakeBroker()
    broker.bars = broker_frame(SETUP)
    worker, audit, _ = make_worker(tmp_path, broker, entry_timeout_minutes=1)
    worker.poll_once()
    pos = audit.get_active_position(SYMBOL)
    audit.conn.execute("UPDATE positions SET opened_ts = opened_ts - 3600 WHERE symbol=?", (SYMBOL,))
    audit.conn.commit()
    worker.monitor_position()
    assert int(pos["entry_order_id"]) not in broker.cancelled
