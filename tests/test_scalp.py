"""Scalp trade management (time stop, break-even) and the starter scalp engine."""
import numpy as np
import pandas as pd
import pytest

import src.orchestrator as orch
from src import scalp_engine, skill_strategy as ss, trading_mode
from tests.test_orchestrator import FakeBroker, SYMBOL, make_worker, open_position


class StopTrackingBroker(FakeBroker):
    def __init__(self):
        super().__init__()
        self.stop_moves = []

    def modify_stop_order(self, order_id, price):
        self.stop_moves.append((int(order_id), price))


@pytest.fixture
def market_open(monkeypatch):
    monkeypatch.setattr(orch, "is_market_open", lambda market: True)


# ------------------------------------------------------------ time stop / break-even

def test_time_stop_closes_a_trade_that_sits(tmp_path, market_open):
    broker = FakeBroker()
    worker, audit, _, _, _ = open_position(tmp_path, broker)
    worker.max_hold_minutes = 15
    broker.price = 71.9                                   # between stop and target
    worker.monitor_position()
    assert audit.get_active_position(SYMBOL) is not None  # not yet 15 minutes

    audit.conn.execute("UPDATE positions SET opened_ts = opened_ts - 16*60 WHERE symbol=?", (SYMBOL,))
    audit.conn.commit()
    broker.placed.clear()
    worker.monitor_position()
    assert broker.kinds() == ["close"]
    assert audit.get_active_position(SYMBOL) is None


def test_breakeven_moves_the_stop_to_entry_after_1r(tmp_path, market_open):
    broker = StopTrackingBroker()
    worker, audit, _, stop_id, _ = open_position(tmp_path, broker)
    worker.breakeven_at_r, worker.breakeven_offset_pct = 1.0, 0.0
    pos = audit.get_active_position(SYMBOL)
    risk = pos["entry_price"] - pos["initial_stop"]

    broker.price = pos["entry_price"] + 0.5 * risk         # half an R: nothing
    worker.monitor_position()
    assert broker.stop_moves == []

    broker.price = pos["entry_price"] + 1.05 * risk        # past 1R: stop to the entry
    worker.monitor_position()
    assert broker.stop_moves == [(stop_id, pos["entry_price"])]
    assert audit.get_active_position(SYMBOL)["current_stop"] == pos["entry_price"]

    worker.monitor_position()                               # not moved again
    assert len(broker.stop_moves) == 1


def test_fill_time_starts_the_hold_clock(tmp_path, market_open):
    broker = FakeBroker()
    worker, audit, _, _, _ = open_position(tmp_path, broker)
    import time
    assert abs(audit.get_active_position(SYMBOL)["opened_ts"] - time.time()) < 5


# ------------------------------------------------------------ the scalp engine

def session(rows, start="2026-11-02 09:30"):
    idx = pd.date_range(start, periods=len(rows), freq="1min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def orb_rows(breakout):
    rows = [(100.5, 101.0, 100.0, 100.5, 1000)] * 15          # opening range 100-101
    rows += [(100.6, 100.8, 100.4, 100.6, 1000)] * 9          # inside the range
    rows.append(breakout)
    return rows


def scalp(**kw):
    s = scalp_engine.ScalpSettings()
    s.orb_minutes = 15            # these charts are 1-minute candles with a 15-minute range
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_orb_long_on_volume():
    st = ss.find_setup(session(orb_rows((100.9, 101.6, 100.9, 101.5, 3000))), scalp())
    assert st.ready and st.direction == "BUY", st.reason
    assert st.entry == 101.5 and round(st.stop, 4) == round(100.5 * 0.9998, 4)
    assert round(st.target, 4) == round(101.5 + 1.5 * (101.5 - st.stop), 4)
    assert st.quality == 1                                       # 3x volume


def test_orb_short_and_sides_filter():
    rows = orb_rows((100.2, 100.3, 99.4, 99.5, 3000))
    st = ss.find_setup(session(rows), scalp())
    assert st.ready and st.direction == "SELL" and st.stop > st.entry > st.target
    assert not ss.find_setup(session(rows), scalp(sides="long")).ready


def test_orb_needs_volume_and_first_close():
    assert not ss.find_setup(session(orb_rows((100.9, 101.6, 100.9, 101.5, 1100))), scalp(setups="orb")).ready
    rows = orb_rows((100.9, 101.6, 100.9, 101.5, 3000)) + [(101.5, 101.9, 101.4, 101.8, 3000)]
    assert not ss.find_setup(session(rows), scalp(setups="orb")).ready     # second close above: no new signal


def test_vwap_pullback_long():
    closes = np.linspace(100, 104, 40)
    rows = [(c - 0.05, c + 0.1, c - 0.1, c, 1000) for c in closes]
    df = session(rows)
    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    j = len(df) - 1
    e = float(ema9.iloc[-1])
    rows.append((104.0, 104.05, e * 0.9999, 103.95, 1000))      # pull back to the fast EMA
    rows.append((103.95, 104.4, 103.9, 104.35, 1000))           # green close above it
    st = ss.find_setup(session(rows), scalp(setups="pullback"))
    assert st.ready and st.direction == "BUY" and "pullback" in st.reason, st.reason
    assert st.stop < min(rows[-2][2], rows[-1][2])


def test_vwap_cross_and_max_hold_exits():
    rows = orb_rows((100.9, 101.6, 100.9, 101.5, 3000))
    opened = session(rows).index[-1].isoformat()
    after = rows + [(101.5, 101.6, 100.0, 100.1, 1000)]          # back below VWAP
    sig = ss.exit_signal(session(after), scalp(), opened, "BUY")
    assert sig.kind == "close_now" and "VWAP" in sig.reason
    assert ss.exit_signal(session(after), scalp(exit_on_vwap_cross=False), opened, "BUY") is None
    held = rows + [(101.5, 101.7, 101.4, 101.6, 1000)] * 3
    sig = ss.exit_signal(session(held), scalp(max_hold_candles=3, exit_on_vwap_cross=False), opened, "BUY")
    assert sig and "max_hold" in sig.reason


def test_scalp_block_parses():
    s, probs = ss.parse("```yaml\nstrategy: scalp_vwap_orb\nsides: long\ntarget_r: 2\norb_stop: weird\n```")
    assert s.strategy == "scalp_vwap_orb" and s.sides == "long" and s.target_r == 2.0
    assert s.orb_stop == "mid" and any("orb_stop" in p for p in probs)


# ------------------------------------------------------------ a short skill trade, end to end

def broker_frame(rows):
    df = session(rows)
    out = df.reset_index(drop=True)
    out["time"] = (df.index.astype("int64") // 10**6).to_numpy()
    out["symbol"] = SYMBOL
    return out


def test_short_scalp_skill_trade(tmp_path, monkeypatch, market_open):
    monkeypatch.setattr(trading_mode, "ROOT", tmp_path)
    monkeypatch.setattr(trading_mode, "SWITCH_PATH", tmp_path / "data" / "trading_mode.json")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "Scalp.md").write_text("```yaml\nstrategy: scalp_vwap_orb\n```\n")
    orch._skill_cache.clear()
    trading_mode.write("skill", "config/Scalp.md")
    try:
        broker = FakeBroker()
        broker.bars = broker_frame(orb_rows((100.2, 100.3, 99.4, 99.5, 3000)))
        worker, audit, risk = make_worker(tmp_path, broker, timeframe="MIN_1")
        risk.config.allow_short = True
        worker.poll_once()
        worker.kronos_agent.confirm_direction.assert_not_called()
        assert broker.kinds() == ["entry"] and broker.placed[0][2]["direction"] == "SELL"
        pos = audit.get_active_position(SYMBOL)
        assert pos["direction"] == "SELL" and pos["initial_stop"] > 99.5 > pos["target"]

        broker.fill(int(pos["entry_order_id"]), price=99.5)
        broker.held = -100
        worker.monitor_position()
        assert broker.kinds()[1:] == ["stop", "target"]
        assert broker.placed[1][2]["direction"] == "BUY"
    finally:
        trading_mode._cache.update(mtime=None, value=None)


def test_orb_on_the_first_3_minute_candle():
    idx = pd.date_range("2026-11-02 09:30", periods=25, freq="3min")
    rows = [(100.5, 101.0, 100.0, 100.5, 1000)] + [(100.6, 100.8, 100.4, 100.6, 1000)] * 23
    rows.append((100.9, 101.6, 100.9, 101.5, 3000))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)
    s = scalp_engine.ScalpSettings()                       # default: opening range = first 3 minutes
    st = ss.find_setup(df, s)
    assert s.orb_minutes == 3 and st.ready and st.details["or_high"] == 101.0 and st.details["or_low"] == 100.0


def test_every_broker_has_3_minute_candles():
    from src import market, moomoo_client
    from src.brokers import angel, ig, india, tiger, zerodha
    assert "MIN_3" in moomoo_client.RESOLUTION_MAP and "MIN_3" in market.BAR_DURATION
    assert zerodha.INTERVAL["MIN_3"] == "3minute" and angel.INTERVAL["MIN_3"] == "THREE_MINUTE"
    assert ig.RESOLUTIONS["MIN_3"] == "MINUTE_3" and tiger.TigerClient.RESOLUTIONS["MIN_3"] == "THREE_MINUTES"
    assert india.history_window("MIN_3", 300).total_seconds() > 0


def test_starter_skill_file_matches_engine_defaults():
    s, probs = ss.load("config/ScalpStarterSkill.md")
    assert not probs and s.orb_minutes == 3 and s.max_hold_candles == 10
