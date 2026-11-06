"""Regression test: stop/target must sit on the correct side of the
actual fill price.

Reproduces the AAPL paper trade on 2026-11-02 (order 3199841): the
entry limit was the last closed bar's close (344.63), the market was
trading lower (~337), so the BUY limit was marketable and filled at
337.06. Stop/target are computed from the signal close, not the fill,
so the "stop" (340.63) ended up ABOVE the fill price: moomoo rejected
the sell stop, and the software monitor treated the position as
stopped out immediately.
"""
from tests.test_orchestrator import SYMBOL, FakeBroker, engulfing_bars, make_worker, market_open  # noqa: F401


def test_stop_below_fill_for_buy_after_gap_down(tmp_path, market_open):  # noqa: F811
    broker = FakeBroker()
    broker.bars = engulfing_bars()          # last close 71.7
    worker, audit, _ = make_worker(tmp_path, broker)  # stop 0.5, target 1.0
    worker.poll_once()

    entry_id = int(audit.get_active_position(SYMBOL)["entry_order_id"])
    broker.fill(entry_id, price=70.9)       # market gapped down; marketable limit filled lower
    broker.held = 100
    worker.monitor_position()

    pos = audit.get_active_position(SYMBOL)
    assert pos["entry_price"] == 70.9
    # A BUY's protective stop must be below where we actually bought.
    assert pos["current_stop"] < pos["entry_price"], (
        f"stop {pos['current_stop']} is above fill {pos['entry_price']}"
    )
