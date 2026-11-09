"""Today's P&L on open positions counts only today's move, not the whole
gain since an entry made on an earlier day."""
from dashboard.app import _apply_today_pnl

DAY_START = 1_000_000.0


def pos(**kw):
    base = {"symbol": "US.AAPL", "side": "LONG", "quantity": 10, "current_price": 110.0,
            "unrealized_pnl": 100.0, "state": "managed", "opened_ts": DAY_START - 86400}
    base.update(kw)
    return base


def test_opened_today_counts_everything_since_entry():
    p = _apply_today_pnl(pos(opened_ts=DAY_START + 60), {"AAPL": 108.0}, DAY_START)
    assert p["today_pnl"] == 100.0 and p["today_basis"] == "entry_today"


def test_held_overnight_uses_previous_close():
    p = _apply_today_pnl(pos(), {"AAPL": 108.0}, DAY_START)
    assert p["today_pnl"] == 20.0 and p["today_basis"] == "prev_close"


def test_short_held_overnight():
    p = _apply_today_pnl(pos(side="SHORT", unrealized_pnl=-100.0), {"AAPL": 108.0}, DAY_START)
    assert p["today_pnl"] == -20.0


def test_broker_today_figure_moved_to_live_price():
    # broker said +15 today at 109.5; live price is 110 -> +5 more on 10 shares
    p = _apply_today_pnl(pos(_broker_today=15.0, _broker_price=109.5), {"AAPL": 108.0}, DAY_START)
    assert p["today_pnl"] == 20.0 and p["today_basis"] == "broker"
    assert "_broker_today" not in p and "_broker_price" not in p


def test_no_previous_close_falls_back_to_since_entry():
    p = _apply_today_pnl(pos(), {}, DAY_START)
    assert p["today_pnl"] == 100.0 and p["today_basis"] == "entry"


def test_pending_entry_has_no_today_pnl():
    p = _apply_today_pnl(pos(state="pending_entry", unrealized_pnl=None), {}, DAY_START)
    assert p["today_pnl"] is None
