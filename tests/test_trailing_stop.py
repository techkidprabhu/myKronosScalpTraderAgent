from src.trailing_stop import TrailConfig, TrailUpdate, compute_update


def test_trailing_stop_buy_engagement():
    cfg = TrailConfig(trigger_points=0.10, trail_distance=0.15)
    # Entry at 10.00, initial stop at 9.85
    # Price moves to 10.05 (less than 0.10 profit): not yet engaged
    res = compute_update(
        direction="BUY",
        entry_price=10.00,
        current_price=10.05,
        current_stop=9.85,
        current_extreme=10.00,
        trail_active=False,
        target=10.50,
        config=cfg,
    )
    assert res.trail_active is False
    assert res.new_stop == 9.85
    assert res.exit_triggered is False

    # Price moves to 10.12 (exceeds 0.10 profit): engages!
    # Extreme is 10.12, stop ratchets to 10.12 - 0.15 = 9.97
    res2 = compute_update(
        direction="BUY",
        entry_price=10.00,
        current_price=10.12,
        current_stop=9.85,
        current_extreme=10.05,
        trail_active=False,
        target=10.50,
        config=cfg,
    )
    assert res2.trail_active is True
    assert round(res2.new_stop, 2) == 9.97
    assert res2.stop_changed is True


def test_trailing_stop_never_loosens():
    cfg = TrailConfig(trigger_points=0.10, trail_distance=0.15)
    # Price pulls back from 10.12 to 10.05: stop stays at 9.97
    res = compute_update(
        direction="BUY",
        entry_price=10.00,
        current_price=10.05,
        current_stop=9.97,
        current_extreme=10.12,
        trail_active=True,
        target=10.50,
        config=cfg,
    )
    assert res.new_stop == 9.97
    assert res.stop_changed is False


def test_trailing_stop_exit_triggered():
    cfg = TrailConfig(trigger_points=0.10, trail_distance=0.15)
    # Price drops below stop of 9.97 to 9.95
    res = compute_update(
        direction="BUY",
        entry_price=10.00,
        current_price=9.95,
        current_stop=9.97,
        current_extreme=10.12,
        trail_active=True,
        target=10.50,
        config=cfg,
    )
    assert res.exit_triggered is True
