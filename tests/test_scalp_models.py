"""The four scalping models: momentum/breakout (ORB), trend pullback, mean
reversion and liquidity sweep."""
import numpy as np
import pytest

from src import scalp_engine as se, skill_strategy as ss
from tests.test_scalp import orb_rows, scalp, session


def ranging(n=60):
    rows = []
    for k in range(n):
        c = 100 + 0.08 * np.sin(k / 2)
        rows.append((c - 0.02, c + 0.05, c - 0.05, c, 1000))
    return rows


def trending(n=60):
    return [(100 + .1 * k - .02, 100 + .1 * k + .05, 100 + .1 * k - .05, 100 + .1 * k, 1000) for k in range(n)]


SPIKE_DOWN_THEN_BACK = [(99.95, 100.0, 98.95, 99.0, 1500), (99.0, 99.55, 98.98, 99.5, 1200)]
SPIKE_UP_THEN_BACK = [(100.05, 101.05, 100.0, 101.0, 1500), (101.0, 101.02, 100.45, 100.5, 1200)]


@pytest.mark.parametrize("value,expected", [
    ("both", {"orb", "pullback"}), ("all", set(se.MODELS)), ("orb", {"orb"}),
    ("orb, mean_reversion", {"orb", "mean_reversion"}), ("['orb', 'sweep']", {"orb", "sweep"}),
    (["pullback", "liquidity-sweep"], {"pullback", "sweep"}), ("breakout reversion", {"orb", "mean_reversion"}),
])
def test_setups_can_name_any_mix_of_models(value, expected):
    assert se.parse_setups(value)[0] == expected


def test_skill_block_accepts_a_yaml_list_and_reports_typos():
    s, probs = ss.parse("```yaml\nstrategy: scalp_vwap_orb\nsetups: [orb, sweep, meanrev]\n```")
    assert se.parse_setups(s.setups)[0] == {"orb", "sweep"} and any("meanrev" in p for p in probs)
    s, probs = ss.parse("```yaml\nstrategy: scalp_vwap_orb\nsetups: nonsense\n```")
    assert s.setups == "both" and probs


def test_existing_files_trade_exactly_as_before():
    df = session(ranging() + SPIKE_DOWN_THEN_BACK)
    assert ss.find_setup(df, scalp(mr_min_r=0.5)).status == "waiting"          # both = orb + pullback only
    r = ss.find_setup(session(orb_rows((100.9, 101.6, 100.9, 101.5, 3000))), scalp())
    assert r.status == "ready" and r.details["setup"] == "ORB"


def test_mean_reversion_long_after_an_oversold_spike_in_a_range():
    r = ss.find_setup(session(ranging() + SPIKE_DOWN_THEN_BACK), scalp(setups="mean_reversion", mr_min_r=0.5))
    assert r.status == "ready" and r.direction == "BUY" and r.details["setup"] == "mean reversion"
    assert r.entry == 99.5 and r.stop < 98.95 and r.target == pytest.approx(r.details["bb_mid"])
    assert r.details["rsi"] <= 30 and r.details["adx"] < 20


def test_mean_reversion_short_after_an_overbought_spike():
    r = ss.find_setup(session(ranging() + SPIKE_UP_THEN_BACK), scalp(setups="mean_reversion", mr_min_r=0.5))
    assert r.status == "ready" and r.direction == "SELL" and r.stop > 101.05 and r.target < r.entry


def test_mean_reversion_needs_room_to_the_middle_band():
    r = ss.find_setup(session(ranging() + SPIKE_DOWN_THEN_BACK), scalp(setups="mean_reversion"))   # mr_min_r 1.0
    assert r.status == "waiting"                                              # only ~0.7R to the band


def test_mean_reversion_stays_out_of_trends():
    r = ss.find_setup(session(trending() + [(105.95, 106.0, 104.9, 105.0, 1500), (105.0, 105.6, 104.95, 105.5, 1200)]),
                      scalp(setups="mean_reversion", mr_min_r=0.1))
    assert r.status == "waiting" and r.details["adx"] > 20


def test_liquidity_sweep_long_through_the_swing_low():
    r = ss.find_setup(session(ranging() + [(99.95, 100.0, 99.70, 99.96, 1500)]), scalp(setups="sweep"))
    assert r.status == "ready" and r.direction == "BUY" and r.details["swept"] == "swing low"
    assert r.stop < 99.70 and r.target == pytest.approx(r.entry + 1.5 * (r.entry - r.stop))


def test_liquidity_sweep_short_through_the_prior_day_high():
    yesterday = session(ranging(), start="2026-10-30 15:00")                 # prior session: high ~100.13
    today = session([(99.9, 99.95, 99.85, 99.9, 1000)] * 5 + [(99.9, 100.40, 99.88, 99.92, 1500)])
    import pandas as pd
    df = pd.concat([yesterday, today])
    r = ss.find_setup(df, scalp(setups="sweep", sweep_levels="prior_day"))
    assert r.status == "ready" and r.direction == "SELL" and r.details["swept"] == "prior day high"


def test_no_sweep_without_a_rejection_wick():
    r = ss.find_setup(session(ranging() + [(99.75, 99.98, 99.70, 99.96, 1500)]), scalp(setups="sweep"))
    assert r.status == "waiting"                    # closed back inside, but the body is the candle: no rejection


def test_vwap_exit_is_only_for_orb_and_pullback_trades():
    rows = orb_rows((100.9, 101.6, 100.9, 101.5, 3000))
    opened = session(rows).index[-1].isoformat()
    after = session(rows + [(101.5, 101.6, 100.0, 100.1, 1000)])            # closes back below VWAP
    assert ss.exit_signal(after, scalp(), opened, "BUY", setup="ORB").kind == "close_now"
    assert ss.exit_signal(after, scalp(), opened, "BUY") .kind == "close_now"      # older trades: as before
    assert ss.exit_signal(after, scalp(), opened, "BUY", setup="mean reversion") is None
    assert ss.exit_signal(after, scalp(), opened, "BUY", setup="liquidity sweep") is None
