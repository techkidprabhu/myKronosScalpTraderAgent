"""Support / resistance levels for the chart (display only)."""
import numpy as np
import pandas as pd

from dashboard.levels import support_resistance, swing_points


def frame(closes, spread=0.2):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c + spread, "low": c - spread, "close": c, "volume": 1000})


def zigzag(tops, bottoms, leg=8, end=None):
    """Price swinging between the given tops and bottoms, ending at `end`."""
    pts, out = [], []
    for t, b in zip(tops, bottoms):
        pts += [b, t]
    pts.append(end if end is not None else (tops[-1] + bottoms[-1]) / 2)
    for a, b in zip(pts, pts[1:]):
        out += list(np.linspace(a, b, leg, endpoint=False))
    return frame(out + [pts[-1]])


def test_swing_points_find_turns():
    df = frame([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4])
    kinds = {(i, k) for i, _, k in swing_points(df, 3, 3)}
    assert (4, "high") in kinds and (8, "low") in kinds


def test_repeated_turns_become_one_level_each_side():
    df = zigzag(tops=[110, 110.2, 109.9], bottoms=[100, 100.1, 99.9], end=105)
    lv = support_resistance(df)
    res = [l for l in lv if l["kind"] == "resistance"]
    sup = [l for l in lv if l["kind"] == "support"]
    assert res and sup
    assert abs(res[0]["price"] - 110.2) < 0.6 and res[0]["touches"] >= 3 and res[0]["strength"] >= 2
    assert abs(sup[0]["price"] - 99.8) < 0.6 and sup[0]["touches"] >= 2
    assert res[0]["distance_pct"] > 0 > sup[0]["distance_pct"]
    assert res[0]["first_index"] < res[0]["last_index"]


def test_broken_resistance_becomes_support():
    df = zigzag(tops=[110, 110.2, 109.9], bottoms=[100, 100.1, 99.9], end=118)   # price broke out above 110
    lv = support_resistance(df)
    near_110 = [l for l in lv if abs(l["price"] - 110) < 1]
    assert near_110 and all(l["kind"] == "support" for l in near_110)


def test_limits_and_short_history():
    assert support_resistance(frame([1, 2, 3])) == []
    rng = np.random.default_rng(3)
    df = frame(100 + np.cumsum(rng.normal(0, 1, 400)))
    lv = support_resistance(df, per_side=3)
    assert sum(l["kind"] == "support" for l in lv) <= 3 and sum(l["kind"] == "resistance" for l in lv) <= 3
    close = df["close"].iloc[-1]
    for kind in ("support", "resistance"):
        d = [abs(l["price"] - close) for l in lv if l["kind"] == kind]
        assert d == sorted(d)                      # nearest first
