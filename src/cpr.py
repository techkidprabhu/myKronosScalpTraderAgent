"""
Central Pivot Range (CPR) - step 1: levels and a scorecard, no trading.

The CPR is three levels worked out before a session from the prior
session's high (H), low (L) and close (C):

    P  = (H + L + C) / 3
    BC = (H + L) / 2
    TC = 2P - BC          (TC and BC swap when TC comes out lower, so TC is
                           always the upper edge)

plus the floor pivots used as targets: R1 = 2P - L, S1 = 2P - H,
R2 = P + (H - L), S2 = P - (H - L). Width% = (TC - BC) / P x 100.

What traders read from it (Frank Ochoa, "Secrets of a Pivot Boss"):
- price above TC = bullish, below BC = bearish, inside = balance;
- a NARROW band (low width for this market) favours a trend day, a WIDE one
  a range day - judged against the market's own last N sessions, not a
  fixed number;
- today's band against yesterday's ("two-day relationship") sets a bias:
  higher value +2, overlapping higher +1, unchanged / outside 0, overlapping
  lower -1, lower value -2; inside value = no bias but a breakout flag.

Nobody publishes statistics showing these tendencies hold, so step 1 only
measures them on the markets this agent watches: build_days() scores every
past session (range against the 14-day average, trend day or not, R1/S1
reached, traded into the band, closed beyond it) and scorecard() sums them
up by width class and by relationship.
"""
from __future__ import annotations

import pandas as pd

DEFAULTS = {
    "enabled": False,
    "backfill_days": 60,     # daily candles fetched the first time a symbol is seen
    "width_lookback": 20,    # sessions the narrow/wide percentile is judged against
    "narrow_pct": 25.0,      # width at or below this percentile = narrow
    "wide_pct": 75.0,        # at or above = wide
    "adr_days": 14,          # average daily range for "how big was the day"
}

RELATIONSHIPS = {
    "higher_value": 2, "overlapping_higher": 1, "unchanged": 0, "outside_value": 0,
    "inside_value": 0, "overlapping_lower": -1, "lower_value": -2,
}
LABELS = {
    "higher_value": "Higher value", "overlapping_higher": "Overlapping higher",
    "unchanged": "Unchanged", "outside_value": "Outside value", "inside_value": "Inside value",
    "overlapping_lower": "Overlapping lower", "lower_value": "Lower value",
}


def settings(config: dict | None) -> dict:
    raw = (config or {}).get("cpr") or {}
    s = {**DEFAULTS, **{k: v for k, v in raw.items() if k in DEFAULTS and v is not None}}
    s["enabled"] = bool(s["enabled"])
    for k in ("backfill_days", "width_lookback", "adr_days"):
        s[k] = max(2, int(s[k]))
    return s


def levels(high: float, low: float, close: float) -> dict:
    h, l, c = float(high), float(low), float(close)
    p = (h + l + c) / 3.0
    bc = (h + l) / 2.0
    tc = 2.0 * p - bc
    if tc < bc:
        tc, bc = bc, tc
    rng = h - l
    return {"P": p, "TC": tc, "BC": bc, "R1": 2 * p - l, "S1": 2 * p - h,
            "R2": p + rng, "S2": p - rng, "width_pct": (tc - bc) / p * 100.0 if p else 0.0}


def relationship(cur: dict, prev: dict) -> str:
    """Today's band against yesterday's (Ochoa's seven two-day relationships)."""
    tol = 0.1 * max(prev["TC"] - prev["BC"], abs(prev["P"]) * 1e-6)
    if abs(cur["TC"] - prev["TC"]) <= tol and abs(cur["BC"] - prev["BC"]) <= tol:
        return "unchanged"
    if cur["BC"] > prev["TC"]:
        return "higher_value"
    if cur["TC"] < prev["BC"]:
        return "lower_value"
    if cur["TC"] <= prev["TC"] and cur["BC"] >= prev["BC"]:
        return "inside_value"
    if cur["TC"] >= prev["TC"] and cur["BC"] <= prev["BC"]:
        return "outside_value"
    return "overlapping_higher" if cur["P"] > prev["P"] else "overlapping_lower"


def width_class(width: float, history: list[float], narrow_pct: float = 25.0, wide_pct: float = 75.0) -> str:
    """narrow / normal / wide against the market's own recent widths."""
    hist = [float(w) for w in history if w == w]
    if len(hist) < 5:
        return "unknown"
    rank = 100.0 * sum(1 for w in hist if w <= width) / len(hist)
    if rank <= narrow_pct:
        return "narrow"
    if rank >= wide_pct:
        return "wide"
    return "normal"


def location(price: float, lv: dict) -> str:
    if price is None or lv is None:
        return "unknown"
    if price > lv["TC"]:
        return "above"
    if price < lv["BC"]:
        return "below"
    return "inside"


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def build_days(daily: pd.DataFrame, cfg: dict | None = None) -> list[dict]:
    """One row per session that has a CPR (from the session before it):
    the levels, width class, relationship, and - when the session itself is
    in `daily` - how it actually went. `daily` = closed daily candles
    (open/high/low/close), oldest first. The last row is the CPR for the
    NEXT session (the one in progress), with no outcome yet."""
    c = {**DEFAULTS, **(cfg or {})}
    if daily is None or len(daily) < 2:
        return []
    d = daily[["open", "high", "low", "close"]].astype(float)
    rows, widths, prev_lv = [], [], None
    ranges = (d["high"] - d["low"]).tolist()
    for i in range(1, len(d) + 1):
        src = d.iloc[i - 1]
        lv = levels(src["high"], src["low"], src["close"])
        row = {**{k: round(v, 6) for k, v in lv.items()},
               "based_on": _day(d.index[i - 1]),
               "width_class": width_class(lv["width_pct"], widths[-int(c["width_lookback"]):],
                                          c["narrow_pct"], c["wide_pct"]),
               "relationship": relationship(lv, prev_lv) if prev_lv else None}
        row["bias"] = RELATIONSHIPS.get(row["relationship"]) if row["relationship"] else None
        if i < len(d):
            day = d.iloc[i]
            hi, lo, cl = float(day["high"]), float(day["low"]), float(day["close"])
            rng = hi - lo
            past = ranges[max(0, i - int(c["adr_days"])):i]
            adr = sum(past) / len(past) if past else 0.0
            where = (cl - lo) / rng if rng > 0 else 0.5
            row.update(day=_day(d.index[i]), open=float(day["open"]), high=hi, low=lo, close=cl,
                       range_x=round(rng / adr, 3) if adr > 0 else None,
                       trend_day=bool(adr > 0 and rng >= adr and (where >= 0.75 or where <= 0.25)),
                       reached_r1=hi >= lv["R1"], reached_s1=lo <= lv["S1"],
                       touched=lo <= lv["TC"] and hi >= lv["BC"],
                       closed_above=cl > lv["TC"], closed_below=cl < lv["BC"])
        else:
            row.update(day=None, current=True)
        rows.append(row)
        widths.append(lv["width_pct"])
        prev_lv = lv
    return rows


def acceptance(bars: pd.DataFrame, lv: dict, closes: int = 2, after_minutes: int = 15,
               session_start=None) -> str:
    """Which side of the band price has ACCEPTED this session: `closes`
    consecutive closes above TC (or below BC) after the first
    `after_minutes`; otherwise "inside". (Used by step 2's setups.)"""
    if bars is None or bars.empty or lv is None:
        return "inside"
    start = pd.Timestamp(session_start) if session_start is not None else bars.index[-1].normalize()
    s = bars[bars.index >= start + pd.Timedelta(minutes=int(after_minutes))]
    if len(s) < closes:
        return "inside"
    tail = s["close"].astype(float).iloc[-int(closes):]
    if (tail > lv["TC"]).all():
        return "above"
    if (tail < lv["BC"]).all():
        return "below"
    return "inside"


def _pct(n: int, total: int):
    return round(100.0 * n / total, 1) if total else None


def scorecard(days: list[dict]) -> dict:
    """Did the CPR tendencies hold? Sessions with an outcome, grouped by
    width class and by two-day relationship, with the all-days base rate
    to compare against."""
    done = [r for r in days if r.get("day")]

    def group(rows):
        n = len(rows)
        rx = [r["range_x"] for r in rows if r.get("range_x") is not None]
        return {"days": n,
                "avg_range_x": round(sum(rx) / len(rx), 2) if rx else None,
                "trend_days_pct": _pct(sum(1 for r in rows if r["trend_day"]), n),
                "reached_r1_or_s1_pct": _pct(sum(1 for r in rows if r["reached_r1"] or r["reached_s1"]), n),
                "touched_pct": _pct(sum(1 for r in rows if r["touched"]), n),
                "closed_above_pct": _pct(sum(1 for r in rows if r["closed_above"]), n),
                "closed_below_pct": _pct(sum(1 for r in rows if r["closed_below"]), n)}

    by_width = [{"class": k, **group([r for r in done if r["width_class"] == k])}
                for k in ("narrow", "normal", "wide")]
    by_rel = [{"relationship": k, "label": LABELS[k], **group([r for r in done if r["relationship"] == k])}
              for k in LABELS]
    return {"all": group(done), "by_width": [g for g in by_width if g["days"]],
            "by_relationship": [g for g in by_rel if g["days"]]}
