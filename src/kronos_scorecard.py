"""
How often were Kronos's verdicts right?

Every time the agent asks Kronos about a signal it logs a decision (with the
confidence) and, on every closed candle, a 'scanning' snapshot with the
symbol's close. So for each verdict we can look up the close at the time and
the close `horizon` candles later - the same horizon Kronos was asked about -
and see whether price went the signal's way. Nothing here calls the model or
the broker; it only reads the audit database.

The useful comparison is "passed" (Kronos agreed, at or above the threshold)
against "rejected" (Kronos disagreed): if passed signals aren't right more
often than rejected ones, Kronos isn't filtering anything on these candles.
"""
from __future__ import annotations

import bisect
import json
import sqlite3
import time

BUCKETS = [(0.0, 0.4, "Below 40%"), (0.4, 0.6, "40-59%"), (0.6, 0.8, "60-79%"), (0.8, 1.01, "80% and up")]


def _bucket(conf: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= conf < hi:
            return name
    return BUCKETS[-1][2]


def _empty(name: str) -> dict:
    return {"name": name, "signals": 0, "scored": 0, "right": 0, "right_pct": None, "avg_move_pct": None,
            "_moves": []}


def _add(g: dict, outcome):
    g["signals"] += 1
    if outcome is None:
        return
    right, move = outcome
    g["scored"] += 1
    g["right"] += int(right)
    g["_moves"].append(move)


def _finish(g: dict) -> dict:
    moves = g.pop("_moves")
    if g["scored"]:
        g["right_pct"] = round(100.0 * g["right"] / g["scored"], 1)
        g["avg_move_pct"] = round(sum(moves) / len(moves), 3)
    return g


def score(conn: sqlite3.Connection, *, horizon: int, bar_seconds: float, threshold: float,
          days: float = 30, now: float | None = None, conf_key: str | None = None) -> dict:
    """Score Kronos verdicts from the last `days` days.

    A verdict is scored once the candle `horizon` candles after it has been
    seen (allowing up to two candles of slack); verdicts whose symbol stopped
    being scanned (dropped by the screener, market closed) stay unscored.
    `conf_key` scores a different confidence stored in the decision details
    (e.g. a shadow measure) instead of the one the agent traded on."""
    now = now if now is not None else time.time()
    since = now - days * 86400
    rows = conn.execute(
        "SELECT ts, epic, pattern_direction, kronos_confirmed, kronos_confidence, details FROM decisions "
        "WHERE ts >= ? AND kronos_confidence IS NOT NULL ORDER BY ts", (since,)).fetchall()
    verdicts = []
    for r in rows:
        conf = r[4]
        if conf_key:
            try:
                conf = json.loads(r[5] or "{}").get(conf_key)
            except (TypeError, ValueError):
                conf = None
            if conf is None:
                continue
        verdicts.append({"ts": float(r[0]), "symbol": r[1], "direction": r[2], "conf": float(conf),
                         "passed": bool(r[3]) and float(conf) >= threshold if not conf_key else float(conf) >= threshold})
    out = {"horizon": horizon, "bar_seconds": bar_seconds, "threshold": threshold, "days": days,
           "conf_key": conf_key, "verdicts": len(verdicts)}
    groups = {"all": _empty("All verdicts"), "passed": _empty("Kronos agreed (traded or sent to risk)"),
              "rejected": _empty("Kronos disagreed (no trade)")}
    buckets = {name: _empty(name) for _, _, name in BUCKETS}
    if verdicts:
        symbols = sorted({v["symbol"] for v in verdicts})
        marks = ",".join("?" * len(symbols))
        snaps: dict[str, tuple[list, list]] = {s: ([], []) for s in symbols}
        for ts, sym, price in conn.execute(
                f"SELECT ts, symbol, price FROM snapshots WHERE stage = 'scanning' AND ts >= ? AND ts <= ? "
                f"AND price IS NOT NULL AND symbol IN ({marks}) ORDER BY ts",
                (since - 2 * bar_seconds, now, *symbols)):
            snaps[sym][0].append(float(ts))
            snaps[sym][1].append(float(price))
        for v in verdicts:
            times, prices = snaps.get(v["symbol"], ([], []))
            outcome = None
            i = bisect.bisect_right(times, v["ts"]) - 1                 # the close Kronos saw
            j = bisect.bisect_left(times, v["ts"] + (horizon - 0.5) * bar_seconds)
            if i >= 0 and j < len(times) and times[j] <= v["ts"] + (horizon + 2) * bar_seconds and prices[i] > 0:
                change = (prices[j] - prices[i]) / prices[i] * 100.0
                move = change if v["direction"] == "BUY" else -change
                outcome = (move > 0, move)
            _add(groups["all"], outcome)
            _add(groups["passed" if v["passed"] else "rejected"], outcome)
            _add(buckets[_bucket(v["conf"])], outcome)
    out["groups"] = [_finish(g) for g in groups.values()]
    out["buckets"] = [_finish(g) for g in buckets.values()]
    p, r = groups["passed"], groups["rejected"]
    out["edge_points"] = (round(p["right_pct"] - r["right_pct"], 1)
                          if p["right_pct"] is not None and r["right_pct"] is not None else None)
    return out
