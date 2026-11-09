"""Performance and decision analytics for the dashboard.

Pure functions over the agent's audit database (sqlite3 connection with
Row factory), so they can be tested without a broker. Everything is
derived from what the agent already records:

- trade_results: one row per closed trade (pnl, exit details)
- decisions:     one row per signal; 'opened' rows carry the planned entry,
                 stop and quantity, and share deal_id with the trade result
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from src.brokers.base import base_symbol

R_BINS = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
CONF_BUCKETS = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
STAGES = ("opened", "indicators", "kronos", "risk", "error")


def _js(text) -> dict:
    try:
        v = json.loads(text or "{}")
        return v if isinstance(v, dict) else {}
    except (TypeError, ValueError):
        return {}


def decision_stage(row: dict) -> str:
    """Which gate a decision stopped at (or 'opened' / 'error')."""
    action = row["action"]
    if action in ("opened", "error"):
        return action
    details = row.get("details") or ""
    has_ind = ("indicators" in details) if isinstance(details, dict) else ('"indicators"' in details)
    if row["kronos_confirmed"] is None and has_ind:
        return "indicators"
    if row["risk_allowed"] == 0:
        return "risk"
    return "kronos"


def _group(trades: list[dict], key) -> list[dict]:
    groups = defaultdict(list)
    for t in trades:
        groups[key(t) or "unknown"].append(t)
    out = []
    for name, ts in groups.items():
        pnls = [t["pnl"] for t in ts]
        rs = [t["r"] for t in ts if t["r"] is not None]
        wins = sum(1 for p in pnls if p > 0)
        out.append({"name": name, "trades": len(ts), "net": round(sum(pnls), 2),
                    "win_rate": round(wins / len(ts) * 100, 1),
                    "avg_r": round(sum(rs) / len(rs), 2) if rs else None})
    return sorted(out, key=lambda g: g["net"], reverse=True)


def _streaks(pnls: list[float]) -> tuple[int, int]:
    best_w = best_l = cur_w = cur_l = 0
    for p in pnls:
        if p > 0:
            cur_w, cur_l = cur_w + 1, 0
        else:
            cur_l, cur_w = cur_l + 1, 0
        best_w, best_l = max(best_w, cur_w), max(best_l, cur_l)
    return best_w, best_l


def closed_trades(conn, since: float) -> list[dict]:
    """Closed trades since `since`, each joined to the decision that opened it."""
    opened = defaultdict(list)          # deal id -> entries (ids are per broker, so match the symbol too)
    for r in conn.execute("SELECT * FROM decisions WHERE action = 'opened' AND deal_id IS NOT NULL ORDER BY ts"):
        opened[str(r["deal_id"])].append(dict(r))
    trades = []
    for r in conn.execute("SELECT * FROM trade_results WHERE ts >= ? ORDER BY ts", (since,)):
        d = dict(r)
        det = _js(d["details"])
        dec = None
        if d["deal_id"] not in (None, "None"):
            cands = [c for c in opened.get(str(d["deal_id"]), [])
                     if base_symbol(c["epic"]) == base_symbol(d["epic"]) and c["ts"] <= d["ts"]]
            dec = cands[-1] if cands else None
        plan = _js(dec["details"]) if dec else {}
        entry = det.get("entry", plan.get("entry"))
        qty = det.get("quantity", plan.get("quantity"))
        risk_amt = None
        if plan.get("entry") is not None and plan.get("stop") is not None and qty:
            risk_amt = abs(float(plan["entry"]) - float(plan["stop"])) * abs(float(qty))
        pnl = float(d["pnl"] or 0.0)
        trades.append({
            "ts": d["ts"], "symbol": d["epic"], "pnl": round(pnl, 2),
            "reason": det.get("reason") or "unknown", "entry": entry, "exit": det.get("exit"), "quantity": qty,
            "pattern": (dec or {}).get("pattern") or "unknown",
            "direction": (dec or {}).get("pattern_direction"),
            "confidence": (dec or {}).get("kronos_confidence"),
            "manual": (dec or {}).get("pattern") == "manual",
            "opened_ts": (dec or {}).get("ts"),
            "hold_sec": (d["ts"] - dec["ts"]) if dec else None,
            "risk": round(risk_amt, 2) if risk_amt else None,
            "r": round(pnl / risk_amt, 2) if risk_amt else None,
        })
    return trades


def performance(conn, since: float, tz: str) -> dict:
    trades = closed_trades(conn, since)
    zone = ZoneInfo(tz)
    pnls = [t["pnl"] for t in trades]
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p <= 0]
    rs = [t["r"] for t in trades if t["r"] is not None]
    holds = [t["hold_sec"] for t in trades if t["hold_sec"]]

    equity, drawdown, run, peak, max_dd, max_dd_pct = [], [], 0.0, 0.0, 0.0, None
    for t in trades:
        run += t["pnl"]
        peak = max(peak, run)
        dd = run - peak
        if dd < max_dd:
            max_dd = dd
            max_dd_pct = round(dd / peak * 100, 1) if peak > 0 else None
        equity.append({"ts": t["ts"], "cum": round(run, 2)})
        drawdown.append({"ts": t["ts"], "dd": round(dd, 2)})

    daily = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
    for t in trades:
        day = datetime.fromtimestamp(t["ts"], zone).date().isoformat()
        daily[day]["pnl"] += t["pnl"]
        daily[day]["trades"] += 1

    r_hist = [0] * (len(R_BINS) + 1)
    for r in rs:
        i = 0
        while i < len(R_BINS) and r >= R_BINS[i]:
            i += 1
        r_hist[i] += 1

    calib = []
    for lo, hi in CONF_BUCKETS:
        ts = [t for t in trades if t["confidence"] is not None and lo <= t["confidence"] < hi and not t["manual"]]
        if ts:
            calib.append({"bucket": f"{int(lo * 100)}–{min(int(hi * 100), 100)}%", "trades": len(ts),
                          "win_rate": round(sum(1 for t in ts if t["pnl"] > 0) / len(ts) * 100, 1),
                          "net": round(sum(t["pnl"] for t in ts), 2)})

    best_w, best_l = _streaks(pnls)
    summary = {
        "net": round(sum(pnls), 2), "trades": len(pnls), "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else None,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "payoff": round((sum(wins) / len(wins)) / abs(sum(losses) / len(losses)), 2)
                  if wins and losses and sum(losses) else None,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if wins and losses and sum(losses) else None,
        "expectancy": round(sum(pnls) / len(pnls), 2) if pnls else None,
        "avg_r": round(sum(rs) / len(rs), 2) if rs else None, "r_trades": len(rs),
        "best": max(pnls) if pnls else None, "worst": min(pnls) if pnls else None,
        "max_drawdown": round(max_dd, 2), "max_drawdown_pct": max_dd_pct,
        "win_streak": best_w, "loss_streak": best_l,
        "avg_hold_sec": round(sum(holds) / len(holds)) if holds else None,
    }
    return {
        "summary": summary, "equity": equity, "drawdown": drawdown,
        "daily": [{"date": k, "pnl": round(v["pnl"], 2), "trades": v["trades"]} for k, v in sorted(daily.items())],
        "by_pattern": _group(trades, lambda t: "manual" if t["manual"] else t["pattern"]),
        "by_symbol": _group(trades, lambda t: t["symbol"]),
        "by_reason": _group(trades, lambda t: str(t["reason"]).replace("_", " ")),
        "by_direction": _group(trades, lambda t: {"BUY": "long", "SELL": "short"}.get(t["direction"], "unknown")),
        "r_hist": {"edges": R_BINS, "counts": r_hist},
        "confidence": calib,
        "trades": list(reversed(trades))[:100],
    }


def decision_stats(conn, since: float, tz: str) -> dict:
    zone = ZoneInfo(tz)
    totals = {s: 0 for s in STAGES}
    daily = defaultdict(lambda: {s: 0 for s in STAGES})
    reasons = defaultdict(int)
    patterns = defaultdict(lambda: {"signals": 0, "opened": 0})
    conf = {"confirmed": [0] * 10, "rejected": [0] * 10}
    signals = 0
    for r in conn.execute("SELECT * FROM decisions WHERE ts >= ? ORDER BY ts", (since,)):
        d = dict(r)
        if d["pattern"] == "manual":
            continue
        signals += 1
        stage = decision_stage(d)
        totals[stage] += 1
        daily[datetime.fromtimestamp(d["ts"], zone).date().isoformat()][stage] += 1
        p = patterns[d["pattern"] or "unknown"]
        p["signals"] += 1
        p["opened"] += stage == "opened"
        if stage in ("indicators", "kronos", "risk"):
            reasons[(stage, (d["risk_reason"] or "no reason given").split(" (")[0][:90])] += 1
        c = d["kronos_confidence"]
        if c is not None and stage != "indicators":
            conf["confirmed" if d["kronos_confirmed"] else "rejected"][min(9, max(0, int(float(c) * 10)))] += 1
    after_ind = signals - totals["indicators"]
    after_kr = after_ind - totals["kronos"]
    return {
        "signals": signals, "totals": totals,
        "funnel": [
            {"name": "Signals found", "count": signals},
            {"name": "Passed indicators", "count": after_ind},
            {"name": "Confirmed by Kronos", "count": after_kr},
            {"name": "Passed risk", "count": after_kr - totals["risk"]},
            {"name": "Orders sent", "count": totals["opened"]},
        ],
        "daily": [{"date": k, **v} for k, v in sorted(daily.items())],
        "top_reasons": [{"stage": k[0], "reason": k[1], "count": v}
                        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]],
        "by_pattern": sorted(({"name": k, **v, "open_rate": round(v["opened"] / v["signals"] * 100, 1)}
                              for k, v in patterns.items()), key=lambda x: -x["signals"]),
        "confidence_hist": conf,
    }
