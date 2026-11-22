"""
Shadow mode: test a fine-tuned Kronos on live signals without letting it
affect any trade.

Whenever the agent asks Kronos for a verdict, the same candles are also
handed to the shadow model on a background thread. Only the original's
answer is used for trading; the shadow's answer is written to
data/shadow.db next to it. The report then shows how often the two agreed
and, using the candles collected for fine-tuning, which of them was right
about where the price went.

Switched on by finetune.shadow_model (a model folder under models/).
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT, settings
from .data import bars_path

from src.market import BAR_DURATION

logger = logging.getLogger(__name__)
SHADOW_DB = ROOT / "data" / "shadow.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, broker TEXT, symbol TEXT, timeframe TEXT, direction TEXT, horizon INTEGER,
    last_close REAL, threshold REAL,
    main_conf REAL, main_pass INTEGER, shadow_conf REAL, shadow_pass INTEGER, shadow_model TEXT
);
CREATE INDEX IF NOT EXISTS shadow_ts ON shadow (ts);
"""


class ShadowRunner:
    """Runs the shadow model off the trading path. `submit` never blocks and
    never raises; if the queue is full the sample is dropped."""

    def __init__(self, agent, model_name: str, db_path: Path | None = None):
        self.agent, self.model_name = agent, str(model_name)
        self.db_path = Path(db_path or SHADOW_DB)
        self.q: queue.Queue = queue.Queue(maxsize=50)
        self.dropped = 0
        self.thread = threading.Thread(target=self._loop, name="kronos-shadow", daemon=True)
        self.thread.start()

    def submit(self, df, direction: str, horizon: int, *, broker: str, symbol: str, timeframe: str,
               last_close: float, threshold: float, main_confirmed: bool, main_confidence: float) -> bool:
        try:
            self.q.put_nowait({
                "df": df.copy(), "direction": direction, "horizon": int(horizon), "ts": time.time(),
                "broker": broker, "symbol": symbol, "timeframe": timeframe, "last_close": float(last_close),
                "threshold": float(threshold), "main_conf": float(main_confidence),
                "main_pass": bool(main_confirmed and main_confidence >= threshold),
            })
            return True
        except queue.Full:
            self.dropped += 1
            return False
        except Exception:  # noqa: BLE001
            logger.debug("Shadow submit failed", exc_info=True)
            return False

    def _loop(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        while True:
            job = self.q.get()
            if job is None:
                break
            try:
                confirmed, conf = self.agent.confirm_direction(
                    job["df"], job["direction"], job["horizon"], bar_duration=BAR_DURATION.get(job["timeframe"]))
                conn.execute(
                    "INSERT INTO shadow (ts, broker, symbol, timeframe, direction, horizon, last_close, threshold, "
                    "main_conf, main_pass, shadow_conf, shadow_pass, shadow_model) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (job["ts"], job["broker"], job["symbol"], job["timeframe"], job["direction"], job["horizon"],
                     job["last_close"], job["threshold"], job["main_conf"], int(job["main_pass"]), float(conf),
                     int(bool(confirmed and conf >= job["threshold"])), self.model_name))
                conn.commit()
            except Exception:  # noqa: BLE001
                logger.exception("Shadow Kronos check failed for %s", job.get("symbol"))
            finally:
                self.q.task_done()

    def close(self):
        self.q.put(None)


def build(config: dict):
    """A ShadowRunner for finetune.shadow_model, or None when shadow mode is
    off or the model can't be loaded (trading carries on either way)."""
    ft = settings(config)
    name = ft.get("shadow_model")
    if not name:
        return None
    try:
        from src.kronos_agent import KronosAgent
        kcfg = dict(config.get("kronos") or {})
        kcfg["model_name"] = str(name)
        if ft.get("shadow_tokenizer"):
            kcfg["tokenizer_name"] = str(ft["shadow_tokenizer"])
        runner = ShadowRunner(KronosAgent(**kcfg), Path(str(name)).name)
        logger.info("Shadow mode: %s is checked alongside the trading model (it cannot affect trades)", name)
        return runner
    except Exception:  # noqa: BLE001
        logger.exception("Shadow model %s could not be loaded - continuing without shadow mode", name)
        return None


# ---------------------------------------------------------------- report

def _outcome(row: dict, cache: dict, root: Path | None) -> bool | None:
    """Did the price move in the signal's direction over the horizon?
    Needs the candles collected for fine-tuning; None if not available yet."""
    key = (row["broker"], row["symbol"], row["timeframe"])
    if key not in cache:
        p = bars_path(*key, root)
        try:
            df = pd.read_csv(p, usecols=["time", "close"]) if p.exists() else None
        except Exception:  # noqa: BLE001
            df = None
        cache[key] = None if df is None else (df["time"].to_numpy(dtype="int64"), df["close"].to_numpy(dtype=float))
    if cache[key] is None:
        return None
    times, closes = cache[key]
    i = int(np.searchsorted(times, row["ts"] * 1000.0, side="right")) - 1
    j = i + int(row["horizon"] or 5)
    if i < 0 or j >= len(closes) or not row["last_close"]:
        return None
    if closes[j] == row["last_close"]:
        return None
    return bool((closes[j] > row["last_close"]) == (str(row["direction"]).upper() == "BUY"))


def report(db_path: Path | None = None, training_root: Path | None = None, model: str | None = None,
           days: float = 60) -> dict:
    db_path = Path(db_path or SHADOW_DB)
    empty = {"signals": 0, "model": model}
    if not db_path.exists():
        return empty
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        sql, args = "SELECT * FROM shadow WHERE ts >= ?", [time.time() - days * 86400]
        if model:
            sql += " AND shadow_model = ?"
            args.append(model)
        rows = [dict(r) for r in conn.execute(sql + " ORDER BY ts", args)]
        conn.close()
    except sqlite3.Error:
        return empty
    if not rows:
        return empty
    cache: dict = {}
    for r in rows:
        r["right"] = _outcome(r, cache, training_root)

    def side(flag: str) -> dict:
        took = [r for r in rows if r[flag]]
        known = [r for r in took if r["right"] is not None]
        return {"confirmed": len(took), "scored": len(known),
                "right_pct": round(sum(r["right"] for r in known) / len(known) * 100, 1) if known else None}

    def bucket(m: int, s: int) -> dict:
        rs = [r for r in rows if r["main_pass"] == m and r["shadow_pass"] == s]
        known = [r for r in rs if r["right"] is not None]
        return {"signals": len(rs), "scored": len(known),
                "right_pct": round(sum(r["right"] for r in known) / len(known) * 100, 1) if known else None}

    disagreements = [r for r in rows if r["main_pass"] != r["shadow_pass"]]
    return {
        "model": model or rows[-1]["shadow_model"], "signals": len(rows), "since": rows[0]["ts"], "last": rows[-1]["ts"],
        "agree_pct": round((len(rows) - len(disagreements)) / len(rows) * 100, 1),
        "scored": sum(r["right"] is not None for r in rows),
        "original": side("main_pass"), "shadow": side("shadow_pass"),
        "only_original": bucket(1, 0), "only_shadow": bucket(0, 1),
        "recent_disagreements": [{k: r[k] for k in ("ts", "broker", "symbol", "direction", "main_conf", "shadow_conf",
                                                    "main_pass", "shadow_pass", "right")} for r in disagreements[-12:][::-1]],
    }
