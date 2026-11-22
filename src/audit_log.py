"""
SQLite audit log. Every stage of the decision pipeline gets a row, even
when the outcome is "rejected" - the point is a full, queryable trail of
why every trade was or wasn't taken.

`snapshots` additionally records a lightweight state row on every poll
cycle (not just when a pattern fires) - this is what the centralized
dashboard (see the agent-dashboard project) polls to animate a live
price line and current pipeline stage per agent between trade events.

`positions` tracks every agent-managed position through its lifecycle:
  pending_entry -> open -> closed      (normal)
  pending_entry -> cancelled           (entry never filled)
along with its broker order ids and trailing-stop state. Because this
lives on disk, a restart picks up exactly where it left off instead of
starting blind - and it's what stops a restart from re-entering a
symbol the agent already holds.

`kv` is a small key/value store for state that must also survive a
restart: the risk manager's daily P&L / halt flag / per-symbol cooldowns,
and the last bar each symbol worker has already processed.
"""
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    epic TEXT NOT NULL,
    pattern TEXT,
    pattern_direction TEXT,
    pattern_strength REAL,
    kronos_confirmed INTEGER,
    kronos_confidence REAL,
    risk_allowed INTEGER,
    risk_reason TEXT,
    action TEXT,           -- 'opened', 'rejected', 'error'
    deal_id TEXT,
    details TEXT           -- free-form JSON for anything else
);

CREATE TABLE IF NOT EXISTS trade_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    epic TEXT NOT NULL,
    deal_id TEXT,
    pnl REAL,
    details TEXT
);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    symbol TEXT NOT NULL,
    price REAL,
    stage TEXT,             -- 'scanning', 'pattern_detected', 'kronos_confirming',
                             -- 'risk_check', 'position_open', 'rejected', etc.
    details TEXT            -- free-form JSON, e.g. {"pattern":..., "confidence":...}
);

CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    direction TEXT NOT NULL,        -- 'BUY' or 'SELL'
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    initial_stop REAL NOT NULL,
    current_stop REAL NOT NULL,
    target REAL,
    stop_order_id TEXT,             -- broker-side resting stop order id, if any
    high_water REAL NOT NULL,       -- best price seen since entry (favorable direction)
    trail_active INTEGER NOT NULL DEFAULT 0,
    opened_ts REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',  -- 'pending_entry', 'open', 'closed', 'cancelled'
    entry_order_id TEXT,
    target_order_id TEXT
);

-- Lets the Kronos scorecard find "the price N candles later" quickly.
CREATE INDEX IF NOT EXISTS idx_snapshots_stage_ts ON snapshots (stage, ts);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Manual buy/sell/close requests from the dashboard. The dashboard only
-- queues them; the orchestrator (which owns the broker session and the
-- position bookkeeping) executes them and writes the outcome back.
CREATE TABLE IF NOT EXISTS manual_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,            -- 'BUY', 'SELL' (new entry) or 'CLOSE'
    quantity INTEGER,
    status TEXT NOT NULL DEFAULT 'queued',  -- queued, working, done, failed
    result TEXT,
    done_ts REAL
);
"""

ACTIVE_STATUSES = ("pending_entry", "open")

# Columns added after the first release - ALTERed into older databases.
_POSITION_MIGRATIONS = {
    "entry_order_id": "TEXT",
    "target_order_id": "TEXT",
}


class AuditLog:
    def __init__(self, db_path: str = "data/audit.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(db_path)
        # WAL mode lets a separate read-only process (e.g. the dashboard)
        # query this database concurrently without lock contention.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        existing = {row[1] for row in self.conn.execute("PRAGMA table_info(positions)")}
        for col, col_type in _POSITION_MIGRATIONS.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE positions ADD COLUMN {col} {col_type}")
        self.conn.commit()

    # ---- key/value state ----

    def get_kv(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_kv(self, key: str, value):
        self.conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def log_decision(
        self,
        epic: str,
        pattern: str | None,
        pattern_direction: str | None,
        pattern_strength: float | None,
        kronos_confirmed: bool | None,
        kronos_confidence: float | None,
        risk_allowed: bool | None,
        risk_reason: str | None,
        action: str,
        deal_id: str | None = None,
        details: dict | None = None,
    ):
        self.conn.execute(
            """INSERT INTO decisions
               (ts, epic, pattern, pattern_direction, pattern_strength,
                kronos_confirmed, kronos_confidence, risk_allowed, risk_reason,
                action, deal_id, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), epic, pattern, pattern_direction, pattern_strength,
                int(kronos_confirmed) if kronos_confirmed is not None else None,
                kronos_confidence,
                int(risk_allowed) if risk_allowed is not None else None,
                risk_reason, action, deal_id,
                json.dumps(details or {}),
            ),
        )
        self.conn.commit()

    def log_trade_result(self, epic: str, deal_id: str, pnl: float, details: dict | None = None):
        self.conn.execute(
            "INSERT INTO trade_results (ts, epic, deal_id, pnl, details) VALUES (?, ?, ?, ?, ?)",
            (time.time(), epic, deal_id, pnl, json.dumps(details or {})),
        )
        self.conn.commit()

    def log_snapshot(self, symbol: str, price: float | None, stage: str, details: dict | None = None):
        self.conn.execute(
            "INSERT INTO snapshots (ts, symbol, price, stage, details) VALUES (?, ?, ?, ?, ?)",
            (time.time(), symbol, price, stage, json.dumps(details or {})),
        )
        self.conn.commit()
        # Keep the table bounded - a dashboard only needs recent history.
        self.conn.execute(
            "DELETE FROM snapshots WHERE id NOT IN "
            "(SELECT id FROM snapshots ORDER BY id DESC LIMIT 2000)"
        )
        self.conn.commit()

    # ---- position / trailing-stop state ----

    def record_pending_entry(self, symbol: str, direction: str, quantity: float,
                             entry_price: float, initial_stop: float, target: float | None,
                             entry_order_id: str):
        """Entry order submitted but not yet filled. No protective orders
        exist yet - they're placed once the fill is confirmed."""
        self.conn.execute(
            """INSERT INTO positions
               (symbol, direction, quantity, entry_price, initial_stop, current_stop,
                target, stop_order_id, high_water, trail_active, opened_ts, status,
                entry_order_id, target_order_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, 0, ?, 'pending_entry', ?, NULL)
               ON CONFLICT(symbol) DO UPDATE SET
                 direction=excluded.direction, quantity=excluded.quantity,
                 entry_price=excluded.entry_price, initial_stop=excluded.initial_stop,
                 current_stop=excluded.current_stop, target=excluded.target,
                 stop_order_id=NULL, high_water=excluded.high_water,
                 trail_active=0, opened_ts=excluded.opened_ts, status='pending_entry',
                 entry_order_id=excluded.entry_order_id, target_order_id=NULL""",
            (symbol, direction, quantity, entry_price, initial_stop, initial_stop,
             target, entry_price, time.time(), entry_order_id),
        )
        self.conn.commit()

    def mark_position_open(self, symbol: str, quantity: float, fill_price: float,
                           stop_order_id: str | None, target_order_id: str | None,
                           stop_price: float | None = None, target_price: float | None = None):
        """Entry filled: record the actual filled quantity/price, the
        broker-side protective order ids and (if given) the stop/target
        re-anchored to the fill price."""
        self.conn.execute(
            """UPDATE positions SET status='open', quantity=?, entry_price=?, high_water=?,
                 stop_order_id=?, target_order_id=?, opened_ts=?,
                 initial_stop=COALESCE(?, initial_stop), current_stop=COALESCE(?, current_stop),
                 target=COALESCE(?, target)
               WHERE symbol=? AND status='pending_entry'""",
            (quantity, fill_price, fill_price, stop_order_id, target_order_id, time.time(),
             stop_price, stop_price, target_price, symbol),
        )
        self.conn.commit()

    def set_target(self, symbol: str, target: float, target_order_id: str | None):
        self.conn.execute("UPDATE positions SET target=?, target_order_id=? WHERE symbol=? AND status='open'",
                          (target, target_order_id, symbol))
        self.conn.commit()

    def set_exit_order_ids(self, symbol: str, stop_order_id: str | None, target_order_id: str | None):
        self.conn.execute(
            "UPDATE positions SET stop_order_id=?, target_order_id=? WHERE symbol=? AND status='open'",
            (stop_order_id, target_order_id, symbol),
        )
        self.conn.commit()

    def cancel_pending_entry(self, symbol: str):
        self.conn.execute(
            "UPDATE positions SET status='cancelled' WHERE symbol=? AND status='pending_entry'", (symbol,)
        )
        self.conn.commit()

    def get_active_position(self, symbol: str) -> dict | None:
        """The symbol's position if it's pending entry or open."""
        cur = self.conn.execute(
            "SELECT * FROM positions WHERE symbol = ? AND status IN (?, ?)", (symbol, *ACTIVE_STATUSES)
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def get_active_positions(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM positions WHERE status IN (?, ?)", ACTIVE_STATUSES)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def update_trailing_stop(self, symbol: str, new_stop: float, new_high_water: float,
                              trail_active: bool, stop_order_id: str | None = None):
        if stop_order_id is not None:
            self.conn.execute(
                """UPDATE positions SET current_stop=?, high_water=?, trail_active=?,
                   stop_order_id=? WHERE symbol=? AND status='open'""",
                (new_stop, new_high_water, int(trail_active), stop_order_id, symbol),
            )
        else:
            self.conn.execute(
                """UPDATE positions SET current_stop=?, high_water=?, trail_active=?
                   WHERE symbol=? AND status='open'""",
                (new_stop, new_high_water, int(trail_active), symbol),
            )
        self.conn.commit()

    def close_position(self, symbol: str):
        self.conn.execute(
            "UPDATE positions SET status='closed' WHERE symbol=? AND status='open'", (symbol,)
        )
        self.conn.commit()

    # ---- manual orders (queued by the dashboard) ----

    def queue_manual_order(self, symbol: str, action: str, quantity: int | None) -> int:
        cur = self.conn.execute(
            "INSERT INTO manual_orders (ts, symbol, action, quantity) VALUES (?, ?, ?, ?)",
            (time.time(), symbol, action, quantity),
        )
        self.conn.commit()
        return cur.lastrowid

    def claim_manual_orders(self, max_age_seconds: float = 300) -> list[dict]:
        """Queued requests, marked 'working' so nothing runs them twice.
        Anything older than max_age_seconds is failed instead: a click from
        before an agent restart shouldn't fire minutes later at a new price."""
        now = time.time()
        self.conn.execute(
            "UPDATE manual_orders SET status='failed', done_ts=?, "
            "result='expired - the agent was not running to pick it up' "
            "WHERE status='queued' AND ts < ?", (now, now - max_age_seconds))
        cur = self.conn.execute("SELECT * FROM manual_orders WHERE status='queued' ORDER BY id")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows:
            self.conn.execute("UPDATE manual_orders SET status='working' WHERE id=? AND status='queued'", (r["id"],))
        self.conn.commit()
        return rows

    def finish_manual_order(self, order_id: int, ok: bool, result: str):
        self.conn.execute("UPDATE manual_orders SET status=?, result=?, done_ts=? WHERE id=?",
                          ("done" if ok else "failed", result, time.time(), order_id))
        self.conn.commit()

    def recent_manual_orders(self, limit: int = 10) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM manual_orders ORDER BY id DESC LIMIT ?", (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def close(self):
        self.conn.close()
