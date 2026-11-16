import argparse
import ipaddress
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from flask import Flask, Response, has_request_context, jsonify, render_template, request, stream_with_context

# Add root directory to sys.path so 'src' can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard import auth
# Ensure Kronos vendor repo is importable
from src.kronos_agent import KronosAgent, _ensure_kronos_importable
from src import market_calendar as cal
from dashboard import analytics
from src.orchestrator import (SCREENER_REPORT_KEY, ACTIVE_SYMBOLS_KEY, BROKER_HEALTH_KEY, PHASE_KEY, UNMANAGED_KEY,
                              bars_to_df)
from src.patterns import scan
from src import indicator_gate
from src.kronos_indicators import indicators as ind_lib
from src.market import drop_unclosed_bar, market_timezone
from src import sentiment
from src import brokers as broker_registry
from src.audit_log import AuditLog
from src.broker_config import broker_specs, spec_for
from src.brokers.base import base_symbol
from src.moomoo_client import RESOLUTION_MAP, MoomooClient, connect_with_timeout
from dashboard import config_writer
from dashboard.schema import schema as settings_schema

load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard")

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

DB_PATH = ROOT_DIR / "data" / "audit.db"
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

DASHBOARD_TOKEN = auth.load_or_create_token(ROOT_DIR / "data" / ".dashboard_token")
auth.install(app, DASHBOARD_TOKEN)

# Kronos inference is CPU-heavy: allow one forecast/scan at a time so
# repeated requests can't pile up and starve the machine.
_kronos_lock = threading.Lock()

# Canonical codes ("US.AAPL", "HK.00700") and IG epics ("UA.D.AAPL.CASH.IP").
SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-_]{0,39}$")

# The only settings the dashboard may change, with their allowed values.
# Everything else - above all the `risk:` section - is read-only here and
# must be edited in config/config.yaml directly.
EDITABLE_SETTINGS = {
    "timeframe": lambda v: v in RESOLUTION_MAP,
    "order_type": lambda v: v in ("LIMIT", "MARKET"),
    "kronos_confidence_threshold": lambda v: isinstance(v, (int, float)) and 0.5 <= v <= 1.0,
    "lookback_bars": lambda v: isinstance(v, int) and 20 <= v <= 512,
    "forecast_horizon": lambda v: isinstance(v, int) and 1 <= v <= 30,
}


class BadRequest(ValueError):
    pass


@app.errorhandler(BadRequest)
def handle_bad_request(exc):
    return jsonify({"error": str(exc)}), 400


def _int_param(source: dict, name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(source.get(name, default))
    except (TypeError, ValueError):
        raise BadRequest(f"{name} must be an integer")
    if not lo <= value <= hi:
        raise BadRequest(f"{name} must be between {lo} and {hi}")
    return value


def _float_param(source: dict, name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(source.get(name, default))
    except (TypeError, ValueError):
        raise BadRequest(f"{name} must be a number")
    if not lo <= value <= hi:
        raise BadRequest(f"{name} must be between {lo} and {hi}")
    return value


def _symbol_param(source: dict) -> str:
    symbol = source.get("symbol")
    if not isinstance(symbol, str) or not SYMBOL_RE.match(symbol):
        raise BadRequest("symbol is required and must be a valid ticker")
    return symbol


def active_symbols(cfg: dict) -> dict:
    """The symbols the agent is actually working, as published by the
    running orchestrator. Before the agent has published anything, falls
    back to what config says it *will* use. Symbols with a pending/open
    position are always included so they stay visible."""
    names = {s["symbol"]: s.get("name", s["symbol"]) for s in cfg.get("symbols", [])}
    published, positioned = None, []
    try:
        conn = get_db()
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (ACTIVE_SYMBOLS_KEY,)).fetchone()
        published = json.loads(row["value"]) if row else None
        positioned = [r["symbol"] for r in conn.execute(
            "SELECT symbol FROM positions WHERE status IN ('pending_entry', 'open')")]
        conn.close()
    except sqlite3.Error:
        pass  # agent not started on this database yet

    screener = cfg.get("screener", {})
    if published and published.get("symbols"):
        source, updated = published.get("source", "agent"), published.get("updated")
        symbols = [s["symbol"] for s in published["symbols"]]
        names.update({s["symbol"]: s["name"] for s in published["symbols"] if s.get("name") != s["symbol"]})
    elif screener.get("enabled"):
        # The screener picks its top_n from these when the agent runs.
        source, updated = "screener_candidates", None
        symbols = list(screener.get("candidate_symbols", []))
    else:
        source, updated = "config", None
        symbols = list(names)

    symbols += [s for s in positioned if s not in symbols]
    return {
        "source": source,
        "updated": updated,
        "symbols": [{"symbol": s, "name": names.get(s, s), "has_position": s in positioned} for s in symbols],
    }


def _timeframe_param(source: dict, default: str) -> str:
    timeframe = source.get("timeframe", default)
    if timeframe not in RESOLUTION_MAP:
        raise BadRequest(f"timeframe must be one of {sorted(RESOLUTION_MAP)}")
    return timeframe

# Global Singletons
BROKER_RETRY_SECONDS = 30  # don't probe a down OpenD on every page poll
BROKER_PING_SECONDS = 30   # how often a connected client is health-checked
# The dashboard must answer page polls quickly, so its connect gives up
# sooner than the agent's.
DASHBOARD_CONNECT_TIMEOUT = 10
_broker_init_lock = threading.Lock()
_kronos_agent = None

MARKET_LABELS = {"US": "United States", "HK": "Hong Kong (HKEX)", "SG": "Singapore (SGX)", "CN": "China A-shares",
                 "NSE": "India (NSE)", "BSE": "India (BSE)"}


MARKET_ENV = {"moomoo": "MOOMOO_TRADE_MARKET", "tiger": "TIGER_MARKET", "ig": "IG_MARKET",
              "zerodha": "KITE_MARKET", "angel": "ANGEL_MARKET"}
DEFAULT_MARKET = {"tiger": "SG", "zerodha": "NSE", "angel": "NSE"}


def load_raw_config() -> dict:
    """config.yaml as written - shared settings plus the brokers: section."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def current_spec(raw: dict | None = None):
    """The broker this request is about: ?broker=<name>, else the first
    enabled broker (moomoo when there's no brokers: section)."""
    raw = load_raw_config() if raw is None else raw
    name = request.args.get("broker") or None if has_request_context() else None
    try:
        return spec_for(raw, name)
    except KeyError:
        raise BadRequest(f"unknown broker {name!r}") from None
    except ValueError as exc:
        raise BadRequest(f"config.yaml brokers section: {exc}") from None


def load_config():
    """Effective settings for the current broker: shared settings with that
    broker's overrides applied."""
    return current_spec().config


def _db_path() -> Path:
    spec = current_spec()
    # The default moomoo DB is DB_PATH (kept as a module setting for tests
    # and for anything that imports it).
    path = DB_PATH if spec.db_path == ROOT_DIR / "data" / "audit.db" else spec.db_path
    if not Path(path).exists():
        AuditLog(str(path)).close()   # a broker that hasn't run yet: empty, valid schema
    return Path(path)


# Currency each market's prices and account are in - shown next to the money
# fields on Config & Risk so a shared value is not mistaken for one currency.
MARKET_CURRENCY = {"US": "USD", "SG": "SGD", "HK": "HKD", "CN": "CNH", "SH": "CNH", "SZ": "CNH",
                   "NSE": "INR", "BSE": "INR", "JP": "JPY", "MY": "MYR", "AU": "AUD", "CA": "CAD", "UK": "GBP"}


def _spec_market(spec) -> str:
    return (spec.market or os.environ.get(MARKET_ENV.get(spec.adapter, ""), "") or
            DEFAULT_MARKET.get(spec.adapter, "US")).upper()


def get_db():
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


_clients: dict = {}   # broker name -> {"client", "failed_at", "checked_at", "error"}


def _client_state(name: str) -> dict:
    return _clients.setdefault(name, {"client": None, "failed_at": 0.0, "checked_at": 0.0, "error": ""})


def get_broker(name: str | None = None):
    """Client for the current (or named) broker, or None while it is
    unreachable. One client per broker. The connect is bounded (never freezes
    a request), a failed connect is retried at most every
    BROKER_RETRY_SECONDS, and a client that stops answering is rebuilt."""
    spec = current_spec() if name is None else spec_for(load_raw_config(), name)
    with _broker_init_lock:
        st = _client_state(spec.name)
        now = time.time()
        if st["client"] is not None and now - st["checked_at"] >= BROKER_PING_SECONDS:
            st["checked_at"] = now
            ok, detail = st["client"].ping()
            if not ok:
                logger.warning("%s unhealthy (%s) - reconnecting", spec.label, detail)
                st["client"].close()
                st.update(client=None, error=detail, failed_at=0.0)
        if st["client"] is None and now - st["failed_at"] >= BROKER_RETRY_SECONDS:
            kwargs = {"connect_timeout": DASHBOARD_CONNECT_TIMEOUT}
            if spec.market:
                kwargs["market"] = spec.market
            try:
                st["client"] = broker_registry.connect(
                    spec.adapter, DASHBOARD_CONNECT_TIMEOUT + 2, data_source=spec.data_source, **kwargs)
                st.update(checked_at=time.time(), error="")
            except Exception as exc:
                st.update(failed_at=time.time(), error=str(exc))
                logger.warning("Could not connect to %s: %s", spec.label, exc)
        return st["client"]


def _not_connected() -> str:
    spec = current_spec()
    return _client_state(spec.name)["error"] or f"{spec.label} not connected"


def _kv(key: str, default=None):
    """Read a value the agent publishes in the kv table (None if absent)."""
    try:
        conn = get_db()
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        conn.close()
        return json.loads(row["value"]) if row else default
    except (sqlite3.Error, ValueError):
        return default


def _agent_market(cfg=None) -> str:
    phase = _kv(PHASE_KEY) or {}
    return (phase.get("market") or _spec_market(current_spec())).upper()


@app.route("/api/markets", methods=["GET"])
def get_markets():
    """Live NASDAQ / NSE sessions (plus the agent's own exchange), the
    agent's sleep/active phase, and upcoming holidays - for the market bar."""
    market = _agent_market()
    own = cal.MARKET_EXCHANGE.get(market)
    codes = ["NASDAQ", "NSE"] + ([own] if own and own not in ("NASDAQ", "NSE") else [])
    sessions = []
    for code in codes:
        sess = cal.session(code)
        sess["upcoming_holidays"] = cal.upcoming_holidays(code, 3)
        sess["agent"] = code == own
        sessions.append(sess)
    return jsonify({
        "server_time": time.time(),
        "agent_market": market,
        "agent_exchange": own,
        "phase": _kv(PHASE_KEY),
        "sessions": sessions,
    })


def get_kronos_agent():
    global _kronos_agent
    if _kronos_agent is None:
        cfg = load_config()
        kronos_cfg = cfg.get("kronos", {})
        try:
            _kronos_agent = KronosAgent(**kronos_cfg)
        except Exception as exc:
            logger.warning("Could not initialize KronosAgent: %s", exc)
    return _kronos_agent


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    cfg = load_config()
    client = get_broker()
    kronos = get_kronos_agent()

    # Determine agent state from latest snapshot
    agent_status = "idle"
    last_poll_ts = None
    recent_snapshots = []
    pipeline = {"scan": None, "cycle": None}
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT ts, stage, symbol, price FROM snapshots ORDER BY id DESC LIMIT 5")
        rows = cur.fetchall()
        if rows:
            last_poll_ts = rows[0]["ts"]
            if time.time() - last_poll_ts < 300:
                agent_status = "active"
            recent_snapshots = [dict(r) for r in rows]
        try:
            pipeline = _pipeline_latest(conn, request.args.get("symbol") or None)
        except Exception:
            logger.exception("Pipeline summary failed")
        conn.close()
    except Exception:
        pass

    # account details (safely masked)
    spec = current_spec()
    account = str(client.account) if client else ""
    market = client.market if client else _spec_market(spec)
    broker_info = {
        "name": spec.name,
        "adapter": spec.adapter,
        "label": spec.label,
        "connected": client is not None,
        "account": (account[:3] + "****" + account[-3:]) if len(account) > 6 else (account or "N/A"),
        "market": MARKET_LABELS.get(market, market),
        "market_code": market,
        "sandbox": client.sandbox if client else True,
        "opend": (f"{os.environ.get('MOOMOO_OPEND_HOST', '127.0.0.1')}:{os.environ.get('MOOMOO_OPEND_PORT', '11111')}"
                  if spec.adapter == "moomoo" else None),
        "funds": None,
    }
    if client:
        try:
            broker_info["funds"] = _account_summary(client)
        except Exception as exc:
            logger.debug("accinfo_query failed: %s", exc)

    # Kronos offline model info
    kronos_info = {
        "ready": kronos is not None,
        "model_name": kronos.model.__class__.__name__ if kronos else "Kronos",
        "model_id": cfg.get("kronos", {}).get("model_name", "NeoQuasar/Kronos-base"),
        "tokenizer_id": cfg.get("kronos", {}).get("tokenizer_name", "NeoQuasar/Kronos-Tokenizer-base"),
        "device": kronos.device if kronos else "cpu",
        "offline": True,
        "max_context": kronos.max_context if kronos else 512,
    }

    # Risk info
    risk_info = {
        "max_daily_loss": cfg.get("risk", {}).get("max_daily_loss", 300.0),
        "max_open_positions": cfg.get("risk", {}).get("max_open_positions", 3),
        "stop_loss_points": cfg.get("risk", {}).get("stop_loss_points", 0.15),
        "take_profit_points": cfg.get("risk", {}).get("take_profit_points", 0.30),
    }

    active = active_symbols(cfg)

    # What the agent itself reports (it may be connected even when this
    # dashboard process isn't, and vice versa).
    agent_broker = _kv(BROKER_HEALTH_KEY)
    broker_info["error"] = None if client else _not_connected()
    unmanaged = _kv(UNMANAGED_KEY) or {"positions": [], "orders": []}

    phase = _kv(PHASE_KEY)
    if phase and time.time() - float(phase.get("ts") or 0) < 180:
        agent_status = {"active": "active", "always": "active"}.get(phase["phase"], phase["phase"])
    if agent_broker and not agent_broker.get("ok") and time.time() - float(agent_broker.get("ts") or 0) < 180:
        agent_status = "broker_down"
    return jsonify({
        "status": "ok",
        "phase": phase,
        "agent_broker": agent_broker,
        "unmanaged": unmanaged,
        "agent_status": agent_status,
        "last_poll_ts": last_poll_ts,
        "broker": broker_info,
        "kronos": kronos_info,
        "risk": risk_info,
        "symbols": [s["symbol"] for s in active["symbols"]],
        "symbol_details": active["symbols"],
        "symbols_source": active["source"],
        "symbols_updated": active["updated"],
        "timeframe": cfg.get("timeframe", "HOUR_4"),
        "forecast_horizon": cfg.get("forecast_horizon", 6),
        "confidence_threshold": cfg.get("kronos_confidence_threshold", 0.62),
        "recent_snapshots": recent_snapshots,
        "pipeline": pipeline,
        "indicator_check": str((cfg.get("indicators") or {}).get("mode") or "off")
                           if (cfg.get("indicators") or {}).get("enabled") else "off",
    })


def _pipeline_latest(conn, symbol: str | None = None) -> dict:
    """The most recent signal's trip through the pipeline, for the Live
    Terminal's Decision Pipeline card.

    Snapshots are logged per symbol as a scan moves along (scanning ->
    pattern_detected -> indicator_check -> kronos_result -> risk_check ->
    risk_plan -> entry_submitted). The card used to show only the single
    newest snapshot of any symbol, which is almost always 'scanning', so the
    later steps never lit up. This gathers every stage of the newest cycle
    that found a pattern - for `symbol` if it has one today, else for any
    symbol - plus the decision that ended it."""
    out = {"scan": None, "cycle": None}
    since = time.time() - 15 * 60
    row = conn.execute("SELECT COUNT(*) AS n, COUNT(DISTINCT symbol) AS syms, MAX(ts) AS last "
                       "FROM snapshots WHERE stage = 'scanning' AND ts >= ?", (since,)).fetchone()
    if row and row["n"]:
        out["scan"] = {"bars": row["n"], "symbols": row["syms"], "last": row["last"], "window_min": 15}

    start = None
    if symbol:
        start = conn.execute("SELECT id, ts, symbol FROM snapshots WHERE stage = 'pattern_detected' AND symbol = ? "
                             "AND ts >= ? ORDER BY id DESC LIMIT 1", (symbol, time.time() - 86400)).fetchone()
    if start is None:
        start = conn.execute("SELECT id, ts, symbol FROM snapshots WHERE stage = 'pattern_detected' "
                             "ORDER BY id DESC LIMIT 1").fetchone()
    if start is None:
        return out
    stages = {}
    for r in conn.execute("SELECT id, ts, stage, price, details FROM snapshots WHERE symbol = ? AND id >= ? "
                          "ORDER BY id LIMIT 20", (start["symbol"], start["id"])):
        if r["stage"] == "scanning":
            break                                   # the next bar's cycle has begun
        try:
            det = json.loads(r["details"]) if r["details"] else {}
        except (TypeError, ValueError):
            det = {}
        stages[r["stage"]] = {"ts": r["ts"], "price": r["price"], **(det if isinstance(det, dict) else {})}
    dec = conn.execute("SELECT action, risk_reason, kronos_confidence, details FROM decisions "
                       "WHERE epic = ? AND ts >= ? ORDER BY id LIMIT 1",
                       (start["symbol"], start["ts"] - 1)).fetchone()
    decision = None
    if dec is not None:
        try:
            ddet = json.loads(dec["details"]) if dec["details"] else {}
        except (TypeError, ValueError):
            ddet = {}
        decision = {"action": dec["action"], "reason": dec["risk_reason"],
                    "error": (ddet or {}).get("error") if isinstance(ddet, dict) else None}
    out["cycle"] = {"symbol": start["symbol"], "ts": start["ts"], "stages": stages, "decision": decision}
    return out


@app.route("/api/symbols", methods=["GET"])
def get_symbols():
    cfg = load_config()
    quantities = {s["symbol"]: s.get("quantity") for s in cfg.get("symbols", [])}
    client = get_broker()

    results = []
    for item in active_symbols(cfg)["symbols"]:
        sym = item["symbol"]
        price = None
        if client:
            try:
                price = client.get_current_price(sym)
            except Exception:
                price = None
        results.append({
            "symbol": sym,
            "name": item["name"],
            "quantity": quantities.get(sym) or cfg.get("default_quantity", 100),
            "latest_price": price,
        })
    return jsonify({"symbols": results})


OVERLAY_WARMUP_BARS = 250   # history beyond the visible bars, for MA/RSI warm-up
OVERLAY_COLORS = {"ma_fast": "#38bdf8", "ma_slow": "#f59e0b", "ma_200": "#e879f9",
                  "vwap": "#a3e635", "rsi": "#c084fc"}


def _series_points(series: pd.Series, times: list[int]) -> list[dict]:
    """Pair a series (aligned with the bars) with bar times, dropping NaNs."""
    out = []
    for t, v in zip(times, series.to_numpy()):
        if v is not None and np.isfinite(v):
            out.append({"time": t, "value": round(float(v), 4)})
    return out


def _chart_overlays(full_df: pd.DataFrame, cfg: dict, timeframe: str, client, limit: int) -> dict:
    """Indicator lines for the chart. Uses the indicator gate's own MA kind
    and periods, so the chart shows exactly what the filter looks at."""
    ind_cfg = indicator_gate.IndicatorConfig.from_dict(cfg.get("indicators"))
    kind = ind_cfg.ma_kind
    tail = slice(-limit, None)
    times = [int(ts.timestamp()) for ts in full_df.index[tail]]
    label = kind.upper()

    lines = []
    for key, period in (("ma_fast", ind_cfg.ma_fast), ("ma_slow", ind_cfg.ma_slow), ("ma_200", 200)):
        if key == "ma_200" and 200 in (ind_cfg.ma_fast, ind_cfg.ma_slow):
            continue
        ma = ind_lib.moving_average(full_df, period, kind)
        lines.append({"id": key, "label": f"{label} {period}", "color": OVERLAY_COLORS[key],
                      "group": "ma", "data": _series_points(ma.iloc[tail], times)})

    if timeframe in indicator_gate.INTRADAY_TIMEFRAMES and "volume" in full_df:
        try:
            tz = market_timezone(client.market)
        except Exception:
            tz = None
        local = indicator_gate._to_market_time(full_df, tz)   # VWAP resets at the local session date
        vw = pd.Series(ind_lib.vwap(local).to_numpy(), index=full_df.index)
        lines.append({"id": "vwap", "label": "VWAP", "color": OVERLAY_COLORS["vwap"],
                      "group": "vwap", "data": _series_points(vw.iloc[tail], times)})

    # Support / resistance from swing points over the loaded history. A level
    # first touched before the visible window is drawn from the left edge.
    levels = []
    try:
        from dashboard.levels import support_resistance
        offset = len(full_df) - len(times)
        for lv in support_resistance(full_df):
            start = max(lv.pop("first_index") - offset, 0)
            lv.pop("last_index", None)
            lv.pop("score", None)
            levels.append({**lv, "from": times[start], "to": times[-1]})
    except Exception:
        logger.exception("Support/resistance levels failed")

    rsi = ind_lib.rsi(full_df["close"], 14)
    return {
        "levels": levels,
        "ma_kind": kind,
        "gate": {"enabled": ind_cfg.enabled, "mode": ind_cfg.mode,
                 "require_ma_trend": ind_cfg.require_ma_trend},
        "lines": lines,
        "rsi": {"period": 14, "color": OVERLAY_COLORS["rsi"],
                "upper": ind_cfg.max_rsi_for_buy, "lower": ind_cfg.min_rsi_for_sell,
                "data": _series_points(rsi.iloc[tail], times)},
    }


@app.route("/api/bars", methods=["GET"])
def get_bars():
    symbol = _symbol_param(request.args)
    limit = _int_param(request.args, "limit", 100, 1, 500)
    cfg = load_config()
    timeframe = _timeframe_param(request.args, cfg.get("timeframe", "HOUR_4"))

    client = get_broker()
    if not client:
        return jsonify({"error": _not_connected()}), 500

    # Fetch extra history so moving averages and RSI are warmed up on the
    # first visible bar, then trim back to `limit` for display.
    want_overlays = request.args.get("overlays", "1") != "0"
    fetch_n = min(limit + OVERLAY_WARMUP_BARS, 1000) if want_overlays else limit
    try:
        raw_bars = client.get_historical_bars(symbol, timeframe, fetch_n)
        full_df = bars_to_df(raw_bars)
        if full_df.empty:
            return jsonify({"bars": [], "patterns": [], "overlays": None})
        overlays = None
        if want_overlays:
            try:
                overlays = _chart_overlays(full_df, cfg, timeframe, client, limit)
            except Exception:
                # An indicator bug must never take the candle chart down.
                logger.exception("Indicator overlays failed for %s", symbol)
        df = full_df.tail(limit)

        bars_data = []
        for ts, row in df.iterrows():
            bars_data.append({
                "time": int(ts.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            })

        # Scan for patterns in the latest 10 windows
        pattern_markers = []
        for i in range(max(2, len(df) - 10), len(df) + 1):
            sub_df = df.iloc[:i]
            signals = scan(sub_df)
            if signals:
                top_signal = max(signals, key=lambda s: s.strength)
                bar_ts = int(sub_df.index[-1].timestamp())
                pattern_markers.append({
                    "time": bar_ts,
                    "position": "aboveBar" if top_signal.direction == "SELL" else "belowBar",
                    "color": "#ef4444" if top_signal.direction == "SELL" else "#10b981",
                    "shape": "arrowDown" if top_signal.direction == "SELL" else "arrowUp",
                    "text": f"{top_signal.name} ({top_signal.direction})",
                    "direction": top_signal.direction,
                    "strength": top_signal.strength,
                })

        return jsonify({
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": bars_data,
            "patterns": pattern_markers,
            "overlays": overlays,
            "last_close": float(df["close"].iloc[-1]),
        })
    except Exception as exc:
        logger.exception("Failed to fetch bars for %s: %s", symbol, exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/forecast", methods=["POST"])
def run_forecast():
    """Run an on-demand forecast with the offline Kronos model, judged the
    same way the agent judges a signal: closed bars only, the configured
    forecast horizon and Kronos sampling settings, and confirmed only when
    more than half of the forecast path - and at least the configured
    threshold - moves the pattern's way. With no pattern on the last closed
    bar there is nothing to confirm, so the reply is Kronos's outlook."""
    data = request.get_json(silent=True) or {}
    symbol = _symbol_param(data)
    cfg = load_config()
    horizon = _int_param(data, "horizon", int(cfg.get("forecast_horizon", 6)), 1, 30)
    # None = the model's own configured setting (config.yaml -> kronos:), like the agent
    sample_count = _int_param(data, "sample_count", 0, 0, 20) or None
    temperature = _float_param(data, "temperature", 0.0, 0.0, 2.0) or None

    timeframe = _timeframe_param(data, cfg.get("timeframe", "HOUR_4"))
    lookback = int(cfg.get("lookback_bars", 100))

    client = get_broker()
    kronos = get_kronos_agent()

    if not client:
        return jsonify({"error": _not_connected()}), 500
    if not kronos:
        return jsonify({"error": "Kronos offline model unavailable"}), 500
    if not _kronos_lock.acquire(blocking=False):
        return jsonify({"error": "Another Kronos forecast is running - try again shortly"}), 429

    try:
        raw_bars = client.get_historical_bars(symbol, timeframe, lookback + 1)
        # The agent never looks at the still-forming bar; neither do we.
        df = drop_unclosed_bar(bars_to_df(raw_bars), timeframe).tail(lookback)
        if df.empty:
            return jsonify({"error": f"No closed bars available for {symbol}"}), 400

        # Pattern scan
        patterns_found = scan(df)
        signal = max(patterns_found, key=lambda s: s.strength) if patterns_found else None

        # Kronos forecast
        pred_df = kronos.forecast(
            df=df,
            forecast_horizon=horizon,
            sample_count=sample_count,
            temperature=temperature,
        )

        last_close = float(df["close"].iloc[-1])
        deltas = pred_df["close"].to_numpy() - last_close
        buy_agreement = float(np.mean(deltas > 0))
        sell_agreement = float(np.mean(deltas < 0))

        threshold = float(cfg.get("kronos_confidence_threshold", 0.62))
        direction = signal.direction if signal else ("BUY" if buy_agreement >= sell_agreement else "SELL")
        confidence = buy_agreement if direction == "BUY" else sell_agreement
        # Same rule as the agent: confirm_direction() needs > 50 %, then the threshold.
        confirmed = bool(signal) and confidence > 0.5 and confidence >= threshold

        forecast_bars = []
        for ts, row in pred_df.iterrows():
            forecast_bars.append({
                "time": int(ts.timestamp()),
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": int(row.get("volume", 0)),
            })

        return jsonify({
            "symbol": symbol,
            "horizon": horizon,
            "last_close": last_close,
            "pattern": {
                "name": signal.name,
                "direction": signal.direction,
                "strength": signal.strength,
            } if signal else None,
            "direction": direction,
            "mode": "confirm" if signal else "outlook",
            "confidence": round(confidence, 4),
            "confirmed": confirmed,
            "threshold": threshold,
            "timeframe": timeframe,
            "based_on_bar": int(df.index[-1].timestamp()),
            "computed_at": int(time.time()),
            "forecast_bars": forecast_bars,
        })
    except Exception as exc:
        logger.exception("Forecast failed: %s", exc)
        return jsonify({"error": str(exc)}), 500
    finally:
        _kronos_lock.release()


def _decision_price(cur, item: dict):
    """Price the agent acted on: the order's entry price for an opened
    trade, otherwise the closed-bar price it snapshotted while evaluating
    the signal (the latest snapshot for the symbol at or just before the
    decision)."""
    details = item.get("details")
    if isinstance(details, dict) and details.get("entry") is not None:
        try:
            return float(details["entry"])
        except (TypeError, ValueError):
            pass
    row = cur.execute(
        "SELECT price FROM snapshots WHERE symbol = ? AND price IS NOT NULL "
        "AND ts <= ? AND ts >= ? ORDER BY ts DESC LIMIT 1",
        (item["epic"], item["ts"] + 1, item["ts"] - 3600),
    ).fetchone()
    return float(row[0]) if row else None


def _attach_current_prices(decisions: list[dict]):
    """Live last price per symbol, plus the move since the decision in the
    signal's direction (positive = the market went the way the signal said)."""
    client = None
    try:
        client = get_broker()
    except Exception:
        pass
    prices = {}
    if client is not None and decisions:
        try:
            prices = _live_prices(client, [d["epic"] for d in decisions])
        except Exception:
            logger.warning("Could not price the audit trail", exc_info=True)
    for d in decisions:
        now = prices.get(base_symbol(d["epic"]))
        d["current_price"] = now
        then = d.get("decision_price")
        d["move_pct"] = None
        if now is not None and then:
            sign = -1 if d.get("pattern_direction") == "SELL" else 1
            d["move_pct"] = round(sign * (now - then) / then * 100, 2)


@app.route("/api/decisions", methods=["GET"])
def get_decisions():
    limit = _int_param(request.args, "limit", 50, 1, 500)
    symbol = request.args.get("symbol")

    try:
        conn = get_db()
        cur = conn.cursor()
        if symbol:
            cur.execute(
                "SELECT * FROM decisions WHERE epic = ? ORDER BY id DESC LIMIT ?",
                (symbol, limit),
            )
        else:
            cur.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()

        results = []
        for r in rows:
            item = dict(r)
            if item.get("details"):
                try:
                    item["details"] = json.loads(item["details"])
                except Exception:
                    pass
            item["decision_price"] = _decision_price(cur, item)
            item["stage"] = "manual" if item.get("pattern") == "manual" else analytics.decision_stage(item)
            results.append(item)
        conn.close()
        _attach_current_prices(results)
        return jsonify({"decisions": results})
    except Exception as exc:
        return jsonify({"decisions": [], "error": str(exc)})


@app.route("/api/snapshots", methods=["GET"])
def get_snapshots():
    limit = _int_param(request.args, "limit", 50, 1, 500)
    symbol = request.args.get("symbol")

    try:
        conn = get_db()
        cur = conn.cursor()
        if symbol:
            cur.execute(
                "SELECT * FROM snapshots WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                (symbol, limit),
            )
        else:
            cur.execute("SELECT * FROM snapshots ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()

        results = []
        for r in rows:
            item = dict(r)
            if item.get("details"):
                try:
                    item["details"] = json.loads(item["details"])
                except Exception:
                    pass
            results.append(item)
        conn.close()
        return jsonify({"snapshots": results})
    except Exception as exc:
        return jsonify({"snapshots": [], "error": str(exc)})


_broker_cache: dict = {}   # broker name -> {"ts", "data"}
_broker_cache_lock = threading.Lock()
BROKER_CACHE_SECONDS = 10  # the page polls every 5s; stay well inside moomoo's rate limits
_funds_cache: dict = {}    # broker name -> {"ts", "data"}


def _broker_snapshot(client):
    """(positions, working orders) straight from moomoo OpenD, cached briefly."""
    with _broker_cache_lock:
        now = time.time()
        cache = _broker_cache.setdefault(current_spec().name, {"ts": 0.0, "data": None})
        if cache["data"] is not None and now - cache["ts"] < BROKER_CACHE_SECONDS:
            return cache["data"]
        positions = client.get_open_positions()
        orders = client.get_open_orders()
        cache.update(ts=now, data=(positions, orders))
        return positions, orders


_quote_cache: dict = {}    # broker name -> {base symbol: (fetched_at, price)}
_prev_close_cache: dict = {}  # broker name -> {base symbol: previous session close}
QUOTE_CACHE_SECONDS = 3


def _live_prices(client, symbols: list[str]) -> dict:
    """base symbol -> last traded price, from a quote snapshot.

    The position query returns the price the broker last stamped on the
    holding - with moomoo that's OpenD's cached account data, which can lag
    the tape by a minute or more. The Live Terminal reads the quote feed, so
    the positions table and the audit trail do too. Prices are cached per
    symbol for a few seconds, so both tables share one quote call."""
    symbols = sorted({s for s in symbols if s})
    if not symbols:
        return {}
    now = time.time()
    with _broker_cache_lock:
        cache = _quote_cache.setdefault(current_spec().name, {})
        prices = {base_symbol(s): cache[base_symbol(s)][1] for s in symbols
                  if base_symbol(s) in cache and now - cache[base_symbol(s)][0] < QUOTE_CACHE_SECONDS}
    stale = [s for s in symbols if base_symbol(s) not in prices]
    if not stale:
        return prices
    try:
        df = client.get_snapshots(stale)
        fetched, prev = {}, {}
        for _, r in df.iterrows():
            price = r.get("last_price")
            if price is not None and price == price and float(price) > 0:
                fetched[base_symbol(str(r["code"]))] = float(price)
            pc = r.get("prev_close_price")
            if pc is not None and pc == pc and float(pc) > 0:
                prev[base_symbol(str(r["code"]))] = float(pc)
    except Exception:
        logger.warning("Live quote snapshot failed - showing last known prices", exc_info=True)
        return prices
    with _broker_cache_lock:
        cache.update({k: (time.time(), v) for k, v in fetched.items()})
        _prev_close_cache.setdefault(current_spec().name, {}).update(prev)
    prices.update(fetched)
    return prices


def _apply_live_price(pos: dict, prices: dict) -> dict:
    price = prices.get(base_symbol(pos["symbol"]))
    if price is None:
        pos["price_source"] = "broker" if pos.get("current_price") is not None else None
        return pos
    pos["current_price"] = price
    pos["price_source"] = "live"
    if pos.get("avg_cost") is not None and pos.get("state") != "pending_entry":
        sign = 1 if pos["side"] == "LONG" else -1
        pos["unrealized_pnl"] = round(sign * (price - float(pos["avg_cost"])) * float(pos["quantity"]), 4)
    return pos


def _market_day_start() -> float:
    """Midnight today in the current broker's market time zone (epoch s)."""
    tz = market_timezone(_spec_market(current_spec()))
    return datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _apply_today_pnl(pos: dict, prev_closes: dict, day_start: float) -> dict:
    """Today's share of a position's P&L, so the Command Center's
    "Today's P&L" doesn't count gains made on earlier days.

    - opened today (agent's record): all of it since entry
    - the broker reports today's P&L (moomoo): that, moved to the live price
    - the previous close is known: change since the previous close
    - otherwise: since entry, and labelled as such
    """
    broker_today = pos.pop("_broker_today", None)
    broker_price = pos.pop("_broker_price", None)
    unreal = pos.get("unrealized_pnl")
    pos["today_pnl"], pos["today_basis"] = None, None
    if unreal is None or pos.get("state") == "pending_entry":
        return pos
    sign = 1 if pos["side"] == "LONG" else -1
    qty = float(pos.get("quantity") or 0)
    price = pos.get("current_price")
    opened = pos.get("opened_ts")
    prev = prev_closes.get(base_symbol(pos["symbol"]))
    if opened and float(opened) >= day_start:
        pos["today_pnl"], pos["today_basis"] = unreal, "entry_today"
    elif broker_today is not None:
        adj = sign * (float(price) - float(broker_price)) * qty if price and broker_price else 0.0
        pos["today_pnl"], pos["today_basis"] = round(broker_today + adj, 4), "broker"
    elif prev and price:
        pos["today_pnl"], pos["today_basis"] = round(sign * (float(price) - prev) * qty, 4), "prev_close"
    else:
        pos["today_pnl"], pos["today_basis"] = unreal, "entry"
    return pos


def _invalidate_broker_cache():
    with _broker_cache_lock:
        for c in (_broker_cache, _quote_cache, _funds_cache):
            c.pop(current_spec().name, None)


def _account_summary(client):
    """Cash / total assets / buying power, cached like positions."""
    with _broker_cache_lock:
        now = time.time()
        cache = _funds_cache.setdefault(current_spec().name, {"ts": 0.0, "data": None})
        if cache["data"] is None or now - cache["ts"] >= BROKER_CACHE_SECONDS:
            cache.update(ts=now, data=client.get_account_summary())
        return cache["data"]


def _agent_fields(row: dict | None) -> dict:
    if row is None:
        return {"stop": None, "target": None, "high_water": None, "trail_active": None,
                "initial_stop": None, "opened_ts": None}
    return {"stop": row["current_stop"], "target": row["target"],
            "high_water": row["high_water"], "trail_active": bool(row["trail_active"]),
            "initial_stop": row["initial_stop"], "opened_ts": row["opened_ts"]}


@app.route("/api/positions", methods=["GET"])
def get_positions():
    """What the broker actually holds and has working, merged with the
    agent's own tracking. Anything at the broker the agent doesn't manage
    is flagged - nothing is moving a stop or target for it."""
    cfg = load_config()
    base = base_symbol
    try:
        conn = get_db()
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM positions WHERE status IN ('pending_entry', 'open')")]
        conn.close()
    except sqlite3.Error:
        rows = []
    agent_by_base = {base(r["symbol"]): r for r in rows}
    managed_order_ids = {
        str(r[k]) for r in rows for k in ("entry_order_id", "stop_order_id", "target_order_id") if r.get(k)
    }

    client = get_broker()
    broker_positions, broker_orders, broker_error = None, None, None
    if client is None:
        broker_error = _not_connected()
    else:
        try:
            broker_positions, broker_orders = _broker_snapshot(client)
        except Exception as exc:
            logger.exception("Broker position/order query failed")
            broker_error = str(exc)

    positions, orders = [], []
    if broker_positions is not None:
        at_broker = set()
        for p in broker_positions:
            qty = int(p["quantity"] or 0)
            if not qty:
                continue
            key = base(p["symbol"])
            at_broker.add(key)
            agent = agent_by_base.get(key)
            positions.append({
                "symbol": agent["symbol"] if agent else p["symbol"],
                "side": "LONG" if qty > 0 else "SHORT",
                "quantity": abs(qty),
                "avg_cost": p["average_cost"],
                "current_price": p["market_price"],
                "unrealized_pnl": p["unrealized_pnl"],
                "_broker_today": p.get("today_pnl"),
                "_broker_price": p["market_price"],
                "currency": p["currency"],
                "state": "managed" if agent and agent["status"] == "open" else "unmanaged",
                **_agent_fields(agent),
            })
        # Agent rows with no broker holding: entries still waiting to fill,
        # or an 'open' row the broker no longer shows (needs attention).
        for key, r in agent_by_base.items():
            if key in at_broker:
                continue
            positions.append({
                "symbol": r["symbol"], "side": "LONG" if r["direction"] == "BUY" else "SHORT",
                "quantity": r["quantity"], "avg_cost": r["entry_price"], "current_price": None,
                "unrealized_pnl": None, "currency": None,
                "state": "pending_entry" if r["status"] == "pending_entry" else "missing_at_broker",
                **_agent_fields(r),
            })
        for o in broker_orders:
            orders.append({**o, "managed": o["id"] in managed_order_ids})
    else:
        # Broker unreachable: show the agent's view only, clearly marked.
        for r in rows:
            positions.append({
                "symbol": r["symbol"], "side": "LONG" if r["direction"] == "BUY" else "SHORT",
                "quantity": r["quantity"], "avg_cost": r["entry_price"], "current_price": None,
                "unrealized_pnl": None, "currency": None,
                "state": r["status"] if r["status"] == "pending_entry" else "managed",
                **_agent_fields(r),
            })

    if client is not None and positions:
        prices = _live_prices(client, [p["symbol"] for p in positions])
        positions = [_apply_live_price(p, prices) for p in positions]
    with _broker_cache_lock:
        prev_closes = dict(_prev_close_cache.get(current_spec().name, {}))
    day_start = _market_day_start()
    positions = [_apply_today_pnl(p, prev_closes, day_start) for p in positions]

    return jsonify({
        "positions": positions,
        "orders": orders,
        "updated_at": time.time(),
        "manual_orders": _recent_manual_orders(),
        "default_quantity": cfg.get("default_quantity", 10),
        "broker_connected": broker_positions is not None,
        "broker_error": broker_error,
        "unmanaged": _kv(UNMANAGED_KEY) or {"positions": [], "orders": []},
        "agent_tracked": len(rows),
        "max_open_positions": cfg.get("risk", {}).get("max_open_positions", 3),
        "holdings_protected": not bool((cfg.get("unmanaged_positions") or {}).get("allow_manual_close", False)),
        "flat_by_minutes_before_close": cfg.get("risk", {}).get("flat_by_minutes_before_close"),
        "trailing_enabled": bool(cfg.get("trailing_stop", {}).get("enabled")),
    })


def _recent_manual_orders(limit: int = 8) -> list[dict]:
    try:
        conn = get_db()
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM manual_orders ORDER BY id DESC LIMIT ?", (limit,))]
        conn.close()
        return rows
    except sqlite3.Error:
        return []


MANUAL_ACTIONS = ("BUY", "SELL", "CLOSE")


@app.route("/api/manual_order", methods=["POST"])
def manual_order():
    """Queue a manual BUY / SELL (new agent-managed entry) or CLOSE for the
    selected broker's agent. The agent process executes it on its next
    cycle - it owns the broker session and the position records, so a
    manual trade gets the same bracket, trailing stop and audit trail."""
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol") or "").strip().upper()
    action = str(body.get("action") or "").strip().upper()
    if not symbol or len(symbol) > 32:
        return jsonify({"error": "symbol is required"}), 400
    if action not in MANUAL_ACTIONS:
        return jsonify({"error": f"action must be one of {', '.join(MANUAL_ACTIONS)}"}), 400
    quantity = None
    if action != "CLOSE":
        try:
            quantity = int(body.get("quantity"))
        except (TypeError, ValueError):
            return jsonify({"error": "quantity must be a whole number"}), 400
        if quantity <= 0:
            return jsonify({"error": "quantity must be positive"}), 400
    from src.audit_log import AuditLog
    audit = AuditLog(str(_db_path()))
    try:
        order_id = audit.queue_manual_order(symbol, action, quantity)
    finally:
        audit.close()
    _invalidate_broker_cache()
    logger.info("Manual %s %s x%s queued (#%s) for %s", action, symbol, quantity, order_id, current_spec().name)
    return jsonify({"ok": True, "id": order_id, "broker": current_spec().name,
                    "message": f"{action} {symbol} queued - the agent picks it up on its next cycle"})


@app.route("/api/scan_now", methods=["POST"])
def scan_now():
    """Trigger an immediate scan cycle across the agent's active symbols."""
    cfg = load_config()
    client = get_broker()
    kronos = get_kronos_agent()
    from src.audit_log import AuditLog

    if not client or not kronos:
        return jsonify({"error": "moomoo OpenD or Kronos model unavailable"}), 500
    if not _kronos_lock.acquire(blocking=False):
        return jsonify({"error": "A scan or forecast is already running"}), 429
    audit = AuditLog(str(_db_path()))
    try:
        return _scan_symbols(cfg, client, kronos, audit)
    finally:
        audit.close()
        _kronos_lock.release()


def _scan_symbols(cfg, client, kronos, audit):
    """Pattern + Kronos pass over the configured symbols. Read-only with
    respect to the broker: never places orders."""
    scanned_results = []
    timeframe = cfg["timeframe"]
    lookback = cfg["lookback_bars"]
    horizon = cfg["forecast_horizon"]
    thresh = cfg["kronos_confidence_threshold"]
    ind_cfg = indicator_gate.IndicatorConfig.from_dict(cfg.get("indicators"))
    try:
        tz = market_timezone(client.market)
    except Exception:
        tz = None

    for item in active_symbols(cfg)["symbols"]:
        sym = item["symbol"]
        try:
            raw_bars = client.get_historical_bars(sym, timeframe, lookback)
            df = bars_to_df(raw_bars)
            if df.empty:
                continue

            last_close = float(df["close"].iloc[-1])
            audit.log_snapshot(sym, last_close, "scanning")

            signals = scan(df)
            if not signals:
                scanned_results.append({
                    "symbol": sym,
                    "stage": "scanning",
                    "price": last_close,
                    "pattern": None,
                })
                continue

            top_sig = max(signals, key=lambda s: s.strength)
            audit.log_snapshot(sym, last_close, "pattern_detected", {
                "pattern": top_sig.name, "direction": top_sig.direction, "strength": top_sig.strength,
            })

            if ind_cfg.enabled:
                verdict = indicator_gate.evaluate(df, top_sig.direction, ind_cfg, timeframe=timeframe, tz=tz)
                audit.log_snapshot(sym, last_close, "indicator_check", {
                    "passed": verdict.passed, "alignment": verdict.alignment, "reasons": verdict.reasons,
                    "skipped": verdict.skipped, "mode": ind_cfg.mode, **verdict.summary,
                })
                if not verdict.passed:
                    reason = "indicators: " + "; ".join(verdict.reasons)
                    audit.log_decision(sym, top_sig.name, top_sig.direction, top_sig.strength,
                                       None, None, None, reason, "rejected",
                                       details={"indicators": verdict.summary})
                    scanned_results.append({
                        "symbol": sym, "stage": "indicator_check", "price": last_close,
                        "pattern": top_sig.name, "direction": top_sig.direction,
                        "action": "rejected", "reason": reason,
                    })
                    continue

            confirmed, conf = kronos.confirm_direction(df, top_sig.direction, horizon)
            audit.log_snapshot(sym, last_close, "kronos_result", {
                "confirmed": confirmed, "confidence": conf,
            })

            if not confirmed or conf < thresh:
                action = "rejected"
                audit.log_decision(
                    sym, top_sig.name, top_sig.direction, top_sig.strength,
                    confirmed, conf, None, None, action,
                )
            else:
                action = "confirmed_passed"

            scanned_results.append({
                "symbol": sym,
                "stage": "kronos_result",
                "price": last_close,
                "pattern": top_sig.name,
                "direction": top_sig.direction,
                "confidence": conf,
                "action": action,
            })
        except Exception as exc:
            scanned_results.append({"symbol": sym, "error": str(exc)})

    return jsonify({"success": True, "results": scanned_results})


def _set_top_level_scalar(text: str, key: str, value) -> str:
    """Replace `key: value` on a top-level line, keeping the file's
    comments and layout intact (yaml.dump would strip every comment)."""
    rendered = json.dumps(value)  # JSON scalars are valid YAML
    pattern = re.compile(rf"^{re.escape(key)}:[^\n#]*?(\s*#.*)?$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(lambda m: f"{key}: {rendered}{m.group(1) or ''}", text, count=1)
    return text.rstrip("\n") + f"\n{key}: {rendered}\n"


@app.route("/api/config", methods=["GET", "POST"])
def handle_config():
    if request.method == "GET":
        return jsonify(load_config())

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequest("expected a JSON object")

    # Legacy endpoint: shared top-level keys only. The full raw file may be
    # posted back (it contains brokers:), so compare against the raw file and
    # leave per-broker settings to /api/settings.
    current = {k: v for k, v in load_raw_config().items() if k != "brokers"}
    data = {k: v for k, v in data.items() if k != "brokers"}
    readonly_changes = sorted(
        k for k, v in data.items() if k not in EDITABLE_SETTINGS and current.get(k) != v
    )
    if readonly_changes:
        return jsonify({"error": "These settings can only be changed in config/config.yaml: "
                                 + ", ".join(readonly_changes)}), 403

    updates = {k: v for k, v in data.items() if k in EDITABLE_SETTINGS and current.get(k) != v}
    for key, value in updates.items():
        if not EDITABLE_SETTINGS[key](value):
            raise BadRequest(f"invalid value for {key}: {value!r}")
    if not updates:
        return jsonify({"success": True, "message": "No changes"})

    try:
        text = CONFIG_PATH.read_text()
        for key, value in updates.items():
            text = _set_top_level_scalar(text, key, value)
        if any(yaml.safe_load(text).get(k) != v for k, v in updates.items()):
            raise RuntimeError("config rewrite did not round-trip")
        CONFIG_PATH.with_suffix(".yaml.bak").write_text(CONFIG_PATH.read_text())
        tmp = CONFIG_PATH.with_suffix(".yaml.tmp")
        tmp.write_text(text)
        os.replace(tmp, CONFIG_PATH)
    except Exception as exc:
        logger.exception("Config update failed")
        return jsonify({"error": str(exc)}), 500

    logger.info("Dashboard updated config: %s", updates)
    return jsonify({"success": True,
                    "message": "Saved. Restart the agent (main.py) for changes to take effect."})


_sentiment_cache: dict = {}            # (broker, timeframe) -> {"ts", "data"}
_sentiment_lock = threading.Lock()
_sentiment_busy: set = set()


def _sentiment_bars(client, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    # The forming bar is kept on purpose: the meter shows the market right now.
    return bars_to_df(client.get_historical_bars(symbol, timeframe, count))


def _compute_sentiment(spec, cfg: dict, s: dict, client) -> dict:
    tf = s["timeframe"]
    market = (client.market or _spec_market(spec)).upper()
    intraday = tf in indicator_gate.INTRADAY_TIMEFRAMES
    results = []
    for symbol, name in sentiment.benchmarks_for(market, s):
        try:
            df = _sentiment_bars(client, symbol, tf, int(s["lookback_bars"]))
            r = sentiment.classify(df, adx_trend=float(s["adx_trend"]), slope_min_pct=float(s["slope_min_pct"]),
                                   stable_bb_percentile=float(s["stable_bb_percentile"]), intraday=intraday)
        except Exception as exc:  # noqa: BLE001 - one bad symbol mustn't hide the others
            r = {"error": str(exc)[:160]}
        results.append({"symbol": symbol, "name": name, **r})

    breadth = None
    n = int(s.get("breadth_symbols") or 0)
    if n > 0:
        frames = {}
        for item in active_symbols(cfg)["symbols"][:n]:
            try:
                frames[item["symbol"]] = _sentiment_bars(client, item["symbol"], tf, 80)
            except Exception:  # noqa: BLE001
                continue
        breadth = sentiment.breadth(frames)
    return {"enabled": True, "broker": spec.name, "source": spec.label, "market": market, "timeframe": tf,
            "overall": sentiment.overall(results), "benchmarks": results, "breadth": breadth,
            "updated": time.time(), "refresh_seconds": int(float(s["refresh_minutes"]) * 60)}


def _clean_num(v):
    """NaN / inf can't go into JSON - send them as null."""
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    return v


# ---------------------------------------------------------------- Kronos self-learning

_ft_inventory_cache = {"ts": 0.0, "rows": []}


def _finetune_in_use(cfg: dict) -> str | None:
    """run_id of the fine-tuned model the agent is configured to use, if any."""
    name = str((cfg.get("kronos") or {}).get("model_name") or "")
    return Path(name).name if name and (Path(name) / "report.json").exists() else None


@app.route("/api/finetune", methods=["GET"])
def finetune_overview():
    from src import finetune
    from src.finetune import data as ftdata, trainer
    cfg = load_config()
    ft = finetune.settings(cfg)
    if time.time() - _ft_inventory_cache["ts"] > 60:
        _ft_inventory_cache.update(ts=time.time(), rows=ftdata.inventory())
    lookback, predict = trainer.window_sizes(cfg)
    base_model, _ = trainer.original_names(cfg)
    runs = [{k: r.get(k) for k in ("run_id", "created", "minutes", "steps", "stopped", "symbols", "symbols_by_broker",
                                   "train_windows", "signal_windows", "original", "fine_tuned", "verdict", "reason",
                                   "has_weights", "base_model")} for r in trainer.list_runs()[:12]]
    return jsonify({
        "enabled": bool(ft["enabled"]), "train_time": ft["train_time"], "train_days": ft["train_days"],
        "max_steps": ft["max_steps"], "max_minutes": ft["max_minutes"], "min_train_windows": ft["min_train_windows"],
        "lookback": lookback, "predict": predict, "base_model": base_model, "in_use": _finetune_in_use(cfg),
        "model_name": (cfg.get("kronos") or {}).get("model_name"),
        "inventory": _ft_inventory_cache["rows"], "status": trainer.read_status(), "runs": runs,
    })


@app.route("/api/finetune/<action>", methods=["POST"])
def finetune_action(action):
    from src import finetune
    from src.finetune import scheduler as ft_scheduler, trainer
    if action == "collect":
        finetune.COLLECT_REQUEST.parent.mkdir(parents=True, exist_ok=True)
        finetune.COLLECT_REQUEST.write_text(str(time.time()))
        return jsonify({"success": True, "message": "Asked the running agent to save candles now. "
                                                    "Each connected broker does it within a minute."})
    if action == "train":
        try:
            pid = ft_scheduler.start_training_process()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        logger.info("Dashboard started Kronos fine-tuning (pid %s)", pid)
        return jsonify({"success": True, "message": "Training started. Progress shows here; details in logs/finetune.log."})
    if action == "use":
        data = request.get_json(silent=True) or {}
        run_id = str(data.get("run_id") or "")
        cfg = load_config()
        if run_id == "original":
            current = Path(str((cfg.get("kronos") or {}).get("model_name") or "")) / "report.json"
            if not current.exists():
                return jsonify({"success": True, "message": "The agent already uses the original model."})
            rep = json.loads(current.read_text())
            model, tok = rep["base_model"], rep["base_tokenizer"]
        else:
            run = next((r for r in trainer.list_runs() if r["run_id"] == run_id), None)
            if run is None or not run["has_weights"]:
                return jsonify({"error": "That run has no saved model"}), 404
            model, tok = run["path"], run.get("tokenizer") or run["base_tokenizer"]
        try:
            config_writer.write_config(CONFIG_PATH, config_writer.set_kronos_model(CONFIG_PATH.read_text(), model, tok))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Switching the Kronos model failed")
            return jsonify({"error": str(exc)}), 400
        logger.info("Dashboard switched the Kronos model to %s", model)
        return jsonify({"success": True, "message": "Saved to config.yaml (backup kept). Restart the agent to load it."})
    return jsonify({"error": "unknown action"}), 404


@app.route("/api/screener", methods=["GET"])
def get_screener_report():
    """What the agent's last screener run saw: each candidate's readings,
    its score and whether it was picked. Read-only; empty until the agent
    has run the screener."""
    cfg = load_config()
    scr = cfg.get("screener", {}) or {}
    report = _kv(SCREENER_REPORT_KEY)
    out = {"enabled": bool(scr.get("enabled")),
           "mode": str((scr.get("score") or {}).get("mode") or "disabled"),
           "report": None}
    if isinstance(report, dict):
        report["candidates"] = [{k: _clean_num(v) for k, v in c.items()}
                                for c in report.get("candidates") or [] if isinstance(c, dict)]
        report["benchmark_return_pct"] = _clean_num(report.get("benchmark_return_pct"))
        out["report"] = report
    return jsonify(out)


@app.route("/api/sentiment", methods=["GET"])
def get_sentiment():
    """Market Sentiment Meter for the current broker's market: regime of its
    benchmark ETFs (Up trend / Down trend / Sideways / Stable), a -100..100
    needle, and breadth over the agent's active symbols. Cached for
    sentiment.refresh_minutes; ?refresh=1 forces a new reading. Display only."""
    cfg = load_config()
    s = sentiment.settings(cfg)
    if not s.get("enabled", True):
        return jsonify({"enabled": False})
    spec = current_spec()
    key = (spec.name, s["timeframe"])
    force = request.args.get("refresh") == "1"
    now = time.time()
    with _sentiment_lock:
        cached = _sentiment_cache.get(key)
        fresh = cached and now - cached["ts"] < float(s["refresh_minutes"]) * 60
        if cached and (fresh and not force or key in _sentiment_busy):
            return jsonify({**cached["data"], "cached": True})
        _sentiment_busy.add(key)
    try:
        client = get_broker()
        if not client:
            body = {"enabled": True, "error": _not_connected(), "source": spec.label}
            if cached:
                body.update({**cached["data"], "stale": True, "error": _not_connected()})
            return jsonify(body)
        data = _compute_sentiment(spec, cfg, s, client)
        with _sentiment_lock:
            _sentiment_cache[key] = {"ts": time.time(), "data": data}
        return jsonify({**data, "cached": False})
    finally:
        with _sentiment_lock:
            _sentiment_busy.discard(key)


@app.route("/api/brokers", methods=["GET"])
def get_brokers():
    """Configured brokers for the switcher: enabled state, market, which
    settings each overrides, the adapter's capabilities, and what that
    broker's agent last reported (health + market phase) from its own DB."""
    raw = load_raw_config()
    try:
        specs = broker_specs(raw)
    except ValueError as exc:
        raise BadRequest(f"config.yaml brokers section: {exc}") from None
    caps = broker_registry.capabilities()
    out = []
    for spec in specs:
        health = phase = None
        if spec.db_path.exists():
            try:
                conn = sqlite3.connect(str(spec.db_path))
                for key in (BROKER_HEALTH_KEY, PHASE_KEY):
                    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
                    if row:
                        if key == BROKER_HEALTH_KEY:
                            health = json.loads(row[0])
                        else:
                            phase = json.loads(row[0])
                conn.close()
            except (sqlite3.Error, ValueError):
                pass
        out.append({**spec.to_dict(), "market": spec.market or _spec_market(spec),
                    "capabilities": caps.get(spec.adapter, {}), "agent_health": health, "agent_phase": phase})
    return jsonify({"brokers": out, "current": current_spec(raw).name,
                    "multi": sum(1 for s in specs if s.enabled) > 1})


def _get_path(cfg: dict, key: str):
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


from src.risk import open_risk_at_stops  # noqa: E402


@app.route("/api/risk_state", methods=["GET"])
def get_risk_state():
    """Today's live risk state for the Risk Model screen: the agent's
    persisted daily P&L / halt flag / cooldowns (read-only - the agent owns
    them), its open + pending positions, and optionally a live reference
    price for drawing the trade model."""
    cfg = load_config()
    risk_cfg = cfg.get("risk", {}) or {}
    state, active = {}, []
    try:
        conn = get_db()
        row = conn.execute("SELECT value FROM kv WHERE key = 'risk_state'").fetchone()
        state = json.loads(row[0]) if row else {}
        active = [dict(r) for r in conn.execute(
            "SELECT symbol, direction, status, quantity, entry_price, current_stop FROM positions "
            "WHERE status IN ('pending_entry', 'open')")]
        conn.close()
    except (sqlite3.Error, ValueError):
        pass

    # A stale day's numbers are reset by the agent at its next check;
    # show them as zero so the gauge doesn't claim yesterday's loss.
    today = None
    try:
        today = datetime.now(market_timezone(_spec_market(current_spec()))).date().isoformat()
    except Exception:
        pass
    stale = bool(today and state.get("trading_day") and state["trading_day"] != today)
    cooldown_min = float(risk_cfg.get("min_minutes_between_trades") or 0)
    now = time.time()
    cooldowns = []
    for sym, ts in (state.get("last_trade_time") or {}).items():
        left = cooldown_min - (now - float(ts)) / 60
        if left > 0:
            cooldowns.append({"symbol": sym, "minutes_left": round(left, 1)})
    cooldowns.sort(key=lambda c: c["minutes_left"])

    reference = None
    symbol = request.args.get("symbol")
    if symbol:
        try:
            client = get_broker()
            if client is not None:
                reference = _live_prices(client, [symbol]).get(base_symbol(symbol))
        except Exception:
            logger.warning("Risk model: no reference price for %s", symbol, exc_info=True)

    return jsonify({
        "daily_pnl": 0.0 if stale else float(state.get("daily_pnl") or 0.0),
        "halted": False if stale else bool(state.get("trading_halted")),
        "trading_day": today or state.get("trading_day"),
        "open_positions": len(active),
        "open_risk": round(open_risk_at_stops(active), 2),
        "trades_today": 0 if stale else int(state.get("trades_today") or 0),
        "positions": active,
        "cooldowns": cooldowns,
        "reference_price": reference,
        "symbol": symbol,
    })


# ---------------------------------------------------------------------------
# Command Center: one summary of today, and a live stream of it
# ---------------------------------------------------------------------------

CLOSE_STAGES = {"entry_cancelled": "Unfilled entry cancelled", "flat_by_cancel_entry": "Entry cancelled (flat by close)",
                "position_closed_externally": "Position closed outside the agent"}


def _decision_stage(row: dict) -> str:
    return analytics.decision_stage(row)


def _overview_data() -> dict:
    """Today at a glance for the Command Center, from the agent's own
    records (audit DB) plus the cached account summary. Cheap: no broker
    calls beyond the cached account figures."""
    cfg = load_config()
    risk_cfg = cfg.get("risk", {}) or {}
    spec = current_spec()
    market = _spec_market(spec)
    tz = market_timezone(market)
    now = time.time()
    day_start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    out = {"generated_at": now, "market": market}

    # session clock
    try:
        code = cal.exchange_for_market(market)
        mins = cal.session_minutes(code)
        out["session"] = {
            "exchange": code, "tz": cal.EXCHANGES[code].tz, "open": mins is not None,
            "minutes_since_open": round(mins[0], 1) if mins else None,
            "minutes_to_close": round(mins[1], 1) if mins else None,
            "length": round(mins[0] + mins[1], 1) if mins else None,
            "next_open": cal.next_open(code).isoformat() if mins is None else None,
            "no_entries_first": risk_cfg.get("no_entries_first_minutes"),
            "no_entries_last": risk_cfg.get("no_entries_last_minutes"),
            "flat_by": risk_cfg.get("flat_by_minutes_before_close"),
        }
    except Exception:
        logger.warning("Session clock unavailable", exc_info=True)
        out["session"] = None

    pipeline = {"scanned": 0, "patterns": 0, "indicators": 0, "kronos": 0, "risk": 0, "opened": 0, "errors": 0}
    reasons, activity, trades = {}, [], []
    state, active, manual = {}, [], []
    try:
        conn = get_db()
        pipeline["scanned"] = conn.execute(
            "SELECT COUNT(*) FROM snapshots WHERE stage = 'scanning' AND ts >= ?", (day_start,)).fetchone()[0]
        for r in conn.execute("SELECT * FROM decisions WHERE ts >= ? ORDER BY id", (day_start,)):
            d = dict(r)
            if d["pattern"] == "manual":
                continue
            pipeline["patterns"] += 1
            stage = _decision_stage(d)
            if stage == "opened":
                pipeline["opened"] += 1
            elif stage == "error":
                pipeline["errors"] += 1
            else:
                pipeline[stage] += 1
                key = (stage, (d["risk_reason"] or "").split(" (")[0][:80])
                reasons[key] = reasons.get(key, 0) + 1
        for r in conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT 15"):
            d = dict(r)
            stage = _decision_stage(d)
            activity.append({"ts": d["ts"], "symbol": d["epic"], "kind": "manual" if d["pattern"] == "manual" else stage,
                             "text": _decision_text(d, stage)})
        for r in conn.execute("SELECT * FROM trade_results ORDER BY id DESC LIMIT 15"):
            d = dict(r)
            reason = ""
            try:
                reason = (json.loads(d["details"] or "{}") or {}).get("reason", "")
            except ValueError:
                pass
            pnl = d["pnl"] or 0.0
            activity.append({"ts": d["ts"], "symbol": d["epic"], "kind": "win" if pnl >= 0 else "loss",
                             "text": f"Closed {d['epic']}{' (' + reason.replace('_', ' ') + ')' if reason else ''}",
                             "pnl": round(pnl, 2)})
        for r in conn.execute("SELECT ts, symbol, stage FROM snapshots WHERE stage IN ({}) ORDER BY id DESC LIMIT 10"
                              .format(",".join("?" * len(CLOSE_STAGES))), tuple(CLOSE_STAGES)):
            activity.append({"ts": r["ts"], "symbol": r["symbol"], "kind": "cancel",
                             "text": f"{CLOSE_STAGES[r['stage']]}: {r['symbol']}"})
        for r in conn.execute("SELECT * FROM manual_orders ORDER BY id DESC LIMIT 5"):
            d = dict(r)
            activity.append({"ts": d["done_ts"] or d["ts"], "symbol": d["symbol"], "kind": "manual",
                             "text": f"Manual {d['action']} {d['symbol']}: {d['status']}"
                                     f"{' - ' + d['result'] if d['result'] else ''}"})
        trades = [dict(r) for r in conn.execute(
            "SELECT ts, epic, pnl, details FROM trade_results WHERE ts >= ? ORDER BY ts", (now - 30 * 86400,))]
        row = conn.execute("SELECT value FROM kv WHERE key = 'risk_state'").fetchone()
        state = json.loads(row[0]) if row else {}
        active = [dict(r) for r in conn.execute(
            "SELECT symbol, direction, status, quantity, entry_price, current_stop FROM positions "
            "WHERE status IN ('pending_entry', 'open')")]
        conn.close()
    except (sqlite3.Error, ValueError):
        logger.warning("Overview: audit DB unavailable", exc_info=True)

    out["pipeline"] = pipeline
    out["pipeline"]["watching"] = len(_kv(ACTIVE_SYMBOLS_KEY) or [])
    out["top_reasons"] = [{"stage": k[0], "reason": k[1], "count": v}
                          for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:4]]
    activity.sort(key=lambda a: a["ts"] or 0, reverse=True)
    out["activity"] = activity[:20]

    # results
    pnls = [t["pnl"] or 0.0 for t in trades]
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p <= 0]
    curve, run = [], 0.0
    for t in trades:
        run += t["pnl"] or 0.0
        curve.append(round(run, 2))
    today = [t["pnl"] or 0.0 for t in trades if t["ts"] >= day_start]
    today_list = []
    for t in trades:
        if t["ts"] < day_start:
            continue
        try:
            det = json.loads(t["details"]) if t.get("details") else {}
        except (TypeError, ValueError):
            det = {}
        reason = det.get("reason") if isinstance(det, dict) else None
        today_list.append({"ts": t["ts"], "symbol": t["epic"], "pnl": round(t["pnl"] or 0.0, 2),
                           "reason": str(reason).replace("_", " ") if reason else None})
    out["results"] = {
        "realized_today": round(sum(today), 2), "closed_today": len(today),
        "closed_today_list": today_list[::-1],
        "pnl_30d": round(sum(pnls), 2), "trades_30d": len(pnls), "wins_30d": len(wins), "losses_30d": len(losses),
        "win_rate_30d": round(len(wins) / len(pnls) * 100, 1) if pnls else None,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if wins and losses and sum(losses) else None,
        "curve_30d": curve[-60:],
    }

    stale = bool(state.get("trading_day") and state["trading_day"] != datetime.now(tz).date().isoformat())
    out["risk"] = {
        "trades_today": 0 if stale else int(state.get("trades_today") or 0),
        "max_trades_per_day": risk_cfg.get("max_trades_per_day"),
        "open_positions": len(active), "max_open_positions": risk_cfg.get("max_open_positions"),
        "open_risk": round(open_risk_at_stops(active), 2),
        "max_daily_loss": risk_cfg.get("max_daily_loss"),
        "halted": False if stale else bool(state.get("trading_halted")),
    }

    account = None
    try:
        client = get_broker()
        if client is not None:
            account = _account_summary(client)
    except Exception:
        logger.warning("Overview: account summary unavailable", exc_info=True)
    out["account"] = account
    out["agent"] = {"phase": _kv(PHASE_KEY), "broker_health": _kv(BROKER_HEALTH_KEY)}
    return out


def _decision_text(d: dict, stage: str) -> str:
    who = f"{d['epic']} {d['pattern'] or ''} {d['pattern_direction'] or ''}".strip()
    if d["pattern"] == "manual":
        return f"Manual {d['pattern_direction']} {d['epic']} submitted"
    if stage == "opened":
        conf = f" · Kronos {d['kronos_confidence'] * 100:.0f}%" if d["kronos_confidence"] is not None else ""
        return f"Entered {who}{conf}"
    if stage == "error":
        return f"Order error on {who}"
    label = {"indicators": "Indicators", "kronos": "Kronos", "risk": "Risk"}[stage]
    return f"{label} rejected {who}{': ' + d['risk_reason'] if d['risk_reason'] else ''}"


@app.route("/api/trade_marks", methods=["GET"])
def get_trade_marks():
    """The agent's trades on one symbol for the chart: entries (agent and
    manual), exits with P&L, and the live position's entry / stop / target.
    `since` (epoch seconds) limits it to what the chart shows."""
    symbol = request.args.get("symbol", "")
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    try:
        since = float(request.args.get("since") or 0)
    except ValueError:
        since = 0.0
    key = base_symbol(symbol)
    entries, exits, position = [], [], None
    try:
        conn = get_db()
        for r in conn.execute("SELECT * FROM decisions WHERE action = 'opened' AND ts >= ? ORDER BY ts", (since,)):
            d = dict(r)
            if base_symbol(d["epic"]) != key:
                continue
            det = _json_or_empty(d["details"])
            entries.append({"ts": d["ts"], "direction": d["pattern_direction"], "manual": d["pattern"] == "manual",
                            "pattern": d["pattern"], "price": det.get("entry"), "quantity": det.get("quantity"),
                            "confidence": d["kronos_confidence"]})
        for r in conn.execute("SELECT * FROM trade_results WHERE ts >= ? ORDER BY ts", (since,)):
            d = dict(r)
            if base_symbol(d["epic"]) != key:
                continue
            det = _json_or_empty(d["details"])
            exits.append({"ts": d["ts"], "pnl": d["pnl"], "price": det.get("exit"),
                          "reason": det.get("reason"), "quantity": det.get("quantity")})
        for r in conn.execute("SELECT * FROM positions WHERE status IN ('pending_entry', 'open')"):
            d = dict(r)
            if base_symbol(d["symbol"]) == key:
                position = {k: d[k] for k in ("direction", "quantity", "entry_price", "initial_stop", "current_stop",
                                              "target", "high_water", "trail_active", "status", "opened_ts")}
                position["trail_active"] = bool(position["trail_active"])
        conn.close()
    except sqlite3.Error:
        logger.warning("Trade marks: audit DB unavailable", exc_info=True)
    return jsonify({"symbol": symbol, "entries": entries, "exits": exits, "position": position})


def _json_or_empty(text) -> dict:
    try:
        v = json.loads(text or "{}")
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


def _since_days() -> float:
    """?days=N -> epoch seconds N days ago (0 / all / missing-but-invalid = everything)."""
    raw = (request.args.get("days") or "30").strip().lower()
    if raw in ("0", "all"):
        return 0.0
    try:
        days = max(1, min(3650, int(raw)))
    except ValueError:
        raise BadRequest("days must be a whole number or 'all'")
    return time.time() - days * 86400


def _market_tz() -> str:
    try:
        return str(market_timezone(_spec_market(current_spec())))
    except Exception:
        return "UTC"


@app.route("/api/performance", methods=["GET"])
def get_performance():
    """Closed-trade analytics for the Performance tab (equity, drawdown,
    calendar, breakdowns, R distribution, Kronos confidence vs results)."""
    since = _since_days()
    try:
        conn = get_db()
        data = analytics.performance(conn, since, _market_tz())
        conn.close()
    except sqlite3.Error as exc:
        return jsonify({"error": f"audit database unavailable: {exc}"}), 500
    data["since"] = since
    data["currency"] = load_config().get("currency")
    return jsonify(data)


@app.route("/api/decision_stats", methods=["GET"])
def get_decision_stats():
    """Signal funnel and rejection analytics for the Decisions tab."""
    since = _since_days()
    try:
        conn = get_db()
        data = analytics.decision_stats(conn, since, _market_tz())
        conn.close()
    except sqlite3.Error as exc:
        return jsonify({"error": f"audit database unavailable: {exc}"}), 500
    data["since"] = since
    return jsonify(data)


@app.route("/api/overview", methods=["GET"])
def get_overview():
    return jsonify(_overview_data())


STREAM_INTERVAL_SECONDS = 2.0
STREAM_LIFETIME_SECONDS = 300      # the browser reconnects on its own


@app.route("/api/stream", methods=["GET"])
def stream():
    """Server-sent events: pushes `overview` and `positions` as soon as they
    change (checked every couple of seconds), instead of the page polling.
    Ends after a few minutes; EventSource reconnects by itself."""
    once = request.args.get("once") == "1"           # tests / debugging

    def payloads():
        yield "overview", _overview_data()
        try:
            yield "positions", get_positions().get_json()
        except Exception:
            logger.warning("Stream: positions unavailable", exc_info=True)

    @stream_with_context
    def gen():
        last, started, last_beat = {}, time.time(), time.time()
        yield "retry: 3000\n\n"
        while True:
            for name, data in payloads():
                body = json.dumps(data, default=str, separators=(",", ":"))
                if last.get(name) != body:
                    last[name] = body
                    yield f"event: {name}\ndata: {body}\n\n"
            if once:
                return
            if time.time() - last_beat > 15:
                last_beat = time.time()
                yield ": ping\n\n"
            if time.time() - started > STREAM_LIFETIME_SECONDS:
                return
            time.sleep(STREAM_INTERVAL_SECONDS)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/settings", methods=["GET", "POST"])
def handle_settings():
    """Schema-driven settings form. GET: the schema plus, for the shared
    scope and every broker, the effective value of each field and whether it
    is a per-broker override. POST {scope, changes: {key: value}, reset: [key]}
    writes config.yaml (comments kept, backup made). Changes apply when the
    agent is restarted."""
    raw = load_raw_config()
    try:
        specs = broker_specs(raw)
    except ValueError as exc:
        raise BadRequest(f"config.yaml brokers section: {exc}") from None

    if request.method == "GET":
        shared = {k: v for k, v in raw.items() if k != "brokers"}
        markets = sorted({m for s in specs for m in broker_registry.capabilities()
                          .get(s.adapter, {}).get("markets", [])})
        fields = [f["key"] for sec in settings_schema() for f in sec["fields"]]
        values = {"shared": {k: _get_path(shared, k)[0] for k in fields}}
        overrides = {}
        broker_meta = {}
        for spec in specs:
            entry = (raw.get("brokers") or {}).get(spec.name) or {}
            vals = {k: _get_path(spec.config, k)[0] for k in fields}
            vals.update(enabled=spec.enabled, market=spec.market, label=(entry.get("label") or None),
                        data_source=(str(entry["data_source"]).lower() if entry.get("data_source") else None))
            values[spec.name] = vals
            overrides[spec.name] = sorted(k for k in fields if k not in ("enabled", "market", "label", "data_source")
                                          and _get_path(entry, k)[1])
            caps = broker_registry.capabilities().get(spec.adapter, {})
            market = _spec_market(spec)
            broker_meta[spec.name] = {"adapter": spec.adapter, "label": spec.label, "enabled": spec.enabled,
                                      "market": market, "currency": MARKET_CURRENCY.get(market),
                                      "markets": caps.get("markets", []), "status": caps.get("status")}
        return jsonify({"schema": settings_schema(markets), "values": values, "overrides": overrides,
                        "brokers": broker_meta, "broker_order": [s.name for s in specs],
                        "current": current_spec(raw).name})

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequest("expected a JSON object")
    scope = str(data.get("scope") or "shared")
    changes = data.get("changes") or {}
    resets = data.get("reset") or []
    if not isinstance(changes, dict) or not isinstance(resets, list):
        raise BadRequest("changes must be an object and reset a list")
    if not changes and not resets:
        return jsonify({"success": True, "message": "No changes"})
    try:
        text = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else ""
        new_text, applied = config_writer.apply_changes(text, scope, changes, resets)
        config_writer.write_config(CONFIG_PATH, new_text)
    except config_writer.SettingsError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Settings update failed")
        return jsonify({"error": str(exc)}), 500
    logger.info("Dashboard settings (%s): set %s reset %s", scope, applied, resets)
    return jsonify({"success": True, "applied": applied, "reset": resets,
                    "message": "Saved to config.yaml (backup kept). Restart the agent (main.py) to apply."})


def main():
    parser = argparse.ArgumentParser(description="moomoo Swing Agent Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: this machine only). Prefer an SSH "
                             "tunnel over binding 0.0.0.0 for remote access.")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    args = parser.parse_args()

    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host == "localhost"
    if not loopback:
        logger.warning(
            "Dashboard is reachable from the network on %s over plain HTTP - the login "
            "token travels unencrypted. Prefer: ssh -L %d:127.0.0.1:%d <this-host>",
            args.host, args.port, args.port,
        )

    display_host = "localhost" if loopback else args.host
    print(f"\n  Dashboard login link (keep it private):\n"
          f"  http://{display_host}:{args.port}/?token={DASHBOARD_TOKEN}\n", flush=True)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
