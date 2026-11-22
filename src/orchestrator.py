"""
Per-symbol worker loop. On each polling interval, pulls the latest
closed bars for a symbol, runs them through the pattern -> indicators ->
Kronos -> risk -> execution pipeline, and reports/logs the outcome at every stage.

Order lifecycle (each step persisted in audit_log.positions):
  1. poll_once submits a DAY limit/market entry -> status 'pending_entry'
  2. monitor_position sees the entry fill -> places the resting stop and
     target for the FILLED quantity -> status 'open'
  3. monitor_position sees a stop/target fill (or the software stop or
     target trigger) -> cancels the other order, closes any remainder,
     records realised P&L -> status 'closed'

Exit orders are GTC on live accounts. moomoo paper accounts reject GTC,
so there they are DAY orders: they expire at the close, are left to the
software monitor overnight, and are re-placed at the next session open.

The Orchestrator follows the exchange calendar (holidays and early closes
included): it scans only while the market is open, warms up shortly
before the open (reconnect check + fresh screener run) and otherwise
sleeps, checking positions at a slow cadence. It also watches the OpenD
connection (bounded connect, health ping, automatic reconnect) and
reports broker positions the agent doesn't manage.
"""
import logging
import os
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

from src import market_calendar as cal
from src.audit_log import AuditLog
from src import indicator_gate
from src import rules as rules_gate
from src import ig_hours
from src import skill_strategy
from src import trading_mode
from src import version as code_version
from src.kronos_agent import DEFAULT_OPTIONS, KronosAgent, is_default, kronos_options
from src.market import BAR_DURATION, drop_unclosed_bar, is_market_open, market_timezone
from src.patterns import scan
from src.kronos_indicators.indicators import atr as atr_series
from src.risk import RiskConfig, RiskManager, open_risk_at_stops
from src.telegram_bot import TelegramReporter, esc
from src.brokers import base_symbol, connect_with_timeout as connect_broker
from src.moomoo_client import MoomooClient, connect_with_timeout
from src.trailing_stop import TrailConfig, compute_update

logger = logging.getLogger(__name__)

ACTIVE_SYMBOLS_KEY = "active_symbols"
SCREENER_REPORT_KEY = "screener_report"
PHASE_KEY = "agent_phase"            # published for the dashboard
BROKER_HEALTH_KEY = "broker_health"
UNMANAGED_KEY = "unmanaged"
UNMANAGED_ALERTED_KEY = "unmanaged_alerted"
MISSING_SINCE_KEY = "missing_at_broker_since"
RECENT_CLOSES_KEY = "recent_closes"

# A symbol the agent just closed can still show at moomoo for a moment -
# don't call it "unmanaged" during this window.
RECENT_CLOSE_GRACE_SECONDS = 180
# An agent 'open' position that the broker no longer holds for this long
# (and whose exit orders didn't fill) was closed outside the agent.
MISSING_CONFIRM_SECONDS = 240
# Minimum gap between attempts to re-arm a stop/target leg the broker
# rejected, so a persistent rejection doesn't spam orders.
REARM_RETRY_SECONDS = 15 * 60
# A cancel is asynchronous at the broker (moomoo reports CANCELLING for a
# moment). Wait this long for it to settle before deferring a close.
CANCEL_SETTLE_SECONDS = 3.0
CANCEL_SETTLE_STEP = 0.5
# A close the agent started (stop, target, flat-by-close) and had to defer
# because a cancel was still pending. While this is set the cancelled exit
# orders are NOT re-placed - re-placing them is what blocked the close.
CLOSING_KEY = "closing:{symbol}"
CLOSING_MAX_AGE = 30 * 60

# Used when trailing is disabled: never engages, so compute_update only
# reports fixed stop / target crossings.
NO_TRAIL = TrailConfig(trigger_points=float("inf"), trail_distance=0.0)
# A trade opened by skill-file trading (trading_mode "skill"): its setup,
# so it keeps the skill's exits even if the switch is flipped back.
IG_STREAM_KEY = "ig_stream"          # around-the-clock IG: stream / allowance status for the dashboard
KRONOS_TIMING_KEY = "kronos_timing"      # how long Kronos takes on this computer (dashboard)
SKILL_TRADE_KEY = "skill_trade:{symbol}"
SKILL_DONE_KEY = "skill_done:{symbol}"
_skill_cache: dict = {}


def load_skill(rel_path: str):
    """(settings | None, problems) of a skill file, re-read when it changes."""
    try:
        path = trading_mode.resolve_file(rel_path)
        mtime = path.stat().st_mtime
    except (ValueError, OSError) as exc:
        return None, [str(exc)]
    hit = _skill_cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1], hit[2]
    s, problems = skill_strategy.load(path)
    _skill_cache[str(path)] = (mtime, s, problems)
    return s, problems


def bars_to_df(bars_response) -> pd.DataFrame:
    """Convert MoomooClient.get_historical_bars output (time = epoch ms)
    into an OHLC(V) DataFrame indexed by timestamp, oldest first."""
    df = bars_response.copy()
    df = df.rename(columns={
        "time": "timestamp", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def _opposite(direction: str) -> str:
    return "SELL" if direction == "BUY" else "BUY"


def note_kronos_time(audit_log, ms: float, symbols: int):
    """Keep a small running record of how long Kronos takes on this
    computer, for the dashboard (Kronos Model Lab)."""
    try:
        rec = audit_log.get_kv(KRONOS_TIMING_KEY) or {}
        day = time.strftime("%Y-%m-%d")
        if rec.get("day") != day:
            rec = {"day": day, "calls": 0, "symbols": 0, "total_ms": 0.0, "max_ms": 0.0}
        rec["calls"] += 1
        rec["symbols"] += symbols
        rec["total_ms"] += ms
        rec["max_ms"] = max(rec["max_ms"], ms)
        rec.update(last_ms=round(ms), last_symbols=symbols, last_ts=time.time(),
                   avg_ms_per_symbol=round(rec["total_ms"] / max(1, rec["symbols"])))
        audit_log.set_kv(KRONOS_TIMING_KEY, rec)
    except Exception:  # noqa: BLE001
        logger.debug("Could not record Kronos timing", exc_info=True)


class SymbolWorker:
    def __init__(
        self,
        symbol_cfg: dict,
        global_cfg: dict,
        broker: MoomooClient,
        kronos_agent: KronosAgent,
        risk_manager: RiskManager,
        audit_log: AuditLog,
        reporter: TelegramReporter,
    ):
        self.symbol = symbol_cfg["symbol"]
        self.name = symbol_cfg.get("name", symbol_cfg["symbol"])
        self.quantity = symbol_cfg.get("quantity", global_cfg.get("default_quantity", 10))
        self.timeframe = global_cfg["timeframe"]
        self.lookback_bars = global_cfg["lookback_bars"]
        self.forecast_horizon = global_cfg["forecast_horizon"]
        self.confidence_threshold = global_cfg["kronos_confidence_threshold"]
        self.kronos_opts = kronos_options(global_cfg)        # kronos_scalp: (defaults = original rule)
        self.global_cfg = global_cfg
        self.order_type = global_cfg.get("order_type", "LIMIT").upper()
        # Cancel an entry still completely unfilled after this many minutes
        # (None = leave it working until the DAY order expires). On short
        # timeframes a stale limit otherwise blocks the symbol all session.
        self.entry_timeout_minutes = global_cfg.get("entry_timeout_minutes")
        self.trade_outside_hours = global_cfg.get("trade_outside_market_hours", False)
        # True for a symbol kept only because it still holds a position
        # (e.g. dropped by the screener): monitored, never scanned.
        self.monitor_only = bool(symbol_cfg.get("monitor_only", False))

        self.broker = broker
        self.kronos_agent = kronos_agent
        self.risk_manager = risk_manager
        self.audit_log = audit_log
        self.reporter = reporter

        trail_cfg = global_cfg.get("trailing_stop", {})
        self.trail_config = TrailConfig(
            trigger_points=trail_cfg.get("trigger_points", 0.0),
            trail_distance=trail_cfg.get("trail_distance", 0.0),
            use_atr=trail_cfg.get("use_atr", False),
            atr_multiplier=trail_cfg.get("atr_multiplier", 3.0),
            trigger_pct=trail_cfg.get("trigger_pct"),
            trail_pct=trail_cfg.get("trail_pct"),
        )
        self._trail_atr = None   # ATR of the latest closed candles, for use_atr (refreshed by poll_once)
        self.trail_enabled = trail_cfg.get("enabled", False)
        # Scalping: a time stop and a break-even stop on every agent trade.
        scalp = global_cfg.get("scalp") or {}
        self.max_hold_minutes = scalp.get("max_hold_minutes")
        self.breakeven_at_r = scalp.get("breakeven_at_r")
        self.breakeven_offset_pct = float(scalp.get("breakeven_offset_pct") or 0.0)

        # Indicator confirmation between pattern and Kronos (off unless
        # config has `indicators: {enabled: true}`).
        self.indicator_cfg = indicator_gate.IndicatorConfig.from_dict(global_cfg.get("indicators"))
        # Your own rules from config/skills.md, after the indicators (off
        # unless `skills: {enabled: true}`); they can only stop a signal.
        self.rules_cfg = rules_gate.settings(global_cfg)
        self.rulebook = rules_gate.RuleBook(global_cfg) if self.rules_cfg["enabled"] else None

        # Persisted so a restart doesn't re-evaluate (and re-trade) a bar
        # that was already processed.
        self._last_bar_key = f"last_bar:{self.symbol}"
        saved = audit_log.get_kv(self._last_bar_key)
        self._last_seen_timestamp = pd.Timestamp(saved) if saved else None

    @property
    def market(self) -> str:
        return self.broker.market

    def _market_open(self) -> bool:
        per_symbol = getattr(self.broker, "symbol_open", None)
        if self.market == ig_hours.MARKET and per_symbol is not None:
            return bool(per_symbol(self.symbol))       # this market's own hours + IG's live status
        return is_market_open(self.market)

    # ------------------------------------------------------------------
    # Entry pipeline
    # ------------------------------------------------------------------

    def _exits_persist(self) -> bool:
        """True if resting exits survive the session close (GTC)."""
        return getattr(self.broker, "exit_time_in_force", "GTC") == "GTC"

    @property
    def bar_duration(self):
        """One candle's length, so Kronos's forecast timestamps are right even
        on the first candle after a gap (the open, a weekend, IG's break)."""
        return BAR_DURATION.get(self.timeframe)

    def poll_once(self):
        """Check the latest closed candle and act on it. The orchestrator
        normally calls prepare_signal / finish_signal itself so it can ask
        Kronos about every symbol that signalled on the same candle at once;
        this does the same for one symbol."""
        ctx = self.prepare_signal()
        if ctx is None:
            return
        result, ctx["kronos_extra"], ms = self.ask_kronos(ctx["df"], ctx["signal"].direction)
        self.finish_signal(ctx, result, ms)

    def kronos_extra_base(self) -> dict:
        """What to record with a decision about how Kronos's answer was read
        (only when the kronos_scalp options differ from the original rule)."""
        o = self.kronos_opts if isinstance(getattr(self, "kronos_opts", None), dict) else DEFAULT_OPTIONS
        if is_default(o):
            return {}
        return {"kronos_method": o["method"], **({"paths": o["paths"]} if o["method"] == "paths" else {}),
                **({"min_move_pct": o["min_move_pct"]} if o["min_move_pct"] else {})}

    def ask_kronos(self, df: pd.DataFrame, direction: str):
        """Ask Kronos about one signal, honouring the kronos_scalp options.
        Returns (result, extra details to record, milliseconds); result is
        (confirmed, confidence) or the exception Kronos raised."""
        o = self.kronos_opts if isinstance(getattr(self, "kronos_opts", None), dict) else DEFAULT_OPTIONS
        t0 = time.perf_counter()
        try:
            if is_default(o):
                result = self.kronos_agent.confirm_direction(
                    df, direction, self.forecast_horizon, bar_duration=self.bar_duration)
            else:
                result = self.kronos_agent.confirm_many(
                    [(df, direction)], self.forecast_horizon, bar_duration=self.bar_duration,
                    method=o["method"], paths=o["paths"], min_move_pct=o["min_move_pct"])[0]
        except Exception as exc:  # noqa: BLE001 - reported by the caller
            result = exc
        ms = (time.perf_counter() - t0) * 1000.0
        extra = self.kronos_extra_base()
        if not isinstance(result, BaseException):
            note_kronos_time(self.audit_log, ms, 1)
            if o["shadow_paths"] and o["method"] == "bars":
                try:
                    sp = self.kronos_agent.confirm_many(
                        [(df, direction)], self.forecast_horizon, bar_duration=self.bar_duration,
                        method="paths", paths=o["paths"], min_move_pct=o["min_move_pct"])[0]
                    if isinstance(sp, tuple):
                        extra["paths_confidence"] = round(float(sp[1]), 3)
                except Exception:  # noqa: BLE001 - recorded only, never affects the trade
                    logger.debug("Shadow paths confidence failed for %s", self.symbol, exc_info=True)
        return result, extra, ms

    def prepare_signal(self):
        """Everything before Kronos: fetch candles, find a pattern, run the
        indicator and rules gates. Returns the context Kronos needs, or None
        when there is nothing to ask it (no new candle, no pattern, rejected
        earlier, skill-file mode, ...)."""
        if self.monitor_only:
            return
        if not self.trade_outside_hours and not self._market_open():
            # Leave the latest closed bar unprocessed - it gets evaluated
            # (and any entry placed) when the session opens.
            return

        bars = self.broker.get_historical_bars(
            self.symbol, self.timeframe, self.lookback_bars
        )
        df = drop_unclosed_bar(bars_to_df(bars), self.timeframe)
        if df.empty:
            return
        if self.trail_config.use_atr:
            try:
                a = float(atr_series(df, 14).iloc[-1])
                self._trail_atr = a if a == a and a > 0 else self._trail_atr
            except Exception:  # noqa: BLE001
                pass

        latest_ts = df.index[-1]
        if self._last_seen_timestamp is not None and latest_ts <= self._last_seen_timestamp:
            return  # no new closed bar yet
        self._last_seen_timestamp = latest_ts
        self.audit_log.set_kv(self._last_bar_key, latest_ts.isoformat())

        last_close = df["close"].iloc[-1]
        self.audit_log.log_snapshot(self.symbol, float(last_close), "scanning")

        trade = self._skill_trade()
        if trade is not None:
            self._skill_manage(df, trade, float(last_close))
        if trading_mode.read()["mode"] in trading_mode.SKILL_MODES:
            self._skill_entry(df, float(last_close))   # Kronos is not called in this mode
            return

        signals = scan(df)
        if not signals:
            return

        signal = max(signals, key=lambda s: s.strength)
        logger.info("%s: pattern %s (%s, strength=%.2f)",
                     self.symbol, signal.name, signal.direction, signal.strength)
        self.audit_log.log_snapshot(
            self.symbol, float(last_close), "pattern_detected",
            {"pattern": signal.name, "direction": signal.direction, "strength": signal.strength},
        )

        ind_verdict = self._indicator_check(df, signal, float(last_close))
        if ind_verdict is not None and not ind_verdict.passed:
            return

        rules_verdict = self._rules_check(df, signal, float(last_close))
        if rules_verdict is not None and not rules_verdict.passed:
            return
        return {"df": df, "signal": signal, "last_close": float(last_close), "ind_verdict": ind_verdict}

    def finish_signal(self, ctx: dict, result, kronos_ms: float | None = None):
        """Everything after Kronos answered (or failed): threshold, risk,
        sizing and the entry order. `result` is (confirmed, confidence) or
        the exception Kronos raised."""
        df, signal, last_close, ind_verdict = ctx["df"], ctx["signal"], ctx["last_close"], ctx["ind_verdict"]
        if isinstance(result, BaseException):
            exc = result
            logger.error("Kronos confirmation failed for %s: %s", self.symbol, exc)
            self.audit_log.log_decision(
                self.symbol, signal.name, signal.direction, signal.strength,
                None, None, None, None, "error", details={"error": str(exc)},
            )
            self.reporter.report_error(f"kronos_agent ({self.symbol})", str(exc))
            return

        confirmed, confidence = result
        kx = ctx.get("kronos_extra") or {}
        self.audit_log.log_snapshot(
            self.symbol, float(last_close), "kronos_result",
            {"confirmed": confirmed, "confidence": confidence,
             **({"ms": round(kronos_ms)} if kronos_ms is not None else {})},
        )
        # Shadow mode (finetune.shadow_model): a second model sees the same
        # candles in the background. Its answer is only recorded.
        try:
            submit = getattr(self.kronos_agent, "shadow_submit", None)
            if callable(submit):
                submit(df, signal.direction, self.forecast_horizon, broker=self.global_cfg.get("_broker_name") or getattr(self.broker, "name", "broker"),
                       symbol=self.symbol, timeframe=self.timeframe, last_close=float(last_close),
                       threshold=self.confidence_threshold, main_confirmed=bool(confirmed),
                       main_confidence=float(confidence))
        except Exception:
            logger.debug("Shadow submit failed", exc_info=True)
        if not confirmed or confidence < self.confidence_threshold:
            reason = f"kronos confidence {confidence:.2f} below threshold {self.confidence_threshold}"
            self.audit_log.log_decision(
                self.symbol, signal.name, signal.direction, signal.strength,
                confirmed, confidence, None, reason, "rejected", details=kx or None,
            )
            self.reporter.report_rejected(self.symbol, signal.name, reason)
            return

        allowed, risk_reason = self._risk_check(signal.direction)
        self.audit_log.log_snapshot(self.symbol, float(last_close), "risk_check", {"allowed": allowed, "reason": risk_reason})
        if not allowed:
            self.audit_log.log_decision(
                self.symbol, signal.name, signal.direction, signal.strength,
                confirmed, confidence, False, risk_reason, "rejected", details=kx or None,
            )
            self.reporter.report_rejected(self.symbol, signal.name, f"risk: {risk_reason}")
            return

        plan = self._plan(float(last_close), self.quantity, df)
        self.audit_log.log_snapshot(self.symbol, float(last_close), "risk_plan", {
            "allowed": plan.allowed, "reason": plan.reason, "quantity": plan.quantity,
            "sized_by": plan.sized_by, "max_loss": round(plan.max_loss, 2),
            "reward_risk": round(plan.reward_risk, 2), **plan.details})
        if not plan.allowed:
            self.audit_log.log_decision(
                self.symbol, signal.name, signal.direction, signal.strength,
                confirmed, confidence, False, plan.reason, "rejected", details={"risk_plan": plan.details, **kx},
            )
            self.reporter.report_rejected(self.symbol, signal.name, f"risk: {plan.reason}")
            return
        quantity = plan.quantity
        stop_points, limit_points = plan.stop_distance, plan.target_distance
        round_price = self.broker.round_price
        stop_price = round_price(last_close - stop_points if signal.direction == "BUY"
                                 else last_close + stop_points)
        limit_price = round_price(last_close + limit_points if signal.direction == "BUY"
                                  else last_close - limit_points)

        # Market orders are rejected outside exchange sessions, so only use one
        # when the market is actually open.
        entry_limit_price = None
        if self.order_type == "LIMIT" or not self._market_open():
            entry_limit_price = round_price(float(last_close))

        try:
            entry_id = self.broker.create_entry(
                self.symbol, signal.direction, quantity, entry_limit_price,
            )
        except Exception as exc:
            logger.exception("Entry order failed for %s", self.symbol)
            self.audit_log.log_decision(
                self.symbol, signal.name, signal.direction, signal.strength,
                confirmed, confidence, True, risk_reason, "error",
                details={"error": str(exc), **kx},
            )
            self.reporter.report_error(f"entry order ({self.symbol})", str(exc))
            return

        # Record before anything else can fail, so the duplicate guard and
        # cooldown hold even if a later step (logging, Telegram) errors.
        self.audit_log.record_pending_entry(
            self.symbol, signal.direction, quantity,
            float(entry_limit_price or last_close), float(stop_price), float(limit_price),
            entry_order_id=str(entry_id),
        )
        self.risk_manager.mark_trade_opened(self.symbol)

        self.audit_log.log_decision(
            self.symbol, signal.name, signal.direction, signal.strength,
            confirmed, confidence, True, risk_reason, "opened",
            deal_id=str(entry_id),
            details={**kx, "status": "entry_submitted", "entry": float(entry_limit_price or last_close),
                     "stop": float(stop_price), "target": float(limit_price), "quantity": quantity,
                     "sized_by": plan.sized_by, "max_loss": round(plan.max_loss, 2),
                     "reward_risk": round(plan.reward_risk, 2),
                     **({"indicators": ind_verdict.summary} if ind_verdict and ind_verdict.summary else {})},
        )
        self.audit_log.log_snapshot(
            self.symbol, float(last_close), "entry_submitted",
            {"direction": signal.direction, "entry": float(entry_limit_price or last_close),
             "stop": float(stop_price), "target": float(limit_price)},
        )
        self.reporter.report_trade_opened(
            self.symbol, signal.direction, quantity,
            entry=entry_limit_price or last_close, stop=stop_price, limit=limit_price,
            pattern=signal.name, kronos_confidence=confidence,
            indicators=ind_verdict.short_text() if ind_verdict is not None else None,
        )

    def _indicator_check(self, df: pd.DataFrame, signal, last_close: float):
        """Run the indicator gate. Returns None when disabled, else the
        verdict (already logged; a failed verdict was already reported)."""
        if not self.indicator_cfg.enabled:
            return None
        try:
            tz = market_timezone(self.market)
        except Exception:
            tz = None
        try:
            verdict = indicator_gate.evaluate(df, signal.direction, self.indicator_cfg,
                                              timeframe=self.timeframe, tz=tz)
        except Exception as exc:
            # A bug in an indicator must not stop trading on patterns + Kronos.
            logger.exception("Indicator check failed for %s - passing signal through", self.symbol)
            self.audit_log.log_snapshot(self.symbol, last_close, "indicator_check",
                                        {"error": str(exc), "passed": True})
            return None

        self.audit_log.log_snapshot(self.symbol, last_close, "indicator_check", {
            "passed": verdict.passed, "alignment": verdict.alignment, "reasons": verdict.reasons,
            "skipped": verdict.skipped, "mode": self.indicator_cfg.mode, **verdict.summary,
        })
        if verdict.reasons and verdict.passed:  # log mode: record, don't block
            logger.info("%s: indicators disagree (log mode, not blocking): %s",
                        self.symbol, "; ".join(verdict.reasons))
        if not verdict.passed:
            reason = "indicators: " + "; ".join(verdict.reasons)
            logger.info("%s: %s", self.symbol, reason)
            self.audit_log.log_decision(
                self.symbol, signal.name, signal.direction, signal.strength,
                None, None, None, reason, "rejected", details={"indicators": verdict.summary},
            )
            self.reporter.report_rejected(self.symbol, signal.name, reason)
        return verdict

    def _rules_check(self, df: pd.DataFrame, signal, last_close: float):
        """Your rules (config/skills.md). Returns None when off, else the
        verdict (already logged; a stopped signal is already reported)."""
        if self.rulebook is None:
            return None
        try:
            tz = market_timezone(self.market)
        except Exception:
            tz = None
        try:
            verdict = self.rulebook.check(df, signal.direction, signal.name, signal.strength, self.timeframe, tz,
                                          self._session_minutes())
        except Exception as exc:
            # A broken rule must never stop trading on patterns + Kronos.
            logger.exception("Rules check failed for %s - passing signal through", self.symbol)
            self.audit_log.log_snapshot(self.symbol, last_close, "rules_check", {"error": str(exc), "passed": True})
            return None
        self.audit_log.log_snapshot(self.symbol, last_close, "rules_check",
                                    {"passed": verdict.passed, **verdict.record()})
        if verdict.hits and verdict.passed:      # log mode: record, don't stop
            logger.info("%s: your rules would stop this (log mode): %s", self.symbol, verdict.reason())
        if not verdict.passed:
            reason = verdict.reason()
            logger.info("%s: %s", self.symbol, reason)
            self.audit_log.log_decision(
                self.symbol, signal.name, signal.direction, signal.strength,
                None, None, None, reason, "rejected", details={"rules": verdict.record()},
            )
            self.reporter.report_rejected(self.symbol, signal.name, reason)
        return verdict

    def _risk_check(self, direction: str) -> tuple[bool, str]:
        # Count the agent's own pending + open positions: broker positions
        # alone miss entries that haven't filled yet, and would include
        # unrelated long-term holdings.
        active_count = len(self.audit_log.get_active_positions())
        has_position = self.audit_log.get_active_position(self.symbol) is not None
        if not has_position:
            try:
                has_position = self.broker.has_exposure(self.symbol)
            except Exception as exc:
                logger.exception("Could not query broker exposure for %s", self.symbol)
                return False, f"could not verify broker exposure: {exc}"
        if self.risk_manager.wants_start_equity():
            try:
                acct = self.broker.get_account_summary() or {}
                self.risk_manager.set_start_equity(float(acct.get("total_assets") or 0))
            except Exception:
                logger.warning("Account value unavailable for the daily profit target", exc_info=True)
        sized = bool(self.risk_manager.config.max_loss_per_trade)
        return self.risk_manager.can_trade(
            self.symbol, None if sized else self.quantity, active_count,
            direction=direction, has_position=has_position,
            session_minutes=self._session_minutes(),
        )

    def _session_minutes(self):
        """(minutes since open, minutes to close) of today's regular session."""
        try:
            if self.market == ig_hours.MARKET and hasattr(self.broker, "session_minutes_for"):
                return self.broker.session_minutes_for(self.symbol)
            return cal.session_minutes(cal.exchange_for_market(self.market))
        except Exception:
            logger.exception("Session clock unavailable for %s", self.symbol)
            return None

    def _atr(self, df: pd.DataFrame | None) -> float | None:
        cfg = self.risk_manager.config
        if not cfg.stop_atr_multiple:
            return None
        try:
            if df is None:
                df = drop_unclosed_bar(bars_to_df(self.broker.get_historical_bars(
                    self.symbol, self.timeframe, max(self.lookback_bars, cfg.atr_period + 5))), self.timeframe)
            value = float(atr_series(df, cfg.atr_period).iloc[-1])
            return value if value == value and value > 0 else None
        except Exception:
            logger.exception("ATR unavailable for %s", self.symbol)
            return None

    def _plan(self, price: float, quantity: int, df: pd.DataFrame | None = None, **fixed):
        """Size the trade and run the trade-level risk checks (risk.plan_entry)."""
        cfg = self.risk_manager.config
        account = None
        if cfg.max_position_pct or cfg.cash_buffer_pct:
            try:
                account = self.broker.get_account_summary()
            except Exception:
                logger.exception("Account summary unavailable for sizing %s", self.symbol)
        open_risk = open_risk_at_stops(self.audit_log.get_active_positions()) \
            if cfg.daily_loss_includes_open_risk else 0.0
        return self.risk_manager.plan_entry(price, quantity, atr=None if fixed else self._atr(df),
                                            account=account, open_risk=open_risk, **fixed)

    # ------------------------------------------------------------------
    # Skill-file trading (trading_mode "skill"): the skill file's engine
    # finds the setup, entry, stop, target and exits; Kronos is not called.
    # Risk limits still apply to every entry.
    # ------------------------------------------------------------------

    def _skill_trade(self) -> dict | None:
        trade = self.audit_log.get_kv(SKILL_TRADE_KEY.format(symbol=self.symbol), None)
        return trade if isinstance(trade, dict) else None

    def _clear_skill_trade(self):
        if self.audit_log.get_kv(SKILL_TRADE_KEY.format(symbol=self.symbol), None) is not None:
            self.audit_log.set_kv(SKILL_TRADE_KEY.format(symbol=self.symbol), None)

    def _trailing_on(self) -> bool:
        trade = self._skill_trade()
        if trade is not None:
            return self.trail_enabled and bool(trade.get("use_trailing_stop"))
        return self.trail_enabled

    def _skill_entry(self, df: pd.DataFrame, last_close: float):
        if self.audit_log.get_active_position(self.symbol) is not None:
            return
        switch = trading_mode.read()
        settings, problems = load_skill(switch["file"])
        if settings is None:
            self.audit_log.log_snapshot(self.symbol, last_close, "skill_error",
                                        {"file": switch["file"], "problems": problems})
            return
        setup = skill_strategy.find_setup(df, settings)
        self.audit_log.log_snapshot(self.symbol, last_close, "skill_scan", setup.summary())
        if not setup.ready:
            return
        key = SKILL_DONE_KEY.format(symbol=self.symbol)
        if self.audit_log.get_kv(key, None) == setup.details.get("breakout"):
            return                       # this breakout was already traded or tried
        name = "praga_setup" if settings.strategy == "praga_price_action" else \
            "scalp " + str(setup.details.get("setup") or settings.strategy)
        direction = setup.direction
        with_kronos = switch["mode"] == "skill_kronos"
        engine = "skill+kronos" if with_kronos else "skill"
        k_ok, k_conf, kx = None, None, {}

        def reject(reason, details=None, prefix="risk: "):
            self.audit_log.set_kv(key, setup.details.get("breakout"))
            self.audit_log.log_decision(self.symbol, name, direction, float(setup.quality), k_ok, k_conf, False,
                                        reason, "rejected",
                                        details={"engine": engine, "file": switch["file"], "setup": setup.summary(),
                                                 **kx, **(details or {})})
            self.reporter.report_rejected(self.symbol, name, f"{prefix}{reason}")

        if with_kronos:
            # The skill found the setup; Kronos must agree with its direction
            # before the entry goes in. It can only say no.
            result, kx, ms = self.ask_kronos(df, direction)
            if isinstance(result, BaseException):
                logger.error("Kronos check of skill setup failed for %s: %s", self.symbol, result)
                self.audit_log.log_decision(self.symbol, name, direction, float(setup.quality), None, None, None,
                                            None, "error", details={"engine": engine, "error": str(result), **kx})
                self.reporter.report_error(f"kronos_agent ({self.symbol})", str(result))
                return
            k_ok, k_conf = result
            self.audit_log.log_snapshot(self.symbol, last_close, "kronos_result",
                                        {"confirmed": k_ok, "confidence": k_conf, "ms": round(ms), "engine": engine})
            if not k_ok or k_conf < self.confidence_threshold:
                return reject(f"kronos confidence {k_conf:.2f} below threshold {self.confidence_threshold}",
                              prefix="")

        allowed, risk_reason = self._risk_check(direction)
        if not allowed:
            return reject(risk_reason)
        round_price = self.broker.round_price
        entry = round_price(setup.entry)
        stop, target = round_price(setup.stop), round_price(setup.target)
        plan = self._plan(entry, self.quantity, df, stop_distance=abs(entry - stop), target_distance=abs(target - entry))
        if not plan.allowed:
            return reject(plan.reason, {"risk_plan": plan.details})
        try:
            entry_id = self.broker.create_entry(self.symbol, direction, plan.quantity, entry)   # limit at the skill's price
        except Exception as exc:
            logger.exception("Skill entry order failed for %s", self.symbol)
            self.audit_log.log_decision(self.symbol, name, direction, float(setup.quality), k_ok, k_conf, True,
                                        risk_reason, "error", details={"engine": engine, "error": str(exc), **kx})
            self.reporter.report_error(f"entry order ({self.symbol})", str(exc))
            return
        self.audit_log.record_pending_entry(self.symbol, direction, plan.quantity, float(entry), float(stop),
                                            float(target), entry_order_id=str(entry_id))
        self.risk_manager.mark_trade_opened(self.symbol)
        self.audit_log.set_kv(key, setup.details.get("breakout"))
        self.audit_log.set_kv(SKILL_TRADE_KEY.format(symbol=self.symbol), {
            "file": switch["file"], "strategy": settings.strategy, "direction": direction,
            "entry": float(entry), "stop": float(stop),
            "target": float(target), "breakout": setup.details.get("breakout"),
            "max_wait_candles": settings.max_wait_candles, "use_trailing_stop": settings.use_trailing_stop,
            "quality": setup.quality, "opened_bar": None})
        self.audit_log.log_decision(
            self.symbol, name, direction, float(setup.quality), k_ok, k_conf, True, risk_reason, "opened",
            deal_id=str(entry_id),
            details={"engine": engine, **kx, "file": switch["file"], "status": "entry_submitted", "entry": float(entry),
                     "stop": float(stop), "target": float(target), "quantity": plan.quantity,
                     "sized_by": plan.sized_by, "max_loss": round(plan.max_loss, 2),
                     "reward_risk": round(plan.reward_risk, 2), "setup": setup.summary()})
        self.audit_log.log_snapshot(self.symbol, last_close, "entry_submitted",
                                    {"direction": direction, "entry": float(entry), "stop": float(stop),
                                     "target": float(target), "engine": engine})
        self.reporter.send(
            f"<b>Entry submitted ({'skill + Kronos' if with_kronos else 'skill'})</b> <code>{esc(self.symbol)}</code>\n"
            f"{direction} {plan.quantity} - limit {entry}  SL: {stop}  TP: {target}\n"
            f"Setup: {esc(name)} (quality {setup.quality})"
            + (f"\nKronos: {k_conf:.0%} agree" if with_kronos else ""))

    def _skill_manage(self, df: pd.DataFrame, trade: dict, last_close: float):
        """On each new closed candle: drop a stale retest order, or exit an
        open skill trade by its rules."""
        pos = self.audit_log.get_active_position(self.symbol)
        if pos is None:
            self._clear_skill_trade()
            return
        if pos["status"] == "pending_entry":
            reason = None
            sell = (trade.get("direction") or pos["direction"]) == "SELL"
            if (last_close > float(trade["stop"])) if sell else (last_close < float(trade["stop"])):
                reason = "a close went past the setup's stop"
            elif trade.get("breakout"):
                try:
                    since = int((df.index > pd.Timestamp(trade["breakout"])).sum())
                except Exception:  # noqa: BLE001
                    since = 0
                if since > int(trade.get("max_wait_candles") or 20):
                    reason = "no retest in time"
            if reason:
                logger.info("%s: cancelling the skill entry - %s", self.symbol, reason)
                self._cancel_quietly(pos["entry_order_id"])
                self.audit_log.log_snapshot(self.symbol, last_close, "skill_setup_cancelled", {"reason": reason})
            return
        settings, _ = load_skill(trade.get("file") or trading_mode.DEFAULT_FILE)
        if settings is None:
            return                       # the bracket still protects the trade
        sig = skill_strategy.exit_signal(df, settings, trade.get("opened_bar"), pos["direction"])
        if sig is None:
            return
        if sig.kind == "close_now":
            if not self._market_open():
                return
            try:
                price = float(self.broker.get_current_price(self.symbol))
            except Exception:  # noqa: BLE001
                price = last_close
            self._software_exit(pos, f"skill: {sig.reason}", price)
            return
        # exit_at: move the target order to the level the skill waits for
        level = self.broker.round_price(float(sig.price))
        if pos.get("target") is not None and abs(float(pos["target"]) - level) < 1e-9:
            return
        if pos.get("target_order_id"):
            self._cancel_quietly(pos["target_order_id"])
        new_id = None
        try:
            new_id = self.broker.place_target_order(self.symbol, _opposite(pos["direction"]),
                                                    int(pos["quantity"]), level)
        except Exception:
            logger.exception("Could not move the target for %s - watched in software", self.symbol)
        self.audit_log.set_target(self.symbol, float(level), str(new_id) if new_id else None)
        self.audit_log.log_snapshot(self.symbol, last_close, "skill_exit_at", {"price": level, "reason": sig.reason})
        self.reporter.send(f"<code>{esc(self.symbol)}</code>: {esc(sig.reason)} - exit set at {level}")

    # ------------------------------------------------------------------
    # Position monitor (fast cadence, independent of the bar scan)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Manual override (dashboard buttons). Skips the pattern / indicator /
    # Kronos / risk gates on purpose - the user is making the call - but
    # never stacks a second position on a symbol and, once filled, the
    # position gets the same bracket, trailing stop and exits as any other.
    # ------------------------------------------------------------------

    def manual_entry(self, direction: str, quantity: int) -> str:
        if self.audit_log.get_active_position(self.symbol) is not None:
            raise RuntimeError(f"the agent already has a position or pending entry in {self.symbol}")
        if self.broker.has_exposure(self.symbol):
            raise RuntimeError(f"the broker already shows a holding or working order in {self.symbol}")
        price = float(self.broker.get_current_price(self.symbol))
        round_price = self.broker.round_price
        stop_points, limit_points = self.risk_manager.bracket_distances(price, self._atr(None))
        sign = 1 if direction == "BUY" else -1
        stop_price = round_price(price - sign * stop_points)
        limit_price = round_price(price + sign * limit_points)
        entry_limit_price = None
        if self.order_type == "LIMIT" or not self._market_open():
            entry_limit_price = round_price(price)

        entry_id = self.broker.create_entry(self.symbol, direction, quantity, entry_limit_price)
        self.audit_log.record_pending_entry(
            self.symbol, direction, quantity, float(entry_limit_price or price),
            float(stop_price), float(limit_price), entry_order_id=str(entry_id),
        )
        self.risk_manager.mark_trade_opened(self.symbol)
        self.audit_log.log_decision(
            self.symbol, "manual", direction, None, None, None, None, "manual override", "opened",
            deal_id=str(entry_id),
            details={"status": "entry_submitted", "manual": True, "entry": float(entry_limit_price or price),
                     "stop": float(stop_price), "target": float(limit_price), "quantity": quantity},
        )
        self.audit_log.log_snapshot(self.symbol, price, "manual_entry_submitted",
                                    {"direction": direction, "quantity": quantity,
                                     "entry": float(entry_limit_price or price)})
        kind = f"LIMIT @ {entry_limit_price}" if entry_limit_price else "MARKET"
        self.reporter.send(
            f"🖐 <b>Manual {esc(direction)}</b> <code>{esc(self.symbol)}</code> x{quantity} ({kind})\n"
            f"SL: {stop_price}  TP: {limit_price} (re-anchored to the fill)"
        )
        return f"{direction} {quantity} submitted ({kind}), order {entry_id}"

    def manual_close(self) -> str:
        pos = self.audit_log.get_active_position(self.symbol)
        if pos is not None and pos["status"] == "pending_entry":
            self.broker.cancel_order(pos["entry_order_id"])
            self.audit_log.log_snapshot(self.symbol, None, "manual_cancel_entry", {})
            # _check_entry_fill books the outcome (and protects any partial fill).
            return f"entry order {pos['entry_order_id']} cancel requested"
        price = float(self.broker.get_current_price(self.symbol))
        if pos is not None:
            self._software_exit(pos, "manual", price)
            if self.audit_log.get_active_position(self.symbol) is not None:
                raise RuntimeError("exit not completed yet (an exit order is still settling) - try again")
            return f"closed {pos['direction']} {pos['quantity']:g} via agent exit"
        held = int(self.broker.get_position_quantity(self.symbol))
        if held == 0:
            raise RuntimeError(f"nothing held in {self.symbol}")
        # A holding the agent did not open (e.g. long-term shares in the same
        # account) is never sold from here unless that has been switched on.
        if not (self.global_cfg.get("unmanaged_positions") or {}).get("allow_manual_close", False):
            raise RuntimeError(
                f"{self.symbol} is a holding the agent did not open - it is protected and was not sold. "
                "Close it in your broker's app (or set unmanaged_positions.allow_manual_close: true)")
        side = "SELL" if held > 0 else "BUY"
        order_id = self.broker.close_position(self.symbol, side, abs(held))
        self.audit_log.log_snapshot(self.symbol, price, "manual_close_unmanaged", {"quantity": held})
        self.reporter.send(f"🖐 <b>Manual close</b> <code>{esc(self.symbol)}</code> {side} {abs(held)} (unmanaged holding)")
        return f"{side} {abs(held)} market close submitted, order {order_id}"

    def monitor_position(self):
        pos = self.audit_log.get_active_position(self.symbol)
        if pos is None:
            return
        if pos["status"] == "pending_entry":
            self._check_entry_fill(pos)
        else:
            self._manage_open_position(pos)

    def _check_entry_fill(self, pos: dict):
        state = self.broker.get_order_state(pos["entry_order_id"])

        if state["active"]:
            # A skill's retest order has its own expiry (max_wait_candles,
            # checked on each candle), not the global entry timeout.
            if state["filled"] == 0 and self.entry_timeout_minutes and self._skill_trade() is None:
                age_min = (time.time() - float(pos["opened_ts"])) / 60
                if age_min >= self.entry_timeout_minutes:
                    logger.info("%s entry unfilled after %.0f min - cancelling",
                                self.symbol, age_min)
                    self.broker.cancel_order(pos["entry_order_id"])
                return
            if state["filled"] > 0:
                # Partially filled: cancel the rest so the filled shares get
                # protected now instead of sitting unprotected until expiry.
                logger.info("%s entry partially filled (%d/%d) - cancelling remainder",
                            self.symbol, state["filled"], state["quantity"])
                self.broker.cancel_order(pos["entry_order_id"])
            return

        if state["filled"] <= 0:
            self.audit_log.cancel_pending_entry(self.symbol)
            self._clear_skill_trade()
            self.audit_log.log_snapshot(self.symbol, None, "entry_cancelled", {"status": state["status"]})
            self.reporter.send(f"Entry for <code>{esc(self.symbol)}</code> ended unfilled ({esc(str(state['status']))})")
            return

        qty = state["filled"]
        fill_price = state["avg_fill_price"] or pos["entry_price"]
        # Anchor the bracket to the actual fill, not the signal bar's close:
        # a marketable limit (or a market order) can fill well away from it,
        # which would otherwise put a BUY's stop above the fill price.
        if self.risk_manager.config.plans_distances:
            # ATR / R-multiple brackets were sized at signal time: keep those
            # distances (the bars' ATR isn't re-read at the fill).
            stop_points = abs(float(pos["entry_price"]) - float(pos["initial_stop"]))
            limit_points = abs(float(pos["target"]) - float(pos["entry_price"])) if pos.get("target") is not None \
                else self.risk_manager.bracket_distances(float(fill_price))[1]
        else:
            stop_points, limit_points = self.risk_manager.bracket_distances(float(fill_price))
        sign = 1 if pos["direction"] == "BUY" else -1
        stop_price = self.broker.round_price(fill_price - sign * stop_points)
        target_price = self.broker.round_price(fill_price + sign * limit_points)
        trade = self._skill_trade()
        if trade is not None:
            # Skill trades keep the setup's own levels (below the breakout
            # candle, under the swing high) when the fill is between them.
            s_stop, s_target = float(trade["stop"]), float(trade["target"])
            lo, hi = (s_stop, s_target) if pos["direction"] == "BUY" else (s_target, s_stop)
            if lo < float(fill_price) < hi:
                stop_price, target_price = self.broker.round_price(s_stop), self.broker.round_price(s_target)
            trade["opened_bar"] = self._last_seen_timestamp.isoformat() if self._last_seen_timestamp is not None else None
            self.audit_log.set_kv(SKILL_TRADE_KEY.format(symbol=self.symbol), trade)
        pos = {**pos, "initial_stop": stop_price, "target": target_price}
        stop_id, target_id = self.broker.place_exit_orders(
            self.symbol, _opposite(pos["direction"]), qty, stop_price, target_price,
        )
        self.audit_log.mark_position_open(
            self.symbol, qty, fill_price,
            str(stop_id) if stop_id else None, str(target_id) if target_id else None,
            stop_price=stop_price, target_price=target_price,
        )
        self.audit_log.log_snapshot(
            self.symbol, fill_price, "position_open",
            {"direction": pos["direction"], "entry": fill_price, "quantity": qty,
             "stop": pos["initial_stop"], "target": pos["target"],
             "broker_stop": stop_id is not None, "broker_target": target_id is not None},
        )
        protection = []
        if stop_id is None:
            protection.append("stop")
        if target_id is None:
            protection.append("target")
        note = f"\nBroker rejected {' & '.join(protection)} order - enforced by software only" if protection else ""
        self.reporter.send(
            f"<b>Entry filled</b> <code>{esc(self.symbol)}</code> {pos['direction']} {qty} @ {fill_price}{note}"
        )

    def _manage_open_position(self, pos: dict):
        stop_id, target_id = pos.get("stop_order_id"), pos.get("target_order_id")

        # 1) Did a broker-side exit fill?
        dead = set()
        for order_id, other_id, reason in ((stop_id, target_id, "stop-loss"), (target_id, stop_id, "target")):
            if not order_id:
                continue
            state = self.broker.get_order_state(order_id)
            if state["filled"] > 0 and not state["active"]:
                if other_id:
                    self._cancel_quietly(other_id)
                self._finish_exit(pos, reason, [(state["filled"], state["avg_fill_price"])])
                return
            if not state["active"]:
                dead.add(order_id)  # cancelled/rejected/expired with nothing filled

        # 2) Software stop / target / trailing - only while the market is
        # open, since a market close order would be rejected otherwise.
        if not self._market_open():
            if dead:
                if self._exits_persist():
                    self._replace_dead_exits(pos, dead)
                else:
                    # DAY exits (paper account) expired at the close. Placing
                    # new ones now would be rejected or expire again - drop
                    # the ids and re-arm at the next session open.
                    self._drop_dead_ids(pos, dead)
            return
        try:
            current_price = self.broker.get_current_price(self.symbol)
        except Exception:
            logger.exception("Failed to fetch current price for %s during position check", self.symbol)
            if dead:
                self._replace_dead_exits(pos, dead)
            return

        update = compute_update(
            direction=pos["direction"], entry_price=pos["entry_price"],
            current_price=current_price, current_stop=pos["current_stop"],
            current_extreme=pos["high_water"], trail_active=bool(pos["trail_active"]),
            target=pos["target"], config=self.trail_config if self._trailing_on() else NO_TRAIL,
            atr=self._trail_atr,
        )
        exiting = update.exit_triggered or update.target_hit
        update = self._breakeven(pos, current_price, update)

        # Scalp time stop: a trade that hasn't worked within max_hold_minutes
        # is closed at the market - a scalp that sits is not a scalp any more.
        if not exiting and self.max_hold_minutes and pos.get("opened_ts") and \
                time.time() - float(pos["opened_ts"]) >= float(self.max_hold_minutes) * 60:
            self._software_exit({**pos, "current_stop": update.new_stop},
                                f"time stop ({self.max_hold_minutes:g} min)", current_price)
            return

        # A close the agent already started (its exit orders were cancelled
        # for it) is finished now - never re-place those orders, which is
        # what kept flat-by-close from ever closing.
        pending = self._closing_reason()
        if pending and not exiting:
            self._software_exit(pos, pending, current_price)
            return

        # 3) An exit order that died unfilled (an interrupted software exit,
        # a manual cancel, broker/DAY expiry) leaves the position unprotected
        # at the broker - put it back, at the up-to-date stop. A leg with no
        # order at all (rejected earlier, or a DAY order dropped overnight)
        # is re-armed too, rate-limited. Skipped when we're about to exit
        # anyway: re-placing just to cancel again would churn.
        stop_fresh = False  # a just-placed stop is already at update.new_stop
        missing_legs = [leg for leg, oid in (("stop", stop_id), ("target", target_id)) if not oid]
        if missing_legs and not exiting and not self._rearm_due():
            missing_legs = []
        if (dead or missing_legs) and not exiting:
            stop_fresh = stop_id in dead or "stop" in missing_legs
            pos = self._replace_dead_exits({**pos, "current_stop": update.new_stop}, dead,
                                           extra_legs=set(missing_legs))
            stop_id = pos.get("stop_order_id")
        elif dead:
            pos = {**pos, **{k: None for k in ("stop_order_id", "target_order_id") if pos.get(k) in dead}}

        if update.stop_changed and stop_id and not stop_fresh and stop_id not in dead:
            try:
                self.broker.modify_stop_order(stop_id, update.new_stop)
            except Exception:
                logger.exception("Failed to modify broker-side stop for %s - stop tracked in software only until next successful update", self.symbol)

        just_activated = update.trail_active and not bool(pos["trail_active"])
        self.audit_log.update_trailing_stop(
            self.symbol, update.new_stop, update.new_extreme, update.trail_active,
        )
        self.audit_log.log_snapshot(
            self.symbol, float(current_price),
            "stop_ratcheted" if update.stop_changed else "monitoring",
            {"stop": update.new_stop, "trail_active": update.trail_active},
        )
        if just_activated:
            self.reporter.send(f"Trailing stop engaged on <code>{esc(self.symbol)}</code> - stop now {update.new_stop:.4f}")

        if exiting:
            if update.exit_triggered:
                reason = "trailing stop" if update.trail_active else "stop-loss"
            else:
                reason = "target"
            self._software_exit({**pos, "current_stop": update.new_stop}, reason, current_price)

    def _breakeven(self, pos: dict, price: float, update):
        """Move the stop to the entry (plus a small offset for costs) once the
        trade is breakeven_at_r times its initial risk in profit."""
        if not self.breakeven_at_r:
            return update
        sign = 1 if pos["direction"] == "BUY" else -1
        entry, initial = float(pos["entry_price"]), float(pos["initial_stop"])
        risk = abs(entry - initial)
        if risk <= 0 or sign * (float(price) - entry) < float(self.breakeven_at_r) * risk:
            return update
        be = self.broker.round_price(entry * (1 + sign * self.breakeven_offset_pct / 100))
        if sign * (be - float(update.new_stop)) <= 0:
            return update                 # the stop is already at or past break-even
        if self.audit_log.get_kv(f"breakeven:{self.symbol}", None) != pos.get("opened_ts"):
            self.audit_log.set_kv(f"breakeven:{self.symbol}", pos.get("opened_ts"))
            self.reporter.send(f"<code>{esc(self.symbol)}</code>: {float(self.breakeven_at_r):g}R in profit - stop moved to break-even {be}")
        return replace(update, new_stop=be, stop_changed=True)

    def _software_exit(self, pos: dict, reason: str, current_price: float):
        """Close the position ourselves without ever selling more than we
        hold: cancel the resting broker orders first, re-check whether one
        of them filled in the meantime, then close only what's left."""
        stop_id, target_id = pos.get("stop_order_id"), pos.get("target_order_id")
        fills = []
        for order_id in (stop_id, target_id):
            if not order_id:
                continue
            self._cancel_quietly(order_id)
            try:
                state = self._settled_state(order_id)
            except Exception:
                logger.exception("Could not re-check order %s for %s", order_id, self.symbol)
                self._mark_closing(reason)
                return  # unknown state - try again next cycle rather than risk a double sell
            if state["active"]:
                logger.warning("Order %s for %s still active after cancel - retrying next cycle", order_id, self.symbol)
                self._mark_closing(reason)
                return
            if state["filled"] > 0:
                fills.append((state["filled"], state["avg_fill_price"]))

        filled_by_broker = sum(q for q, _ in fills)
        remaining = int(pos["quantity"]) - filled_by_broker
        if remaining > 0:
            held = abs(self.broker.get_position_quantity(self.symbol))
            if held == 0 and not fills:
                # Nothing filled through us and nothing is held: the position
                # was closed elsewhere. Don't book a made-up P&L for it.
                self.close_external(pos)
                return
            to_close = min(remaining, held)
            if held < remaining:
                logger.warning("%s: expected %d shares, broker shows %d - closing %d",
                               self.symbol, remaining, held, to_close)
            if to_close > 0:
                try:
                    close_id = self.broker.close_position(self.symbol, _opposite(pos["direction"]), to_close)
                except Exception as exc:
                    logger.exception("Software close failed for %s (%s) - re-arming broker orders", self.symbol, reason)
                    self.reporter.report_error(f"closing {self.symbol}", str(exc))
                    self._clear_closing()
                    self._rearm_exit_orders(pos, remaining)
                    return
                fill_price = current_price
                try:
                    state = self.broker.get_order_state(close_id)
                    fill_price = state["avg_fill_price"] or current_price
                except Exception:
                    logger.warning("Could not read close fill for %s; using last price for P&L", self.symbol)
                fills.append((to_close, fill_price))

        self._finish_exit(pos, reason, fills or [(int(pos["quantity"]), current_price)])

    def _settled_state(self, order_id) -> dict:
        """Order state after a cancel, waiting up to CANCEL_SETTLE_SECONDS
        for the broker to finish cancelling (or report a fill)."""
        state = self.broker.get_order_state(order_id)
        waited = 0.0
        while state["active"] and waited < CANCEL_SETTLE_SECONDS:
            time.sleep(CANCEL_SETTLE_STEP)
            waited += CANCEL_SETTLE_STEP
            state = self.broker.get_order_state(order_id)
        return state

    def _mark_closing(self, reason: str):
        self.audit_log.set_kv(CLOSING_KEY.format(symbol=self.symbol), {"reason": reason, "ts": time.time()})

    def _closing_reason(self) -> str | None:
        """The reason of a close the agent started and is still finishing."""
        mark = self.audit_log.get_kv(CLOSING_KEY.format(symbol=self.symbol), None)
        if isinstance(mark, dict) and time.time() - float(mark.get("ts") or 0) < CLOSING_MAX_AGE:
            return str(mark.get("reason") or "close")
        return None

    def _clear_closing(self):
        if self.audit_log.get_kv(CLOSING_KEY.format(symbol=self.symbol), None) is not None:
            self.audit_log.set_kv(CLOSING_KEY.format(symbol=self.symbol), None)

    def _rearm_exit_orders(self, pos: dict, quantity: int):
        stop_id, target_id = self.broker.place_exit_orders(
            self.symbol, _opposite(pos["direction"]), quantity, pos["current_stop"], pos["target"],
        )
        self.audit_log.set_exit_order_ids(
            self.symbol, str(stop_id) if stop_id else None, str(target_id) if target_id else None,
        )

    def _rearm_due(self) -> bool:
        key = f"rearm_retry:{self.symbol}"
        last = float(self.audit_log.get_kv(key, 0) or 0)
        if time.time() - last < REARM_RETRY_SECONDS:
            return False
        self.audit_log.set_kv(key, time.time())
        return True

    def _drop_dead_ids(self, pos: dict, dead: set) -> dict:
        stop_id = None if pos.get("stop_order_id") in dead else pos.get("stop_order_id")
        target_id = None if pos.get("target_order_id") in dead else pos.get("target_order_id")
        self.audit_log.set_exit_order_ids(self.symbol, stop_id, target_id)
        logger.info("%s: DAY exit order(s) ended with the session - re-arming at the next open", self.symbol)
        return {**pos, "stop_order_id": stop_id, "target_order_id": target_id}

    def _replace_dead_exits(self, pos: dict, dead: set, extra_legs: set | None = None) -> dict:
        """Re-place whichever of stop/target is in `dead` (or named in
        extra_legs: legs with no order at all), for the quantity the broker
        still holds (never more than the position). Returns pos with the new
        order ids; a leg that can't be placed is left as None and enforced
        by the software monitor."""
        extra_legs = extra_legs or set()
        held = min(abs(self.broker.get_position_quantity(self.symbol)), int(pos["quantity"]))
        stop_id, target_id = pos.get("stop_order_id"), pos.get("target_order_id")
        stop_id = None if stop_id in dead else stop_id
        target_id = None if target_id in dead else target_id
        if held <= 0:
            # Nothing left to protect; the software monitor / next cycle will
            # reconcile. Just drop the dead ids so we stop polling them.
            self.audit_log.set_exit_order_ids(self.symbol, stop_id, target_id)
            return {**pos, "stop_order_id": stop_id, "target_order_id": target_id}

        want_stop = pos.get("stop_order_id") in dead or ("stop" in extra_legs and not stop_id)
        want_target = pos.get("target_order_id") in dead or ("target" in extra_legs and not target_id)
        closing = _opposite(pos["direction"])
        replaced, failed = [], []
        if want_stop:
            try:
                stop_id = str(self.broker.place_stop_order(self.symbol, closing, held, pos["current_stop"]))
                replaced.append(f"stop @ {pos['current_stop']}")
            except Exception:
                failed.append("stop")
                logger.exception("Could not re-place stop for %s - software monitor enforces it", self.symbol)
        if want_target:
            try:
                target_id = str(self.broker.place_target_order(self.symbol, closing, held, pos["target"]))
                replaced.append(f"target @ {pos['target']}")
            except Exception:
                failed.append("target")
                logger.exception("Could not re-place target for %s - software monitor enforces it", self.symbol)

        self.audit_log.set_exit_order_ids(self.symbol, stop_id, target_id)
        if dead:
            msg = (f"Exit order(s) for {self.symbol} were no longer working at the broker - "
                   f"re-placed: {', '.join(replaced) or 'none (software monitor only)'}")
            logger.warning(msg)
            self.reporter.send(f"⚠️ {esc(msg)}")
        elif replaced:
            tif = getattr(self.broker, "exit_time_in_force", "GTC")
            msg = f"Session protection armed for {self.symbol} ({tif}): {', '.join(replaced)}"
            if failed:
                msg += f"; broker rejected {' & '.join(failed)} - software monitor enforces it"
            logger.info(msg)
            self.reporter.send(f"🛡️ {esc(msg)}")
        return {**pos, "stop_order_id": stop_id, "target_order_id": target_id}

    def close_external(self, pos: dict):
        """The broker no longer holds this position and none of our exit
        orders filled: it was closed outside the agent (manually, or by
        another program). Stop tracking it; realised P&L is unknown here."""
        for order_id in (pos.get("stop_order_id"), pos.get("target_order_id")):
            if order_id:
                self._cancel_quietly(order_id)
        self.audit_log.close_position(self.symbol)
        self._clear_closing()
        self._clear_skill_trade()
        self.audit_log.log_snapshot(self.symbol, None, "position_closed_externally",
                                    {"quantity": pos["quantity"], "entry": pos["entry_price"]})
        msg = (f"{self.symbol}: the broker no longer holds this {pos['direction']} {pos['quantity']:g} "
               "position and none of the agent's exit orders filled - it was closed outside the agent. "
               "Marked closed; P&L not recorded.")
        logger.warning(msg)
        self.reporter.send(f"ℹ️ {esc(msg)}")

    def _cancel_quietly(self, order_id):
        try:
            self.broker.cancel_order(order_id)
        except Exception as exc:
            # Usually means it already filled or was already cancelled.
            logger.info("Cancel of order %s for %s failed: %s", order_id, self.symbol, exc)

    def _finish_exit(self, pos: dict, reason: str, fills: list[tuple[int, float | None]]):
        sign = 1 if pos["direction"] == "BUY" else -1
        pnl = sum(
            sign * ((price if price is not None else pos["entry_price"]) - pos["entry_price"]) * qty
            for qty, price in fills
        )
        exit_qty = sum(q for q, _ in fills)
        avg_exit = (sum(q * (p or pos["entry_price"]) for q, p in fills) / exit_qty) if exit_qty else None

        self.audit_log.close_position(self.symbol)
        self._clear_closing()
        self._clear_skill_trade()
        recent = self.audit_log.get_kv(RECENT_CLOSES_KEY, {}) or {}
        recent = {k: v for k, v in recent.items() if time.time() - v < 3600}
        recent[base_symbol(self.symbol)] = time.time()
        self.audit_log.set_kv(RECENT_CLOSES_KEY, recent)
        self.risk_manager.record_trade_result(pnl, self.symbol)
        deal_id = str(pos.get("entry_order_id"))
        self.audit_log.log_trade_result(
            self.symbol, deal_id, pnl,
            {"reason": reason, "entry": pos["entry_price"], "exit": avg_exit, "quantity": exit_qty,
             "direction": pos.get("direction"), "initial_stop": pos.get("initial_stop"),
             "opened_ts": pos.get("opened_ts"), "closed_ts": time.time()},
        )
        self.audit_log.log_snapshot(self.symbol, avg_exit, "position_closed", {"reason": reason, "pnl": pnl})
        self.reporter.report_trade_closed(self.symbol, deal_id, pnl)
        if self.risk_manager.state.trading_halted:
            self.reporter.send("<b>Daily loss cap breached</b> - no new entries until tomorrow")


class Orchestrator:
    def __init__(self, config: dict, broker=None, *, broker_name: str = "moomoo", adapter: str = "moomoo",
                 market: str | None = None, label: str | None = None, audit_log=None,
                 kronos_agent=None, reporter=None, data_source: str | None = None):
        """One trading loop for one broker. With several brokers configured,
        src/multi_broker.py builds one of these per broker - each with its
        own connection, audit DB, risk state and market calendar - and
        shares a single offline Kronos model between them."""
        self.config = config
        self.broker_name, self.adapter, self.market_override = broker_name, adapter, market
        config.setdefault("_broker_name", broker_name)      # lets symbol workers label shadow records
        self.data_source = data_source   # adapter supplying price data, None = the broker itself
        self.broker_label = label or ("moomoo OpenD" if adapter == "moomoo" else adapter)
        sched = config.get("schedule", {}) or {}
        self.follow_market = bool(sched.get("follow_market", True)) and not config.get(
            "trade_outside_market_hours", False)
        self.warmup_minutes = float(sched.get("warmup_minutes", 15))
        self.notify_open_close = bool(sched.get("notify_open_close", True))
        self.sleep_monitor_seconds = float(sched.get("sleep_monitor_seconds", 300))
        conn = config.get("connection", {}) or {}
        self.connect_timeout = float(conn.get("connect_timeout_seconds", 35))
        self.ping_interval = float(conn.get("health_check_seconds", 60))
        self.reconnect_after = int(conn.get("reconnect_after_failures", 3))
        unmanaged = config.get("unmanaged_positions", {}) or {}
        self.unmanaged_mode = str(unmanaged.get("mode", "alert")).lower()   # alert | off
        self.unmanaged_interval = float(unmanaged.get("check_interval_seconds", 300))
        self.reconcile_missing = bool(unmanaged.get("close_missing_positions", True))

        self.reporter = reporter if reporter is not None else TelegramReporter()
        self.audit_log = audit_log if audit_log is not None else AuditLog()
        # which code this agent process is running (shown on the dashboard)
        self.audit_log.set_kv(code_version.AGENT_VERSION_KEY, {**code_version.RUNNING, "started": time.time(),
                                                              "pid": os.getpid()})
        self.broker_ok = False
        self._ping_failures = 0
        self._last_ping = 0.0
        self._broker_detail = ""
        self.broker = broker if broker is not None else self._connect_until_ready()
        self.broker_ok = True
        self._publish_health("connected")

        if kronos_agent is not None:
            self.kronos_agent = kronos_agent          # shared offline model (multi-broker)
        else:
            kronos_cfg = config.get("kronos", {})
            self.kronos_agent = KronosAgent(**kronos_cfg)

        risk_cfg = RiskConfig(**config["risk"])
        # Daily loss cap resets at midnight in the exchange's timezone.
        self.risk_manager = RiskManager(
            risk_cfg, store=self.audit_log, timezone=str(market_timezone(self.broker.market)),
        )

        self.screener = None
        self.screener_refresh_seconds = 0
        self._last_screener_run = 0.0
        self.workers: dict[str, SymbolWorker] = {}
        self.phase = None
        self._last_unmanaged_check = 0.0

        screener_cfg = config.get("screener", {})
        self.around_the_clock = self.broker.market == ig_hours.MARKET and hasattr(self.broker, "enable_around_the_clock")
        self._stream_published = 0.0
        if self.around_the_clock:
            from src import ig_screener
            self.broker.enable_around_the_clock(ig_screener.settings(config), config["timeframe"])
        if screener_cfg.get("enabled") and self.around_the_clock:
            from src.ig_screener import IG24Screener
            self.screener = IG24Screener(self.broker, config, open_symbols=lambda: [
                p["symbol"] for p in self.audit_log.get_active_positions()])
            self.screener_refresh_seconds = float(self.screener.cfg["refresh_minutes"]) * 60
            self._refresh_symbols_from_screener()
        elif screener_cfg.get("enabled"):
            from src.screener import Screener
            self.screener = Screener(self.broker, config)
            self.screener_refresh_seconds = screener_cfg.get("refresh_interval_minutes", 360) * 60
            self._refresh_symbols_from_screener()
        else:
            self._set_workers(config.get("symbols", []))

        self._check_unmanaged()

    # ------------------------------------------------------------------
    # Broker connection: bounded connect, health ping, auto-reconnect
    # ------------------------------------------------------------------

    def _publish_health(self, detail: str):
        prev = self.audit_log.get_kv(BROKER_HEALTH_KEY, {}) or {}
        since = prev.get("since") if prev.get("ok") == self.broker_ok else time.time()
        self.audit_log.set_kv(BROKER_HEALTH_KEY, {
            "ok": self.broker_ok, "detail": detail, "ts": time.time(), "since": since or time.time(),
            "failures": self._ping_failures,
        })

    def _connect(self):
        """Bounded connect to this orchestrator's broker."""
        adapter = getattr(self, "adapter", "moomoo")
        kwargs = {"market": self.market_override} if getattr(self, "market_override", None) else {}
        data_source = getattr(self, "data_source", None)
        if data_source and data_source != adapter:
            from src.brokers import connect as connect_routed
            return connect_routed(adapter, self.connect_timeout, data_source=data_source, **kwargs)
        if adapter == "moomoo":
            return connect_with_timeout(self.connect_timeout, **kwargs)
        return connect_broker(adapter, self.connect_timeout, **kwargs)

    @property
    def _label(self) -> str:
        return getattr(self, "broker_label", "moomoo OpenD")

    def _connect_until_ready(self) -> MoomooClient:
        """Block until OpenD answers and is logged in, retrying with a
        bounded wait each time - the agent never freezes on a gateway that
        accepts the connection but never responds."""
        attempt, alerted = 0, False
        while True:
            attempt += 1
            try:
                client = self._connect()
                if alerted:
                    self.reporter.send(f"✅ {esc(self._label)} connected - agent starting")
                return client
            except Exception as exc:
                detail = str(exc)
                logger.warning("%s connect attempt %d failed: %s", self._label, attempt, detail)
                self.broker_ok = False
                self._publish_health(detail)
                if not alerted:
                    self.reporter.send(f"⚠️ Waiting for {esc(self._label)}: {esc(detail)}\nRetrying every 30s.")
                    alerted = True
                time.sleep(30)

    def _swap_broker(self, new_broker):
        old = self.broker
        self.broker = new_broker
        for worker in self.workers.values():
            worker.broker = new_broker
        if self.screener is not None:
            self.screener.broker = new_broker
        if getattr(self, "around_the_clock", False) and hasattr(new_broker, "enable_around_the_clock"):
            from src import ig_screener
            try:
                old_stream = getattr(old, "stream", None)
                if old_stream is not None:
                    old_stream.stop()
                new_broker.enable_around_the_clock(ig_screener.settings(self.config), self.config["timeframe"])
                new_broker.classes.update(getattr(old, "classes", {}) or {})
                new_broker.watch(list(self.workers))
            except Exception:
                logger.exception("Restarting the IG stream after the reconnect failed")
        if old is not None and old is not new_broker:
            try:
                old.close()
            except Exception:
                logger.debug("Closing old broker failed", exc_info=True)

    def check_broker(self, now: float | None = None, force: bool = False) -> bool:
        """Ping OpenD at most every ping_interval seconds. After
        reconnect_after consecutive failures, rebuild the connection
        (bounded). Returns whether the broker is usable right now."""
        now = now if now is not None else time.time()
        if not force and now - self._last_ping < self.ping_interval:
            return self.broker_ok
        self._last_ping = now
        ping = getattr(self.broker, "ping", None)
        ok, detail = ping() if ping else (True, "connected")
        if not ok:
            self._ping_failures += 1
            if self._ping_failures >= self.reconnect_after:
                try:
                    self._swap_broker(self._connect())
                    ok, detail = True, "reconnected"
                except Exception as exc:
                    detail = str(exc)
        was_ok = self.broker_ok
        self.broker_ok = ok
        if ok:
            self._ping_failures = 0
            if not was_ok:
                logger.info("%s back: %s", self._label, detail)
                self.reporter.send(f"✅ {esc(self._label)} connection restored")
        elif was_ok:
            logger.warning("%s unhealthy: %s", self._label, detail)
            self.reporter.send(f"⚠️ {esc(self._label)} problem: {esc(detail)}\n"
                               "Trading paused; reconnecting automatically.")
        if ok != was_ok or detail != self._broker_detail or not ok:
            self._broker_detail = detail
            self._publish_health(detail)
        return ok

    # ------------------------------------------------------------------
    # Market calendar: active / warmup / sleeping
    # ------------------------------------------------------------------

    @property
    def exchange(self) -> str | None:
        return cal.MARKET_EXCHANGE.get(self.broker.market)

    def compute_phase(self, now: datetime | None = None) -> tuple[str, float | None]:
        """(phase, next_open_epoch). phase: always | active | warmup | sleeping."""
        if not self.follow_market:
            return "always", None
        if getattr(self, "around_the_clock", False):
            now = now or datetime.now(timezone.utc)
            if self.broker.any_open(now):
                return "active", None
            nxt = self.broker.next_open(now)
            nxt = nxt.timestamp() if nxt is not None else None
            if nxt is not None and nxt - now.timestamp() <= self.warmup_minutes * 60:
                return "warmup", nxt
            return "sleeping", nxt
        market_open = is_market_open(self.broker.market, now) if now else is_market_open(self.broker.market)
        now = now or datetime.now(timezone.utc)
        if market_open:
            return "active", None
        nxt = cal.next_open(self.exchange, now).timestamp() if self.exchange else None
        if nxt is not None and nxt - now.timestamp() <= self.warmup_minutes * 60:
            return "warmup", nxt
        return "sleeping", nxt

    def _finetune_tick(self, old: str | None, phase: str) -> None:
        """Save candles for fine-tuning (finetune.enabled): when this market
        closes, the first time there is nothing saved yet, and when the
        dashboard asks. Runs in the background; never touches trading."""
        from src import finetune
        from src.finetune import data as ftdata
        ft = finetune.settings(self.config)
        name = self.broker_name or getattr(self.broker, "name", "broker")
        requested = 0.0
        try:
            requested = finetune.COLLECT_REQUEST.stat().st_mtime
        except OSError:
            pass
        last = getattr(self, "_ft_last_collect", None)
        if last is None:                               # first tick after start
            self._ft_last_collect = last = time.time() if (finetune.TRAINING_DIR / ftdata._safe(name)).exists() else 0.0
        on_request = requested > last
        if not on_request:
            if not ft["enabled"]:
                return
            closed_now = ft["collect_on_close"] and phase == "sleeping" and old in ("active", "always")
            if not (closed_now or (last == 0.0 and phase in ("active", "always", "sleeping"))):
                return
        thread = getattr(self, "_ft_thread", None)
        if thread is not None and thread.is_alive():
            return
        if not self.broker_ok:
            return
        self._ft_last_collect = time.time()
        timeframe = self.config.get("timeframe")
        symbols = list(self.workers)
        symbols += ftdata.symbols_from_audit(self.audit_log.db_path)
        folder = finetune.TRAINING_DIR / ftdata._safe(name)
        known = {ftdata._safe(s): s for s in symbols}
        symbols += [f.stem.split("__")[0] for f in folder.glob(f"*__{ftdata._safe(timeframe)}.csv")
                    if f.stem.split("__")[0] not in known]
        market_open = phase in ("active", "always", "warmup")

        def work():
            try:
                ftdata.collect(self.broker, name, timeframe, symbols, int(ft["bars_per_symbol"]),
                               int(ft["max_symbols"]), drop_last=market_open)
            except Exception:
                logger.exception("Collecting fine-tuning candles failed")

        self._ft_thread = threading.Thread(target=work, name=f"finetune-collect-{name}", daemon=True)
        self._ft_thread.start()

    def update_phase(self, now: datetime | None = None) -> str:
        phase, nxt = self.compute_phase(now)
        old, self.phase = self.phase, phase
        name = self.exchange or self.broker.market
        if phase != old:
            logger.info("Market phase %s -> %s (%s)", old, phase, name)
            if phase == "warmup":
                self.check_broker(force=True)
                if self.screener:
                    self._last_screener_run = 0.0  # fresh picks for the session
            if old is not None and self.notify_open_close:
                when = (datetime.fromtimestamp(nxt, cal.ZoneInfo(cal.EXCHANGES[self.exchange].tz))
                        .strftime("%a %d %b %H:%M %Z") if nxt and self.exchange else "next session")
                if phase == "active":
                    self.reporter.send(f"🔔 <b>{esc(name)} open</b> - scanning {len(self.workers)} symbol(s)")
                elif phase == "warmup":
                    self.reporter.send(f"⏰ {esc(name)} opens soon - agent warming up")
                elif phase == "sleeping" and old in ("active", "always"):
                    self.reporter.send(f"🌙 <b>{esc(name)} closed</b> - sleeping until {esc(when)}")
        try:
            self._finetune_tick(old, phase)
        except Exception:
            logger.exception("Fine-tuning data collection check failed")
        self.audit_log.set_kv(PHASE_KEY, {
            "phase": phase, "exchange": name, "market": self.broker.market,
            "next_open": nxt, "ts": time.time(), "follow_market": self.follow_market,
        })
        return phase

    # ------------------------------------------------------------------
    # Broker positions the agent doesn't manage (and the reverse)
    # ------------------------------------------------------------------

    def _check_unmanaged(self, now: float | None = None):
        """Flag broker positions/orders this agent isn't tracking (left over
        from an older version, placed by hand, ...). They block new entries
        in that symbol via the exposure check, but nothing manages their
        exits. Also closes agent rows the broker no longer holds."""
        if self.unmanaged_mode == "off":
            return
        now = now if now is not None else time.time()
        self._last_unmanaged_check = now
        base = base_symbol
        rows = self.audit_log.get_active_positions()
        tracked = {base(r["symbol"]) for r in rows}
        managed_ids = {str(r[k]) for r in rows
                       for k in ("entry_order_id", "stop_order_id", "target_order_id") if r.get(k)}
        try:
            held = {base(p["symbol"]): p for p in self.broker.get_open_positions() if p["quantity"]}
            orders = [o for o in self.broker.get_open_orders() if str(o["id"]) not in managed_ids]
        except Exception:
            logger.exception("Broker reconciliation failed")
            return

        recent = self.audit_log.get_kv(RECENT_CLOSES_KEY, {}) or {}
        positions = [
            {"symbol": p["symbol"], "quantity": p["quantity"], "average_cost": p.get("average_cost"),
             "market_price": p.get("market_price"), "unrealized_pnl": p.get("unrealized_pnl")}
            for key, p in sorted(held.items())
            if key not in tracked and now - float(recent.get(key, 0)) > RECENT_CLOSE_GRACE_SECONDS
        ]
        stray_orders = [
            {"id": o["id"], "symbol": o["symbol"], "action": o.get("action"), "quantity": o.get("quantity"),
             "status": o.get("status")}
            for o in orders if base(o["symbol"]) not in tracked
        ]
        self.audit_log.set_kv(UNMANAGED_KEY, {"ts": now, "positions": positions, "orders": stray_orders})

        current = sorted({base(p["symbol"]) for p in positions} | {base(o["symbol"]) for o in stray_orders})
        alerted = set(self.audit_log.get_kv(UNMANAGED_ALERTED_KEY, []) or [])
        new = [s for s in current if s not in alerted]
        if new:
            detail = []
            for p in positions:
                if base(p["symbol"]) in new:
                    side = "LONG" if p["quantity"] > 0 else "SHORT"
                    detail.append(f"{p['symbol']} {side} {abs(p['quantity']):g}")
            for o in stray_orders:
                if base(o["symbol"]) in new:
                    detail.append(f"{o['symbol']} order {o['id']} ({o.get('action')})")
            msg = (f"{self._label} holds positions/orders this agent does not manage - no stop/target is "
                   f"being managed for them: {'; '.join(detail)}")
            logger.warning(msg)
            self.reporter.send(f"⚠️ {esc(msg)}")
        # Remember only what is still there, so a symbol re-alerts if it
        # comes back later.
        self.audit_log.set_kv(UNMANAGED_ALERTED_KEY, current)

        if self.reconcile_missing:
            self._reconcile_missing(rows, held, now)

    def _reconcile_missing(self, rows: list[dict], held: dict, now: float):
        base = base_symbol
        since = self.audit_log.get_kv(MISSING_SINCE_KEY, {}) or {}
        still = {}
        for r in rows:
            key = base(r["symbol"])
            if r["status"] != "open" or key in held:
                continue
            first = float(since.get(key, now))
            if now - first < MISSING_CONFIRM_SECONDS:
                still[key] = first
                continue
            worker = self.workers.get(r["symbol"]) or self._monitor_worker(r["symbol"])
            try:
                worker.monitor_position()  # an exit order may have filled meanwhile
            except Exception:
                # e.g. its old order ids are unknown to this account - that
                # must not keep a stale row alive forever.
                logger.warning("Monitor check for %s failed during reconcile", r["symbol"], exc_info=True)
            try:
                pos = self.audit_log.get_active_position(r["symbol"])
                if pos is not None and pos["status"] == "open" and \
                        worker.broker.get_position_quantity(r["symbol"]) == 0:
                    worker.close_external(pos)
            except Exception:
                logger.exception("Could not reconcile %s", r["symbol"])
                still[key] = first
        self.audit_log.set_kv(MISSING_SINCE_KEY, still)

    # ------------------------------------------------------------------
    # Symbols
    # ------------------------------------------------------------------

    def _monitor_worker(self, symbol: str) -> SymbolWorker:
        worker = SymbolWorker(
            {"symbol": symbol, "monitor_only": True}, self.config, self.broker, self.kronos_agent,
            self.risk_manager, self.audit_log, self.reporter,
        )
        self.workers[symbol] = worker
        return worker

    def _set_workers(self, symbol_cfgs: list[dict]):
        new_workers = {}
        for symbol_cfg in symbol_cfgs:
            symbol = symbol_cfg["symbol"]
            # Keep the existing worker (and its _last_seen_timestamp state)
            # if this symbol was already being tracked.
            existing = self.workers.get(symbol)
            if existing is not None and not existing.monitor_only:
                new_workers[symbol] = existing
            else:
                new_workers[symbol] = SymbolWorker(
                    symbol_cfg, self.config, self.broker, self.kronos_agent,
                    self.risk_manager, self.audit_log, self.reporter,
                )
        # A symbol the screener drops - or any agent position in the
        # database, e.g. from before a restart - keeps a monitor-only
        # worker so its exits keep being managed until it closes.
        for pos in self.audit_log.get_active_positions():
            symbol = pos["symbol"]
            if symbol not in new_workers:
                worker = self.workers.get(symbol)
                if worker is None:
                    worker = SymbolWorker(
                        {"symbol": symbol, "monitor_only": True}, self.config, self.broker,
                        self.kronos_agent, self.risk_manager, self.audit_log, self.reporter,
                    )
                new_workers[symbol] = worker
        dropped = set(self.workers) - set(new_workers)
        if dropped:
            logger.info("Screener dropped symbols: %s", dropped)
        self.workers = new_workers
        # Published for the dashboard, which runs in a separate process and
        # otherwise can't know what the screener picked.
        self.audit_log.set_kv(ACTIVE_SYMBOLS_KEY, {
            "source": "screener" if self.screener else "config",
            "updated": time.time(),
            "symbols": [{"symbol": w.symbol, "name": w.name} for w in new_workers.values()],
        })

    def _refresh_symbols_from_screener(self):
        try:
            selected = self.screener.run()
        except Exception:
            logger.exception("Screener run failed - keeping existing symbol list")
            return
        # Publish what the screener saw even when it picked nothing, so the
        # dashboard can say why (markets closed, still being discovered, ...).
        report = getattr(self.screener, "last_report", None)
        if isinstance(report, dict):
            try:
                self.audit_log.set_kv(SCREENER_REPORT_KEY, report)
            except Exception:
                logger.exception("Could not publish the screener report")
        if not selected:
            if not getattr(self, "_empty_logged", False):
                logger.warning("Screener returned no candidates - keeping existing symbol list")
                self._empty_logged = True
            return
        self._empty_logged = False
        self._set_workers(selected)
        self._last_screener_run = time.time()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_cycle(self, state: dict, now: float | None = None):
        """One pass of the main loop. `state` carries the loop timers."""
        now = now if now is not None else time.time()
        interval = self.config.get("polling_interval_seconds", 60)

        if not self.check_broker(now):
            return  # every broker call would fail - wait for the reconnect

        phase = self.update_phase()
        awake = phase in ("active", "warmup", "always")

        if getattr(self, "around_the_clock", False):
            try:
                self.broker.keep_up()            # save streamed candles, live status, stream health
            except Exception:
                logger.exception("IG around-the-clock upkeep failed")
            if now - self._stream_published >= 15:
                self._stream_published = now
                self.audit_log.set_kv(IG_STREAM_KEY, {**self.broker.stream_summary(), "ts": now})

        if self.screener and awake and now - self._last_screener_run >= self.screener_refresh_seconds:
            self._refresh_symbols_from_screener()
        elif self.screener and getattr(self, "around_the_clock", False) and not awake and \
                now - getattr(self, "_last_closed_report", 0.0) >= self.screener_refresh_seconds:
            # Every chosen market is closed (e.g. the weekend): keep the
            # dashboard's list current - which markets, and when they reopen -
            # without scanning anything.
            self._last_closed_report = now
            self._refresh_symbols_from_screener()

        candle_due = self._candle_closed_since(state, now)
        if phase in ("active", "always") and (candle_due or now - state.get("last_full_poll", 0.0) >= interval):
            self.poll_workers()
            state["last_full_poll"] = now

        # Positions: every cycle while awake; at a slow cadence while the
        # market sleeps (fills can't happen, but DAY exits expire and a
        # restart may need reconciling).
        if awake or now - state.get("last_monitor", 0.0) >= self.sleep_monitor_seconds:
            for worker in list(self.workers.values()):
                try:
                    worker.monitor_position()
                except Exception:
                    logger.exception("Position monitor error for %s", worker.symbol)
            state["last_monitor"] = now

        if now - self._last_unmanaged_check >= self.unmanaged_interval:
            self._check_unmanaged(now)

        self.process_manual_orders()
        self.flat_by_close()

    # ---- candle-close timing and one Kronos pass per candle ------------------

    def _candle_seconds(self) -> float | None:
        """Length of the trading candle when it's short enough for its closes
        to line up with the clock (1-15 minutes); None otherwise."""
        d = BAR_DURATION.get(self.config.get("timeframe", ""))
        secs = d.total_seconds() if d is not None else None
        return secs if secs and secs <= 900 else None

    def _candle_closed_since(self, state: dict, now: float) -> bool:
        """True once per candle, a few seconds after it closes (so the broker
        has the finished bar). Polling then happens right away instead of
        up to polling_interval_seconds later."""
        secs = self._candle_seconds()
        if not secs or not self.config.get("candle_close_poll", True):
            return False
        settle = float(self.config.get("candle_close_delay_seconds", 3))
        boundary = (now - settle) // secs * secs
        last = state.get("last_candle_boundary")
        state["last_candle_boundary"] = boundary
        return last is not None and boundary > last

    def seconds_until_next_candle(self, now: float | None = None) -> float | None:
        secs = self._candle_seconds()
        if not secs or not self.config.get("candle_close_poll", True):
            return None
        now = now if now is not None else time.time()
        settle = float(self.config.get("candle_close_delay_seconds", 3))
        return (now - settle) // secs * secs + secs + settle - now

    def poll_workers(self):
        """Poll every symbol for its latest closed candle. Symbols whose
        pattern survived the gates are confirmed by Kronos together - one
        batched model pass where possible - so the tenth symbol to signal on
        a candle isn't kept waiting behind the first nine."""
        pending = []
        for worker in list(self.workers.values()):
            try:
                ctx = worker.prepare_signal()
            except Exception:
                logger.exception("Worker error for %s", worker.symbol)
                continue
            if ctx is not None:
                pending.append((worker, ctx))
        if not pending:
            return
        groups: dict = {}
        for worker, ctx in pending:
            o = worker.kronos_opts if isinstance(getattr(worker, "kronos_opts", None), dict) else DEFAULT_OPTIONS
            key = (worker.forecast_horizon, worker.bar_duration, tuple(sorted(o.items())))
            groups.setdefault(key, []).append((worker, ctx))
        for (horizon, bar, okey), items in groups.items():
            o = dict(okey)
            custom = not is_default(o)
            results, per_ms = None, None
            many = getattr(self.kronos_agent, "confirm_many", None)
            jobs = [(c["df"], c["signal"].direction) for _, c in items]
            if (len(items) > 1 or custom) and callable(many):
                t0 = time.perf_counter()
                kw = {"bar_duration": bar}
                if custom:
                    kw.update(method=o["method"], paths=o["paths"], min_move_pct=o["min_move_pct"])
                try:
                    results = many(jobs, horizon, **kw)
                except Exception:  # noqa: BLE001
                    logger.exception("Kronos batch failed; asking one symbol at a time")
                    results = None
                if not (isinstance(results, list) and len(results) == len(items)):
                    results = None
                else:
                    per_ms = (time.perf_counter() - t0) * 1000.0 / len(items)
                    self._note_kronos_time(per_ms * len(items), len(items))
            extras = [None] * len(items)
            if results is not None:
                extras = [w.kronos_extra_base() if isinstance(getattr(w, "kronos_opts", None), dict) else {}
                          for w, _ in items]
                if o["shadow_paths"] and o["method"] == "bars":
                    try:          # recorded only: the 'paths' confidence, for the scorecard
                        sp = many(jobs, horizon, bar_duration=bar, method="paths", paths=o["paths"],
                                  min_move_pct=o["min_move_pct"])
                        for k, r in enumerate(sp if isinstance(sp, list) else []):
                            if isinstance(r, tuple):
                                extras[k]["paths_confidence"] = round(float(r[1]), 3)
                    except Exception:  # noqa: BLE001
                        logger.debug("Shadow paths confidence failed", exc_info=True)
            for i, (worker, ctx) in enumerate(items):
                try:
                    if results is not None:
                        ctx["kronos_extra"] = extras[i]
                        worker.finish_signal(ctx, results[i], per_ms)
                        continue
                    if custom:
                        res, ctx["kronos_extra"], ms = worker.ask_kronos(ctx["df"], ctx["signal"].direction)
                        worker.finish_signal(ctx, res, ms)
                        continue
                    t0 = time.perf_counter()
                    try:
                        res = worker.kronos_agent.confirm_direction(
                            ctx["df"], ctx["signal"].direction, horizon, bar_duration=bar)
                    except Exception as exc:  # noqa: BLE001
                        res = exc
                    ms = (time.perf_counter() - t0) * 1000.0
                    if not isinstance(res, BaseException):
                        self._note_kronos_time(ms, 1)
                    worker.finish_signal(ctx, res, ms)
                except Exception:
                    logger.exception("Worker error for %s", worker.symbol)

    def _note_kronos_time(self, ms: float, symbols: int):
        note_kronos_time(self.audit_log, ms, symbols)

    def flat_by_close(self):
        """Day-trade mode: in the last N minutes of the session, close every
        position the agent manages and cancel unfilled entries, win or lose.
        Holdings the agent doesn't manage are left alone."""
        minutes = getattr(self.risk_manager.config, "flat_by_minutes_before_close", None)
        if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or minutes <= 0:
            return
        if getattr(self, "around_the_clock", False):
            return self._flat_by_each_close(float(minutes))
        try:
            mins = cal.session_minutes(cal.exchange_for_market(self.broker.market))
        except Exception:
            logger.exception("Session clock unavailable - flat-by-close skipped")
            return
        if mins is None:
            return                      # market closed: nothing can be sold now
        in_window = mins[1] <= minutes
        # A position opened on an earlier day should not exist in day-trade
        # mode (the agent was down at yesterday's close, or the exit failed).
        # Close it as soon as the market is open instead of holding it all day.
        try:
            tz = cal.ZoneInfo(cal.EXCHANGES[cal.exchange_for_market(self.broker.market)].tz)
            day_start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        except Exception:
            day_start = None
        for pos in self.audit_log.get_active_positions():
            overdue = bool(day_start and pos.get("opened_ts") and float(pos["opened_ts"]) < day_start)
            if not (in_window or overdue):
                continue
            worker = self._worker_for(pos["symbol"])
            reason = "flat_by_close" if in_window else "flat_overnight_leftover"
            try:
                if pos["status"] == "pending_entry":
                    if pos.get("entry_order_id"):
                        self.broker.cancel_order(pos["entry_order_id"])
                        self.audit_log.log_snapshot(pos["symbol"], None, "flat_by_cancel_entry", {})
                    continue
                price = float(self.broker.get_current_price(pos["symbol"]))
                logger.info("%s: %s (%.0f min left) - closing %s %g",
                            pos["symbol"], reason, mins[1], pos["direction"], pos["quantity"])
                worker._software_exit(pos, reason, price)
            except Exception as exc:
                logger.exception("Flat-by-close failed for %s", pos["symbol"])
                # Tell the user once per symbol per day: this is the one exit
                # that has a deadline. It is retried every cycle until the close.
                alerted = getattr(self, "_flat_alerted", None)
                if alerted is None:
                    alerted = self._flat_alerted = set()
                key = (pos["symbol"], day_start)
                if key not in alerted:
                    alerted.add(key)
                    try:
                        self.reporter.report_error(
                            f"closing {pos['symbol']} before the market close", f"{exc} - retrying; check it manually")
                    except Exception:
                        logger.debug("Could not send the flat-by-close alert", exc_info=True)

    def _flat_by_each_close(self, minutes: float):
        """Around the clock: each trade is closed `minutes` before ITS market's
        next close (the daily break or the weekend). Crypto never closes."""
        for pos in self.audit_log.get_active_positions():
            mins = self.broker.session_minutes_for(pos["symbol"])
            if mins is None or mins[1] > minutes:
                continue
            worker = self._worker_for(pos["symbol"])
            try:
                if pos["status"] == "pending_entry":
                    if pos.get("entry_order_id"):
                        self.broker.cancel_order(pos["entry_order_id"])
                        self.audit_log.log_snapshot(pos["symbol"], None, "flat_by_cancel_entry", {})
                    continue
                price = float(self.broker.get_current_price(pos["symbol"]))
                logger.info("%s: its market closes in %.0f min - closing %s %g",
                            pos["symbol"], mins[1], pos["direction"], pos["quantity"])
                worker._software_exit(pos, "flat_by_close", price)
            except Exception:
                logger.exception("Flat-before-close failed for %s", pos["symbol"])

    def _worker_for(self, symbol: str) -> SymbolWorker:
        """Existing worker for this symbol (matched with or without the
        market prefix), or a monitor-only one so the position is managed."""
        if symbol in self.workers:
            return self.workers[symbol]
        key = base_symbol(symbol)
        for sym, worker in self.workers.items():
            if base_symbol(sym) == key:
                return worker
        return self._monitor_worker(symbol)

    def process_manual_orders(self):
        try:
            requests = self.audit_log.claim_manual_orders()
        except Exception:
            logger.exception("Could not read manual order queue")
            return
        for req in requests:
            try:
                worker = self._worker_for(req["symbol"])
                if req["action"] == "CLOSE":
                    result = worker.manual_close()
                else:
                    result = worker.manual_entry(req["action"], int(req["quantity"] or worker.quantity))
                ok = True
            except Exception as exc:
                logger.exception("Manual %s %s failed", req["action"], req["symbol"])
                result, ok = str(exc), False
            logger.info("Manual %s %s: %s", req["action"], req["symbol"], result)
            self.audit_log.finish_manual_order(req["id"], ok, str(result))

    def run_forever(self):
        interval = self.config.get("polling_interval_seconds", 60)
        monitor_interval = self.config.get("trailing_stop", {}).get("check_interval_seconds", 20)
        logger.info("Starting orchestrator loop (scan interval=%ss, position monitor=%ss, follow market=%s)",
                    interval, monitor_interval, self.follow_market)
        state: dict = {}
        while True:
            try:
                self.run_cycle(state)
            except Exception:
                logger.exception("Orchestrator cycle failed")
            if not self.broker_ok:
                time.sleep(10)
            else:
                wait = min(monitor_interval, interval)
                to_candle = self.seconds_until_next_candle()
                if to_candle is not None:
                    wait = max(0.5, min(wait, to_candle))   # wake just after the candle closes
                time.sleep(wait)
