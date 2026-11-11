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
import time
from datetime import datetime, timezone

import pandas as pd

from src import market_calendar as cal
from src.audit_log import AuditLog
from src import indicator_gate
from src.kronos_agent import KronosAgent
from src.market import drop_unclosed_bar, is_market_open, market_timezone
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

# Used when trailing is disabled: never engages, so compute_update only
# reports fixed stop / target crossings.
NO_TRAIL = TrailConfig(trigger_points=float("inf"), trail_distance=0.0)


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
        )
        self.trail_enabled = trail_cfg.get("enabled", False)

        # Indicator confirmation between pattern and Kronos (off unless
        # config has `indicators: {enabled: true}`).
        self.indicator_cfg = indicator_gate.IndicatorConfig.from_dict(global_cfg.get("indicators"))

        # Persisted so a restart doesn't re-evaluate (and re-trade) a bar
        # that was already processed.
        self._last_bar_key = f"last_bar:{self.symbol}"
        saved = audit_log.get_kv(self._last_bar_key)
        self._last_seen_timestamp = pd.Timestamp(saved) if saved else None

    @property
    def market(self) -> str:
        return self.broker.market

    def _market_open(self) -> bool:
        return is_market_open(self.market)

    # ------------------------------------------------------------------
    # Entry pipeline
    # ------------------------------------------------------------------

    def _exits_persist(self) -> bool:
        """True if resting exits survive the session close (GTC)."""
        return getattr(self.broker, "exit_time_in_force", "GTC") == "GTC"

    def poll_once(self):
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

        latest_ts = df.index[-1]
        if self._last_seen_timestamp is not None and latest_ts <= self._last_seen_timestamp:
            return  # no new closed bar yet
        self._last_seen_timestamp = latest_ts
        self.audit_log.set_kv(self._last_bar_key, latest_ts.isoformat())

        last_close = df["close"].iloc[-1]
        self.audit_log.log_snapshot(self.symbol, float(last_close), "scanning")

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

        try:
            confirmed, confidence = self.kronos_agent.confirm_direction(
                df, signal.direction, self.forecast_horizon
            )
        except Exception as exc:
            logger.exception("Kronos confirmation failed for %s", self.symbol)
            self.audit_log.log_decision(
                self.symbol, signal.name, signal.direction, signal.strength,
                None, None, None, None, "error", details={"error": str(exc)},
            )
            self.reporter.report_error(f"kronos_agent ({self.symbol})", str(exc))
            return

        self.audit_log.log_snapshot(
            self.symbol, float(last_close), "kronos_result",
            {"confirmed": confirmed, "confidence": confidence},
        )
        if not confirmed or confidence < self.confidence_threshold:
            reason = f"kronos confidence {confidence:.2f} below threshold {self.confidence_threshold}"
            self.audit_log.log_decision(
                self.symbol, signal.name, signal.direction, signal.strength,
                confirmed, confidence, None, reason, "rejected",
            )
            self.reporter.report_rejected(self.symbol, signal.name, reason)
            return

        allowed, risk_reason = self._risk_check(signal.direction)
        self.audit_log.log_snapshot(self.symbol, float(last_close), "risk_check", {"allowed": allowed, "reason": risk_reason})
        if not allowed:
            self.audit_log.log_decision(
                self.symbol, signal.name, signal.direction, signal.strength,
                confirmed, confidence, False, risk_reason, "rejected",
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
                confirmed, confidence, False, plan.reason, "rejected", details={"risk_plan": plan.details},
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
                details={"error": str(exc)},
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
            details={"status": "entry_submitted", "entry": float(entry_limit_price or last_close),
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
        sized = bool(self.risk_manager.config.max_loss_per_trade)
        return self.risk_manager.can_trade(
            self.symbol, None if sized else self.quantity, active_count,
            direction=direction, has_position=has_position,
            session_minutes=self._session_minutes(),
        )

    def _session_minutes(self):
        """(minutes since open, minutes to close) of today's regular session."""
        try:
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

    def _plan(self, price: float, quantity: int, df: pd.DataFrame | None = None):
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
        return self.risk_manager.plan_entry(price, quantity, atr=self._atr(df),
                                            account=account, open_risk=open_risk)

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
            if state["filled"] == 0 and self.entry_timeout_minutes:
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
            target=pos["target"], config=self.trail_config if self.trail_enabled else NO_TRAIL,
        )
        exiting = update.exit_triggered or update.target_hit

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
                state = self.broker.get_order_state(order_id)
            except Exception:
                logger.exception("Could not re-check order %s for %s", order_id, self.symbol)
                return  # unknown state - try again next cycle rather than risk a double sell
            if state["active"]:
                logger.warning("Order %s for %s still active after cancel - retrying next cycle", order_id, self.symbol)
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
        recent = self.audit_log.get_kv(RECENT_CLOSES_KEY, {}) or {}
        recent = {k: v for k, v in recent.items() if time.time() - v < 3600}
        recent[base_symbol(self.symbol)] = time.time()
        self.audit_log.set_kv(RECENT_CLOSES_KEY, recent)
        self.risk_manager.record_trade_result(pnl)
        deal_id = str(pos.get("entry_order_id"))
        self.audit_log.log_trade_result(
            self.symbol, deal_id, pnl,
            {"reason": reason, "entry": pos["entry_price"], "exit": avg_exit, "quantity": exit_qty},
        )
        self.audit_log.log_snapshot(self.symbol, avg_exit, "position_closed", {"reason": reason, "pnl": pnl})
        self.reporter.report_trade_closed(self.symbol, deal_id, pnl)
        if self.risk_manager.state.trading_halted:
            self.reporter.send("<b>Daily loss cap breached</b> - no new entries until tomorrow")


class Orchestrator:
    def __init__(self, config: dict, broker=None, *, broker_name: str = "moomoo", adapter: str = "moomoo",
                 market: str | None = None, label: str | None = None, audit_log=None,
                 kronos_agent=None, reporter=None):
        """One trading loop for one broker. With several brokers configured,
        src/multi_broker.py builds one of these per broker - each with its
        own connection, audit DB, risk state and market calendar - and
        shares a single offline Kronos model between them."""
        self.config = config
        self.broker_name, self.adapter, self.market_override = broker_name, adapter, market
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
        if screener_cfg.get("enabled"):
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
        market_open = is_market_open(self.broker.market, now) if now else is_market_open(self.broker.market)
        now = now or datetime.now(timezone.utc)
        if market_open:
            return "active", None
        nxt = cal.next_open(self.exchange, now).timestamp() if self.exchange else None
        if nxt is not None and nxt - now.timestamp() <= self.warmup_minutes * 60:
            return "warmup", nxt
        return "sleeping", nxt

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
        if not selected:
            logger.warning("Screener returned no candidates - keeping existing symbol list")
            return
        self._set_workers(selected)
        self._last_screener_run = time.time()
        report = getattr(self.screener, "last_report", None)
        if isinstance(report, dict):
            try:
                self.audit_log.set_kv(SCREENER_REPORT_KEY, report)
            except Exception:
                logger.exception("Could not publish the screener report")

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

        if self.screener and awake and now - self._last_screener_run >= self.screener_refresh_seconds:
            self._refresh_symbols_from_screener()

        if phase in ("active", "always") and now - state.get("last_full_poll", 0.0) >= interval:
            for worker in list(self.workers.values()):
                try:
                    worker.poll_once()
                except Exception:
                    logger.exception("Worker error for %s", worker.symbol)
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

    def flat_by_close(self):
        """Day-trade mode: in the last N minutes of the session, close every
        position the agent manages and cancel unfilled entries, win or lose.
        Holdings the agent doesn't manage are left alone."""
        minutes = getattr(self.risk_manager.config, "flat_by_minutes_before_close", None)
        if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or minutes <= 0:
            return
        try:
            mins = cal.session_minutes(cal.exchange_for_market(self.broker.market))
        except Exception:
            logger.exception("Session clock unavailable - flat-by-close skipped")
            return
        if mins is None or mins[1] > minutes:
            return
        for pos in self.audit_log.get_active_positions():
            worker = self._worker_for(pos["symbol"])
            try:
                if pos["status"] == "pending_entry":
                    if pos.get("entry_order_id"):
                        self.broker.cancel_order(pos["entry_order_id"])
                        self.audit_log.log_snapshot(pos["symbol"], None, "flat_by_cancel_entry", {})
                    continue
                price = float(self.broker.get_current_price(pos["symbol"]))
                logger.info("%s: flat by close (%.0f min left) - closing %s %g",
                            pos["symbol"], mins[1], pos["direction"], pos["quantity"])
                worker._software_exit(pos, "flat_by_close", price)
            except Exception:
                logger.exception("Flat-by-close failed for %s", pos["symbol"])

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
                time.sleep(min(monitor_interval, interval))
