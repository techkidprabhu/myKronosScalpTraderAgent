"""
Hard-coded risk rules. Nothing in this file is influenced by pattern
detection or Kronos output - it is the last, non-overridable gate before
any order reaches MoomooClient.create_entry. Keep it that way: extend
these checks in plain Python, never by asking a model to approve/deny.

State (daily P&L, halt flag, per-symbol cooldowns) is persisted through
an optional `store` (AuditLog's kv table) so that restarting the agent
can't reset the daily loss cap or the cooldown between trades.
"""
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from src.brokers.base import base_symbol as _base

logger = logging.getLogger(__name__)

STATE_KEY = "risk_state"


@dataclass
class RiskConfig:
    max_open_positions: int
    max_stake_per_point: float  # max order quantity per position
    stop_loss_points: float
    take_profit_points: float
    max_daily_loss: float
    min_minutes_between_trades: int
    # Short selling needs a margin account (and borrow) at moomoo.
    # With this off, SELL signals are rejected instead of opening shorts.
    allow_short: bool = False
    # Quantities must be a whole multiple of this (US = 1, SGX = 100, HK varies by stock).
    lot_size: int = 1
    # Optional: stop/target as a percentage of the entry price (0.5 = 0.5%).
    # When set, these replace stop_loss_points / take_profit_points, so a
    # screener mixing $20 and $500 stocks gets the same relative risk on each.
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    # ---- Day-trader rules (all off by default: None / False / empty) ----
    # Stop as K x ATR of the signal bars; wins over pct / points when set.
    stop_atr_multiple: float | None = None
    atr_period: int = 14
    # Target as R x the stop distance (2.0 = 2R); wins over pct / points.
    target_r_multiple: float | None = None
    # Refuse a stop closer than this % of the price - too tight to survive
    # the spread and normal noise (the day-trader skill uses 0.05).
    min_stop_pct: float | None = None
    # Refuse trades whose target / stop distance is below this ratio.
    min_reward_risk: float | None = None
    # Risk-based sizing: quantity = this / stop distance (in account
    # currency), capped by max_stake_per_point, max_position_pct and the
    # cash buffer. When unset, the fixed configured quantity is used.
    max_loss_per_trade: float | None = None
    max_position_pct: float | None = None    # % of account value in one position
    cash_buffer_pct: float | None = None     # % of account value always left in cash
    max_trades_per_day: int | None = None    # new entries per exchange day
    # Count the open risk at the stops (plus the new trade's) against
    # max_daily_loss, not just closed P&L.
    daily_loss_includes_open_risk: bool = False
    blocked_symbols: list = field(default_factory=list)
    # Entry window: no new entries in the first / last N minutes of the
    # regular session (1 pm early closes and holidays handled).
    no_entries_first_minutes: int | None = None
    no_entries_last_minutes: int | None = None
    # Day-trade mode: close every agent position N minutes before the close.
    flat_by_minutes_before_close: int | None = None
    # Per-stock limits for the day (exchange day): no new entries in a symbol
    # once its closed trades today have lost this much in total, or after this
    # many losing trades in it. Other symbols keep trading. Exits never blocked.
    max_symbol_loss_per_day: float | None = None
    max_losses_per_symbol_per_day: int | None = None
    # Daily profit target: once today's closed P&L reaches this (an amount in
    # the account currency, or % of the account value at the start of the
    # day), no new entries until tomorrow. Everything else keeps running:
    # patterns, indicators, your rules and Kronos are still evaluated and
    # logged, candles are still collected, and open positions keep their
    # stop, target and trailing stop.
    daily_profit_target: float | None = None
    daily_profit_target_pct: float | None = None

    @property
    def plans_distances(self) -> bool:
        """Stop / target are fixed at signal time (ATR or R based), so the
        bracket keeps those distances when re-anchored to the fill."""
        return bool(self.stop_atr_multiple) or bool(self.target_r_multiple)


@dataclass
class EntryPlan:
    """What the agent intends to send, after sizing and the trade-level
    checks. `allowed` False means skip the trade, with `reason`."""
    allowed: bool
    reason: str
    quantity: int = 0
    stop_distance: float = 0.0
    target_distance: float = 0.0
    risk_per_share: float = 0.0
    max_loss: float = 0.0
    reward_risk: float = 0.0
    sized_by: str = "fixed quantity"
    details: dict = field(default_factory=dict)


@dataclass
class RiskState:
    daily_pnl: float = 0.0
    last_trade_time: dict = field(default_factory=dict)  # symbol -> epoch seconds
    trading_halted: bool = False
    trading_day: str = ""  # ISO date in the exchange timezone
    trades_today: int = 0
    symbol_pnl: dict = field(default_factory=dict)      # symbol -> closed P&L today
    symbol_losses: dict = field(default_factory=dict)   # symbol -> losing trades today
    start_equity: float | None = None                   # account value at the start of the day
    profit_locked: bool = False                         # daily profit target reached
    profit_locked_at: float | None = None               # the target it was reached against


class RiskManager:
    def __init__(self, config: RiskConfig, store=None, timezone: str = "Asia/Singapore"):
        self.config = config
        self.store = store
        self.tz = ZoneInfo(timezone)
        self.state = RiskState()
        if store is not None:
            saved = store.get_kv(STATE_KEY)
            if saved:
                known = RiskState.__dataclass_fields__
                self.state = RiskState(**{k: v for k, v in saved.items() if k in known})
                logger.info("Restored risk state: %s", saved)
        self._roll_day()
        self._recheck_halt()
        self._check_profit_target()      # a changed target applies straight after a restart

    def _recheck_halt(self):
        """Keep the halt in step with the configured cap. A restart alone
        never lifts it (today's loss is still over the cap), but if the cap
        has been raised above today's loss, trading may resume - and if it
        has been lowered below it, trading stops."""
        over = self.state.daily_pnl <= -abs(self.config.max_daily_loss)
        if self.state.trading_halted and not over:
            logger.warning("Daily loss cap is now %.2f and today's P&L is %.2f - trading halt lifted",
                           self.config.max_daily_loss, self.state.daily_pnl)
            self.state.trading_halted = False
            self._save()
        elif over and not self.state.trading_halted:
            logger.warning("Today's P&L %.2f is beyond the daily loss cap %.2f - trading halted",
                           self.state.daily_pnl, self.config.max_daily_loss)
            self.state.trading_halted = True
            self._save()

    def _save(self):
        if self.store is not None:
            self.store.set_kv(STATE_KEY, asdict(self.state))

    def _roll_day(self):
        today = datetime.now(self.tz).date().isoformat()
        if self.state.trading_day != today:
            if self.state.trading_day:
                logger.info("New trading day %s - resetting daily P&L (was %.2f)", today, self.state.daily_pnl)
            self.state.daily_pnl = 0.0
            self.state.trading_halted = False
            self.state.trades_today = 0
            self.state.symbol_pnl = {}
            self.state.symbol_losses = {}
            self.state.start_equity = None
            self.state.profit_locked = False
            self.state.profit_locked_at = None
            self.state.trading_day = today
            self._save()

    def record_trade_result(self, pnl: float, symbol: str | None = None):
        self._roll_day()
        self.state.daily_pnl += pnl
        if symbol:
            key = _base(symbol)
            self.state.symbol_pnl[key] = round(self.state.symbol_pnl.get(key, 0.0) + pnl, 6)
            if pnl < 0:
                self.state.symbol_losses[key] = self.state.symbol_losses.get(key, 0) + 1
            reason = self.symbol_limit_reason(symbol)
            if reason:
                logger.warning("%s: %s - no more entries in it today", symbol, reason)
        self._check_profit_target()
        if self.state.daily_pnl <= -abs(self.config.max_daily_loss):
            self.state.trading_halted = True
            logger.warning(
                "Daily loss cap breached (%.2f <= -%.2f). Trading halted.",
                self.state.daily_pnl, self.config.max_daily_loss,
            )
        self._save()

    def reset_daily(self):
        self.state.daily_pnl = 0.0
        self.state.trading_halted = False
        self._save()

    def can_trade(self, epic: str, stake: float | None, open_position_count: int,
                  direction: str = "BUY", has_position: bool = False,
                  session_minutes: tuple | None = None) -> tuple[bool, str]:
        """Cheap gates checked before a trade is planned. `stake` None means
        the quantity is decided later by risk-based sizing (plan_entry).
        `session_minutes` is (minutes since the open, minutes to the close)
        of the regular session, or None when that isn't known."""
        self._roll_day()
        if self.state.trading_halted:
            return False, (f"daily loss cap breached (today {self.state.daily_pnl:+.2f}, "
                           f"cap -{abs(self.config.max_daily_loss):.2f})")

        blocked = {_base(b) for b in (self.config.blocked_symbols or [])}
        if blocked and _base(epic) in blocked:
            return False, f"{epic} is on blocked_symbols"

        self._check_profit_target()
        if self.state.profit_locked:
            return False, (f"daily profit target reached (today {self.state.daily_pnl:+.2f}, target "
                           f"{self.profit_target():.2f}) - no new entries today; analysis continues")

        sym_reason = self.symbol_limit_reason(epic)
        if sym_reason:
            return False, sym_reason

        limit = self.config.max_trades_per_day
        if limit and self.state.trades_today >= limit:
            return False, f"max_trades_per_day reached ({self.state.trades_today} of {limit} today)"

        window = self.entry_window_reason(session_minutes)
        if window:
            return False, window

        if direction == "SELL" and not self.config.allow_short:
            return False, "short selling disabled (allow_short: false)"

        if has_position:
            return False, "already have a position or working order in this symbol"

        if open_position_count >= self.config.max_open_positions:
            return False, "max open positions reached"

        if stake is not None:
            if stake <= 0 or stake % self.config.lot_size != 0:
                return False, f"quantity {stake} is not a multiple of lot_size {self.config.lot_size}"
            if stake > self.config.max_stake_per_point:
                return False, "stake exceeds max_stake_per_point"

        last_time = self.state.last_trade_time.get(epic)
        if last_time is not None:
            elapsed_minutes = (time.time() - last_time) / 60
            if elapsed_minutes < self.config.min_minutes_between_trades:
                return False, "min_minutes_between_trades not elapsed for this symbol"

        return True, "ok"

    def wants_start_equity(self) -> bool:
        """True when the % profit target needs today's starting account value."""
        self._roll_day()
        return bool(self.config.daily_profit_target_pct) and not self.state.start_equity

    def set_start_equity(self, account_value: float):
        """Account value now, minus what was already made today = the start
        of the day. Recorded once per exchange day."""
        self._roll_day()
        if account_value and account_value > 0 and not self.state.start_equity:
            self.state.start_equity = round(float(account_value) - self.state.daily_pnl, 2)
            logger.info("Start-of-day account value: %.2f", self.state.start_equity)
            self._save()

    def profit_target(self) -> float | None:
        """Today's profit target as an amount, or None when off / not known yet."""
        c = self.config
        targets = [float(c.daily_profit_target)] if c.daily_profit_target else []
        if c.daily_profit_target_pct and self.state.start_equity:
            targets.append(self.state.start_equity * c.daily_profit_target_pct / 100)
        return min(targets) if targets else None

    def _check_profit_target(self):
        """Lock new entries once the target is reached. The lock holds for the
        rest of the day even if an open trade later gives some back; only
        switching the target off, or raising it above today's P&L, lifts it."""
        target = self.profit_target()
        reached = target is not None and target > 0 and self.state.daily_pnl >= target
        if reached and not self.state.profit_locked:
            self.state.profit_locked, self.state.profit_locked_at = True, round(target, 2)
            logger.warning("Daily profit target reached: today %+.2f >= %.2f. No new entries until tomorrow; "
                           "analysis, Kronos checks and data collection continue.", self.state.daily_pnl, target)
            self._save()
        elif self.state.profit_locked:
            off = not (self.config.daily_profit_target or self.config.daily_profit_target_pct)
            raised = (target is not None and self.state.profit_locked_at is not None
                      and target > self.state.profit_locked_at + 1e-9 and self.state.daily_pnl < target)
            if off or raised:
                self.state.profit_locked, self.state.profit_locked_at = False, None
                logger.warning("Daily profit target %s - new entries resume (today %+.2f)",
                               "switched off" if off else f"raised to {target:.2f}", self.state.daily_pnl)
                self._save()

    def symbol_limit_reason(self, epic: str) -> str | None:
        """Why this symbol may not be entered again today, or None."""
        key = _base(epic)
        cap = self.config.max_symbol_loss_per_day
        pnl = self.state.symbol_pnl.get(key, 0.0)
        if cap and pnl <= -abs(cap):
            return f"{key} lost {abs(pnl):.2f} today (per-stock limit {abs(cap):.2f})"
        n_cap = self.config.max_losses_per_symbol_per_day
        n = self.state.symbol_losses.get(key, 0)
        if n_cap and n >= n_cap:
            return f"{key} had {n} losing trade{'s' if n != 1 else ''} today (per-stock limit {n_cap})"
        return None

    def mark_trade_opened(self, epic: str):
        self._roll_day()
        self.state.last_trade_time[epic] = time.time()
        self.state.trades_today += 1
        self._save()

    def entry_window_reason(self, session_minutes: tuple | None) -> str | None:
        """Why a new entry is outside the allowed window, or None."""
        c = self.config
        if session_minutes is None or not (c.no_entries_first_minutes or c.no_entries_last_minutes
                                           or c.flat_by_minutes_before_close):
            return None
        since_open, to_close = session_minutes
        if c.no_entries_first_minutes and since_open < c.no_entries_first_minutes:
            return (f"entry window: no new entries in the first {c.no_entries_first_minutes} min "
                    f"of the session ({since_open:.0f} min in)")
        last = max(c.no_entries_last_minutes or 0, c.flat_by_minutes_before_close or 0)
        if last and to_close <= last:
            return f"entry window: no new entries in the last {last} min of the session ({to_close:.0f} min left)"
        return None

    def bracket_distances(self, price: float | None = None, atr: float | None = None) -> tuple[float, float]:
        """(stop distance, target distance) in price units for an entry at
        `price`. ATR stop > percentage > points; an R-multiple target is
        set from the final stop distance."""
        c = self.config
        stop = c.stop_loss_points
        target = c.take_profit_points
        if price:
            if c.stop_loss_pct:
                stop = price * c.stop_loss_pct / 100
            if c.take_profit_pct:
                target = price * c.take_profit_pct / 100
        if c.stop_atr_multiple and atr and atr > 0:
            stop = atr * c.stop_atr_multiple
        if c.target_r_multiple:
            target = stop * c.target_r_multiple
        return stop, target

    def plan_entry(self, price: float, default_quantity: int, *, atr: float | None = None,
                   account: dict | None = None, open_risk: float = 0.0,
                   stop_distance: float | None = None, target_distance: float | None = None,
                   spec: dict | None = None) -> EntryPlan:
        """Size the trade and run the trade-level checks: stop distance,
        reward:risk, risk-based quantity and the daily cap including open
        risk. With every day-trader rule off this returns the configured
        quantity and the usual distances, allowed.

        `spec` is the broker's own description of one unit of size, when it
        isn't "one share": value_per_point (account currency per 1.0 price
        move per 1 size), min_size and size_step. IG needs it - one forex
        contract is 100,000 of the base currency."""
        c = self.config
        fixed = stop_distance is not None and target_distance is not None
        stop_d, target_d = (float(stop_distance), float(target_distance)) if fixed \
            else self.bracket_distances(price, atr)
        details = {"price": round(price, 6), "atr": round(atr, 6) if atr else None,
                   "stop_rule": "skill" if fixed else
                   ("atr" if (c.stop_atr_multiple and atr) else "pct" if c.stop_loss_pct else "points"),
                   "target_rule": "skill" if fixed else
                   ("R" if c.target_r_multiple else "pct" if c.take_profit_pct else "points")}

        def deny(reason: str, **kw) -> EntryPlan:
            return EntryPlan(False, reason, stop_distance=stop_d, target_distance=target_d,
                             details=details, **kw)

        if c.stop_atr_multiple and not atr and not fixed:
            return deny("ATR stop configured but no ATR available for these bars")
        if stop_d <= 0 or target_d <= 0:
            return deny("stop / target distance must be above zero")
        if c.min_stop_pct and stop_d / price * 100 < c.min_stop_pct:
            return deny(f"stop {stop_d / price * 100:.3f}% from entry is tighter than min_stop_pct "
                        f"{c.min_stop_pct}% - too tight to survive the spread")
        rr = target_d / stop_d
        if c.min_reward_risk and rr + 1e-9 < c.min_reward_risk:
            return deny(f"reward:risk {rr:.2f} below min_reward_risk {c.min_reward_risk}")

        vpp = float((spec or {}).get("value_per_point") or 1.0)
        if spec is not None and not vpp > 0:
            return deny("the market's contract size is unknown - not sizing a trade blind")
        lot = max(1, int(c.lot_size or 1))
        if spec is not None:
            details.update(value_per_point=round(vpp, 6), min_size=spec.get("min_size"),
                           size_step=spec.get("size_step"), account_currency=spec.get("account_currency"),
                           unit=spec.get("unit"))
        caps = {"max quantity per position": float(c.max_stake_per_point)}
        if c.max_loss_per_trade:
            caps["max loss per trade"] = c.max_loss_per_trade / (stop_d * vpp)
        else:
            caps["configured quantity"] = float(default_quantity)
        if account and (c.max_position_pct or c.cash_buffer_pct):
            total = float(account.get("total_assets") or 0)
            cash = float(account.get("cash") if account.get("cash") is not None else total)
            if c.max_position_pct and total > 0:
                caps["max position % of account"] = total * c.max_position_pct / 100 / (price * vpp)
            if c.cash_buffer_pct and total > 0:
                caps["cash after buffer"] = max(0.0, cash - total * c.cash_buffer_pct / 100) / (price * vpp)
        elif c.max_position_pct or c.cash_buffer_pct:
            return deny("account value unavailable for max_position_pct / cash_buffer_pct")
        sized_by, raw = min(caps.items(), key=lambda kv: kv[1])
        details["caps"] = {k: round(v, 4) for k, v in caps.items()}
        if spec is not None:
            step = float(spec.get("size_step") or 0.01)
            qty = round(int(raw / step + 1e-9) * step, 6)
            qty = int(qty) if qty == int(qty) else qty
            min_size = float(spec.get("min_size") or step)
            if qty < min_size:
                at_min = min_size * stop_d * vpp
                return deny(f"too small for this market: {sized_by} allows size {raw:.4f}, under the minimum "
                            f"{min_size:g}; even the minimum would risk {at_min:.2f} {spec.get('account_currency') or ''}"
                            .rstrip() + " at the stop", sized_by=sized_by)
        else:
            qty = int(raw // lot) * lot
            if qty < lot:
                return deny(f"position too small: {sized_by} allows {raw:.2f} shares, under one lot of {lot}",
                            sized_by=sized_by)

        max_loss = qty * stop_d * vpp
        if c.daily_loss_includes_open_risk:
            self._roll_day()
            worst = self.state.daily_pnl - open_risk - max_loss
            details["daily_worst_case"] = round(worst, 2)
            if worst <= -abs(c.max_daily_loss):
                return deny(f"daily loss cap: today's P&L {self.state.daily_pnl:+.2f}, open risk {open_risk:.2f} "
                            f"and this trade's {max_loss:.2f} at the stops could reach {worst:.2f} "
                            f"(cap -{abs(c.max_daily_loss):.2f})", quantity=qty, max_loss=max_loss, sized_by=sized_by)

        return EntryPlan(True, "ok", quantity=qty, stop_distance=stop_d, target_distance=target_d,
                         risk_per_share=stop_d * vpp, max_loss=max_loss, reward_risk=rr,
                         sized_by=sized_by, details=details)


def open_risk_at_stops(positions: list[dict], value_per_point=None) -> float:
    """What the agent's pending and open positions would lose if every stop
    filled at its price. A stop already past the entry (profit locked in by
    the trailing stop) counts as zero. `value_per_point(symbol)` gives the
    account-currency value of a 1.0 move per 1 size where that isn't 1 (IG)."""
    total = 0.0
    for p in positions:
        entry, stop = float(p["entry_price"]), float(p["current_stop"])
        loss = (entry - stop) if p["direction"] == "BUY" else (stop - entry)
        per = 1.0
        if value_per_point is not None:
            try:
                per = float(value_per_point(p["symbol"]) or 1.0)
            except Exception:  # noqa: BLE001
                per = 1.0
        total += max(0.0, loss) * abs(float(p["quantity"])) * per
    return total
