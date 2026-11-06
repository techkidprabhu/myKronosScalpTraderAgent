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


@dataclass
class RiskState:
    daily_pnl: float = 0.0
    last_trade_time: dict = field(default_factory=dict)  # symbol -> epoch seconds
    trading_halted: bool = False
    trading_day: str = ""  # ISO date in the exchange timezone


class RiskManager:
    def __init__(self, config: RiskConfig, store=None, timezone: str = "Asia/Singapore"):
        self.config = config
        self.store = store
        self.tz = ZoneInfo(timezone)
        self.state = RiskState()
        if store is not None:
            saved = store.get_kv(STATE_KEY)
            if saved:
                self.state = RiskState(**saved)
                logger.info("Restored risk state: %s", saved)
        self._roll_day()

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
            self.state.trading_day = today
            self._save()

    def record_trade_result(self, pnl: float):
        self._roll_day()
        self.state.daily_pnl += pnl
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

    def can_trade(self, epic: str, stake: float, open_position_count: int,
                  direction: str = "BUY", has_position: bool = False) -> tuple[bool, str]:
        self._roll_day()
        if self.state.trading_halted:
            return False, "daily loss cap breached"

        if direction == "SELL" and not self.config.allow_short:
            return False, "short selling disabled (allow_short: false)"

        if has_position:
            return False, "already have a position or working order in this symbol"

        if open_position_count >= self.config.max_open_positions:
            return False, "max open positions reached"

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

    def mark_trade_opened(self, epic: str):
        self.state.last_trade_time[epic] = time.time()
        self._save()

    def bracket_distances(self, price: float | None = None) -> tuple[float, float]:
        """(stop distance, target distance) in price units for an entry at
        `price`. Percentage settings win when set and a price is given."""
        stop = self.config.stop_loss_points
        target = self.config.take_profit_points
        if price:
            if self.config.stop_loss_pct:
                stop = price * self.config.stop_loss_pct / 100
            if self.config.take_profit_pct:
                target = price * self.config.take_profit_pct / 100
        return stop, target
