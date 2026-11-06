"""
Minimal Telegram reporting - plain requests calls against the Bot API,
no extra dependency needed beyond `requests`.

Messages use HTML parse mode. (Legacy Markdown mode rejects any message
containing an unpaired `_`, e.g. pattern names like `inside_bar`, so
those notifications were silently dropped.) Escape every interpolated
value with `esc`.
"""
import logging
import os
from html import escape as esc

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramReporter:
    def __init__(self, prefix: str = ""):
        # prefix tags every message with its broker when several run at once,
        # e.g. "[Tiger] " (already HTML-escaped by the caller)
        self.prefix = prefix
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and not self.token.startswith("your_"))
        self.url = API_BASE.format(token=self.token) if self.enabled else ""

    def send(self, text: str):
        if not self.enabled:
            logger.debug("Telegram notifications disabled or placeholder token configured.")
            return
        try:
            resp = requests.post(
                self.url,
                data={"chat_id": self.chat_id, "text": self.prefix + text, "parse_mode": "HTML"},
                timeout=10,
            )
            resp.raise_for_status()
        except requests.RequestException:
            logger.exception("Failed to send Telegram message")

    def report_trade_opened(self, epic: str, direction: str, size: float,
                             entry: float, stop: float, limit: float,
                             pattern: str, kronos_confidence: float,
                             indicators: str | None = None):
        text = (
            f"<b>Entry submitted</b> <code>{esc(epic)}</code>\n"
            f"Direction: <b>{esc(direction)}</b>  Size: {size}\n"
            f"Entry: {entry}  SL: {stop}  TP: {limit}\n"
            f"Pattern: {esc(pattern)}  Kronos confidence: {kronos_confidence:.2f}"
        )
        if indicators:
            text += f"\nIndicators: {esc(indicators)}"
        self.send(text)

    def report_trade_closed(self, epic: str, deal_id: str, pnl: float):
        emoji = "✅" if pnl >= 0 else "❌"
        self.send(f"{emoji} <b>Trade closed</b> <code>{esc(epic)}</code> ({esc(str(deal_id))})\nP&amp;L: {pnl:.2f}")

    def report_rejected(self, epic: str, pattern: str, reason: str):
        self.send(f"<b>Signal skipped</b> <code>{esc(epic)}</code>\nPattern: {esc(pattern)}\nReason: {esc(reason)}")

    def report_daily_summary(self, pnl: float, trades: int):
        self.send(f"<b>Daily summary</b>\nTrades: {trades}\nNet P&amp;L: {pnl:.2f}")

    def report_error(self, context: str, error: str):
        self.send(f"⚠️ <b>Error</b> in {esc(context)}:\n{esc(error)}")
