"""
Settings form schema for the dashboard (pattern borrowed from the
ensemble project, but listing only THIS agent's settings).

Each field: key (dotted path in config.yaml), label, type, options.
Types: bool, int, float, select, list (comma separated), text.
Options: min / max, options (select), nullable (empty = off), help,
scope: "any" (shared or per-broker override - default), "shared" (one
value for every broker - e.g. the shared Kronos model) or "broker"
(per-broker only: enabled, market, label).

Not editable here on purpose: the Kronos model / tokenizer / device /
offline flag (the core model setup), the static `symbols:` list (edit
config.yaml) and credentials (.env).
"""
from __future__ import annotations

TIMEFRAMES = ["MIN_1", "MIN_5", "MIN_15", "HOUR_1", "HOUR_4", "DAY"]

SECTIONS = [
    {"id": "broker", "title": "Broker", "fields": [
        ("enabled", "Run this broker", "bool", {"scope": "broker",
         "help": "Several brokers can run at once - each with its own DB, risk state and calendar"}),
        ("market", "Market", "select", {"scope": "broker", "options": [], "nullable": True,
         "help": "Empty = the market set in .env for this broker"}),
        ("label", "Display name", "text", {"scope": "broker", "nullable": True}),
        ("data_source", "Price data from", "select", {"scope": "broker", "nullable": True,
         "options": ["moomoo", "tiger", "ig", "zerodha", "angel"], "blank": "(this broker)",
         "help": "Read all candles and live prices from another broker's API - e.g. angel for "
                 "Zerodha's free plan, which has no market data. Orders stay with this broker. "
                 "Needs that broker's login in .env; restart to apply"}),
    ]},
    {"id": "schedule", "title": "Timeframe & schedule", "fields": [
        ("timeframe", "Candle timeframe", "select", {"options": TIMEFRAMES}),
        ("polling_interval_seconds", "Scan interval (s)", "int", {"min": 10, "max": 3600}),
        ("lookback_bars", "Lookback bars fed to Kronos", "int", {"min": 20, "max": 512}),
        ("trade_outside_market_hours", "Trade outside market hours", "bool", {}),
        ("schedule.follow_market", "Follow the exchange session (sleep when closed)", "bool", {}),
        ("schedule.warmup_minutes", "Warm-up before the open (min)", "int", {"min": 0, "max": 120}),
        ("schedule.notify_open_close", "Telegram at open / close", "bool", {}),
    ]},
    {"id": "kronos", "title": "Kronos confirmation", "fields": [
        ("kronos_confidence_threshold", "Min Kronos confidence", "float", {"min": 0.5, "max": 1.0,
         "help": "Share of forecast bars that must move the pattern's way"}),
        ("forecast_horizon", "Forecast horizon (bars)", "int", {"min": 1, "max": 30}),
        ("kronos.sample_count", "Sample paths averaged", "int", {"min": 1, "max": 20, "scope": "shared",
         "help": "Shared Kronos model - one value for every broker"}),
        ("kronos.temperature", "Sampling temperature", "float", {"min": 0.05, "max": 2.0, "scope": "shared"}),
        ("kronos.top_p", "Top-p", "float", {"min": 0.1, "max": 1.0, "scope": "shared"}),
    ]},
    {"id": "indicators", "title": "Indicator filter", "fields": [
        ("indicators.enabled", "Indicator filter on", "bool", {}),
        ("indicators.mode", "Mode", "select", {"options": ["filter", "log"],
         "help": "log = record only, never blocks"}),
        ("indicators.require_ma_trend", "Trade only with the MA trend", "bool", {}),
        ("indicators.ma_kind", "MA type", "select", {"options": ["ema", "sma", "wma", "hma", "vwma"]}),
        ("indicators.ma_fast", "Fast MA period", "int", {"min": 2, "max": 200}),
        ("indicators.ma_slow", "Slow MA period", "int", {"min": 3, "max": 400}),
        ("indicators.min_alignment", "Min indicator bias toward the trade", "float", {"min": -100, "max": 100}),
        ("indicators.max_rsi_for_buy", "No BUY above RSI", "float", {"min": 50, "max": 100, "nullable": True}),
        ("indicators.min_rsi_for_sell", "No SELL below RSI", "float", {"min": 0, "max": 50, "nullable": True}),
        ("indicators.min_bars", "Min closed bars for the filter", "int", {"min": 20, "max": 500}),
    ]},
    {"id": "screener", "title": "Universe & screener", "fields": [
        ("screener.enabled", "Dynamic screener", "bool", {"help": "Off = the static symbols: list in config.yaml"}),
        ("screener.candidate_symbols", "Candidate symbols", "list", {"help": "e.g. US.AAPL, SG.D05 - IG uses epics"}),
        ("screener.min_price", "Min price", "float", {"min": 0, "unit": "money"}),
        ("screener.max_price", "Max price", "float", {"min": 0, "unit": "money"}),
        ("screener.min_avg_volume", "Min avg volume (per bar)", "int", {"min": 0}),
        ("screener.atr_period", "ATR period", "int", {"min": 2, "max": 100}),
        ("screener.min_atr_pct", "Min ATR %", "float", {"min": 0, "max": 20}),
        ("screener.max_atr_pct", "Max ATR %", "float", {"min": 0, "max": 50}),
        ("screener.top_n", "Symbols traded at once", "int", {"min": 1, "max": 50}),
        ("screener.refresh_interval_minutes", "Re-screen every (min)", "int", {"min": 5, "max": 1440}),
        ("default_quantity", "Order quantity (screener picks)", "int", {"min": 1}),
        ("screener.score.mode", "Stock score", "select", {"options": ["disabled", "log", "rank"],
            "help": "log = show the score but keep ranking by ATR %; rank = pick the top scores"}),
        ("screener.score.benchmark", "Score benchmark", "text", {"nullable": True,
            "help": "e.g. US.SPY - empty uses the market's main index ETF"}),
        ("screener.score.rs_lookback", "Relative strength lookback (bars)", "int", {"min": 5, "max": 500}),
        ("screener.score.weight_relative_strength", "Weight: relative strength", "float", {"min": 0, "max": 100}),
        ("screener.score.weight_relative_volume", "Weight: relative volume", "float", {"min": 0, "max": 100}),
        ("screener.score.weight_squeeze", "Weight: squeeze", "float", {"min": 0, "max": 100}),
        ("screener.score.min_relative_strength", "Min relative strength % (rank mode)", "float",
            {"min": -100, "max": 100, "nullable": True, "help": "Drop symbols lagging the benchmark by more than this"}),
    ]},
    {"id": "orders", "title": "Orders & exits", "fields": [
        ("order_type", "Entry order type", "select", {"options": ["LIMIT", "MARKET"]}),
        ("entry_timeout_minutes", "Cancel unfilled entry after (min)", "int", {"min": 1, "max": 390, "nullable": True}),
        ("trailing_stop.enabled", "Trailing stop", "bool", {}),
        ("trailing_stop.check_interval_seconds", "Position check every (s)", "int", {"min": 5, "max": 600}),
        ("trailing_stop.trigger_points", "Trail starts after profit of", "float", {"min": 0, "unit": "price"}),
        ("trailing_stop.trail_distance", "Trail distance", "float", {"min": 0, "unit": "price"}),
        ("trailing_stop.use_atr", "Trail distance from ATR", "bool", {}),
        ("trailing_stop.atr_multiplier", "ATR multiplier", "float", {"min": 0.1, "max": 20}),
    ]},
    {"id": "risk", "title": "Risk limits", "danger": True, "fields": [
        ("risk.max_open_positions", "Max open positions", "int", {"min": 1, "max": 50}),
        ("risk.max_stake_per_point", "Max quantity per position", "int", {"min": 1}),
        ("risk.stop_loss_pct", "Stop loss % of fill", "float", {"min": 0.05, "max": 25, "nullable": True,
         "help": "Empty = use the fixed points below"}),
        ("risk.take_profit_pct", "Take profit % of fill", "float", {"min": 0.05, "max": 50, "nullable": True}),
        ("risk.stop_loss_points", "Stop loss distance", "float", {"min": 0.0001, "unit": "price"}),
        ("risk.take_profit_points", "Take profit distance", "float", {"min": 0.0001, "unit": "price"}),
        ("risk.max_daily_loss", "Max daily loss", "float", {"min": 1, "unit": "money"}),
        ("risk.min_minutes_between_trades", "Per-symbol cooldown (min)", "int", {"min": 0, "max": 10080}),
        ("risk.allow_short", "Allow short selling", "bool", {}),
        ("risk.lot_size", "Lot size", "int", {"min": 1, "max": 100000}),
    ]},
    {"id": "daytrader", "title": "Day-trader rules (empty / off = not used)", "danger": True, "fields": [
        ("risk.max_loss_per_trade", "Max loss per trade", "float", {"min": 0.01, "nullable": True, "unit": "money",
         "help": "Sizes each order: quantity = this / stop distance. Empty = fixed quantity"}),
        ("risk.max_position_pct", "Max position % of account", "float", {"min": 1, "max": 100, "nullable": True}),
        ("risk.cash_buffer_pct", "Cash buffer % of account", "float", {"min": 0, "max": 90, "nullable": True}),
        ("risk.min_reward_risk", "Min reward:risk", "float", {"min": 0.1, "max": 10, "nullable": True,
         "help": "Skip trades whose target is closer than this many times the stop"}),
        ("risk.max_trades_per_day", "Max new trades per day", "int", {"min": 1, "max": 200, "nullable": True}),
        ("risk.daily_loss_includes_open_risk", "Daily cap counts open risk at the stops", "bool", {
         "help": "Closed P&L + what open trades lose if every stop fills + the new trade's max loss"}),
        ("risk.stop_atr_multiple", "Stop = ATR \u00d7", "float", {"min": 0.1, "max": 10, "nullable": True,
         "help": "Overrides the % / price-unit stop"}),
        ("risk.atr_period", "ATR period (bars)", "int", {"min": 2, "max": 100}),
        ("risk.target_r_multiple", "Target = R \u00d7 stop distance", "float", {"min": 0.25, "max": 10, "nullable": True,
         "help": "2 = 2R. Overrides the % / price-unit target"}),
        ("risk.min_stop_pct", "Min stop distance % of price", "float", {"min": 0.001, "max": 5, "nullable": True,
         "help": "Refuse stops tighter than this (spread and noise)"}),
        ("risk.no_entries_first_minutes", "No entries in first N min of session", "int", {"min": 0, "max": 240, "nullable": True}),
        ("risk.no_entries_last_minutes", "No entries in last N min of session", "int", {"min": 0, "max": 240, "nullable": True}),
        ("risk.flat_by_minutes_before_close", "Intraday only: close all agent positions N min before close", "int",
         {"min": 1, "max": 240, "nullable": True, "help": "Day-trade mode: no overnight holds. Empty = swing (hold)"}),
        ("risk.blocked_symbols", "Never trade these symbols", "list", {"nullable": True}),
    ]},
    {"id": "sentiment", "title": "Market sentiment meter (display only)", "fields": [
        ("sentiment.enabled", "Show the sentiment meter", "bool", {}),
        ("sentiment.timeframe", "Candles used", "select", {"options": TIMEFRAMES,
         "help": "DAY = the market's trend; HOUR_1 / MIN_15 for intraday mood"}),
        ("sentiment.refresh_minutes", "Re-read every (min)", "int", {"min": 1, "max": 1440}),
        ("sentiment.adx_trend", "ADX that counts as trending", "float", {"min": 10, "max": 40}),
        ("sentiment.breadth_symbols", "Breadth: active symbols checked", "int", {"min": 0, "max": 50,
         "help": "0 = off"}),
        ("sentiment.benchmarks.US", "US benchmarks", "list", {"nullable": True,
         "help": "Empty = US.SPY, US.QQQ, US.IWM"}),
        ("sentiment.benchmarks.SG", "SGX benchmarks", "list", {"nullable": True, "help": "Empty = SG.ES3, SG.G3B"}),
        ("sentiment.benchmarks.HK", "HK benchmarks", "list", {"nullable": True, "help": "Empty = HK.02800, HK.03033"}),
        ("sentiment.benchmarks.NSE", "NSE benchmarks", "list", {"nullable": True,
         "help": "Empty = NSE.NIFTYBEES, NSE.BANKBEES, NSE.JUNIORBEES"}),
    ]},
    {"id": "safety", "title": "Connection & safety", "fields": [
        ("connection.health_check_seconds", "Broker health check every (s)", "int", {"min": 10, "max": 3600}),
        ("connection.reconnect_after_failures", "Reconnect after failed checks", "int", {"min": 1, "max": 20}),
        ("unmanaged_positions.mode", "Positions not opened by the agent", "select", {"options": ["alert", "off"]}),
        ("unmanaged_positions.close_missing_positions", "Close agent rows the broker no longer holds", "bool", {}),
        ("unmanaged_positions.allow_manual_close", "Allow closing my own holdings from the dashboard", "bool", {
            "help": "Off (default) = holdings the agent did not open are protected: no Close button, and a close request is refused"}),
    ]},
]

FIELDS = {k: {"key": k, "label": lbl, "type": t, "section": s["id"], **o}
          for s in SECTIONS for k, lbl, t, o in s["fields"]}


def schema(markets: list[str] | None = None) -> list[dict]:
    out = []
    for s in SECTIONS:
        fields = []
        for k, lbl, t, o in s["fields"]:
            f = {"key": k, "label": lbl, "type": t, "scope": o.get("scope", "any"), **o}
            if k == "market" and markets is not None:
                f["options"] = markets
            fields.append(f)
        out.append({"id": s["id"], "title": s["title"], "danger": s.get("danger", False), "fields": fields})
    return out


def coerce(key: str, value):
    """Validate and normalise one value for `key`. Raises ValueError."""
    f = FIELDS.get(key)
    if f is None:
        raise ValueError(f"{key}: not editable from the dashboard")
    t = f["type"]
    if value is None or (isinstance(value, str) and value.strip() == "" and t != "text"):
        if f.get("nullable"):
            return None
        raise ValueError(f"{f['label']}: a value is required")
    if t == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{f['label']}: must be true or false")
        return value
    if t in ("int", "float"):
        if isinstance(value, bool):
            raise ValueError(f"{f['label']}: must be a number")
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{f['label']}: must be a number") from None
        if t == "int":
            if num != int(num):
                raise ValueError(f"{f['label']}: must be a whole number")
            num = int(num)
        if "min" in f and num < f["min"]:
            raise ValueError(f"{f['label']}: must be at least {f['min']}")
        if "max" in f and num > f["max"]:
            raise ValueError(f"{f['label']}: must be at most {f['max']}")
        return num
    if t == "select":
        opts = f.get("options") or []
        if opts and value not in opts:
            raise ValueError(f"{f['label']}: one of {', '.join(map(str, opts))}")
        return value
    if t == "list":
        items = value if isinstance(value, list) else str(value).split(",")
        items = [str(x).strip().upper() for x in items if str(x).strip()]
        if not items:
            raise ValueError(f"{f['label']}: at least one entry")
        return items
    if t == "text":
        return str(value).strip() or None
    raise ValueError(f"{key}: unsupported type {t}")


def cross_check(cfg: dict) -> list[str]:
    """Rules across fields, checked on each broker's effective settings."""
    errs = []
    ind = cfg.get("indicators", {}) or {}
    if ind.get("ma_fast") and ind.get("ma_slow") and ind["ma_fast"] >= ind["ma_slow"]:
        errs.append("Fast MA period must be smaller than the slow one")
    scr = cfg.get("screener", {}) or {}
    if scr.get("min_price") is not None and scr.get("max_price") is not None and scr["min_price"] > scr["max_price"]:
        errs.append("Screener min price is above max price")
    if (scr.get("min_atr_pct") is not None and scr.get("max_atr_pct") is not None
            and scr["min_atr_pct"] >= scr["max_atr_pct"]):
        errs.append("Screener min ATR % must be below max ATR %")
    sc = scr.get("score") or {}
    if str(sc.get("mode") or "disabled") != "disabled":
        if not scr.get("enabled"):
            errs.append("Stock score needs the dynamic screener switched on")
        if not any((sc.get(k) or 0) > 0 for k in
                   ("weight_relative_strength", "weight_relative_volume", "weight_squeeze")) and any(
                k in sc for k in ("weight_relative_strength", "weight_relative_volume", "weight_squeeze")):
            errs.append("Stock score weights are all zero")
    lb, mb = cfg.get("lookback_bars"), ind.get("min_bars")
    if ind.get("enabled") and lb and mb and mb > lb:
        errs.append("Indicator min bars is larger than lookback bars - the filter would never run")
    risk = cfg.get("risk", {}) or {}
    if risk.get("stop_atr_multiple") and lb and (risk.get("atr_period") or 14) >= lb:
        errs.append("ATR period must be smaller than lookback bars for the ATR stop")
    if risk.get("max_loss_per_trade") and risk.get("max_daily_loss") \
            and risk["max_loss_per_trade"] > risk["max_daily_loss"]:
        errs.append("Max loss per trade is larger than the max daily loss - one stop-out would halt the day")
    first, last = risk.get("no_entries_first_minutes") or 0, risk.get("no_entries_last_minutes") or 0
    if first + max(last, risk.get("flat_by_minutes_before_close") or 0) >= 375:
        errs.append("Entry window leaves no time to trade (NSE's session is 375 min)")
    if risk.get("flat_by_minutes_before_close") and cfg.get("timeframe") in ("DAY", "WEEK", "MONTH"):
        errs.append("Flat-by-close needs an intraday timeframe - on daily bars every trade would close the day it opens")
    return errs
