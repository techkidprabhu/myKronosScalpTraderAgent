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

TIMEFRAMES = ["MIN_1", "MIN_3", "MIN_5", "MIN_15", "HOUR_1", "HOUR_4", "DAY"]

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
        ("trading_day_timezone", "Trading day time zone", "text", {"nullable": True, "timezone": True,
         "help": "Where 'today' starts for P&L, the daily loss limit and profit target, e.g. Asia/Singapore. "
                 "Blank = the market's own time zone"}),
        ("candle_close_poll", "Also scan right after each candle closes", "bool", {}),
        ("candle_close_delay_seconds", "Wait after the close for the finished candle (s)", "int", {"min": 1, "max": 30}),
        ("lookback_bars", "Lookback bars fed to Kronos", "int", {"min": 20, "max": 512}),
        ("trade_outside_market_hours", "Trade outside market hours", "bool", {}),
        ("schedule.follow_market", "Follow the exchange session (sleep when closed)", "bool", {}),
        ("schedule.warmup_minutes", "Warm-up before the open (min)", "int", {"min": 0, "max": 120}),
        ("schedule.sleep_monitor_seconds", "While closed, check positions every (s)", "int", {"min": 30, "max": 3600}),
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
        ("kronos_scalp.confidence_method", "How confidence is measured", "select", {"options": ["bars", "paths"],
         "help": "bars (default) = share of the averaged forecast's candles beyond the last close, the original rule. "
                 "paths = share of separate forecasts that end beyond it"}),
        ("kronos_scalp.paths", "Separate forecasts for 'paths'", "int", {"min": 2, "max": 50,
         "help": "Default 10. More = steadier answer, slower (each one is a full forecast)"}),
        ("kronos_scalp.shadow_paths", "Also record the 'paths' confidence", "bool", {
         "help": "Recorded only, never decides a trade. Kronos Model Lab scores it next to the rule in use"}),
        ("kronos_scalp.min_move_pct", "Min forecast move (%)", "float", {"min": 0.0, "max": 5.0,
         "help": "A forecast close only counts if it's at least this far beyond the last close. "
                 "Set it to cover spread + fees. Blank or 0 = off (default)"}),
    ]},
    {"id": "selflearning", "title": "Self-learning (fine-tuning Kronos)", "fields": [
        ("finetune.enabled", "Collect candles automatically", "bool", {"scope": "shared",
            "help": "Saves candles of screened and traded symbols at each market close (data/training)"}),
        ("finetune.train_time", "Train at (local HH:MM)", "text", {"scope": "shared", "nullable": True,
            "help": "Empty = only when you start it. Runs only while all markets are closed"}),
        ("finetune.train_days", "Training days", "list", {"scope": "shared", "lowercase": True,
            "help": "mon, tue, wed, thu, fri, sat, sun"}),
        ("finetune.max_steps", "Training steps (max)", "int", {"scope": "shared", "min": 10, "max": 20000}),
        ("finetune.max_minutes", "Time limit per run (min)", "int", {"scope": "shared", "min": 5, "max": 1440}),
        ("finetune.collect_on_close", "Collect when each market closes", "bool", {"scope": "shared"}),
        ("finetune.bars_per_symbol", "Candles requested per symbol", "int", {"scope": "shared", "min": 200, "max": 20000}),
        ("finetune.max_symbols", "Symbols collected per broker (max)", "int", {"scope": "shared", "min": 1, "max": 1000}),
        ("finetune.batch_size", "Batch size", "int", {"scope": "shared", "min": 1, "max": 256,
            "help": "Windows per training step on this computer (the GPU kit uses its own, larger default)"}),
        ("finetune.learning_rate", "Learning rate", "float", {"scope": "shared", "min": 0.000001, "max": 0.01}),
        ("finetune.signal_fraction", "Share of each batch from the agent's signals", "float", {"scope": "shared", "min": 0, "max": 0.9,
            "help": "0.3 = 30% of windows end where the agent asked Kronos for a verdict"}),
        ("finetune.val_fraction", "Newest candles held out for the comparison", "float", {"scope": "shared", "min": 0.05, "max": 0.5,
            "help": "0.15 = the newest 15% are never trained on"}),
        ("finetune.train_tokenizer", "Also fine-tune the tokenizer", "bool", {"scope": "shared", "help": "Needs far more data; usually off"}),
        ("finetune.device", "Train on", "select", {"scope": "shared", "options": ["auto", "cpu", "cuda"],
            "help": "auto = an NVIDIA GPU if PyTorch can use one, else the CPU"}),
        ("finetune.keep_runs", "Training runs kept on this computer", "int", {"scope": "shared", "min": 1, "max": 50,
            "help": "Older runs' weights are deleted (the model in use and the shadow model are always kept)"}),
        ("finetune.releases_repo", "Saved versions repository (owner/name)", "text", {"scope": "shared", "nullable": True,
            "help": "GitHub repository for Save as version; empty = this folder's GitHub origin. Needs GITHUB_TOKEN in .env"}),
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
    {"id": "skills", "title": "Your rules (skills.md)", "fields": [
        ("skills.enabled", "Check signals against your rules", "bool", {
            "help": "Runs after the indicator filter, before Kronos. Rules can only stop a signal. "
                    "Edit them on Strategies & Tactics -> Your rules. Restart the agent after switching on or off"}),
        ("skills.mode", "Mode", "select", {"options": ["log", "filter"],
         "help": "log = record what a rule would stop, never stops anything; filter = stops the signal"}),
        ("skills.file", "Rules file", "text", {"nullable": True,
         "help": "A .md file inside the project; empty = config/skills.md"}),
    ]},
    {"id": "screener", "title": "Universe & screener", "fields": [
        ("screener.enabled", "Dynamic screener", "bool", {"help": "Off = the static symbols: list in config.yaml"}),
        ("screener.candidate_symbols", "Candidate symbols", "list", {"help": "e.g. US.AAPL, SG.D05 - IG uses epics"}),
        ("screener.sources", "Start from moomoo Market Movers lists", "list", {"lowercase": True, "nullable": True,
            "help": "moomoo US/HK only: top_gainers, top_losers, hot_list, movers_5min, short_selling. "
                    "Merged with the candidate symbols above; empty = off"}),
        ("screener.per_source", "Symbols taken from each list", "int", {"min": 1, "max": 200}),
        ("screener.min_price", "Min price", "float", {"min": 0, "unit": "money"}),
        ("screener.max_price", "Max price", "float", {"min": 0, "unit": "money"}),
        ("screener.min_avg_volume", "Min avg volume (per bar)", "int", {"min": 0}),
        ("screener.atr_period", "ATR period", "int", {"min": 2, "max": 100}),
        ("screener.min_atr_pct", "Min ATR %", "float", {"min": 0, "max": 20}),
        ("screener.max_atr_pct", "Max ATR %", "float", {"min": 0, "max": 50}),
        ("screener.top_n", "Symbols traded at once", "int", {"min": 1, "max": 50}),
        ("screener.refresh_interval_minutes", "Re-screen every (min)", "int", {"min": 5, "max": 1440}),
        ("default_quantity", "Order quantity (screener picks)", "int", {"min": 1}),
        ("focus_mode", "Single-stock focus (live switch)", "bool", {"live": "focus", "scope": "shared",
         "help": "On = trade only the one cleanest volume breakout among the screener's picks, one position at a time. "
                 "Takes effect on the next candle, no restart. Same switch as on the Command Center"}),
        ("screener.focus.level_bars", "Focus: breakout level = high of last N candles", "int", {"min": 5, "max": 200}),
        ("screener.focus.volume_period", "Focus: average volume over (candles)", "int", {"min": 5, "max": 200}),
        ("screener.focus.min_volume_ratio", "Focus: min volume spike (x average)", "float", {"min": 1, "max": 20, "help": "1.5-3x is the usual range"}),
        ("screener.focus.strong_volume_ratio", "Focus: spike that scores full marks (x)", "float", {"min": 1, "max": 50}),
        ("screener.focus.min_break_pct", "Focus: min close beyond the level (%)", "float", {"min": 0, "max": 5}),
        ("screener.focus.min_accumulation", "Focus: min up-volume / down-volume", "float", {"min": 0, "max": 10, "help": "Volume growing on rises and fading on dips; 1.0 = balanced"}),
        ("screener.focus.require_vwap", "Focus: price must be on the right side of VWAP", "bool", {}),
        ("screener.focus.require_ema", "Focus: EMA fast/slow must agree", "bool", {}),
        ("screener.focus.rsi_min", "Focus: RSI at least", "float", {"min": 0, "max": 100, "help": "Shorts use 100 minus these"}),
        ("screener.focus.rsi_max", "Focus: RSI at most", "float", {"min": 0, "max": 100}),
        ("screener.focus.sides", "Focus: directions", "select", {"options": ["long", "both"], "help": "both = also breakdowns below support (needs Allow short selling for stocks)"}),
        ("screener.focus.prime_minutes", "Focus: pick only in the first N minutes after the open", "int", {"min": 0, "max": 600, "help": "Markets with a session (stocks, index CFDs). 0 = any time. 24-hour markets have no window"}),
        ("screener.focus.max_wait_bars", "Focus: drop it after N candles with no entry", "int", {"min": 1, "max": 100}),
        ("screener.focus.fade_ratio", "Focus: drop it when volume fades below (x average)", "float", {"min": 0, "max": 5}),
        ("screener.focus.entry_on_breakout", "Focus: the breakout candle itself is a signal (Kronos still confirms)", "bool", {}),
        ("screener.focus.stop_below_level", "Focus: stop just beyond the breakout level", "bool", {}),
        ("screener.focus.stop_buffer_pct", "Focus: stop buffer past the level (%)", "float", {"min": 0, "max": 5}),
        ("screener.focus.target_r", "Focus: target (x the risk)", "float", {"min": 0.5, "max": 10}),
        ("screener.score.mode", "Stock score", "select", {"options": ["disabled", "log", "rank"],
            "help": "log = show the score but keep ranking by ATR %; rank = pick the top scores"}),
        ("screener.score.benchmark", "Score benchmark", "text", {"nullable": True,
            "help": "e.g. US.SPY - empty uses the market's main index ETF"}),
        ("screener.score.rs_lookback", "Relative strength lookback (bars)", "int", {"min": 5, "max": 500}),
        ("screener.score.rvol_recent", "Relative volume: recent bars", "int", {"min": 1, "max": 50}),
        ("screener.score.rvol_period", "Relative volume: usual over (bars)", "int", {"min": 5, "max": 200}),
        ("screener.score.squeeze_period", "Squeeze: Bollinger period", "int", {"min": 5, "max": 200}),
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
        ("trailing_stop.trigger_pct", "Trail starts after profit of (% of entry)", "float", {"min": 0.01, "max": 50,
         "nullable": True, "help": "Replaces the price-units trigger above when set, e.g. 0.5"}),
        ("trailing_stop.trail_pct", "Trail distance (% of the best price)", "float", {"min": 0.01, "max": 50,
         "nullable": True, "help": "Replaces the price-units distance above when set, e.g. 0.4. ATR mode still wins"}),
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
        ("risk.min_minutes_between_trades", "Per-symbol cooldown (min)", "int", {"min": 0, "max": 10080}),
        ("risk.allow_short", "Allow short selling", "bool", {}),
        ("risk.lot_size", "Lot size", "int", {"min": 1, "max": 100000}),
    ]},
    {"id": "around", "title": "IG around the clock (broker market IG24)", "fields": [
        ("screener.around_the_clock.markets", "Markets to scan", "list", {"lowercase": True,
         "help": "forex, indices, commodities, crypto. Used when the IG broker's market is IG24 and its screener is on"}),
        ("screener.around_the_clock.max_watch", "Markets streamed and scanned (max)", "int", {"min": 1, "max": 40,
         "help": "IG allows 40 streaming subscriptions; markets with an open trade are always kept"}),
        ("screener.around_the_clock.refresh_minutes", "Re-rank every (min)", "int", {"min": 2, "max": 240}),
        ("screener.around_the_clock.max_spread_pct", "Max spread (% of price)", "float", {"min": 0.001, "max": 5,
         "help": "Forex, indices and commodities, e.g. 0.2"}),
        ("screener.around_the_clock.crypto_max_spread_pct", "Max spread for crypto (% of price)", "float", {"min": 0.01, "max": 10,
         "help": "Crypto CFD spreads are wider, often 0.5-1%"}),
        ("screener.around_the_clock.min_range_pct", "Min range today (% of price)", "float", {"min": 0, "max": 20,
         "help": "Skip markets that have hardly moved today"}),
        ("screener.around_the_clock.seed_bars", "History candles fetched once per new market", "int", {"min": 30, "max": 500,
         "help": "Only the first time a market is streamed (IG's weekly history allowance is 10,000 points)"}),
        ("screener.around_the_clock.universe_refresh_hours", "Re-discover IG's markets every (h)", "int", {"min": 1, "max": 720}),
        ("screener.around_the_clock.max_nav_nodes", "Market-menu pages read per discovery", "int", {"min": 10, "max": 500,
         "help": "IG allows 30 a minute; 120 takes about 4 minutes in the background"}),
    ]},
    {"id": "scalp", "title": "Scalp trade management", "fields": [
        ("scalp.max_hold_minutes", "Time stop: close a trade after (min)", "int", {"min": 1, "max": 390, "nullable": True,
         "help": "A trade that hits neither its stop nor its target in this time is closed at the market. Empty = off"}),
        ("scalp.breakeven_at_r", "Move the stop to break-even at (R in profit)", "float", {"min": 0.1, "max": 10, "nullable": True,
         "help": "R = the trade's initial risk. 1 = once the trade is up as much as it risked. Empty = off"}),
        ("scalp.breakeven_offset_pct", "Break-even stop beyond the entry by (%)", "float", {"min": 0, "max": 1,
         "help": "So a break-even exit still covers fees, e.g. 0.02"}),
    ]},
    {"id": "daily", "title": "Daily limits (loss cap, profit target, per stock)", "fields": [
        ("risk.max_daily_loss", "Max daily loss", "float", {"min": 1, "unit": "money"}),
        ("risk.daily_loss_includes_open_risk", "Daily cap counts open risk at the stops", "bool", {
         "help": "Closed P&L + what open trades lose if every stop fills + the new trade's max loss"}),
        ("risk.daily_profit_target_pct", "Daily profit target (% of account)", "float", {"min": 0.01, "max": 100,
         "nullable": True, "help": "No new trades for the rest of the day once today's closed profit reaches this % of the "
                                   "start-of-day account value. Analysis, Kronos checks and data collection continue; "
                                   "open positions keep their stops and targets"}),
        ("risk.daily_profit_target", "Daily profit target (amount)", "float", {"min": 1, "unit": "money", "nullable": True,
         "help": "The same as an amount; with both set, the lower one applies"}),
        ("risk.max_trades_per_day", "Max new trades per day", "int", {"min": 1, "max": 200, "nullable": True}),
        ("risk.max_symbol_loss_per_day", "Per stock: stop after losing", "float", {"min": 1, "unit": "money",
         "nullable": True, "help": "No new entries in a stock once its closed trades today lost this much; others keep trading"}),
        ("risk.max_losses_per_symbol_per_day", "Per stock: stop after N losing trades", "int", {"min": 1, "max": 50,
         "nullable": True, "help": "No new entries in a stock after this many losing trades in it today"}),
    ]},
    {"id": "daytrader", "title": "Day-trader rules (empty / off = not used)", "danger": True, "fields": [
        ("risk.max_loss_per_trade", "Max loss per trade", "float", {"min": 0.01, "nullable": True, "unit": "money",
         "help": "Sizes each order: quantity = this / stop distance. Empty = fixed quantity"}),
        ("risk.max_position_pct", "Max position % of account", "float", {"min": 1, "max": 100, "nullable": True}),
        ("risk.cash_buffer_pct", "Cash buffer % of account", "float", {"min": 0, "max": 90, "nullable": True}),
        ("risk.min_reward_risk", "Min reward:risk", "float", {"min": 0.1, "max": 10, "nullable": True,
         "help": "Skip trades whose target is closer than this many times the stop"}),
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
         {"min": 1, "max": 240, "nullable": True, "help": "Scalps: close every agent trade this long before the close. Empty = hold overnight (not for scalping)"}),
        ("risk.blocked_symbols", "Never trade these symbols", "list", {"nullable": True}),
    ]},
    {"id": "cpr", "title": "CPR levels & scorecard (watching only)", "fields": [
        ("cpr.enabled", "Collect daily candles for CPR", "bool", {
         "help": "Once a day per watched symbol. Shows the Central Pivot Range and a scorecard of how past "
                 "sessions went. Nothing trades on it yet"}),
        ("cpr.backfill_days", "History fetched the first time (days)", "int", {"min": 25, "max": 250,
         "help": "After that only the last 5 days are fetched each day"}),
        ("cpr.width_lookback", "Narrow / wide judged against (sessions)", "int", {"min": 5, "max": 100}),
        ("cpr.narrow_pct", "Narrow = width at or below percentile", "float", {"min": 1, "max": 50}),
        ("cpr.wide_pct", "Wide = width at or above percentile", "float", {"min": 50, "max": 99}),
        ("cpr.adr_days", "Average daily range over (days)", "int", {"min": 5, "max": 60}),
    ]},
    {"id": "sentiment", "title": "Market sentiment meter (display only)", "fields": [
        ("sentiment.enabled", "Show the sentiment meter", "bool", {}),
        ("sentiment.timeframe", "Candles used", "select", {"options": TIMEFRAMES,
         "help": "MIN_3 = the mood on the scalping chart; MIN_15 / HOUR_1 for the intraday trend; DAY = the market's trend"}),
        ("sentiment.lookback_bars", "Candles read per benchmark", "int", {"min": 60, "max": 1000}),
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
        ("connection.connect_timeout_seconds", "Give up a connection attempt after (s)", "int", {"min": 5, "max": 300}),
        ("unmanaged_positions.check_interval_seconds", "Check for positions not opened by the agent every (s)", "int", {"min": 30, "max": 3600}),
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
    if f.get("live"):
        raise ValueError(f"{f['label']}: switched live, not saved to config.yaml")
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
        case = str.lower if f.get("lowercase") else str.upper      # symbols upper, list names lower
        items = [case(str(x).strip()) for x in items if str(x).strip()]
        if not items:
            raise ValueError(f"{f['label']}: at least one entry")
        return items
    if t == "text":
        text = str(value).strip() or None
        if text and f.get("timezone"):
            from zoneinfo import ZoneInfo
            try:
                ZoneInfo(text)
            except Exception:  # noqa: BLE001
                raise ValueError(f"{f['label']}: unknown time zone {text!r} - use a name like Asia/Singapore")
        return text
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
    known = {"top_gainers", "top_losers", "hot_list", "movers_5min", "short_selling"}
    bad = [s for s in (scr.get("sources") or []) if str(s).strip().lower() not in known]
    if bad:
        errs.append(f"Unknown market list(s): {', '.join(map(str, bad))} (use {', '.join(sorted(known))})")
    sk = (cfg.get("skills") or {}).get("file")
    if sk and not str(sk).lower().endswith(".md"):
        errs.append("The rules file must be a .md file")
    if scr.get("sources") and not scr.get("enabled"):
        errs.append("Market lists need the dynamic screener switched on")
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
