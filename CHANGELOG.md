# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/):

- **MAJOR** (2.0.0) - a change that breaks existing config or behaviour
  (e.g. a renamed config key, a different order flow).
- **MINOR** (1.1.0) - a new feature that keeps existing setups working.
- **PATCH** (1.0.1) - a bug fix only.

Add new entries under **Unreleased** as you work; when you release, rename
that heading to the new version and date, and tag the commit
(see [docs/VERSIONING.md](docs/VERSIONING.md)).

## [Unreleased]

### Added
- **Live Terminal, phase 2**: the Kronos forecast is drawn as a shaded cone
  (predicted highs to lows) fanning out from the last closed candle, with
  the dashed path now joined to it. The agent's own trades are marked on
  the candles - entries (cyan; manual orders amber) and exits with P&L and
  reason - and a held position gets entry, stop / trailing stop and target
  lines that follow the live stream. The price flashes on each tick.
  `GET /api/trade_marks?symbol=` serves the marks.
- **Position cards** (Positions tab, with a Cards / Table switch): live
  price, P&L, R from the original stop, a stop -> entry -> target track
  with the price moving along it, profit locked by the stop, best price,
  time held, and Chart / Close buttons. A trailing stop that moves our way
  makes the card glow and pops a notification.
- **Command Center** (new home tab, dashboard redesign phase 1): today's
  session bar with the entry window and flat-by time, animated KPI tiles
  (today's P&L closed + open, 30-day P&L with sparkline, win rate ring and
  profit factor, trades today vs limit, risk at the stops vs the daily cap,
  account equity and cash split), the agent pipeline for today (watching ->
  bars checked -> patterns -> indicators -> Kronos -> risk -> orders) with
  rejection counts and top reasons, open-position cards showing where price
  sits between stop and target in R, and a live activity feed with toasts
  for entries, exits and errors. Respects reduced-motion settings.
- **Live updates**: `GET /api/stream` (server-sent events) pushes the
  overview and positions as soon as they change; the Positions tab uses it
  too. Polling pauses while the stream is connected and resumes as a
  fallback if it drops. `GET /api/overview` returns the same summary.

### Fixed
- Header, tab bar and sentiment bar no longer push the page sideways on
  phone-width screens.
- Flat-by-close check ignores a non-numeric setting instead of raising.

### Added
- **Day-trader risk rules** (from the Day Trader skill pack), all off by
  default: risk-based position sizing (max loss per trade, max % of
  account, cash buffer), min reward:risk, max trades per day, daily loss
  cap counting open risk at the stops, ATR-based stop and R-multiple
  target (kept through the fill), min stop distance, entry window (first /
  last N minutes, early closes handled), flat-by-close day-trade mode and a
  blocked-symbols list. They act after Kronos approves a signal and never
  block exits. New Settings section, Risk Model "Day-trader rules" card
  with a session timeline, and config cross-checks.
- `/api/risk_state` also returns `trades_today` and `open_risk`.

### Added
- **Risk Model - Live market graph**: switch the trade graph from the
  example path to real candles for any watched or held symbol, with the
  live quote updating every 3 s. For a symbol you hold it shows your actual
  entry, live P&L, the stop / target the agent is really using and how far
  price is from each; otherwise it shows where your settings would place a
  long or short entered at the live price. Levels stay draggable.
- **Risk Model screen** (new tab): an interactive diagram of one trade -
  entry, stop loss, take profit, trailing-stop trigger and trail distance on
  an example price path (long or short), using the same trailing logic as
  the agent. Drag a level or type a value to edit it. Shows risk and reward
  per trade, reward:risk, break-even win rate and how many stop-outs the
  daily cap allows, plus account limits: open-position slots (live), daily
  loss cap with today's P&L, per-symbol cooldown with symbols cooling down
  now, and order size / lot size / shorts. Saves through the Settings API
  (validated, comments kept, backup made) to shared or per-broker settings.
- `GET /api/risk_state`: today's daily P&L, halt flag, open positions and
  cooldowns from the agent's own records (read-only).
- **Audit trail prices**: new "Price at Decision -> Now" column on the
  Decision & Audit Log screen. Shows the entry price for opened orders (or
  the closed-bar price the agent evaluated for rejected signals), the live
  price now, and the % move in the signal's direction ("with" / "against
  signal").

### Fixed
- Risk Model example graph: shaded risk / reward zones were not drawn for
  short trades (negative SVG height).

### Changed
- Live quotes are cached per symbol for 3 s and shared by the Positions
  table and the audit trail, so the two don't make separate broker calls.

## [1.0.0] - 2026-11-07

First version under Git. Everything below was built before the project
moved to GitHub, so it is recorded here as one release instead of
separate commits.

### Core (unchanged by everything below)
- Closed-bar candlestick pattern detection (`patterns.py`) -> indicator gate
  -> offline Kronos foundation-model confirmation (`kronos_agent.py`) ->
  hard risk rules (`risk.py`) -> bracket entry on moomoo OpenD.
- Software trailing stop (`trailing_stop.py`), SQLite audit log, Telegram
  reports, token-protected Flask dashboard.

### Added
- **Indicator gate & skills library** (`src/kronos_indicators`,
  `indicator_gate.py`): EMA stack, SMA, RSI, MACD, ADX, Supertrend,
  Bollinger, Donchian, volume/VWAP confirmation between pattern and Kronos;
  chart overlays for them on the Live Terminal.
- **Dynamic screener** with configurable universe and refresh interval;
  symbols dropped while holding a position stay monitored.
- **Multi-broker framework** (`src/brokers`, `broker_config.py`,
  `multi_broker.py`): one orchestrator thread per broker, shared Kronos
  model, per-broker settings overrides, audit DB, risk state and calendar.
  Adapters: moomoo, Tiger, IG (experimental), Zerodha Kite Connect (GTT
  exits, daily `kite_login.py`), Angel One SmartAPI (TOTP, marketable limits).
- **NSE/BSE market support** with tick-size rounding and holiday calendar.
- **Dashboard**: broker switcher; schema-driven Settings tab that writes
  `config.yaml` while preserving comments (with timestamped backups);
  market clocks and agent phase bar.
- **Market Sentiment Meter** (`sentiment.py`): benchmark trend + breadth
  gauge showing Up trend / Down trend / Sideways / Stable.
- **Full-screen live chart** on the Live Terminal.
- **Live Kronos badge** on the chart, re-run on every new closed bar.
- **Manual override** on the Positions tab: Buy, Sell / Short and per-row
  Close / Cancel, queued to the agent so manual trades get the usual
  bracket, trailing stop, audit trail and Telegram alert.
- Faster positions refresh (about 3 s) while the Positions tab is open.

### Fixed
- Bracket stop/target are anchored to the actual fill price, not the signal
  bar close (a gap-down fill had put a BUY's stop above the entry).
- Positions table showed stale prices and P&L from OpenD's cached position
  data; it now uses live quotes, like the Live Terminal.
- Kronos badge was stale or wrong: it ran only on symbol change, included
  the still-forming bar, used a hard-coded horizon and showed an
  unconfirmed SELL in green.
- Sentiment needle could point bullish while the label said Sideways.
- Chart did not shrink back after leaving full screen until a page refresh.
- Angel / Kite stop prices were rounded the wrong way (now directional).
- Legacy `/api/config` save rejected configs containing a `brokers` section.
