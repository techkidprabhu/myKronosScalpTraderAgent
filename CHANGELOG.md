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
