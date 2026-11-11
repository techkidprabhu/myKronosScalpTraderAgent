# moomoo Swing Trading Agent

Same architecture as the Tiger and Zerodha agents, retargeted at the
[moomoo OpenAPI](https://openapi.moomoo.com/moomoo-api-doc/en/intro/intro.html)
(`moomoo-api` Python SDK talking to the local **OpenD** gateway).
Deterministic candlestick pattern detection triggers a confirmation
pass from a local Kronos foundation model, gated by hard-coded risk
rules the model cannot override. Every decision is logged to SQLite,
reported via Telegram, and shown in a web dashboard.

Supported markets per moomoo broker entry: **US** (default), **HK**, **SG**,
**CN** A-shares.

**Multi-broker:** one agent process can trade several brokers at once -
moomoo, Tiger (ported from tiger-swing-agent), and experimentally IG,
Zerodha Kite and Angel One (NSE/BSE) - each
with its own settings overrides, audit DB, risk state and market calendar,
all sharing the same offline Kronos model. See
[Multi-broker setup](#multi-broker-setup). With no `brokers:` section the
agent runs moomoo alone, exactly as before.

**Version & history:** see [CHANGELOG.md](CHANGELOG.md) for every feature and
fix, and [docs/VERSIONING.md](docs/VERSIONING.md) for the branch / commit /
release workflow. Current version: see `VERSION`.

## Architecture

```
   ┌──────────────┐  TCP :11111   ┌─────────────────────┐
   │ moomoo OpenD │◀─────────────▶│  MoomooClient        │  OpenQuoteContext: K-lines, snapshots
   │  (gateway,   │               │  (moomoo_client.py)  │  OpenSecTradeContext: orders, positions
   │  logged in)  │               └──────────┬───────────┘
   └──────────────┘                          │ closed bar
                                             ▼
                                 ┌─────────────────────┐
                                 │  Pattern Detector    │  engulfing, hammer, pin bar, inside bar...
                                 │  (patterns.py)       │
                                 └──────────┬───────────┘
                                    pattern found?
                                             ▼
                                 ┌─────────────────────┐
                                 │  Indicator Gate      │  EMA stack, ADX, Supertrend, RSI, MACD,
                                 │  (indicator_gate.py) │  Bollinger, Donchian, volume/VWAP
                                 └──────────┬───────────┘
                                    indicators agree?
                                             ▼
                                 ┌─────────────────────┐
                                 │  Kronos Subagent     │  local K-line foundation model
                                 │  (kronos_agent.py)   │  → direction/volatility forecast
                                 └──────────┬───────────┘
                                    forecast confirms?
                                             ▼
                                 ┌─────────────────────┐
                                 │  Risk Manager        │  hard-coded, non-overridable:
                                 │  (risk.py)           │  max exposure, SL/TP, loss caps
                                 └──────────┬───────────┘
                                    allowed?
                                             ▼
                                 ┌─────────────────────┐        ┌──────────────────┐
                                 │  place_order → OpenD │───────▶│  Telegram Bot     │
                                 └──────────┬───────────┘        └──────────────────┘
                                             ▼
                                 ┌─────────────────────┐        ┌──────────────────┐
                                 │  SQLite Audit Log    │◀───────│  Web Dashboard    │
                                 │  (audit_log.py)      │        │  (dashboard/)     │
                                 └─────────────────────┘        └──────────────────┘
```

`patterns.py`, `kronos_agent.py`, `risk.py`, `trailing_stop.py`,
`audit_log.py`, `telegram_bot.py` and the dashboard UI are shared with
the Tiger version. Only the broker integration (`moomoo_client.py`,
`streaming.py`, `screener.py`) and exchange rules (`market.py`) differ.

## Setup

1. **Python 3.10+**. The agents in `mySwingTraders/` share `../venv`:

   ```bash
   source ../venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Kronos model**: the shared clone at `../vendor/Kronos` is detected
   automatically (or set `KRONOS_PATH`). Weights load offline
   (`local_files_only: true`).

3. **moomoo OpenD**: download, install, and log in with your moomoo ID
   ([OpenD guide](https://openapi.moomoo.com/moomoo-api-doc/en/quick/opend-base.html)).
   Leave it running. The default listen address is `127.0.0.1:11111`.
   The agent and dashboard **fail fast** with a clear error if OpenD isn't
   reachable. The SDK on its own would retry forever.

   For a headless Linux box, use the command-line OpenD and set
   `login_account` / `login_pwd_md5` in its `OpenD.xml`.

4. **Environment**: copy `.env.example` to `.env`:
   - `MOOMOO_TRADING_ENV=SIMULATE` (paper) until you've validated the
     pipeline. `REAL` also needs `MOOMOO_TRADING_PASSWORD` (the trade
     unlock PIN) or a manual unlock in the OpenD GUI.
   - `MOOMOO_TRADE_MARKET`: `US`, `HK`, `SG` or `CN`.
   - `MOOMOO_SECURITY_FIRM`: the entity your account is with, e.g.
     `FUTUSG` for moomoo Singapore or `FUTUINC` for moomoo US.
   - `MOOMOO_ACC_ID`: optional. If empty, the first account for the env
     that is authorised for the market is used, preferring stock paper
     accounts.
   - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

5. **Config**: copy `config/config.example.yaml` to `config/config.yaml`.
   Symbols are moomoo codes (`US.AAPL`, `HK.00700`, `SG.D05`) or bare
   tickers, which get the market prefix added. Set the risk distances
   in your market's currency, and set `lot_size` (US 1, SGX 100, HK
   per-stock).

6. **Run**, always on `SIMULATE` first:

   ```bash
   python main.py --config config/config.yaml
   ```

7. **Web GUI dashboard**: a real-time trading terminal and Kronos
   forecast visualiser.

   ```bash
   python run_dashboard.py --port 5000
   ```
   Open the login link it prints (`http://localhost:5000/?token=...`).

   **Or run both in the background with one script**, from any folder:

   ```bash
   ./start.sh            # agent + dashboard; prints the login link
   ./start.sh status     # what is running
   ./start.sh logs       # follow both logs (Ctrl+C stops watching, not the agent)
   ./start.sh stop       # stop both
   ```
   It uses the virtualenv that is active, or `venv/` in this folder or the
   one above. Logs go to `logs/agent.log` and `logs/dashboard.log`.

   - Header badges show agent state, moomoo market, PAPER/LIVE, masked
     account, total assets, and whether OpenD is online. The broker badge
     turns red when OpenD is down.
   - Tabs: candlestick chart with detected patterns and an on-demand
     Kronos forecast path, the 5-stage pipeline, positions and working
     orders (broker data merged with the agent's own tracking, with
     unmanaged exposure flagged), decisions and snapshots, strategy
     cards, and settings.
   - Binds to `127.0.0.1` by default. For remote access, use an SSH
     tunnel (`ssh -L 5000:127.0.0.1:5000 <host>`).
   - The token is `DASHBOARD_TOKEN` from `.env`, or one generated into
     `data/.dashboard_token`. Delete that file to revoke sessions.
   - The dashboard can change only `timeframe`, `order_type`,
     `kronos_confidence_threshold`, `lookback_bars` and
     `forecast_horizon`. Risk settings are read-only there.
   - If OpenD is down, the dashboard retries the connection every 30s
     rather than on every poll.

## Multi-broker setup

The Kronos pipeline (patterns → indicator gate → Kronos → risk) is the same
for every broker. What changes per broker is only the connection - each
adapter implements the same small interface (`src/brokers/base.py`).

| Adapter | Status | Needs |
|---|---|---|
| `moomoo` | ready | OpenD running and logged in; `MOOMOO_*` in `.env` |
| `tiger` | ready (ported) | `pip install tigeropen`; `TIGER_*` + private key in `.env` |
| `ig` | **experimental** | `IG_*` in `.env`; written from IG's REST docs, test on a DEMO account first |
| `zerodha` | **experimental** | `pip install kiteconnect`; `KITE_*` in `.env`; `python kite_login.py` every morning; registered static IP |
| `angel` | **experimental** | `pip install smartapi-python pyotp`; `ANGEL_*` in `.env`; registered static IP |

**Config.** Every top-level setting in `config/config.yaml` is shared. The
`brokers:` section lists brokers; any setting written under a broker
overrides the shared value for that broker only, and nested sections merge,
so you write just what differs:

```yaml
brokers:
  moomoo:
    enabled: true          # shared settings as-is
  tiger:
    enabled: true
    market: SG
    timeframe: HOUR_1      # only Tiger trades hourly candles
    risk: {lot_size: 100}  # other risk keys still come from the shared risk:
```

Per-broker-only keys: `enabled`, `adapter` (defaults to the entry name - two
moomoo accounts can run as `moomoo` and `moomoo_hk: {adapter: moomoo}`),
`market`, `label`, `db_path`. The `kronos:` model section is shared-only.

**What each broker gets of its own:** connection and reconnects, audit DB
(`data/audit.db` for moomoo, `data/audit_<name>.db` for others), daily loss
cap and cooldowns, screener picks, market calendar (sleeps when its exchange
is closed), and Telegram messages tagged `[Tiger]`, `[moomoo OpenD]` ...

**What they share:** one Kronos model, loaded once. Forecasts are taken one at
a time, so with several brokers scanning 5-minute candles a busy bar can queue
a few forecasts (about 20 s each on CPU).

**Symbols** are canonical codes everywhere - `US.AAPL`, `SG.D05`,
`HK.00700`; the Tiger adapter converts to `D05.SI` itself. IG uses its own
epics (`UA.D.AAPL.CASH.IP`) and can't list a whole market, so set
`screener.candidate_symbols` for it.

**Dashboard.** A *Broker* switcher appears in the header when more than one
broker is configured; every tab then shows that broker's data. The *Config
& Risk* tab is built from `dashboard/schema.py`: edit shared settings, or
pick a broker to set overrides (marked *override*, ↺ returns to the shared
value). Saving keeps the comments in `config.yaml`, writes a backup to
`config/backups/`, and checks the whole file first (e.g. fast MA < slow MA,
at least one broker enabled). Risk changes ask for confirmation. Restart
`main.py` to apply. Not editable there: the Kronos model files/device, the
static `symbols:` list, and anything in `.env`.

### Indian brokers (Zerodha Kite, Angel One)

Ported from `moomoo-ensemble-agent/src/brokers/` and tested against fake SDK
clients only. Neither broker has a paper account, so start with a few shares.

- **Static IP (both).** Since 1 Apr 2026 (SEBI retail-algo framework) API
  *orders* are accepted only from the static IP registered with the broker:
  Kite developer dashboard → profile → IP whitelist; SmartAPI dashboard →
  app → static IP. A home connection whose IP changes will have its orders
  rejected - run the agent from a machine/VPS with a static IP. Market data
  works from anywhere.
- **Zerodha login.** Access tokens expire every morning (~06:00 IST). Run
  `python kite_login.py` before the open; it saves the token to
  `data/kite_access_token`. A running agent waits (Telegram tells you) and
  reconnects by itself within 30 s once the token is there.
- **Angel login** is automatic - a TOTP code is generated from
  `ANGEL_TOTP_SECRET` on every connect, and the agent reconnects after
  Angel's midnight logout.
- **Market orders.** Zerodha: every MARKET / SL-M order carries
  `market_protection` (default `-1`, automatic), now mandatory. Angel:
  market and IOC orders are banned for algos, so the adapter never sends one -
  market entries/closes become LIMIT orders 0.5 % past the last price
  (`ANGEL_LIMIT_PCT`) and stops are STOPLOSS-LIMIT.
- **Exits.** Zerodha: stop and target are GTT triggers (`KITE_EXIT_MODE=gtt`,
  default) - they last up to a year and don't lock your CNC shares; the stop's
  limit sits `KITE_STOP_LIMIT_PCT` (0.5 %) beyond the trigger and the software
  monitor closes at market if a gap skips it. Angel: DAY stop + target orders
  re-armed at every session open; if Angel refuses the second sell for the
  same shares, the software monitor enforces the target.
- **Delivery only.** Product is CNC (Zerodha) / DELIVERY (Angel), so no short
  selling: `allow_short` is switched off automatically for these brokers.
- **Ticks.** NSE price-band ticks (₹0.01 below ₹250, ₹0.05 to ₹1,000, then
  ₹0.10 / 0.50 / 1 / 5) are in `src/market.py`; before each order the adapters
  re-round to the instrument's own published tick.
- **Sessions & holidays.** NSE/BSE 09:15-15:30 IST with the NSE holiday list
  from `src/market_calendar.py` (e.g. Diwali).

**Adding a broker:** implement `BrokerClient` in `src/brokers/<name>.py` and add
one line to `REGISTRY` in `src/brokers/__init__.py`.

## Repo layout

```
moomoo-swing-agent/
├── main.py                  entry point: one orchestrator per enabled broker
├── kite_login.py            daily Zerodha Kite login (saves data/kite_access_token)
├── run_dashboard.py         launches the web GUI dashboard (Flask)
├── requirements.txt
├── .env.example
├── config/config.example.yaml
├── data/audit.db            created on first run (SQLite, WAL mode)
├── dashboard/               Flask API + offline SPA terminal (lightweight-charts)
│   ├── schema.py            Settings form fields, ranges and scopes
│   └── config_writer.py     comment-preserving config.yaml writes + backups
├── src/
│   ├── brokers/             broker interface (base.py), registry, Tiger / IG / Zerodha / Angel adapters
│   ├── broker_config.py     brokers: section - per-broker overrides over shared settings
│   ├── multi_broker.py      runs every enabled broker, sharing one Kronos model
│   ├── moomoo_client.py     moomoo adapter: OpenD contexts, K-lines, orders, positions
│   ├── streaming.py         optional real-time quote push (subscribe + handler)
│   ├── market.py            sessions + tick tables for US / HK / SG / CN
│   ├── market_calendar.py   NASDAQ/NSE holidays, early closes, live session state
│   ├── screener.py          dynamic symbol selection (price, liquidity, ATR%)
│   ├── patterns.py          deterministic candlestick pattern detection
│   ├── indicator_gate.py    indicator confirmation between pattern and Kronos
│   ├── kronos_indicators/   pandas-only indicator library (MAs, RSI, MACD, ADX, ATR...)
│   ├── kronos_agent.py      Kronos forecast wrapper (offline foundation model)
│   ├── risk.py              hard-coded risk rules, non-overridable
│   ├── trailing_stop.py     trailing stop maths
│   ├── orchestrator.py      per-symbol worker loop, wires everything
│   ├── audit_log.py         SQLite schema + writes
│   └── telegram_bot.py      trade/status reporting
└── tests/                   pytest suite, using fake OpenD contexts (no gateway needed)
```

Run the tests with `python -m pytest -q`.

## Indicator gate

After a pattern fires, `src/indicator_gate.py` checks it against the
indicators before Kronos runs:

- **Bias score**, from −100 to +100: a weighted vote of price vs EMA200,
  the EMA 9/20/50/200 stack, EMA20/50 and SMA50/200 crosses, Supertrend,
  RSI zone, MACD, Stochastic, Bollinger/Donchian breakouts, volume surges
  and, on intraday timeframes, session VWAP. The score is damped when
  ADX < 20 (weak trend). A trade needs `min_alignment` or more in its own
  direction.
- **RSI exhaustion**: skips BUYs at `max_rsi_for_buy` or higher, and SELLs
  at `min_rsi_for_sell` or lower.
- **MA trend** (optional, `require_ma_trend`): the fast MA must be on the
  trade's side of the slow MA.

Rejections reach the audit log (`indicator_check` snapshot plus a
`rejected` decision), Telegram ("Signal skipped") and the dashboard's
"scan now" results. Accepted entries carry a one-line summary in the
Telegram message, for example `Indicators: bias bullish +62, RSI 58, ADX 27,
EMA20/EMA50 up`. Set `mode: log` to record verdicts without blocking, so
you can compare them with outcomes first. Remove the section or set
`enabled: false` to turn it off. If an indicator ever raises, the signal
passes through on patterns + Kronos, so trading doesn't stop.

RSI, ATR and ADX use Wilder smoothing, matching TradingView and broker
charts. EMA200 needs 200+ bars: with `lookback_bars: 200` it has just
enough, and `lookback_bars: 300` gives a better-settled EMA200 (Kronos
then also sees 300 bars of context).

## moomoo-specific details

| Concern | How it's handled |
|---|---|
| Bars | `request_history_kline` with paging. moomoo stamps intraday bars with their **end** time in exchange-local time; they are converted to UTC **start** times so the still-forming bar is dropped correctly. |
| `HOUR_4` | moomoo has no historical 4h K-line, so 60-minute bars are grouped four at a time within each trading day. |
| Quota | Each distinct symbol's history uses moomoo's 7-day historical K-line quota. Keep `candidate_symbols` lists reasonable. |
| Entry | `OrderType.NORMAL` (limit) or `OrderType.MARKET`, time-in-force DAY. MARKET is only sent while the session is open. |
| Exits | `OrderType.STOP` (trigger = `aux_price`) and limit target, placed only after the entry fills. **GTC** on REAL accounts; **DAY** on paper accounts, because moomoo paper rejects GTC ("Paper trading does not support GTC orders"). DAY exits expire at the close and are re-placed at the next session open. Override with `MOOMOO_EXIT_TIF`. |
| Connect | `OpenQuoteContext(is_async_connect=True)`, then `get_global_state` is polled until `qot_logined`/`trd_logined` (`MOOMOO_CONNECT_TIMEOUT`, default 20s). An OpenD that accepts the socket but isn't logged in raises a clear error instead of hanging, and the half-open context is closed. |
| Trailing | `modify_order(NORMAL, ...)` moves the stop's `aux_price`, keeping its quantity. |
| Order status | Today's orders via `order_list_query`. Older ones (e.g. a GTC stop filled while the agent was down) come from `history_order_list_query` (30-day lookback). |
| Rate limits | Order and position reads use OpenD's cache (`refresh_cache=False`), so they don't count against the 10 per 30s refresh limit. The dashboard caches broker data for 10s. |

## Dynamic symbol screener

Set `screener.enabled: true` to re-select symbols every
`refresh_interval_minutes` by price range, average volume, and ATR% band,
ranked by ATR%, keeping the top `top_n`. With `candidate_symbols` empty, the
whole market comes from `get_stock_basicinfo`. Batched
`get_market_snapshot` calls (400 codes each) prefilter on price and
volume before any history is pulled.

## Always-on operation

The agent is meant to run continuously (e.g. as a systemd service).

- **Market calendar.** `schedule.follow_market` (default on) makes the
  agent follow the exchange calendar. NYSE/Nasdaq holidays (Good Friday,
  observed days) and 1 pm early closes are computed for every year. NSE
  2026 holidays are built in; add later years or extra closures in
  `config/holidays.yaml`.
  - **Active:** scans and monitors.
  - **Warm-up:** starts `warmup_minutes` before the open. Checks OpenD and
    runs the screener fresh.
  - **Sleeping:** no scans and no screener runs. Positions are checked
    every `sleep_monitor_seconds`.
  - **Notifications:** Telegram gets "open" and "closed" messages.
- **OpenD connection.**
  - Every connect has a hard time limit, so the agent can't freeze on a
    gateway that isn't ready, and it can start before OpenD.
  - A health ping runs every minute.
  - After `reconnect_after_failures` failed pings, the connection is
    rebuilt. Trading pauses meanwhile, with a Telegram message and a
    dashboard banner.
- **Unmanaged positions.** Every `check_interval_seconds` the agent
  compares what moomoo holds with what it manages.
  - **Unmanaged positions or orders:** you're alerted once per symbol,
    by Telegram and a dashboard banner.
  - **Positions closed outside the agent:** if the agent's database says
    a position is open but moomoo no longer holds it (and none of its
    exit orders filled), it's marked closed after about 4 minutes.
  - **Leftover positions:** positions from before a restart, or symbols
    the screener dropped, get a monitor-only worker, so their exits keep
    being managed.
- **Dashboard.**
  - The market bar shows NASDAQ and NSE local clocks, open/closed/holiday
    state, a countdown to the next open or close, the next holiday, and
    the agent's phase.
  - Banners flag OpenD problems and unmanaged positions.

## Day-trader risk rules

Optional rules adapted from the Day Trader skill pack (`trader-risk`,
`trader-budget`, `trader-exit`). They are **all off** until you set them
(in `config.yaml` under `risk:`, the Settings tab, or the Risk Model
screen). They act only after patterns, the indicator gate and Kronos have
approved a signal, and never block an exit.

| rule | config | what it does |
| --- | --- | --- |
| Risk-based sizing | `max_loss_per_trade`, `max_position_pct`, `cash_buffer_pct` | quantity = max loss / stop distance, capped by max qty, % of account and cash after the buffer, rounded down to `lot_size`; a stop too wide for one lot skips the trade |
| Reward:risk floor | `min_reward_risk` | skip trades whose target is closer than N x the stop |
| Trades per day | `max_trades_per_day` | new entries per exchange day (manual orders count) |
| Daily cap with open risk | `daily_loss_includes_open_risk` | closed P&L + every open stop + the new trade's max loss must stay inside `max_daily_loss` |
| ATR stop, R target | `stop_atr_multiple`, `atr_period`, `target_r_multiple` | stop = K x ATR of the signal bars, target = R x stop; the distances are kept when the bracket is re-anchored to the fill |
| Too-tight stops | `min_stop_pct` | refuse a stop closer than N% of the price |
| Entry window | `no_entries_first_minutes`, `no_entries_last_minutes` | no new entries right after the open / before the close (early closes handled) |
| Flat by close | `flat_by_minutes_before_close` | day-trade mode: close every agent position (and cancel unfilled entries) N min before the close; needs an intraday timeframe |
| Blocked list | `blocked_symbols` | never enter these (with or without market prefix) |

Every rejection is logged in the Decision & Audit Log with the reason, and
each entry's snapshot records the quantity, what decided it and the max
loss at the stop. Manual orders from the Positions tab still skip the
entry gates on purpose, but are closed by flat-by-close like any agent
position.

## Order lifecycle & safety rules

1. **pending_entry**: a DAY entry is submitted. No exits rest yet.
2. **open**: once the entry fills, a stop and target (GTC live / DAY
   paper) are placed for the *filled* quantity. A partial fill cancels
   the unfilled remainder. A leg the broker rejects is retried every 15
   minutes while the market is open; until then the software monitor
   enforces it.
3. **closed**: when one exit fills, the other is cancelled. If the
   software monitor triggers the exit instead, it cancels both,
   re-checks fills, and market-closes only what's still held. If that
   close fails, the stop and target are re-placed.

Rules in `src/risk.py` persist across restarts:
- One position per symbol, including exposure the agent didn't create.
- `max_open_positions`.
- `max_daily_loss`, which resets at midnight in the exchange's timezone.
- A per-symbol cooldown.
- `allow_short`.
- `lot_size`.

## Important notes

- **This is a starting scaffold, not a finished trading system.**
  Backtest before connecting real capital, and run on `SIMULATE` first.
- moomoo paper trading covers US, HK and A-shares. Check the
  [docs](https://openapi.moomoo.com/moomoo-api-doc/en/intro/intro.html)
  for SG paper support and for STOP-order support on your account and
  market. If a stop is rejected, the software monitor enforces it instead,
  but only while the agent is running.
- Exchange holidays are not modelled.
- Nothing here is financial advice.
