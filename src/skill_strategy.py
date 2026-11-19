"""
The price-action engine behind skill-file trading (trading_mode "skill").

A skill file (e.g. config/PragaPriceActionSkill.md) describes a method in
words for people and AI, and carries one fenced ```yaml block with a
`strategy:` key. That block picks the engine and its numbers; the agent
reads nothing else from the file for trading. Today there is one engine:

    strategy: praga_price_action

Long only. Everything is worked out from CLOSED candles, so the same chart
always gives the same answer.

Setup (all in order)
  A  a bullish engulfing at a swing low (the "previous" engulfing)
  B  a turnaround: a later, higher swing low (require_second_low)
  C  breakout: a candle closes above A's high (the breakout candle)
  D  retest: price comes back to A. The entry is a resting LIMIT buy at
     A's red candle open - or its high when that candle's upper wick is
     longer than its body - placed once `min_retest_candles` candles have
     closed after the breakout, so a retest that comes straight back is
     not taken. The setup ends if a close falls below the stop or nothing
     fills within `max_wait_candles`.

Stop     below the breakout candle's low (`stop_at: breakout_candle`), or
         below A's low when the breakout candle's low is not under the entry.
Target   just under the nearest swing high above the breakout close; with
         none, `target_r_fallback` x the risk.
Exits    at the close of a bearish engulfing; when that candle is bigger
         than `big_candle_atr` ATRs, exit when price comes back up to the
         low of the green candle before it (the target order is moved
         there). Close everything when a close is below both the latest
         bullish engulfing since entry and the latest swing low since entry.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ENGINES = ("praga_price_action",)


@dataclass
class StrategySettings:
    strategy: str = "praga_price_action"
    swing_candles: int = 2              # candles each side that make a swing low / high
    require_second_low: bool = True     # step B: a higher swing low after A
    strong_body_pct: float = 60.0       # strong candle: body >= this % of its range
    strong_close_top_pct: float = 25.0  # ...and closes within this top % of its range
    min_retest_candles: int = 2         # closed candles after the breakout before the order rests
    max_wait_candles: int = 20          # setup expires this many candles after the breakout
    min_quality: int = 0                # 0-2: strong breakout candle, 2+ green candles after it
    stop_at: str = "breakout_candle"    # breakout_candle | engulfing_low
    stop_buffer_pct: float = 0.05       # stop this % below the level
    target_buffer_pct: float = 0.1      # target this % under the swing high
    target_r_fallback: float = 2.0      # target = entry + R x risk when no swing high is above
    exit_on_bearish_engulfing: bool = True
    big_candle_atr: float = 2.0         # a bearish engulfing bigger than this many ATRs is "too big"
    close_on_trend_end: bool = True
    atr_period: int = 14
    lookback_candles: int = 150
    use_trailing_stop: bool = False     # the agent's trailing stop on skill trades

    def summary(self) -> dict:
        return asdict(self)


_FENCE = re.compile(r"```(?:yaml|yml)?\s*\n(.*?)```", re.S | re.I)


def load(path: str | Path) -> tuple[StrategySettings | None, list[str]]:
    """Settings from the skill file's ```yaml block, and any problems.
    None when the file can't drive trading (no block, unknown engine)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return None, [f"Can't read {path}: {exc.strerror or exc}"]
    return parse(text)


def parse(text: str) -> tuple[StrategySettings | None, list[str]]:
    block = None
    for m in _FENCE.finditer(text):
        try:
            data = yaml.safe_load(m.group(1))
        except yaml.YAMLError as exc:
            if re.search(r"^\s*strategy\s*:", m.group(1), re.M):
                return None, [f"The settings block isn't valid YAML: {exc}"]
            continue
        if isinstance(data, dict) and "strategy" in data:
            block = data
            break
    if block is None:
        return None, ["No settings block: add a ```yaml block with 'strategy: praga_price_action'"]
    problems: list[str] = []
    engine = str(block.get("strategy") or "").strip().lower()
    if engine not in ENGINES:
        return None, [f"Unknown strategy {engine!r}; the agent knows: {', '.join(ENGINES)}"]
    s = StrategySettings()
    types = {f.name: f.type for f in fields(StrategySettings)}
    for key, val in block.items():
        k = str(key).strip().lower()
        if k == "strategy":
            continue
        if k not in types:
            problems.append(f"Unknown setting {key!r} ignored")
            continue
        default = getattr(s, k)
        try:
            if isinstance(default, bool):
                v = val if isinstance(val, bool) else str(val).strip().lower() in ("true", "yes", "on", "1")
            elif isinstance(default, int):
                v = int(val)
            elif isinstance(default, float):
                v = float(val)
            else:
                v = str(val).strip().lower()
        except (TypeError, ValueError):
            problems.append(f"{key}: {val!r} is not a valid value - using {default}")
            continue
        setattr(s, k, v)
    s.strategy = engine
    if s.stop_at not in ("breakout_candle", "engulfing_low"):
        problems.append(f"stop_at {s.stop_at!r} unknown - using breakout_candle")
        s.stop_at = "breakout_candle"
    for name, lo in (("swing_candles", 1), ("min_retest_candles", 0), ("max_wait_candles", 1),
                     ("atr_period", 2), ("lookback_candles", 30)):
        if getattr(s, name) < lo:
            problems.append(f"{name} below {lo} - using {lo}")
            setattr(s, name, lo)
    if s.max_wait_candles < s.min_retest_candles:
        problems.append("max_wait_candles is below min_retest_candles - the setup could never be entered")
    return s, problems


# ---------------------------------------------------------------- candles

class _Bars:
    def __init__(self, df: pd.DataFrame, s: StrategySettings):
        d = df.tail(int(s.lookback_candles))
        self.index = d.index
        self.o, self.h = d["open"].to_numpy(float), d["high"].to_numpy(float)
        self.l, self.c = d["low"].to_numpy(float), d["close"].to_numpy(float)
        self.n = len(d)
        self.k = int(s.swing_candles)
        self.s = s

    def green(self, i):
        return self.c[i] > self.o[i]

    def red(self, i):
        return self.c[i] < self.o[i]

    def bull_engulf(self, i):
        return i >= 1 and self.green(i) and self.red(i - 1) \
            and self.o[i] <= self.c[i - 1] and self.c[i] >= self.o[i - 1]

    def bear_engulf(self, i):
        return i >= 1 and self.red(i) and self.green(i - 1) \
            and self.o[i] >= self.c[i - 1] and self.c[i] <= self.o[i - 1]

    def swing_low(self, i):
        k = self.k
        if i - k < 0 or i + k >= self.n:
            return False
        return self.l[i] < self.l[i - k:i].min() and self.l[i] <= self.l[i + 1:i + k + 1].min()

    def swing_high(self, i):
        k = self.k
        if i - k < 0 or i + k >= self.n:
            return False
        return self.h[i] > self.h[i - k:i].max() and self.h[i] >= self.h[i + 1:i + k + 1].max()

    def engulf_at_low(self, g):
        """Bullish engulfing (red g-1, green g) whose low is a swing low."""
        if not self.bull_engulf(g) or g + self.k >= self.n:
            return False
        lo = min(self.l[g - 1], self.l[g])
        a, b = max(0, g - 1 - self.k), min(self.n, g + 1 + self.k)
        return lo <= self.l[a:b].min()

    def strong(self, i):
        rng = self.h[i] - self.l[i]
        if not self.green(i) or rng <= 0:
            return False
        body = self.c[i] - self.o[i]
        return body / rng * 100 >= self.s.strong_body_pct and \
            (self.h[i] - self.c[i]) / rng * 100 <= self.s.strong_close_top_pct

    def atr(self, upto=None):
        end = self.n if upto is None else upto + 1
        p = int(self.s.atr_period)
        if end < 2:
            return None
        h, l, c = self.h[:end], self.l[:end], self.c[:end]
        tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
        tr = tr[-p:]
        return float(tr.mean()) if len(tr) else None

    def t(self, i):
        return pd.Timestamp(self.index[i]).isoformat()


@dataclass
class Setup:
    status: str                     # none | waiting | ready | invalid
    reason: str
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    quality: int = 0
    details: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def summary(self) -> dict:
        return {"status": self.status, "reason": self.reason, "entry": self.entry, "stop": self.stop,
                "target": self.target, "quality": self.quality, **self.details}


def find_setup(df: pd.DataFrame, s: StrategySettings) -> Setup:
    """The current setup on the closed candles in df (newest last)."""
    if df is None or len(df) < 2 * s.swing_candles + 4:
        return Setup("none", "not enough candles")
    b = _Bars(df, s)
    last = b.n - 1
    best, best_rank = None, -1
    # Newest base first. The turnaround (step B) is often itself a bullish
    # engulfing at a swing low, so an older base can be the one that is ready.
    # Otherwise report the base that got furthest (a breakout beats none).
    for a in range(last, 0, -1):
        if not b.engulf_at_low(a):
            continue
        res = _evaluate(b, a, last)
        if res.ready:
            return res
        rank = (2 if "breakout" in res.details else 0) + (1 if res.status == "waiting" else 0)
        if rank > best_rank:
            best, best_rank = res, rank
    return best or Setup("none", "no bullish engulfing at a swing low")


def _evaluate(b: _Bars, a: int, last: int) -> Setup:
    s = b.s
    red = a - 1
    a_low = min(b.l[red], b.l[a])
    zone_high = max(b.h[red], b.h[a])
    body = abs(b.c[red] - b.o[red])
    upper_wick = b.h[red] - max(b.o[red], b.c[red])
    long_wick = upper_wick > body
    entry = float(b.h[red] if long_wick else b.o[red])
    det = {"base": b.t(a), "base_low": round(a_low, 6), "base_high": round(zone_high, 6),
           "entry_rule": "red candle high (long upper wick)" if long_wick else "red candle open"}

    start = a
    if s.require_second_low:
        turn = None
        for k in range(a + 1, last + 1):
            if b.c[k] < a_low:
                return Setup("invalid", "price closed below the engulfing's low before a turnaround", details=det)
            if b.swing_low(k) and b.l[k] > a_low and b.l[k] < zone_high:
                turn = k
                break
        if turn is None:
            return Setup("waiting", "waiting for a higher swing low after the engulfing", details=det)
        start = turn
        det["turnaround"] = b.t(turn)

    brk = None
    for k in range(start + 1, last + 1):
        if b.c[k] < a_low:
            return Setup("invalid", "price closed below the engulfing's low", details=det)
        if b.c[k] > zone_high:
            brk = k
            break
    if brk is None:
        return Setup("waiting", f"waiting for a close above {zone_high:g}", details=det)
    det["breakout"] = b.t(brk)

    level = float(b.l[brk])
    if s.stop_at == "engulfing_low" or level >= entry:
        level = float(a_low)
        det["stop_rule"] = "below the engulfing low"
    else:
        det["stop_rule"] = "below the breakout candle low"
    stop = level * (1 - s.stop_buffer_pct / 100)
    if stop >= entry:
        return Setup("invalid", "stop would be above the entry", details=det)

    after = last - brk
    for k in range(brk + 1, last + 1):
        if b.c[k] < stop:
            return Setup("invalid", "a close fell below the stop after the breakout", details=det)
        if b.l[k] <= entry:
            why = "retest came straight back (fewer than min_retest_candles)" \
                if k - brk <= s.min_retest_candles else "the retest already happened"
            return Setup("invalid", why, details=det)
    if after > s.max_wait_candles:
        return Setup("invalid", f"no retest within {s.max_wait_candles} candles", details=det)

    quality = 0
    if b.strong(brk):
        quality += 1
        det["strong_breakout"] = True
    greens = 0
    for k in range(brk + 1, last + 1):
        if not b.green(k):
            break
        greens += 1
    if greens >= 2:
        quality += 1
        det["greens_after_breakout"] = greens

    # Target: just under the nearest swing high above the breakout close.
    above = [float(b.h[k]) for k in range(0, last + 1) if b.swing_high(k) and b.h[k] > b.c[brk]]
    risk = entry - stop
    if above:
        target = min(above) * (1 - s.target_buffer_pct / 100)
        det["target_rule"] = "swing high"
    else:
        target = entry + s.target_r_fallback * risk
        det["target_rule"] = f"{s.target_r_fallback:g}R (no swing high above)"
    if target <= entry:
        target = entry + s.target_r_fallback * risk
        det["target_rule"] = f"{s.target_r_fallback:g}R (swing high below the entry)"

    if after < s.min_retest_candles:
        return Setup("waiting", f"breakout - waiting {s.min_retest_candles - after} more candle(s) before the retest order",
                     entry=entry, stop=stop, target=target, quality=quality, details=det)
    if quality < s.min_quality:
        return Setup("invalid", f"quality {quality} below min_quality {s.min_quality}",
                     entry=entry, stop=stop, target=target, quality=quality, details=det)
    det["expires_after"] = s.max_wait_candles - after
    return Setup("ready", "breakout above the engulfing - buy the retest", entry=entry, stop=stop,
                 target=target, quality=quality, details=det)


@dataclass
class ExitSignal:
    kind: str          # close_now | exit_at
    reason: str
    price: float | None = None


def exit_signal(df: pd.DataFrame, s: StrategySettings, opened_at: str | None) -> ExitSignal | None:
    """Exit for an open skill trade, judged on the newest closed candle."""
    if df is None or len(df) < 3:
        return None
    b = _Bars(df, s)
    last = b.n - 1
    since = 0
    if opened_at:
        ts = pd.Timestamp(opened_at)
        try:
            idx = b.index.tz_localize(None) if getattr(b.index, "tz", None) is not None else b.index
            ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
            after = np.nonzero(np.asarray(idx > ts))[0]
        except Exception:  # noqa: BLE001
            after = np.arange(b.n)
        if not len(after):
            return None
        since = int(after[0])
        if last < since:
            return None

    if s.exit_on_bearish_engulfing and b.bear_engulf(last):
        rng = b.h[last] - b.l[last]
        atr = b.atr(last - 1)
        prior_low = float(b.l[last - 1])
        if s.big_candle_atr and atr and rng > s.big_candle_atr * atr and prior_low > b.c[last]:
            return ExitSignal("exit_at", f"big bearish engulfing ({rng / atr:.1f} ATR) - exit near the prior green candle's low",
                              price=prior_low)
        return ExitSignal("close_now", "bearish engulfing close")

    if s.close_on_trend_end and last > since:
        eng = [g for g in range(max(since, 1), last + 1) if b.bull_engulf(g)]
        lows = [k for k in range(since, last + 1) if b.swing_low(k)]
        if eng and lows:
            g = eng[-1]
            if b.c[last] < min(b.l[g - 1], b.l[g]) and b.c[last] < b.l[lows[-1]]:
                return ExitSignal("close_now", "trend ended - closed below the last bullish engulfing and swing low")
    return None
