"""
Your own trading rules, written in plain Markdown (config/skills.md).

Runs after the indicator check and before Kronos. Like the indicator check
it can only STOP a signal, never create one, and it never changes what
Kronos sees. Off by default (`skills.enabled`); `skills.mode: log` records
which rule would have stopped a signal without stopping it.

The file
--------
Each rule is a "## " heading followed by "- key: value" lines. Anything
else (titles, notes, blank lines) is ignored.

    ## No chasing above VWAP
    - applies to: buy                       buy | sell | both (default both)
    - patterns: hammer, bullish_engulfing   optional - only for these patterns
    - block if: price_vs_vwap_pct > 1.5     stop the signal when this is true
    - why: Buying far above VWAP chases a move that is often over.
    - off                                   optional - keeps the rule but skips it

    `block if` lines: the signal is stopped if ANY of them is true.
    `only if`  lines: the signal is stopped unless ALL of them are true.
    Inside one line, "and" joins conditions that must all hold.

A condition is `<measure> <op> <value>`: ops > >= < <= = != for numbers,
`is` / `is not` for words (trend, trend_vs_trade, pattern, direction).
The measures are listed in METRICS. A rule whose measure can't be worked
out right now (e.g. VWAP on daily candles, time to the close outside the
session) is skipped for that signal, never guessed - it lets the signal
pass. A line the agent can't read makes that rule an error: it is listed
on the dashboard and skipped, and the other rules still run.

The file is re-read when it changes; no restart is needed for rule edits.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.kronos_indicators import indicators as ind

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = ROOT / "config" / "skills.example.md"
DEFAULTS = {"enabled": False, "mode": "log", "file": "config/skills.md"}
INTRADAY = {"MIN_1", "MIN_3", "MIN_5", "MIN_15", "HOUR_1"}

# name -> (kind, what it means). kind: num | word
METRICS = {
    "price_vs_vwap_pct": ("num", "Signal candle's close vs today's VWAP, in % (intraday candles only)"),
    "price_vs_ema20_pct": ("num", "Close vs the 20-candle EMA, in %"),
    "price_vs_ema50_pct": ("num", "Close vs the 50-candle EMA, in %"),
    "price_vs_ema200_pct": ("num", "Close vs the 200-candle EMA, in %"),
    "rsi": ("num", "RSI(14), 0-100"),
    "adx": ("num", "ADX(14): trend strength, 0-100 (above ~20 = trending)"),
    "atr_pct": ("num", "ATR(14) as % of the close: how much it usually moves per candle"),
    "volume_ratio": ("num", "Signal candle's volume vs the 20-candle average (1 = normal)"),
    "candle_range_atr": ("num", "Signal candle's high-low range in ATRs (above ~2 = an unusually big candle)"),
    "body_pct": ("num", "Signal candle's body as % of its range"),
    "gap_pct": ("num", "Today's open vs the previous session's close, in % (daily candles: the signal candle's open)"),
    "day_change_pct": ("num", "Close vs the previous session's close, in %"),
    "room_to_high_pct": ("num", "Distance up to the highest high of the last 20 candles, in % (0 = at the high)"),
    "room_to_low_pct": ("num", "Distance down to the lowest low of the last 20 candles, in % (0 = at the low)"),
    "minutes_since_open": ("num", "Minutes since today's session opened (empty outside the session)"),
    "minutes_to_close": ("num", "Minutes until today's close (empty outside the session)"),
    "pattern_strength": ("num", "The pattern's strength (inside bar 0.45 ... engulfing 0.7)"),
    "trend": ("word", "up / down / flat: EMA 20 vs EMA 50"),
    "trend_vs_trade": ("word", "with / against / flat: the trend compared with the signal's direction"),
    "pattern": ("word", "bullish_engulfing, bearish_engulfing, hammer, shooting_star, inside_bar"),
    "direction": ("word", "buy / sell"),
}
OPS = (">=", "<=", "!=", "==", ">", "<", "=")
_COND = re.compile(r"^\s*([a-z0-9_ ]+?)\s*(>=|<=|!=|==|>|<|=|\bis not\b|\bis\b)\s*(.+?)\s*$", re.I)


@dataclass
class Condition:
    metric: str
    op: str
    value: object
    text: str

    def check(self, metrics: dict):
        """True / False, or None when the measure isn't available now."""
        have = metrics.get(self.metric)
        if have is None or (isinstance(have, float) and not np.isfinite(have)):
            return None
        if METRICS[self.metric][0] == "word":
            same = str(have).lower() == str(self.value).lower()
            return same if self.op == "is" else not same
        a, b = float(have), float(self.value)
        return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b, "=": a == b, "==": a == b, "!=": a != b}[self.op]


@dataclass
class Rule:
    name: str
    line: int
    applies: str = "both"
    patterns: list = field(default_factory=list)
    block_if: list = field(default_factory=list)     # list of lists (OR of ANDs)
    only_if: list = field(default_factory=list)      # list of lists (all must hold)
    why: str = ""
    enabled: bool = True
    error: str | None = None

    def summary(self) -> dict:
        txt = lambda groups: [" and ".join(c.text for c in g) for g in groups]  # noqa: E731
        return {"name": self.name, "line": self.line, "applies": self.applies, "patterns": self.patterns,
                "block_if": txt(self.block_if), "only_if": txt(self.only_if), "why": self.why,
                "enabled": self.enabled, "error": self.error}


def _norm_metric(s: str) -> str:
    return re.sub(r"[\s-]+", "_", s.strip().lower())


def _parse_conditions(text: str) -> list[Condition]:
    out = []
    for part in re.split(r"\s+and\s+", text.strip(), flags=re.I):
        m = _COND.match(part)
        if not m:
            raise ValueError(f"can't read the condition {part.strip()!r} - write it like 'rsi > 70'")
        metric, op, raw = _norm_metric(m.group(1)), m.group(2).lower(), m.group(3).strip().strip("'\"")
        op = re.sub(r"\s+", " ", op)
        if metric not in METRICS:
            close = difflib.get_close_matches(metric, list(METRICS), n=3, cutoff=0.6) or \
                [k for k in METRICS if metric.split("_")[0] and metric.split("_")[0] in k][:3]
            raise ValueError(f"unknown measure {metric!r}" + (f" - did you mean {', '.join(close)}?" if close else ""))
        kind = METRICS[metric][0]
        if kind == "word":
            if op in ("=", "=="):
                op = "is"
            elif op == "!=":
                op = "is not"
            if op not in ("is", "is not"):
                raise ValueError(f"{metric} is a word - use 'is' or 'is not'")
            value = raw.lower().replace(" ", "_")
        else:
            if op in ("is", "is not"):
                raise ValueError(f"{metric} is a number - use >, >=, <, <=, = or !=")
            try:
                value = float(raw.rstrip("%").replace(",", ""))
            except ValueError:
                raise ValueError(f"{raw!r} is not a number (in {part.strip()!r})") from None
        out.append(Condition(metric, op, value, f"{metric} {op} {raw}"))
    return out


def parse(text: str) -> tuple[list[Rule], list[str]]:
    """Rules from the Markdown text, and file-level problems."""
    rules, problems, cur = [], [], None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line.startswith("## "):
            cur = Rule(name=line[3:].strip() or f"Rule on line {n}", line=n)
            rules.append(cur)
            continue
        if line.startswith("#") or not line.startswith(("- ", "* ")):
            continue
        if cur is None:
            continue                     # a list before the first rule: notes
        body = line[2:].strip()
        if cur.error:
            continue
        low = body.lower()
        if low in ("off", "disabled", "enabled: no", "enabled: false", "on: no"):
            cur.enabled = False
            continue
        key, sep, val = body.partition(":")
        key = key.strip().lower()
        if not sep:
            cur.error = f"line {n}: expected 'key: value', got {body!r}"
            continue
        try:
            if key in ("applies to", "applies", "side", "direction"):
                v = val.strip().lower()
                if v not in ("buy", "sell", "both"):
                    raise ValueError("applies to must be buy, sell or both")
                cur.applies = v
            elif key in ("pattern", "patterns"):
                cur.patterns = [p.strip().lower().replace(" ", "_") for p in val.split(",") if p.strip()]
            elif key in ("block if", "block when", "skip if"):
                cur.block_if.append(_parse_conditions(val))
            elif key in ("only if", "only when", "require", "requires"):
                cur.only_if.append(_parse_conditions(val))
            elif key in ("why", "note", "reason"):
                cur.why = val.strip()
            elif key in ("enabled", "on"):
                cur.enabled = val.strip().lower() not in ("no", "false", "off", "0")
            else:
                raise ValueError(f"unknown key {key!r} - use applies to, patterns, block if, only if, why or off")
        except ValueError as exc:
            cur.error = f"line {n}: {exc}"
    for r in rules:
        if not r.error and not r.block_if and not r.only_if:
            r.error = f"line {r.line}: the rule has no 'block if' or 'only if' line"
    names = [r.name.lower() for r in rules]
    for r in rules:
        if names.count(r.name.lower()) > 1 and not r.error:
            problems.append(f"Two rules are called {r.name!r}; give each its own name")
    return rules, sorted(set(problems))


# ---------------------------------------------------------------- measures

def _pct(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return None
    return round((a - b) / b * 100, 4)


def _last(s):
    try:
        v = float(s.iloc[-1])
    except (IndexError, TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def compute_metrics(df: pd.DataFrame, direction: str, pattern: str | None = None, strength: float | None = None,
                    timeframe: str = "DAY", tz=None, session_minutes: tuple | None = None) -> dict:
    """Every measure in METRICS for the latest closed candle (None when it
    can't be worked out from these candles)."""
    m = {k: None for k in METRICS}
    m["direction"] = "buy" if str(direction).upper() == "BUY" else "sell"
    m["pattern"] = (pattern or "").lower() or None
    m["pattern_strength"] = strength
    if session_minutes:
        m["minutes_since_open"], m["minutes_to_close"] = (round(float(x), 1) for x in session_minutes)
    if df is None or df.empty:
        return m
    d = df.copy()
    idx = pd.DatetimeIndex(d.index)
    d.index = idx.tz_localize("UTC") if idx.tz is None else idx
    if tz is not None:
        d.index = d.index.tz_convert(tz)
    close = d["close"]
    c = _last(close)
    for n in (20, 50, 200):
        if len(d) >= n:
            m[f"price_vs_ema{n}_pct"] = _pct(c, _last(ind.ema(close, n)))
    if len(d) >= 15:
        m["rsi"] = _last(ind.rsi(close, 14))
        atr = ind.atr(d, 14)
        a = _last(atr)
        m["atr_pct"] = round(a / c * 100, 4) if a is not None and c else None
        prev_atr = _last(atr.iloc[:-1]) if len(atr) > 1 else None
        rng = float(d["high"].iloc[-1] - d["low"].iloc[-1])
        m["candle_range_atr"] = round(rng / prev_atr, 3) if prev_atr else None
    if len(d) >= 30:
        m["adx"] = _last(ind.adx(d, 14)["adx"])
    if len(d) >= 21 and d["volume"].iloc[:-1].tail(20).mean() > 0:
        m["volume_ratio"] = round(float(d["volume"].iloc[-1] / d["volume"].iloc[-21:-1].mean()), 3)
    rng = float(d["high"].iloc[-1] - d["low"].iloc[-1])
    if rng > 0:
        m["body_pct"] = round(abs(float(d["close"].iloc[-1] - d["open"].iloc[-1])) / rng * 100, 2)
    if len(d) >= 20:
        hi, lo = float(d["high"].iloc[-20:].max()), float(d["low"].iloc[-20:].min())
        m["room_to_high_pct"] = _pct(hi, c)
        m["room_to_low_pct"] = _pct(c, lo)
    if len(d) >= 50:
        e20, e50 = _last(ind.ema(close, 20)), _last(ind.ema(close, 50))
        if e20 is not None and e50:
            gap = (e20 - e50) / e50 * 100
            m["trend"] = "flat" if abs(gap) < 0.1 else ("up" if gap > 0 else "down")
            want = "up" if m["direction"] == "buy" else "down"
            m["trend_vs_trade"] = "flat" if m["trend"] == "flat" else ("with" if m["trend"] == want else "against")
    if timeframe in INTRADAY:
        try:
            m["price_vs_vwap_pct"] = _pct(c, _last(ind.vwap(d)))
        except Exception:  # noqa: BLE001
            pass
        days = d.index.normalize()
        today = days[-1]
        prev = d[days < today]
        cur = d[days == today]
        if len(prev) and len(cur):
            m["gap_pct"] = _pct(cur["open"].iloc[0], prev["close"].iloc[-1])
            m["day_change_pct"] = _pct(c, prev["close"].iloc[-1])
    elif len(d) >= 2:
        m["gap_pct"] = _pct(d["open"].iloc[-1], d["close"].iloc[-2])
        m["day_change_pct"] = _pct(c, d["close"].iloc[-2])
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


# ---------------------------------------------------------------- verdict

@dataclass
class RulesVerdict:
    passed: bool                       # False = stop the signal (filter mode)
    hits: list                         # [{rule, why, detail}] rules that would stop it
    unchecked: list                    # rule names that couldn't be checked now
    checked: int
    mode: str
    metrics: dict

    def reason(self) -> str:
        h = self.hits[0]
        more = f" (+{len(self.hits) - 1} more)" if len(self.hits) > 1 else ""
        return f"rule '{h['rule']}': {h['detail']}{more}"

    def record(self) -> dict:
        return {"mode": self.mode, "hits": self.hits, "unchecked": self.unchecked, "checked": self.checked,
                "metrics": {k: v for k, v in self.metrics.items() if v is not None}}


def _fmt(v):
    return f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def evaluate(rules: list[Rule], metrics: dict, mode: str = "filter") -> RulesVerdict:
    hits, unchecked, checked = [], [], 0
    direction, pattern = metrics.get("direction"), metrics.get("pattern")
    for r in rules:
        if r.error or not r.enabled:
            continue
        if r.applies != "both" and r.applies != direction:
            continue
        if r.patterns and pattern not in r.patterns:
            continue
        checked += 1
        detail, unknown = None, False
        for group in r.block_if:
            results = [c.check(metrics) for c in group]
            if None in results:
                unknown = True
                continue
            if all(results):
                detail = " and ".join(f"{c.metric} {_fmt(metrics[c.metric])} {c.op} {_fmt(c.value)}" for c in group)
                break
        if detail is None:
            for group in r.only_if:
                results = [c.check(metrics) for c in group]
                if None in results:
                    unknown = True
                    continue
                failed = [c for c, ok in zip(group, results) if not ok]
                if failed:
                    detail = "needs " + " and ".join(c.text for c in group) + \
                             f" ({', '.join(f'{c.metric} is {_fmt(metrics[c.metric])}' for c in failed)})"
                    break
        if detail:
            hits.append({"rule": r.name, "why": r.why, "detail": detail})
        elif unknown:
            unchecked.append(r.name)
    passed = not hits or mode == "log"
    return RulesVerdict(passed, hits, unchecked, checked, mode, metrics)


# ---------------------------------------------------------------- the file

def settings(config: dict) -> dict:
    s = {**DEFAULTS, **((config or {}).get("skills") or {})}
    s["mode"] = s["mode"] if s["mode"] in ("filter", "log") else "log"
    return s


def rules_path(config: dict) -> tuple[Path, bool]:
    """(file to read, True if it is the example because yours doesn't exist yet)."""
    p = Path(settings(config)["file"])
    p = p if p.is_absolute() else ROOT / p
    if p.exists():
        return p, False
    return EXAMPLE_PATH, True


class RuleBook:
    """The rules file, re-read whenever it changes."""

    def __init__(self, config: dict):
        self.config = config
        self._mtime, self.rules, self.problems, self.path, self.example = None, [], [], None, False

    def load(self) -> list[Rule]:
        path, example = rules_path(self.config)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            self.rules, self.problems, self.path = [], [f"{path} not found"], path
            return self.rules
        if (path, mtime) != (self.path, self._mtime):
            self.rules, self.problems = parse(path.read_text(encoding="utf-8"))
            self.path, self._mtime, self.example = path, mtime, example
            bad = [r for r in self.rules if r.error]
            logger.info("Loaded %d rule(s) from %s%s", len(self.rules) - len(bad), path,
                        f"; {len(bad)} with errors skipped" if bad else "")
            for r in bad:
                logger.warning("Rule %r skipped: %s", r.name, r.error)
        return self.rules

    def check(self, df, direction, pattern=None, strength=None, timeframe="DAY", tz=None,
              session_minutes=None) -> RulesVerdict:
        rules = self.load()
        metrics = compute_metrics(df, direction, pattern, strength, timeframe, tz, session_minutes)
        return evaluate(rules, metrics, settings(self.config)["mode"])
