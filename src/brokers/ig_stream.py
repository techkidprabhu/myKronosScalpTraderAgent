"""
IG live candles for around-the-clock trading.

IG allows 10,000 historical price points a week, which round-the-clock
3-minute scanning would use up within days. So candles are built from IG's
streaming API (Lightstreamer, its own allowance: 40 subscriptions) instead:

  * CandleStore keeps the agent-timeframe candles of every watched market,
    on disk in data/ig_candles/<epic>__<TF>.csv, so a restart (or the
    dashboard) never downloads them again. History is fetched from the REST
    API only to seed a market the first time, or to fill a gap after the
    agent was down - never on every scan.
  * IGStream subscribes CHART:<epic>:1MINUTE for each watched market and
    folds the 1-minute candles into the agent's timeframe (MIN_3 = three of
    them). Prices are the mid of bid and offer, like the REST candles; the
    candle's tick count stands in for volume (IG's CFDs have none).

The Lightstreamer package (`lightstreamer-client-lib`) is only imported when
streaming starts; without it the IG adapter falls back to REST candles.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

try:                                    # listeners subclass the SDK's own when it's installed
    from lightstreamer.client import ClientListener as _ClientBase, SubscriptionListener as _SubBase
except ImportError:                     # pragma: no cover - REST fallback / tests
    _ClientBase = _SubBase = object

ROOT = Path(__file__).resolve().parent.parent.parent
CANDLE_DIR = ROOT / "data" / "ig_candles"
MAX_SUBSCRIPTIONS = 40
FIELDS = ["BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE", "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE",
          "CONS_TICK_COUNT", "CONS_END", "UTM"]
TF_MINUTES = {"MIN_1": 1, "MIN_3": 3, "MIN_5": 5, "MIN_15": 15, "HOUR_1": 60}
COLUMNS = ["time", "open", "high", "low", "close", "volume"]
KEEP_BARS = 3000


def _safe(epic: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", epic)


def candle_path(epic: str, timeframe: str, root: Path | None = None) -> Path:
    return Path(root or CANDLE_DIR) / f"{_safe(epic)}__{timeframe}.csv"


def stored_frame(epic: str, timeframe: str, n: int, max_age_s: float = 900, root: Path | None = None):
    """Candles the agent saved for this market, if fresh - lets a second
    process (the dashboard) read them without using IG's allowance."""
    p = candle_path(epic, timeframe, root)
    try:
        if time.time() - p.stat().st_mtime > max_age_s:
            return None
        df = pd.read_csv(p)
    except (OSError, ValueError):
        return None
    if df.empty:
        return None
    df = df.tail(int(n)).reset_index(drop=True)
    df["symbol"] = epic
    return df


class CandleStore:
    """Agent-timeframe candles per market: seeded from history, extended by
    the live 1-minute stream, saved to disk."""

    def __init__(self, timeframe: str, root: Path | None = None):
        if timeframe not in TF_MINUTES:
            raise ValueError(f"Around-the-clock IG needs an intraday timeframe, not {timeframe}")
        self.tf = timeframe
        self.ms = TF_MINUTES[timeframe] * 60_000
        self.root = Path(root or CANDLE_DIR)
        self.bars: dict[str, dict[int, list]] = {}       # epic -> bucket start ms -> [o, h, l, c, v]
        self._minutes: dict[str, dict[int, dict[int, tuple]]] = {}   # epic -> bucket -> minute -> candle
        self._dirty: set[str] = set()
        self._saved = 0.0
        self.lock = threading.RLock()

    # ---- disk
    def load(self, epic: str) -> int:
        with self.lock:
            if epic in self.bars:
                return len(self.bars[epic])
            rows = {}
            try:
                df = pd.read_csv(candle_path(epic, self.tf, self.root))
                for r in df.itertuples(index=False):
                    rows[int(r.time)] = [float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume)]
            except (OSError, ValueError, AttributeError):
                pass
            self.bars[epic] = rows
            return len(rows)

    def save(self, force: bool = False):
        with self.lock:
            if not self._dirty or (not force and time.time() - self._saved < 30):
                return
            self.root.mkdir(parents=True, exist_ok=True)
            for epic in list(self._dirty):
                rows = self.bars.get(epic) or {}
                keys = sorted(rows)[-KEEP_BARS:]
                df = pd.DataFrame([[k, *rows[k]] for k in keys], columns=COLUMNS)
                tmp = candle_path(epic, self.tf, self.root).with_suffix(".tmp")
                df.to_csv(tmp, index=False)
                tmp.replace(candle_path(epic, self.tf, self.root))
            self._dirty.clear()
            self._saved = time.time()

    # ---- data in
    def merge_history(self, epic: str, df: pd.DataFrame):
        """REST candles (time = bucket start, ms) - fill in, never overwrite
        a bucket the stream is building."""
        with self.lock:
            rows = self.bars.setdefault(epic, {})
            for r in df.itertuples(index=False):
                t = int(r.time)
                if t not in rows or t not in self._minutes.get(epic, {}):
                    rows[t] = [float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume or 0)]
            self._dirty.add(epic)

    def on_minute(self, epic: str, minute_ms: int, o: float, h: float, l: float, c: float, v: float):
        """A 1-minute candle from the stream (forming or complete)."""
        with self.lock:
            bucket = minute_ms - minute_ms % self.ms
            mins = self._minutes.setdefault(epic, {}).setdefault(bucket, {})
            mins[minute_ms] = (o, h, l, c, v)
            ks = sorted(mins)
            self.bars.setdefault(epic, {})[bucket] = [
                mins[ks[0]][0], max(m[1] for m in mins.values()), min(m[2] for m in mins.values()),
                mins[ks[-1]][3], float(sum(m[4] for m in mins.values()))]
            old = [b for b in self._minutes[epic] if b < bucket - 2 * self.ms]
            for b in old:
                del self._minutes[epic][b]
            self._dirty.add(epic)

    # ---- data out
    def has(self, epic: str) -> bool:
        with self.lock:
            return bool(self.bars.get(epic))

    def last_time(self, epic: str) -> int | None:
        with self.lock:
            rows = self.bars.get(epic)
            return max(rows) if rows else None

    def frame(self, epic: str, n: int) -> pd.DataFrame:
        with self.lock:
            rows = self.bars.get(epic) or {}
            keys = sorted(rows)[-int(n):]
            df = pd.DataFrame([[k, *rows[k]] for k in keys], columns=COLUMNS)
        df["symbol"] = epic
        return df


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


class IGStream:
    """CHART:<epic>:1MINUTE subscriptions feeding a CandleStore."""

    def __init__(self, endpoint: str, account: str, cst: str, xst: str, store: CandleStore,
                 client_factory=None, max_subscriptions: int = MAX_SUBSCRIPTIONS):
        self.endpoint, self.account = endpoint, account
        self.password = f"CST-{cst}|XST-{xst}"
        self.store = store
        self.max = int(max_subscriptions)
        self._factory = client_factory
        self.client = None
        self.status = "DISCONNECTED"
        self.status_since = time.time()
        self.subs: dict[str, object] = {}
        self.last: dict[str, tuple[float, float]] = {}       # epic -> (mid, when)
        self.errors: dict[str, str] = {}
        self.lock = threading.RLock()

    # ---- connection
    def start(self):
        if self._factory is not None:
            client = self._factory(self.endpoint, "DEFAULT")
        else:
            from lightstreamer.client import LightstreamerClient   # pip install lightstreamer-client-lib
            client = LightstreamerClient(self.endpoint, "DEFAULT")
        client.connectionDetails.setUser(self.account)
        client.connectionDetails.setPassword(self.password)
        client.addListener(_ClientListener(self))
        client.connect()
        self.client = client
        logger.info("IG stream connecting to %s", self.endpoint)

    def stop(self):
        with self.lock:
            if self.client is not None:
                try:
                    self.client.disconnect()
                except Exception:  # noqa: BLE001
                    logger.debug("IG stream disconnect failed", exc_info=True)
            self.client = None
            self.subs.clear()
            self.status = "DISCONNECTED"

    def _on_status(self, status: str):
        if status != self.status:
            logger.info("IG stream: %s", status)
            self.status, self.status_since = status, time.time()

    @property
    def connected(self) -> bool:
        return str(self.status).startswith("CONNECTED")

    # ---- subscriptions
    def set_epics(self, epics: list[str]) -> list[str]:
        """Stream exactly these markets (at most max_subscriptions); returns
        the ones streamed."""
        want = list(dict.fromkeys(epics))[: self.max]
        with self.lock:
            for epic in [e for e in self.subs if e not in want]:
                try:
                    self.client.unsubscribe(self.subs.pop(epic))
                except Exception:  # noqa: BLE001
                    logger.debug("IG unsubscribe %s failed", epic, exc_info=True)
            for epic in [e for e in want if e not in self.subs]:
                sub = self._subscription(epic)
                self.client.subscribe(sub)
                self.subs[epic] = sub
        return want

    def _subscription(self, epic: str):
        items, fields = [f"CHART:{epic}:1MINUTE"], FIELDS
        if self._factory is not None and hasattr(self._factory, "subscription"):
            sub = self._factory.subscription("MERGE", items, fields)
        else:
            from lightstreamer.client import Subscription
            sub = Subscription("MERGE", items, fields)
        sub.addListener(_SubListener(self, epic))
        return sub

    # ---- updates
    def on_update(self, epic: str, get):
        bo, bh, bl, bc = (_num(get(f)) for f in ("BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE"))
        oo, oh, ol, oc = (_num(get(f)) for f in ("OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE"))
        utm = _num(get("UTM"))
        if None in (bc, oc, utm):
            return

        def mid(b, o, fallback):
            return (b + o) / 2 if b is not None and o is not None else fallback
        c = (bc + oc) / 2
        o = mid(bo, oo, c)
        h = mid(bh, oh, max(o, c))
        l = mid(bl, ol, min(o, c))
        minute = int(utm) - int(utm) % 60_000
        self.store.on_minute(epic, minute, o, max(h, o, c), min(l, o, c), c, _num(get("CONS_TICK_COUNT")) or 0.0)
        self.last[epic] = (c, time.time())

    def price(self, epic: str, max_age_s: float = 60) -> float | None:
        v = self.last.get(epic)
        return v[0] if v and time.time() - v[1] <= max_age_s else None

    def summary(self) -> dict:
        now = time.time()
        return {"status": self.status, "since": self.status_since, "subscriptions": len(self.subs),
                "max": self.max, "errors": dict(list(self.errors.items())[:10]),
                "stale": sorted(e for e in self.subs if now - self.last.get(e, (0, 0))[1] > 300)}


class _ClientListener(_ClientBase):
    def __init__(self, stream: IGStream):
        self.stream = stream

    def onStatusChange(self, status):
        self.stream._on_status(str(status))

    def onServerError(self, code, message):
        logger.warning("IG stream server error %s: %s", code, message)
        self.stream.errors["server"] = f"{code}: {message}"

    def onListenStart(self, *a):
        pass

    def onListenEnd(self, *a):
        pass

    def onPropertyChange(self, *a):
        pass


class _SubListener(_SubBase):
    def __init__(self, stream: IGStream, epic: str):
        self.stream, self.epic = stream, epic

    def onItemUpdate(self, update):
        try:
            self.stream.on_update(self.epic, update.getValue)
        except Exception:  # noqa: BLE001
            logger.exception("IG stream update for %s failed", self.epic)

    def onSubscriptionError(self, code, message):
        logger.warning("IG stream: %s refused (%s %s)", self.epic, code, message)
        self.stream.errors[self.epic] = f"{code}: {message}"

    def __getattr__(self, name):            # the other SubscriptionListener callbacks: ignore
        if name.startswith("on"):
            return lambda *a, **k: None
        raise AttributeError(name)
