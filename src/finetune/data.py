"""Collecting candles for fine-tuning and turning them into training windows."""
from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import TRAINING_DIR

logger = logging.getLogger(__name__)

COLUMNS = ["time", "open", "high", "low", "close", "volume"]
FEATURES = ["open", "high", "low", "close", "volume", "amount"]
CLIP = 5.0


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._&-]", "_", str(name))


def bars_path(broker: str, symbol: str, timeframe: str, root: Path | None = None) -> Path:
    return (root or TRAINING_DIR) / _safe(broker) / f"{_safe(symbol)}__{_safe(timeframe)}.csv"


def merge_bars(path: Path, new: pd.DataFrame) -> int:
    """Add candles to a symbol's file (time = epoch ms), newest version of a
    bar wins. Returns how many bars the file holds afterwards."""
    new = new[[c for c in COLUMNS if c in new.columns]].copy()
    if "volume" not in new:
        new["volume"] = 0.0
    new = new.dropna(subset=["time", "open", "high", "low", "close"])
    if path.exists():
        try:
            new = pd.concat([pd.read_csv(path), new], ignore_index=True)
        except Exception:
            logger.warning("Unreadable training file %s - rewriting it", path)
    new["time"] = new["time"].astype("int64")
    new = new.drop_duplicates("time", keep="last").sort_values("time")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    new[COLUMNS].to_csv(tmp, index=False)
    tmp.replace(path)
    return len(new)


def symbols_from_audit(db_path, days: float = 30) -> list[str]:
    """Symbols the agent decided on or traded recently (read-only)."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return []
    out = []
    try:
        since = time.time() - days * 86400
        for sql in ("SELECT DISTINCT epic FROM decisions WHERE ts >= ?",
                    "SELECT DISTINCT epic FROM trade_results WHERE ts >= ?"):
            try:
                out += [r[0] for r in conn.execute(sql, (since,)) if r[0]]
            except sqlite3.Error:
                pass
    finally:
        conn.close()
    return list(dict.fromkeys(out))


def signal_times(db_path) -> dict[str, list[float]]:
    """{symbol: [epoch seconds of every signal the agent took to Kronos]}."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return {}
    out: dict[str, list[float]] = {}
    try:
        for sym, ts in conn.execute("SELECT epic, ts FROM decisions WHERE kronos_confidence IS NOT NULL"):
            out.setdefault(sym, []).append(float(ts))
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return out


def collect(broker, broker_name: str, timeframe: str, symbols: list[str], bars: int = 1500,
            max_symbols: int = 150, drop_last: bool = False, root: Path | None = None) -> dict:
    """Fetch recent candles for `symbols` and merge them into the training
    files. `drop_last` leaves out the newest bar (still forming while the
    market is open). One failing symbol doesn't stop the rest."""
    done, failed, total = 0, [], 0
    for symbol in list(dict.fromkeys(symbols))[:max_symbols]:
        try:
            df = broker.get_historical_bars(symbol, timeframe, bars)
            if df is None or len(df) < 2:
                raise ValueError("no bars")
            if drop_last:
                df = df.iloc[:-1]
            total += merge_bars(bars_path(broker_name, symbol, timeframe, root), df)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed.append(symbol)
            logger.info("Training bars for %s not collected: %s", symbol, str(exc)[:120])
    summary = {"broker": broker_name, "timeframe": timeframe, "symbols": done, "failed": len(failed),
               "bars": total, "ts": time.time()}
    logger.info("Fine-tuning data: %d symbols (%d bars in total) saved for %s %s%s", done, total, broker_name,
                timeframe, f", {len(failed)} failed" if failed else "")
    return summary


def inventory(root: Path | None = None) -> list[dict]:
    """What has been collected: one row per broker + timeframe."""
    root = root or TRAINING_DIR
    out = {}
    if not root.exists():
        return []
    for f in sorted(root.glob("*/*.csv")):
        tf = f.stem.split("__")[-1]
        row = out.setdefault((f.parent.name, tf), {"broker": f.parent.name, "timeframe": tf, "symbols": 0,
                                                    "bars": 0, "updated": 0.0})
        row["symbols"] += 1
        with open(f, "rb") as fh:
            row["bars"] += max(0, sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b"")) - 1)
        row["updated"] = max(row["updated"], f.stat().st_mtime)
    return list(out.values())


# ---------------------------------------------------------------- windows

class Series:
    """One symbol's candles as arrays, prepared the way the agent feeds
    Kronos: UTC timestamps, amount = volume x mean price."""

    def __init__(self, path: Path):
        df = pd.read_csv(path).dropna(subset=["open", "high", "low", "close"])
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].sort_values("time")
        df["volume"] = df["volume"].fillna(0.0) if "volume" in df else 0.0
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
        ts = pd.to_datetime(df["time"], unit="ms")
        self.path = path
        self.symbol = path.stem.split("__")[0]
        self.broker = path.parent.name
        self.time = df["time"].to_numpy(dtype="int64")
        self.x = df[FEATURES].to_numpy(dtype=np.float32)
        self.stamp = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], axis=1).astype(np.float32)
        self.timestamps = ts.reset_index(drop=True)

    def __len__(self):
        return len(self.time)


class WindowSet:
    """Training and held-out windows over all collected series.

    The split is by date: everything after the cut-off (the most recent
    `val_fraction` of all candles) is held out. A training window lies
    entirely before the cut-off; a held-out window's predicted part lies
    entirely after it."""

    def __init__(self, series: list[Series], lookback: int, predict: int, val_fraction: float = 0.15,
                 signals: dict[tuple[str, str], list[float]] | None = None, seed: int = 7):
        self.series = [s for s in series if len(s) >= lookback + predict + 20]
        self.lookback, self.predict, self.window = lookback, predict, lookback + predict
        self.rng = np.random.default_rng(seed)
        self.train: list[tuple[int, int]] = []
        self.val: list[tuple[int, int]] = []
        self.train_signal: list[tuple[int, int]] = []
        self.val_signal: list[tuple[int, int]] = []
        self.cutoff_ms = None
        if not self.series:
            return
        all_times = np.concatenate([s.time for s in self.series])
        self.cutoff_ms = int(np.quantile(all_times, 1 - val_fraction))
        for si, s in enumerate(self.series):
            n_starts = len(s) - self.window + 1
            ends = s.time[self.window - 1:]                        # last bar of each window
            first_pred = s.time[lookback:lookback + n_starts]      # first predicted bar of each window
            train_idx = np.nonzero(ends < self.cutoff_ms)[0]
            val_idx = np.nonzero(first_pred >= self.cutoff_ms)[0]
            self.train += [(si, int(i)) for i in train_idx]
            self.val += [(si, int(i)) for i in val_idx]
            sig = (signals or {}).get((s.broker, s.symbol)) or []
            if sig:
                # the window whose lookback ends on the bar the signal fired on
                bar = np.searchsorted(s.time, np.asarray(sig) * 1000.0, side="right") - 1
                starts = np.unique(bar + 1 - lookback)
                starts = starts[(starts >= 0) & (starts < n_starts)]
                tset, vset = set(train_idx.tolist()), set(val_idx.tolist())
                self.train_signal += [(si, int(i)) for i in starts if int(i) in tset]
                self.val_signal += [(si, int(i)) for i in starts if int(i) in vset]

    def _window(self, si: int, start: int):
        s = self.series[si]
        x = s.x[start:start + self.window]
        past = x[:self.lookback]
        x = np.clip((x - past.mean(axis=0)) / (past.std(axis=0) + 1e-5), -CLIP, CLIP)
        return x, s.stamp[start:start + self.window]

    def batch(self, items):
        xs, st = zip(*(self._window(si, i) for si, i in items))
        return np.stack(xs).astype(np.float32), np.stack(st).astype(np.float32)

    def sample_train(self, batch_size: int, signal_fraction: float = 0.3):
        n_sig = int(round(batch_size * signal_fraction)) if self.train_signal else 0
        picks = [self.train_signal[i] for i in self.rng.integers(0, len(self.train_signal), n_sig)] if n_sig else []
        picks += [self.train[i] for i in self.rng.integers(0, len(self.train), batch_size - n_sig)]
        return self.batch(picks)

    def val_items(self, n: int) -> list[tuple[int, int]]:
        """A fixed, evenly spread set of held-out windows (signal windows first)."""
        items = list(dict.fromkeys(self.val_signal))[: n // 2]
        rest = [v for v in self.val if v not in set(items)]
        if rest and n - len(items) > 0:
            step = max(1, len(rest) // (n - len(items)))
            items += rest[::step][: n - len(items)]
        return items


def all_signals(audit_dbs: dict[str, Path] | None) -> dict[str, dict[str, list[float]]]:
    """{broker: {symbol: [signal times]}} from every broker's audit log."""
    return {broker: signal_times(db) for broker, db in (audit_dbs or {}).items()}


def load_window_set(lookback: int, predict: int, val_fraction: float, audit_dbs: dict[str, Path] | None = None,
                    root: Path | None = None, seed: int = 7,
                    signals_by_broker: dict[str, dict[str, list[float]]] | None = None) -> WindowSet:
    root = root or TRAINING_DIR
    series = []
    for f in sorted(root.glob("*/*.csv")):
        try:
            series.append(Series(f))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping training file %s: %s", f, exc)
    signals = {}
    by_broker = signals_by_broker if signals_by_broker is not None else all_signals(audit_dbs)
    for broker, per_symbol in by_broker.items():
        for sym, ts in per_symbol.items():
            signals[(_safe(broker), _safe(sym))] = ts
    return WindowSet(series, lookback, predict, val_fraction, signals, seed)
