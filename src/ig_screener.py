"""
Around-the-clock screener for IG (broker market "IG24").

1. Universe - forex, index, commodity and crypto CFDs, found by browsing
   IG's market navigation. That takes a few minutes (IG allows 30 such
   requests a minute), so it runs in the background and is cached in
   data/ig_universe.json for `universe_refresh_hours`.
2. Ranking - every `refresh_minutes`, one price snapshot per 50 markets (no
   history allowance used). A market qualifies when IG says TRADEABLE, its
   market hours are open, its spread is at most `max_spread_pct`
   (`crypto_max_spread_pct` for crypto) and today's
   range at least `min_range_pct`. Score (0-100): 70% how much it moves for
   what it costs (range / spread), 30% how far it has moved today.
3. The best `max_watch` (at most 40, IG's streaming limit) are streamed and
   scanned. Markets with an open agent trade are always kept.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from src import ig_hours

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_FILE = ROOT / "data" / "ig_universe.json"

DEFAULTS = {
    "enabled": True,
    "markets": ["forex", "indices", "commodities", "crypto"],
    "max_watch": 30,
    "refresh_minutes": 15,
    "universe_refresh_hours": 24,
    "max_spread_pct": 0.2,
    "crypto_max_spread_pct": 1.5,      # crypto CFD spreads are wider (often 0.5-1%)
    "min_range_pct": 0.1,
    "seed_bars": 150,
    "max_nav_nodes": 120,
}


def settings(config: dict) -> dict:
    s = {**DEFAULTS, **(((config.get("screener") or {}).get("around_the_clock")) or {})}
    s["markets"] = [str(m).strip().lower() for m in (s.get("markets") or []) if str(m).strip().lower() in ig_hours.CLASSES] \
        or list(ig_hours.CLASSES)
    s["max_watch"] = max(1, min(40, int(s.get("max_watch") or 30)))
    return s


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    n = len(values)
    for rank, i in enumerate(order):
        out[i] = 100.0 * rank / (n - 1) if n > 1 else 100.0
    return out


class IG24Screener:
    def __init__(self, broker, config: dict, universe_file: Path | None = None, background: bool = True,
                 open_symbols=None):
        self.broker = broker
        self.cfg = settings(config)
        self.timeframe = config.get("timeframe")
        self.file = Path(universe_file or UNIVERSE_FILE)
        self.background = background
        self.open_symbols = open_symbols or (lambda: [])
        self.candidate_symbols = list((config.get("screener") or {}).get("candidate_symbols") or [])
        self.universe: list[dict] = []
        self.universe_updated = 0.0
        self.discovering = None            # {"nodes": n, "markets": m} while a discovery runs
        self._thread = None
        self.last_report: dict | None = None
        self._load()

    # ---- universe
    def _load(self):
        try:
            data = json.loads(self.file.read_text())
            if sorted(data.get("classes") or []) == sorted(self.cfg["markets"]):
                self.universe = data.get("markets") or []
                self.universe_updated = float(data.get("updated") or 0)
        except (OSError, ValueError):
            pass

    def _discover(self):
        try:
            markets = self.broker.discover_markets(
                self.cfg["markets"], max_nodes=int(self.cfg["max_nav_nodes"]),
                progress=lambda n, m: self.__setattr__("discovering", {"nodes": n, "markets": m}))
            if markets:
                self.universe, self.universe_updated = markets, time.time()
                self.file.parent.mkdir(parents=True, exist_ok=True)
                self.file.write_text(json.dumps({"updated": self.universe_updated, "classes": self.cfg["markets"],
                                                 "markets": markets}, indent=1))
                logger.info("IG universe: %d markets (%s)", len(markets), ", ".join(self.cfg["markets"]))
            else:
                logger.warning("IG found no markets for %s - trying again in 10 min", self.cfg["markets"])
                self._failed_at = time.time()
        except Exception:
            logger.exception("IG market discovery failed - trying again in 10 min")
            self._failed_at = time.time()
        finally:
            self.discovering = None

    def _ensure_universe(self):
        stale = time.time() - self.universe_updated > float(self.cfg["universe_refresh_hours"]) * 3600
        if not stale or (self._thread is not None and self._thread.is_alive()):
            return
        if time.time() - getattr(self, "_failed_at", 0.0) < 600:
            return                                  # the last attempt failed: wait 10 min
        self.discovering = {"nodes": 0, "markets": 0}
        if self.background:
            self._thread = threading.Thread(target=self._discover, name="ig-discover", daemon=True)
            self._thread.start()
        else:
            self._discover()

    # ---- ranking
    def run(self) -> list[dict]:
        # The agent retries an empty result every cycle; don't re-poll IG
        # more than once a minute for it.
        if time.time() - getattr(self, "_ran", 0.0) < 60 and getattr(self, "_last", None) is not None:
            return self._last
        self._ran = time.time()
        self._last = self._run()
        return self._last

    def _run(self) -> list[dict]:
        self._ensure_universe()
        meta = {m["epic"]: m for m in self.universe}
        for sym in self.candidate_symbols:
            meta.setdefault(sym, {"epic": sym, "name": sym, "class": ig_hours.classify(None, sym)})
        if hasattr(self.broker, "classes"):
            self.broker.classes.update({e: m.get("class") for e, m in meta.items() if m.get("class")})
            self.broker.names.update({e: m.get("name") for e, m in meta.items()})
        if not meta:
            self._report([], 0)
            return []
        snaps = self.broker.market_snapshots(list(meta))
        cands = []
        for epic, m in meta.items():
            s = snaps.get(epic) or {}
            bid, offer = s.get("bid"), s.get("offer")
            if not bid or not offer:
                continue
            mid = (bid + offer) / 2
            cls = m.get("class") or "indices"
            spread = (offer - bid) / mid * 100
            rng = ((s.get("high") or mid) - (s.get("low") or mid)) / mid * 100
            c = {"symbol": epic, "name": m.get("name") or epic, "class": cls, "status": s.get("status"),
                 "price": round(mid, 6), "spread_pct": round(spread, 4), "range_pct": round(rng, 3),
                 "change_pct": s.get("change_pct"), "open_now": ig_hours.is_open(cls)}
            why = None
            if c["status"] != "TRADEABLE":
                why = "not tradeable now"
            elif not c["open_now"]:
                why = "market hours closed"
            elif spread > float(self.cfg["crypto_max_spread_pct" if cls == "crypto" else "max_spread_pct"]):
                why = "spread too wide"
            elif rng < float(self.cfg["min_range_pct"]):
                why = "too quiet today"
            c["excluded"] = why
            cands.append(c)
        ok = [c for c in cands if not c["excluded"]]
        if ok:
            room = _ranks([c["range_pct"] / max(c["spread_pct"], 0.005) for c in ok])
            move = _ranks([abs(c["change_pct"] or 0.0) for c in ok])
            for c, r, mv in zip(ok, room, move):
                c["score"] = round(0.7 * r + 0.3 * mv, 1)
        ok.sort(key=lambda c: c["score"], reverse=True)
        keep = [s for s in self.open_symbols() if s in meta]
        picks = list(dict.fromkeys(keep + [c["symbol"] for c in ok]))[: self.cfg["max_watch"]]
        watched = self.broker.watch(picks) if hasattr(self.broker, "watch") else picks
        streamed = set(watched)
        for c in cands:
            c["picked"] = c["symbol"] in streamed
        cands.sort(key=lambda c: (c["excluded"] is not None, -(c.get("score") or 0)))
        self._report(cands, len(meta))
        by = {c["symbol"]: c for c in cands}
        return [{"symbol": e, "name": by.get(e, {}).get("name", e)} for e in watched]

    def _report(self, cands: list[dict], universe: int):
        summary = self.broker.stream_summary() if hasattr(self.broker, "stream_summary") else {}
        self.last_report = {
            "updated": time.time(), "mode": "ig24", "timeframe": self.timeframe, "ranked_by": "score",
            "universe": universe, "top_n": self.cfg["max_watch"], "sources": [],
            "classes": self.cfg["markets"], "universe_updated": self.universe_updated,
            "discovering": self.discovering, "filters": {k: self.cfg[k] for k in ("max_spread_pct", "crypto_max_spread_pct", "min_range_pct")},
            "stream": summary,
            "next_open": {c: (ig_hours.next_open(c).timestamp() if ig_hours.next_open(c) else None)
                          for c in self.cfg["markets"]},
            "candidates": cands[:80],
        }
