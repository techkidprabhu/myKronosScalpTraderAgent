"""
Starter scalp engine for skill-file trading: `strategy: scalp_vwap_orb`.

A common 3-minute scalping method, written as a placeholder until your
own scalp rules replace it. Two setups, long and short, on CLOSED candles of
the current session only:

Opening-range breakout (ORB)
  The opening range is the high / low of the first `orb_minutes` of the
  session - by default the first 3-minute candle. A candle that closes beyond it - on the same side of VWAP, with
  volume at least `volume_mult` x the recent average, within
  `orb_window_minutes` of the open - is the signal. Stop at the middle of
  the range (or beyond the breakout candle, `orb_stop: candle`); target
  `target_r` x the risk.

VWAP pullback
  In a trend (EMA fast above EMA slow and price above VWAP for longs; the
  mirror for shorts), price pulls back to VWAP or the fast EMA within
  `pullback_touch_pct`, and the next candle closes back in the trend's
  direction beyond the fast EMA. Stop beyond the pull-back's extreme; target
  `target_r` x the risk.

Entry: a limit at the signal candle's close (it fills at once or within
`max_wait_candles`). Exits, besides the stop, target, time stop and
break-even stop: a close back across VWAP against the trade
(`exit_on_vwap_cross`), and `max_hold_candles`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class ScalpSettings:
    strategy: str = "scalp_vwap_orb"
    setups: str = "both"                # both | orb | pullback
    sides: str = "both"                 # both | long | short
    orb_minutes: int = 3                # 3 = the first 3-minute candle
    orb_window_minutes: int = 90        # ORB signals only this long after the open
    orb_stop: str = "mid"               # mid (of the range) | candle (beyond the breakout candle)
    volume_mult: float = 1.5            # signal candle volume vs the last volume_period candles
    volume_period: int = 20
    ema_fast: int = 9
    ema_slow: int = 20
    pullback_touch_pct: float = 0.05    # how close to VWAP / the fast EMA counts as a touch
    stop_buffer_pct: float = 0.02
    target_r: float = 1.5
    max_wait_candles: int = 2           # cancel the entry if not filled after this many candles
    max_hold_candles: int = 10          # close after this many candles in the trade
    exit_on_vwap_cross: bool = True
    use_trailing_stop: bool = True      # the agent's trailing stop on these trades
    min_quality: int = 0                # 0-1: +1 when volume is 2x the average

    def summary(self) -> dict:
        return asdict(self)


def _session(df: pd.DataFrame) -> pd.DataFrame:
    """Today's candles: those on the same calendar day as the newest one."""
    idx = pd.DatetimeIndex(df.index)
    day = idx[-1].date()
    return df[np.asarray([d.date() == day for d in idx])]


def _vwap(d: pd.DataFrame) -> np.ndarray:
    tp = (d["high"].to_numpy(float) + d["low"].to_numpy(float) + d["close"].to_numpy(float)) / 3
    vol = d["volume"].to_numpy(float) if "volume" in d else np.ones(len(d))
    vol = np.where(vol > 0, vol, 0)
    cum_v = np.cumsum(vol)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.cumsum(tp * vol) / cum_v
    return np.where(cum_v > 0, v, tp)


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def find_setup(df: pd.DataFrame, s: ScalpSettings, Setup):
    if df is None or len(df) < max(s.ema_slow, s.volume_period) + 2:
        return Setup("none", "not enough candles")
    full_c = df["close"].to_numpy(float)
    ema_f, ema_s = _ema(full_c, s.ema_fast), _ema(full_c, s.ema_slow)
    d = _session(df)
    n = len(d)
    if n < 2:
        return Setup("none", "the session has just started")
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    vol = d["volume"].to_numpy(float) if "volume" in d else np.ones(n)
    vw = _vwap(d)
    off = len(df) - n                       # session index -> full index
    i = n - 1
    ef, es = ema_f[off + i], ema_s[off + i]
    all_vol = df["volume"].to_numpy(float) if "volume" in df else np.ones(len(df))
    avg_vol = float(np.mean(all_vol[-s.volume_period - 1:-1])) or 1.0
    vol_ratio = float(vol[i]) / avg_vol
    long_ok, short_ok = s.sides in ("both", "long"), s.sides in ("both", "short")
    t0 = pd.Timestamp(d.index[0])
    mins = (pd.Timestamp(d.index[i]) - t0).total_seconds() / 60
    det = {"vwap": round(float(vw[i]), 6), "ema_fast": round(float(ef), 6), "ema_slow": round(float(es), 6),
           "volume_x": round(vol_ratio, 2), "minutes_since_open": int(mins)}

    def make(direction, kind, entry, stop):
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        target = entry + s.target_r * risk if direction == "BUY" else entry - s.target_r * risk
        q = 1 if vol_ratio >= 2 else 0
        if q < s.min_quality:
            return Setup("invalid", f"{kind}: quality {q} below min_quality {s.min_quality}", details=det)
        return Setup("ready", f"{kind} {'long' if direction == 'BUY' else 'short'}", entry=float(entry),
                     stop=float(stop), target=float(target), quality=q, direction=direction,
                     details={**det, "setup": kind, "signal_bar": pd.Timestamp(d.index[i]).isoformat(),
                              "breakout": pd.Timestamp(d.index[i]).isoformat()})

    buf = s.stop_buffer_pct / 100
    # --- opening-range breakout
    if s.setups in ("both", "orb"):
        rng = np.asarray([(pd.Timestamp(t) - t0).total_seconds() / 60 < s.orb_minutes for t in d.index])
        if rng.sum() and not rng[i] and mins <= s.orb_window_minutes:
            or_h, or_l = float(h[rng].max()), float(l[rng].min())
            mid = (or_h + or_l) / 2
            det.update(or_high=or_h, or_low=or_l)
            # only the FIRST close beyond the range is a breakout signal
            first_up = c[i] > or_h and (rng[i - 1] or c[i - 1] <= or_h)
            first_dn = c[i] < or_l and (rng[i - 1] or c[i - 1] >= or_l)
            if vol_ratio >= s.volume_mult:
                if long_ok and first_up and c[i] > vw[i]:
                    stop = (mid if s.orb_stop == "mid" else l[i]) * (1 - buf)
                    r = make("BUY", "ORB", c[i], stop)
                    if r:
                        return r
                if short_ok and first_dn and c[i] < vw[i]:
                    stop = (mid if s.orb_stop == "mid" else h[i]) * (1 + buf)
                    r = make("SELL", "ORB", c[i], stop)
                    if r:
                        return r
    # --- VWAP / EMA pullback
    if s.setups in ("both", "pullback") and i >= 1:
        tol = s.pullback_touch_pct / 100
        j = i - 1
        ef_j = ema_f[off + j]
        if long_ok and ef > es and c[i] > vw[i] and c[i] > o[i] and c[i] > ef:
            touched = l[j] <= max(vw[j], ef_j) * (1 + tol) and c[j] >= min(vw[j], ef_j) * (1 - tol)
            if touched:
                stop = min(l[j], l[i]) * (1 - buf)
                r = make("BUY", "VWAP pullback", c[i], stop)
                if r:
                    return r
        if short_ok and ef < es and c[i] < vw[i] and c[i] < o[i] and c[i] < ef:
            touched = h[j] >= min(vw[j], ef_j) * (1 - tol) and c[j] <= max(vw[j], ef_j) * (1 + tol)
            if touched:
                stop = max(h[j], h[i]) * (1 + buf)
                r = make("SELL", "VWAP pullback", c[i], stop)
                if r:
                    return r
    trend = "up" if ef > es and c[i] > vw[i] else "down" if ef < es and c[i] < vw[i] else "mixed"
    return Setup("waiting", f"no scalp signal (trend {trend}, volume {vol_ratio:.1f}x)", details=det)


def exit_signal(df: pd.DataFrame, s: ScalpSettings, opened_at: str | None, direction: str, ExitSignal):
    if df is None or len(df) < 2:
        return None
    idx = pd.DatetimeIndex(df.index)
    since = 0
    if opened_at:
        ts = pd.Timestamp(opened_at)
        try:
            ii = idx.tz_localize(None) if idx.tz is not None else idx
            ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
            since = int((ii > ts).sum())
        except Exception:  # noqa: BLE001
            since = 0
        if since == 0:
            return None
    if s.max_hold_candles and since >= s.max_hold_candles:
        return ExitSignal("close_now", f"held {since} candles (max_hold_candles)")
    if s.exit_on_vwap_cross:
        d = _session(df)
        vw = _vwap(d)
        c = float(d["close"].iloc[-1])
        if direction == "BUY" and c < vw[-1]:
            return ExitSignal("close_now", "closed back below VWAP")
        if direction == "SELL" and c > vw[-1]:
            return ExitSignal("close_now", "closed back above VWAP")
    return None
