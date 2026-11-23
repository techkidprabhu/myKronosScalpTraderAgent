"""
Starter scalp engine for skill-file trading: `strategy: scalp_vwap_orb`.

Common 1-5 minute scalping methods, written as a placeholder until your own
scalp rules replace it. Four setups ("models"), long and short, on CLOSED
candles; `setups:` picks which run (`both` = ORB + pullback, as before):

  orb             Momentum / breakout      - opening-range breakout on volume
  pullback        Trend pullback           - VWAP / fast-EMA pullback in a trend
  mean_reversion  Mean reversion           - Bollinger + RSI extreme, then a
                                             candle back inside; ranging only
  sweep           Liquidity sweep          - a wick through a prior high/low that
                                             closes back inside (spring/upthrust)

When several could fire on the same candle the order is: orb, sweep,
pullback, mean_reversion.

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

Mean reversion
  The previous candle closed outside the Bollinger band (`bb_period`,
  `bb_std`) with RSI at an extreme (<= `rsi_low` for longs, >= `rsi_high`
  for shorts); the newest candle closes back inside the band, in the
  reversal's colour. Only when the market was ranging before the spike: ADX
  below `mr_max_adx` (0 = no check). Stop beyond the spike; target the middle band,
  and only if that is at least `mr_min_r` x the risk.

Liquidity sweep
  The newest candle trades through a level where stops sit - the prior
  session's high/low and/or the recent swing high/low over `sweep_lookback`
  candles (`sweep_levels`) - then closes back inside it, with a rejection
  wick of at least `sweep_wick_ratio` of the candle and the close in the
  half away from the sweep. Stop beyond the wick; target `target_r` x risk.

Entry: a limit at the signal candle's close (it fills at once or within
`max_wait_candles`). Exits, besides the stop, target, time stop and
break-even stop: for ORB and pullback trades a close back across VWAP against
the trade (`exit_on_vwap_cross`) - not for mean-reversion or sweep trades,
which often start on the far side of VWAP - and `max_hold_candles`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class ScalpSettings:
    strategy: str = "scalp_vwap_orb"
    setups: str = "both"                # both (= orb, pullback) | all | a list of: orb, pullback,
                                        #   mean_reversion, sweep
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
    # mean reversion
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_low: float = 30.0
    rsi_high: float = 70.0
    adx_period: int = 14
    mr_max_adx: float = 20.0            # only when ADX is below this (ranging); 0 = no check
    mr_min_r: float = 1.0               # the middle band must be at least this many R away
    # liquidity sweep
    sweep_levels: str = "both"          # both | prior_day | swing
    sweep_lookback: int = 20            # swing high / low over this many candles
    sweep_wick_ratio: float = 0.5       # rejection wick at least this share of the candle

    def summary(self) -> dict:
        return asdict(self)


MODELS = ("orb", "pullback", "mean_reversion", "sweep")
ALIASES = {"both": ("orb", "pullback"), "all": MODELS, "breakout": ("orb",), "momentum": ("orb",),
           "mr": ("mean_reversion",), "reversion": ("mean_reversion",), "liquidity_sweep": ("sweep",)}
VWAP_EXIT_SETUPS = (None, "", "ORB", "VWAP pullback")


def parse_setups(value) -> tuple[set, list]:
    """`setups:` as written in the skill file -> (models, unknown names).
    Accepts both / all / one name / a comma list / a YAML list."""
    raw = value if isinstance(value, (list, tuple)) else str(value or "both")
    if not isinstance(raw, (list, tuple)):
        raw = raw.strip().strip("[]").replace(";", ",").replace(" ", ",").split(",")
    chosen, unknown = set(), []
    for item in raw:
        name = str(item).strip().strip("'\"").lower().replace("-", "_")
        if not name:
            continue
        if name in MODELS:
            chosen.add(name)
        elif name in ALIASES:
            chosen.update(ALIASES[name])
        else:
            unknown.append(name)
    return chosen, unknown


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


def _rsi(c: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = up / dn
    return (100 - 100 / (1 + rs)).fillna(50).to_numpy()


def _bands(c: np.ndarray, n: int, k: float):
    sr = pd.Series(c)
    mid = sr.rolling(n, min_periods=n).mean()
    sd = sr.rolling(n, min_periods=n).std(ddof=0)
    return (mid - k * sd).to_numpy(), mid.to_numpy(), (mid + k * sd).to_numpy()


def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    """Wilder's ADX: trend strength, 0-100 (below ~20 = ranging)."""
    up, dn = np.diff(h, prepend=h[0]), -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    w = lambda x: pd.Series(x).ewm(alpha=1 / n, adjust=False).mean()  # noqa: E731
    atr = w(tr)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi, ndi = 100 * w(pdm) / atr, 100 * w(ndm) / atr
        dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    return w(dx.fillna(0)).to_numpy()


def find_setup(df: pd.DataFrame, s: ScalpSettings, Setup):
    models, _ = parse_setups(s.setups)
    models = models or set(ALIASES["both"])
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

    def make(direction, kind, entry, stop, target=None):
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        if target is None:
            target = entry + s.target_r * risk if direction == "BUY" else entry - s.target_r * risk
        q = 1 if vol_ratio >= 2 else 0
        if q < s.min_quality:
            return Setup("invalid", f"{kind}: quality {q} below min_quality {s.min_quality}", details=det)
        return Setup("ready", f"{kind} {'long' if direction == 'BUY' else 'short'}", entry=float(entry),
                     stop=float(stop), target=float(target), quality=q, direction=direction,
                     details={**det, "setup": kind, "signal_bar": pd.Timestamp(d.index[i]).isoformat(),
                              "breakout": pd.Timestamp(d.index[i]).isoformat()})

    buf = s.stop_buffer_pct / 100
    # --- opening-range breakout (momentum / breakout)
    if "orb" in models:
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
    # --- liquidity sweep: a wick through a prior high / low that closes back inside
    if "sweep" in models:
        r = _sweep(df, d, s, i, o, h, l, c, buf, long_ok, short_ok, det, make)
        if r:
            return r
    # --- VWAP / EMA pullback (trend pullback)
    if "pullback" in models and i >= 1:
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
    # --- mean reversion: back inside the band after an RSI extreme, ranging market only
    if "mean_reversion" in models and i >= 1:
        r = _mean_reversion(df, s, off, i, o, h, l, c, buf, long_ok, short_ok, det, make)
        if r:
            return r
    trend = "up" if ef > es and c[i] > vw[i] else "down" if ef < es and c[i] < vw[i] else "mixed"
    return Setup("waiting", f"no scalp signal (trend {trend}, volume {vol_ratio:.1f}x)", details=det)


def _mean_reversion(df, s, off, i, o, h, l, c, buf, long_ok, short_ok, det, make):
    full_c = df["close"].to_numpy(float)
    if len(full_c) < max(s.bb_period, s.rsi_period, 2 * s.adx_period) + 2:
        return None
    lo, mid, hi = _bands(full_c, s.bb_period, s.bb_std)
    rsi = _rsi(full_c, s.rsi_period)
    adx = _adx(df["high"].to_numpy(float), df["low"].to_numpy(float), full_c, s.adx_period)
    k, kj = off + i, off + i - 1
    # The market's state is judged BEFORE the spike: the spike itself lifts ADX.
    kr = max(0, kj - 1)
    det.update(bb_low=round(float(lo[k]), 6), bb_mid=round(float(mid[k]), 6), bb_high=round(float(hi[k]), 6),
               rsi=round(float(rsi[kj]), 1), adx=round(float(adx[kr]), 1))
    if np.isnan(mid[k]) or np.isnan(lo[kj]):
        return None
    if s.mr_max_adx and adx[kr] >= s.mr_max_adx:
        return None                                   # trending: fading it is the wrong model
    j = i - 1
    if long_ok and c[j] < lo[kj] and rsi[kj] <= s.rsi_low and c[i] > lo[k] and c[i] > o[i]:
        entry, stop, target = c[i], min(l[j], l[i]) * (1 - buf), float(mid[k])
        if target - entry >= s.mr_min_r * (entry - stop) > 0:
            return make("BUY", "mean reversion", entry, stop, target)
    if short_ok and c[j] > hi[kj] and rsi[kj] >= s.rsi_high and c[i] < hi[k] and c[i] < o[i]:
        entry, stop, target = c[i], max(h[j], h[i]) * (1 + buf), float(mid[k])
        if entry - target >= s.mr_min_r * (stop - entry) > 0:
            return make("SELL", "mean reversion", entry, stop, target)
    return None


def _sweep(df, d, s, i, o, h, l, c, buf, long_ok, short_ok, det, make):
    rng = h[i] - l[i]
    if rng <= 0:
        return None
    lows, highs = {}, {}
    if s.sweep_levels in ("both", "prior_day"):
        idx = pd.DatetimeIndex(df.index)
        today = idx[-1].date()
        prev_days = sorted({t.date() for t in idx if t.date() < today})
        if prev_days:
            pdm = np.asarray([t.date() == prev_days[-1] for t in idx])
            lows["prior day low"] = float(df["low"].to_numpy(float)[pdm].min())
            highs["prior day high"] = float(df["high"].to_numpy(float)[pdm].max())
    if s.sweep_levels in ("both", "swing"):
        fh, fl = df["high"].to_numpy(float), df["low"].to_numpy(float)
        win = slice(max(0, len(df) - 1 - s.sweep_lookback), len(df) - 1)
        if win.stop - win.start >= 3:
            lows["swing low"] = float(fl[win].min())
            highs["swing high"] = float(fh[win].max())
    body_lo, body_hi = min(o[i], c[i]), max(o[i], c[i])
    if long_ok:
        swept = [(lvl, name) for name, lvl in lows.items() if l[i] < lvl < c[i]]
        if swept and (body_lo - l[i]) >= s.sweep_wick_ratio * rng and c[i] >= l[i] + rng / 2:
            lvl, name = max(swept)
            det.update(swept_level=round(lvl, 6), swept=name)
            r = make("BUY", "liquidity sweep", c[i], l[i] * (1 - buf))
            if r:
                return r
    if short_ok:
        swept = [(lvl, name) for name, lvl in highs.items() if c[i] < lvl < h[i]]
        if swept and (h[i] - body_hi) >= s.sweep_wick_ratio * rng and c[i] <= h[i] - rng / 2:
            lvl, name = min(swept)
            det.update(swept_level=round(lvl, 6), swept=name)
            r = make("SELL", "liquidity sweep", c[i], h[i] * (1 + buf))
            if r:
                return r
    return None


def exit_signal(df: pd.DataFrame, s: ScalpSettings, opened_at: str | None, direction: str, ExitSignal,
                setup: str | None = None):
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
    if s.exit_on_vwap_cross and setup in VWAP_EXIT_SETUPS:
        d = _session(df)
        vw = _vwap(d)
        c = float(d["close"].iloc[-1])
        if direction == "BUY" and c < vw[-1]:
            return ExitSignal("close_now", "closed back below VWAP")
        if direction == "SELL" and c > vw[-1]:
            return ExitSignal("close_now", "closed back above VWAP")
    return None
