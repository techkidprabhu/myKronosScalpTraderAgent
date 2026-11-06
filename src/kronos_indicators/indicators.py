"""
Candle-based technical indicators for the Kronos swing-trading agent.

Pure pandas/numpy (no TA-Lib needed). Every function takes the same OHLCV
DataFrame format Kronos uses:

    columns: open, high, low, close, [volume], [amount]
    index or 'timestamps' column: datetimes, sorted ascending

and returns a pd.Series / pd.DataFrame aligned to the input index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, set a DatetimeIndex, sort, coerce to float."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    for ts_col in ("timestamps", "timestamp", "datetime", "date", "time"):
        if ts_col in out.columns:
            out[ts_col] = pd.to_datetime(out[ts_col])
            out = out.set_index(ts_col)
            break
    out = out.sort_index()
    missing = {"open", "high", "low", "close"} - set(out.columns)
    if missing:
        raise ValueError(f"OHLC columns missing: {sorted(missing)}")
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def _wilder(s: pd.Series, n: int) -> pd.Series:
    """
    Wilder's smoothing (RMA) seeded with the SMA of the first n values - the same
    convention TradingView, Zerodha/Angel charts and TA-Lib use for RSI / ATR / ADX.
    """
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    valid = np.flatnonzero(~np.isnan(vals))
    if len(valid) < n:
        return pd.Series(out, index=s.index)
    start = valid[0]
    seed_end = start + n
    if np.isnan(vals[start:seed_end]).any():  # gap inside the seed window: fall back to ewm
        return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    out[seed_end - 1] = vals[start:seed_end].mean()
    for i in range(seed_end, len(vals)):
        v = vals[i]
        out[i] = out[i - 1] if np.isnan(v) else out[i - 1] + (v - out[i - 1]) / n
    return pd.Series(out, index=s.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


# --------------------------------------------------------------------------- #
# moving averages
# --------------------------------------------------------------------------- #
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s: pd.Series, n: int) -> pd.Series:
    """Hull moving average: faster, less lag."""
    half, root = max(int(n / 2), 1), max(int(np.sqrt(n)), 1)
    return wma(2 * wma(s, half) - wma(s, n), root)


def vwma(df: pd.DataFrame, n: int) -> pd.Series:
    pv = (df["close"] * df["volume"]).rolling(n, min_periods=n).sum()
    v = df["volume"].rolling(n, min_periods=n).sum()
    return pv / v.replace(0, np.nan)


MA_FUNCS = {"sma": sma, "ema": ema, "wma": wma, "hma": hma}


def moving_average(df: pd.DataFrame, n: int, kind: str = "ema", source: str = "close") -> pd.Series:
    kind = kind.lower()
    if kind == "vwma":
        return vwma(df, n)
    if kind not in MA_FUNCS:
        raise ValueError(f"unknown MA kind '{kind}', use one of {list(MA_FUNCS) + ['vwma']}")
    return MA_FUNCS[kind](df[source], n)


# --------------------------------------------------------------------------- #
# momentum
# --------------------------------------------------------------------------- #
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain, loss = _wilder(d.clip(lower=0), n), _wilder(-d.clip(upper=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3) -> pd.DataFrame:
    ll = df["low"].rolling(k, min_periods=k).min()
    hh = df["high"].rolling(k, min_periods=k).max()
    raw = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    k_line = raw.rolling(smooth, min_periods=smooth).mean()
    return pd.DataFrame({"k": k_line, "d": k_line.rolling(d, min_periods=d).mean()})


def roc(s: pd.Series, n: int = 10) -> pd.Series:
    return 100 * (s / s.shift(n) - 1)


# --------------------------------------------------------------------------- #
# trend strength
# --------------------------------------------------------------------------- #
def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_ = _wilder(true_range(df), n)
    plus_di = 100 * _wilder(plus_dm, n) / atr_
    minus_di = 100 * _wilder(minus_dm, n) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": _wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Returns supertrend line and direction (+1 up-trend, -1 down-trend)."""
    atr_ = atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    upper_basic = (hl2 + mult * atr_).to_numpy()
    lower_basic = (hl2 - mult * atr_).to_numpy()
    close = df["close"].to_numpy()
    size = len(df)
    upper, lower = upper_basic.copy(), lower_basic.copy()
    direction = np.full(size, np.nan)
    line = np.full(size, np.nan)
    for i in range(size):
        if np.isnan(upper_basic[i]):
            continue
        if i == 0 or np.isnan(direction[i - 1]):
            direction[i] = 1.0
            line[i] = lower[i]
            continue
        upper[i] = upper_basic[i] if (upper_basic[i] < upper[i - 1] or close[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = lower_basic[i] if (lower_basic[i] > lower[i - 1] or close[i - 1] < lower[i - 1]) else lower[i - 1]
        if direction[i - 1] == 1 and close[i] < lower[i]:
            direction[i] = -1.0
        elif direction[i - 1] == -1 and close[i] > upper[i]:
            direction[i] = 1.0
        else:
            direction[i] = direction[i - 1]
        line[i] = lower[i] if direction[i] == 1 else upper[i]
    return pd.DataFrame({"supertrend": line, "direction": direction}, index=df.index)


# --------------------------------------------------------------------------- #
# volatility
# --------------------------------------------------------------------------- #
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _wilder(true_range(df), n)


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(s, n)
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower) / mid
    pct_b = (s - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "width": width, "pct_b": pct_b})


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 1.5) -> pd.DataFrame:
    mid = ema(df["close"], n)
    a = atr(df, n)
    return pd.DataFrame({"mid": mid, "upper": mid + mult * a, "lower": mid - mult * a})


def donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    hi = df["high"].rolling(n, min_periods=n).max()
    lo = df["low"].rolling(n, min_periods=n).min()
    return pd.DataFrame({"upper": hi, "lower": lo, "mid": (hi + lo) / 2})


# --------------------------------------------------------------------------- #
# volume
# --------------------------------------------------------------------------- #
def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df["volume"]).cumsum()


def vwap(df: pd.DataFrame, reset: str = "D") -> pd.Series:
    """Anchored VWAP that resets each session (default daily) - for intraday bars."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    key = df.index.to_period(reset)
    pv = (tp * df["volume"]).groupby(key).cumsum()
    v = df["volume"].groupby(key).cumsum()
    return pv / v.replace(0, np.nan)


def volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Current volume relative to its n-bar average (1.0 = normal)."""
    return df["volume"] / sma(df["volume"], n).replace(0, np.nan)


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    flow = tp * df["volume"]
    pos = flow.where(tp > tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    neg = flow.where(tp < tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


# --------------------------------------------------------------------------- #
# levels
# --------------------------------------------------------------------------- #
def pivot_points(df: pd.DataFrame) -> dict:
    """Classic floor pivots from the last *completed* bar (use daily bars for swing)."""
    last = df.iloc[-1]
    p = (last["high"] + last["low"] + last["close"]) / 3
    rng = last["high"] - last["low"]
    return {
        "pivot": p,
        "r1": 2 * p - last["low"], "s1": 2 * p - last["high"],
        "r2": p + rng, "s2": p - rng,
        "r3": last["high"] + 2 * (p - last["low"]), "s3": last["low"] - 2 * (last["high"] - p),
    }


# --------------------------------------------------------------------------- #
# one-shot: compute a standard indicator set
# --------------------------------------------------------------------------- #
def compute_all(df: pd.DataFrame, ma_periods=(9, 20, 50, 200), intraday: bool = False) -> pd.DataFrame:
    df = prepare(df)
    out = df.copy()
    c = df["close"]
    for n in ma_periods:
        out[f"ema_{n}"] = ema(c, n)
        out[f"sma_{n}"] = sma(c, n)
    out["rsi_14"] = rsi(c, 14)
    out[["macd", "macd_signal", "macd_hist"]] = macd(c).to_numpy()
    out[["stoch_k", "stoch_d"]] = stochastic(df).to_numpy()
    out[["adx", "plus_di", "minus_di"]] = adx(df).to_numpy()
    out["atr_14"] = atr(df, 14)
    out[["bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pct_b"]] = bollinger(c).to_numpy()
    out[["st_line", "st_dir"]] = supertrend(df).to_numpy()
    out[["dc_upper", "dc_lower", "dc_mid"]] = donchian(df).to_numpy()
    if "volume" in df.columns and df["volume"].fillna(0).sum() > 0:
        out["obv"] = obv(df)
        out["vol_ratio"] = volume_ratio(df)
        out["mfi_14"] = mfi(df)
        if intraday:
            out["vwap"] = vwap(df)
    return out
