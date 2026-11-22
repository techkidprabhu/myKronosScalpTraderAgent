"""
Wrapper around the Kronos foundation model
(https://github.com/shiyu-coder/Kronos) used as the confirmation
subagent: given a candidate pattern from patterns.py, ask Kronos to
forecast the next `forecast_horizon` bars and check whether its
predicted path agrees with the pattern's implied direction.
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_kronos_importable():
    """Ensure the Kronos repository directory is in sys.path so 'from model import ...' succeeds."""
    candidates = [
        os.environ.get("KRONOS_PATH"),
        str(Path(__file__).resolve().parent.parent / "vendor" / "Kronos"),
        str(Path(__file__).resolve().parent.parent.parent / "vendor" / "Kronos"),
        str(Path.cwd() / "vendor" / "Kronos"),
        str(Path.cwd().parent / "vendor" / "Kronos"),
        "/home/prabhu/mySwingTraders/vendor/Kronos",
    ]
    for c in candidates:
        if c and os.path.isdir(c) and os.path.isdir(os.path.join(c, "model")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return True
    return False


_ensure_kronos_importable()

try:
    from model import Kronos, KronosPredictor, KronosTokenizer
except ImportError:  # pragma: no cover
    Kronos = KronosPredictor = KronosTokenizer = None


METHODS = ("bars", "paths")
DEFAULT_OPTIONS = {"method": "bars", "paths": 10, "shadow_paths": False, "min_move_pct": 0.0}


def kronos_options(cfg: dict | None) -> dict:
    """Optional scalp settings for how Kronos's answer is read (config
    `kronos_scalp:`). The defaults are exactly the original behaviour:

      confidence_method: bars   share of the averaged forecast's candles that
                                close beyond the last close (original rule)
                         paths  share of `paths` separate forecasts that end
                                beyond the last close - a real probability
      paths: 10                 forecasts for the 'paths' method (2-50)
      shadow_paths: false       also work out the 'paths' confidence and only
                                record it (Kronos Model Lab scores both)
      min_move_pct: 0           a close only counts as agreeing if it is at
                                least this % beyond the last close (0 = off);
                                set it to cover spread + fees
    """
    raw = (cfg or {}).get("kronos_scalp") or {}
    out = dict(DEFAULT_OPTIONS)
    m = str(raw.get("confidence_method") or "bars").lower()
    out["method"] = m if m in METHODS else "bars"
    try:
        out["paths"] = max(2, min(50, int(raw.get("paths", 10))))
    except (TypeError, ValueError):
        pass
    out["shadow_paths"] = bool(raw.get("shadow_paths", False))
    try:
        out["min_move_pct"] = max(0.0, min(5.0, float(raw.get("min_move_pct") or 0.0)))
    except (TypeError, ValueError):
        pass
    return out


def is_default(opts: dict | None) -> bool:
    o = opts or DEFAULT_OPTIONS
    return o.get("method", "bars") == "bars" and not o.get("min_move_pct")


class KronosAgent:
    def __init__(
        self,
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        model_name: str = "NeoQuasar/Kronos-base",
        device: str | None = "cpu",
        max_context: int = 512,
        local_files_only: bool = True,
        sample_count: int = 3,
        temperature: float = 0.5,
        top_p: float = 0.9,
        verbose: bool = False,
    ):
        if Kronos is None:
            raise RuntimeError(
                "Kronos is not installed or importable. Ensure vendor/Kronos "
                "or KRONOS_PATH is present before using KronosAgent."
            )

        self.device = device or "cpu"
        self.max_context = max_context
        self.local_files_only = local_files_only
        self.sample_count = sample_count
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose

        if self.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"

        logger.info(
            "Loading Kronos model: model=%s, tokenizer=%s, device=%s, local_files_only=%s",
            model_name, tokenizer_name, self.device, self.local_files_only,
        )

        load_kwargs = {}
        if self.local_files_only:
            load_kwargs["local_files_only"] = True

        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_name, **load_kwargs)
        self.model = Kronos.from_pretrained(model_name, **load_kwargs)
        self.predictor = KronosPredictor(
            self.model, self.tokenizer, device=self.device, max_context=self.max_context
        )
        logger.info("Kronos model initialized successfully.")

    @staticmethod
    def bar_step(df: pd.DataFrame, bar_duration=None) -> pd.Timedelta:
        """Length of one candle, for the forecast's future timestamps.

        Use the timeframe's own length when the caller knows it. Otherwise
        take the most common gap between recent bars - NOT the last gap: on
        the first candle after the open (or after a weekend, IG's daily
        break or a missing bar) the last gap is the whole night, which would
        tell Kronos it is forecasting candles a day ahead."""
        if bar_duration is not None:
            return pd.Timedelta(bar_duration)
        if df.index.freq is not None:
            return pd.Timedelta(df.index.freq)
        diffs = pd.Series(df.index[-60:]).diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        return diffs.mode().iloc[0] if not diffs.empty else pd.Timedelta(minutes=1)

    def _inputs(self, df: pd.DataFrame, forecast_horizon: int, bar_duration=None):
        if len(df) < 2:
            raise ValueError(f"Need at least 2 bars for forecasting, got {len(df)}")
        if len(df) > self.max_context:
            df = df.iloc[-self.max_context:]
        x_timestamp = pd.Series(df.index.to_numpy())
        last_ts = df.index[-1]
        step = self.bar_step(df, bar_duration)
        y_timestamp = pd.Series([last_ts + step * (i + 1) for i in range(forecast_horizon)])
        return df, x_timestamp, y_timestamp

    def forecast(
        self,
        df: pd.DataFrame,
        forecast_horizon: int,
        sample_count: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        bar_duration=None,
    ) -> pd.DataFrame:
        """
        df: OHLCV history, columns open/high/low/close (volume/amount
        optional), indexed by timestamp, most recent last.
        Returns a DataFrame of predicted OHLC(V) for the next
        `forecast_horizon` bars. `bar_duration` is one candle's length
        (e.g. 3 minutes); without it the usual gap between bars is used.
        """
        df, x_timestamp, y_timestamp = self._inputs(df, forecast_horizon, bar_duration)

        pred_df = self.predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=forecast_horizon,
            T=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            sample_count=sample_count if sample_count is not None else self.sample_count,
            verbose=self.verbose,
        )
        return pred_df

    @staticmethod
    def agreement(pred_close, last_close: float, direction: str, min_move_pct: float = 0.0) -> tuple[bool, float]:
        """Share of the forecast closes beyond the last close in the signal's
        direction, and whether that's a majority. With min_move_pct, a close
        must be at least that % beyond the last close to count (0 = the
        original rule)."""
        deltas = np.asarray(pred_close, dtype=float) - float(last_close)
        edge = abs(float(last_close)) * float(min_move_pct or 0.0) / 100.0
        agree = np.mean(deltas > edge) if direction == "BUY" else np.mean(deltas < -edge)
        return bool(agree > 0.5), float(agree)

    def confirm_direction(
        self,
        df: pd.DataFrame,
        direction: str,
        forecast_horizon: int,
        sample_count: int | None = None,
        temperature: float | None = None,
        bar_duration=None,
    ) -> tuple[bool, float]:
        """
        Runs a forecast and checks whether Kronos's predicted close at the
        end of the horizon agrees with `direction` ("BUY" or "SELL").

        Returns (confirmed, confidence) where confidence is the
        proportion of the forecast path in the expected direction from
        the last known close.
        """
        pred_df = self.forecast(
            df,
            forecast_horizon,
            sample_count=sample_count,
            temperature=temperature,
            bar_duration=bar_duration,
        )
        return self.agreement(pred_df["close"].to_numpy(), df["close"].iloc[-1], direction)

    def confirm_many(self, jobs: list, forecast_horizon: int, bar_duration=None, method: str = "bars",
                     paths: int = 10, min_move_pct: float = 0.0) -> list:
        """confirm_direction for several symbols that signalled on the same
        candle, in as few model passes as possible: series with the same
        number of candles go through Kronos's own predict_batch together.
        Same model, same settings and the same answer rule as
        confirm_direction. `jobs` is [(df, direction), ...]; returns
        [(confirmed, confidence) or an Exception, ...] in the same order.

        method="paths" asks for `paths` separate forecasts per symbol (one
        sample each, in one batched pass) and reports the share whose final
        close is beyond the last close - see kronos_options()."""
        if method == "paths":
            return self._confirm_paths(jobs, forecast_horizon, bar_duration, paths, min_move_pct)
        out: list = [None] * len(jobs)
        groups: dict[int, list[int]] = {}
        prepared = {}
        for i, (df, _direction) in enumerate(jobs):
            try:
                prepared[i] = self._inputs(df, forecast_horizon, bar_duration)
                groups.setdefault(len(prepared[i][0]), []).append(i)
            except Exception as exc:  # noqa: BLE001
                out[i] = exc
        batch_ok = hasattr(self.predictor, "predict_batch")
        for idx in groups.values():
            if len(idx) > 1 and batch_ok:
                try:
                    preds = self.predictor.predict_batch(
                        [prepared[i][0] for i in idx], [prepared[i][1] for i in idx],
                        [prepared[i][2] for i in idx], pred_len=forecast_horizon,
                        T=self.temperature, top_p=self.top_p, sample_count=self.sample_count,
                        verbose=self.verbose)
                    for i, pred in zip(idx, preds):
                        out[i] = self.agreement(pred["close"].to_numpy(), jobs[i][0]["close"].iloc[-1], jobs[i][1],
                                                min_move_pct)
                    continue
                except Exception:  # noqa: BLE001 - fall back to one at a time
                    logger.warning("Kronos batch of %d failed; running them one at a time", len(idx), exc_info=True)
            for i in idx:
                try:
                    if min_move_pct:
                        pred = self.forecast(jobs[i][0], forecast_horizon, bar_duration=bar_duration)
                        out[i] = self.agreement(pred["close"].to_numpy(), jobs[i][0]["close"].iloc[-1], jobs[i][1],
                                                min_move_pct)
                    else:
                        out[i] = self.confirm_direction(jobs[i][0], jobs[i][1], forecast_horizon,
                                                        bar_duration=bar_duration)
                except Exception as exc:  # noqa: BLE001
                    out[i] = exc
        return out

    def _confirm_paths(self, jobs, forecast_horizon, bar_duration, paths, min_move_pct):
        """`paths` independent forecasts per symbol (sample_count=1 each, so
        Kronos doesn't average them away); confidence = share whose close at
        the end of the horizon is beyond the last close (+ min_move_pct)."""
        paths = max(2, int(paths))
        out: list = [None] * len(jobs)
        ends: dict[int, list] = {}
        groups: dict[int, list[int]] = {}
        prepared = {}
        for i, (df, _direction) in enumerate(jobs):
            try:
                prepared[i] = self._inputs(df, forecast_horizon, bar_duration)
                groups.setdefault(len(prepared[i][0]), []).append(i)
            except Exception as exc:  # noqa: BLE001
                out[i] = exc
        kw = dict(pred_len=forecast_horizon, T=self.temperature, top_p=self.top_p, sample_count=1, verbose=self.verbose)
        for idx in groups.values():
            done = False
            if hasattr(self.predictor, "predict_batch"):
                flat = [i for i in idx for _ in range(paths)]
                try:
                    preds = self.predictor.predict_batch([prepared[i][0] for i in flat], [prepared[i][1] for i in flat],
                                                         [prepared[i][2] for i in flat], **kw)
                    for i, pred in zip(flat, preds):
                        ends.setdefault(i, []).append(float(pred["close"].iloc[-1]))
                    done = True
                except Exception:  # noqa: BLE001
                    logger.warning("Kronos paths batch failed; running them one at a time", exc_info=True)
                    for i in idx:
                        ends.pop(i, None)
            if not done:
                for i in idx:
                    try:
                        d, x, y = prepared[i]
                        ends[i] = [float(self.predictor.predict(df=d, x_timestamp=x, y_timestamp=y, **kw)["close"].iloc[-1])
                                   for _ in range(paths)]
                    except Exception as exc:  # noqa: BLE001
                        out[i] = exc
        for i, vals in ends.items():
            if out[i] is None:
                out[i] = self.agreement(vals, jobs[i][0]["close"].iloc[-1], jobs[i][1], min_move_pct)
        return out
