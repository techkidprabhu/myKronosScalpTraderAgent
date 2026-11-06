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

    def forecast(
        self,
        df: pd.DataFrame,
        forecast_horizon: int,
        sample_count: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> pd.DataFrame:
        """
        df: OHLCV history, columns open/high/low/close (volume/amount
        optional), indexed by timestamp, most recent last.
        Returns a DataFrame of predicted OHLC(V) for the next
        `forecast_horizon` bars.
        """
        if len(df) < 2:
            raise ValueError(f"Need at least 2 bars for forecasting, got {len(df)}")

        if len(df) > self.max_context:
            df = df.iloc[-self.max_context:]

        x_timestamp = pd.Series(df.index.to_numpy())
        last_ts = df.index[-1]
        freq = df.index.freq or (df.index[-1] - df.index[-2])
        y_timestamp = pd.Series([last_ts + freq * (i + 1) for i in range(forecast_horizon)])

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

    def confirm_direction(
        self,
        df: pd.DataFrame,
        direction: str,
        forecast_horizon: int,
        sample_count: int | None = None,
        temperature: float | None = None,
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
        )
        last_close = df["close"].iloc[-1]
        deltas = pred_df["close"].to_numpy() - last_close

        if direction == "BUY":
            agree = np.mean(deltas > 0)
        else:
            agree = np.mean(deltas < 0)

        confirmed = bool(agree > 0.5)
        return confirmed, float(agree)
