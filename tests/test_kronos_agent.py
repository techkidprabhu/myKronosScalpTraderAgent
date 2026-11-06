import numpy as np
import pandas as pd
import pytest

from src.kronos_agent import KronosAgent


@pytest.fixture(scope="module")
def kronos_agent():
    return KronosAgent(
        model_name="NeoQuasar/Kronos-base",
        tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
        device="cpu",
        local_files_only=True,
        sample_count=1,
        temperature=0.5,
        verbose=False,
    )


def test_kronos_forecast(kronos_agent):
    dates = pd.date_range("2026-10-01", periods=30, freq="4h")
    df = pd.DataFrame(
        {
            "open": np.linspace(30, 32, 30),
            "high": np.linspace(30.5, 32.5, 30),
            "low": np.linspace(29.5, 31.5, 30),
            "close": np.linspace(30.2, 32.2, 30),
            "volume": np.full(30, 10000),
        },
        index=dates,
    )

    pred = kronos_agent.forecast(df, forecast_horizon=4)
    assert len(pred) == 4
    assert all(col in pred.columns for col in ["open", "high", "low", "close"])


def test_kronos_confirm_direction(kronos_agent):
    dates = pd.date_range("2026-10-01", periods=30, freq="4h")
    df = pd.DataFrame(
        {
            "open": np.linspace(30, 32, 30),
            "high": np.linspace(30.5, 32.5, 30),
            "low": np.linspace(29.5, 31.5, 30),
            "close": np.linspace(30.2, 32.2, 30),
            "volume": np.full(30, 10000),
        },
        index=dates,
    )

    confirmed, confidence = kronos_agent.confirm_direction(df, "BUY", forecast_horizon=4)
    assert isinstance(confirmed, bool)
    assert 0.0 <= confidence <= 1.0


def test_kronos_insufficient_bars(kronos_agent):
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError):
        kronos_agent.forecast(df_empty, forecast_horizon=4)
