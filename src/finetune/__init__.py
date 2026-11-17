"""
Self-learning for the offline Kronos model.

Follows the fine-tuning method in the Kronos repository (finetune/): sliding
windows of candles, normalised on the lookback part only, tokenised with the
Kronos tokenizer, and the predictor trained to predict the next token.

What it learns from
-------------------
The candles of every symbol the agent screened, decided on or traded, on
every broker (US via moomoo / Tiger, India via Zerodha / Angel), collected
at each market close into data/training/. The agent's own signals steer the
sampling: a share of each batch is windows that end exactly where the agent
asked Kronos for a verdict - winners AND losers, because training only on
winning trades would teach the model that such setups always work.

What it never does
------------------
It never changes the model the agent trades with. Each run writes a new
model folder under models/ with a report comparing it with the original on
candles it did not train on. Switching is a deliberate step (dashboard
button or kronos.model_name) and is undone the same way.

    python -m src.finetune check      can this computer train, and how long?
    python -m src.finetune status     data collected, runs so far
    python -m src.finetune train      train + evaluate now
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_DIR = ROOT / "data" / "training"
MODELS_DIR = ROOT / "models"
STATUS_PATH = ROOT / "data" / "finetune_status.json"
COLLECT_REQUEST = ROOT / "data" / "finetune_collect.request"

DEFAULTS = {
    "enabled": False,            # automatic collection (and scheduled training) by the agent
    "collect_on_close": True,    # save candles when each market closes
    "bars_per_symbol": 1500,     # history asked for per symbol on each collection
    "max_symbols": 150,          # per broker
    "train_time": "",            # local "HH:MM" to train automatically; empty = only when you start it
    "train_days": ["sat"],       # which days the scheduled training runs
    "base_model": None,          # start from this model; None = the original kronos.model_name
    "base_tokenizer": None,
    "lookback": None,            # candles of context; None = lookback_bars (max 400)
    "predict": None,             # candles predicted; None = max(forecast_horizon, 10)
    "batch_size": 8,
    "max_steps": 1500,
    "max_minutes": 240,          # stop here even if max_steps isn't reached
    "learning_rate": 4e-5,       # the Kronos repository's predictor rate
    "eval_every": 100,
    "patience": 5,               # stop after this many evaluations without improvement
    "val_fraction": 0.15,        # the most recent share of candles is held out, never trained on
    "val_windows": 160,
    "signal_fraction": 0.3,      # share of each batch drawn from the agent's own signal bars
    "min_train_windows": 3000,   # refuse to train on less
    "eval_forecasts": 100,       # held-out forecasts compared, original vs fine-tuned
    "train_tokenizer": False,    # also fine-tune the tokenizer first (official step 1) - needs far more data
    "tokenizer_steps": 300,
    "threads": None,             # CPU threads; None = all cores
    "device": "auto",            # auto = NVIDIA GPU if PyTorch can use one, else CPU
    "shadow_model": None,        # model folder checked alongside the trading model; never affects trades
    "shadow_tokenizer": None,
    "keep_runs": 3,              # model folders kept (the one in use is always kept)
    "seed": 7,
}


def ensure_kronos() -> bool:
    """Put the Kronos source (github.com/shiyu-coder/Kronos) on sys.path.
    Looks at KRONOS_PATH, then next to this code, the project and the
    current folder - so it works inside the agent and in the exported
    training kit alike."""
    import os
    import sys
    here = Path(__file__).resolve().parent
    candidates = [os.environ.get("KRONOS_PATH"), here.parent / "Kronos", ROOT / "vendor" / "Kronos",
                  ROOT.parent / "vendor" / "Kronos", Path.cwd() / "Kronos", Path.cwd() / "vendor" / "Kronos",
                  Path.cwd().parent / "vendor" / "Kronos"]
    for c in candidates:
        if c and (Path(c) / "model").is_dir():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return True
    return False


def settings(config: dict) -> dict:
    return {**DEFAULTS, **((config or {}).get("finetune") or {})}
