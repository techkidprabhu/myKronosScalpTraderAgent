"""Train on another computer (one with an NVIDIA GPU) and bring the model back.

    build_export()   one zip with the collected candles, the agent's signal
                     times, the settings the windows must match, and the
                     trainer itself - nothing else. No broker credentials,
                     no config.yaml, no audit databases.
    import_run()     takes the result zip the trainer wrote and puts the run
                     under models/, where it shows up like a run made here.

An imported model is treated exactly like a local one: it is never used for
trading until "Use this model" is chosen.
"""
from __future__ import annotations

import json
import re
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

from . import MODELS_DIR, TRAINING_DIR, settings
from .data import all_signals
from .trainer import audit_dbs, original_names, window_sizes

KIT_DIR = Path(__file__).resolve().parent / "kit_files"
KIT_FOLDER = "kronos-training-kit"
TRAINER_SOURCES = ("__init__.py", "data.py", "trainer.py")
RUN_NAME = re.compile(r"^kronos-ft-[A-Za-z0-9_-]{1,60}$")
RUN_FILES = {"report.json", "config.json", "model.safetensors", "tokenizer/config.json", "tokenizer/model.safetensors"}
MAX_IMPORT_BYTES = 3 * 1024 ** 3
# what the GPU computer decides for itself (bigger batches, more steps); the
# rest follows the agent's settings so the windows match what Kronos is asked
GPU_DEFAULTS = {"batch_size": 32, "max_steps": 4000, "max_minutes": 180, "val_windows": 320, "eval_forecasts": 200,
                "patience": 8, "threads": None, "device": "auto"}
CARRIED = ("learning_rate", "eval_every", "val_fraction", "signal_fraction", "min_train_windows", "train_tokenizer",
           "tokenizer_steps", "seed", "keep_runs")


def export_settings(config: dict) -> dict:
    ft, k = settings(config), (config.get("kronos") or {})
    lookback, predict = window_sizes(config)
    base, tok = original_names(config)
    return {
        "exported": time.time(),
        "config": {
            "lookback_bars": config.get("lookback_bars"), "forecast_horizon": config.get("forecast_horizon"),
            "kronos": {"model_name": base, "tokenizer_name": tok, "local_files_only": False,
                       "max_context": k.get("max_context"), "temperature": k.get("temperature"),
                       "top_p": k.get("top_p"), "sample_count": k.get("sample_count")},
            "finetune": {**{key: ft[key] for key in CARRIED}, **GPU_DEFAULTS, "lookback": lookback,
                         "predict": predict, "base_model": base, "base_tokenizer": tok},
        },
    }


def export_name() -> str:
    return f"kronos-training-{datetime.now():%Y%m%d-%H%M}.zip"


def build_export(config: dict, out, training_root: Path | None = None) -> dict:
    """Write the training kit zip to `out` (a path or a binary file object)."""
    root = training_root or TRAINING_DIR
    files = sorted(root.glob("*/*.csv")) if root.exists() else []
    if not files:
        raise RuntimeError("No candles collected yet - use 'Collect candles now' first.")
    signals = all_signals(audit_dbs(config))
    here = Path(__file__).resolve().parent
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(KIT_DIR.iterdir()):
            if f.is_file():
                info = zipfile.ZipInfo(f"{KIT_FOLDER}/{f.name}", time.localtime()[:6])
                info.external_attr = (0o755 if f.suffix in (".sh", ".py") else 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, f.read_bytes())
        for name in TRAINER_SOURCES:
            z.write(here / name, f"{KIT_FOLDER}/finetune/{name}")
        z.writestr(f"{KIT_FOLDER}/settings.json", json.dumps(export_settings(config), indent=2))
        z.writestr(f"{KIT_FOLDER}/signals.json", json.dumps(signals))
        for f in files:
            z.write(f, f"{KIT_FOLDER}/training/{f.parent.name}/{f.name}")
    return {"files": len(files), "brokers": sorted({f.parent.name for f in files}),
            "signals": sum(len(t) for per in signals.values() for t in per.values())}


def import_run(zip_path, models_dir: Path | None = None) -> dict:
    """Unpack a trainer result zip into models/<run>/ after checking it holds
    exactly one run folder with the expected files. Returns the report."""
    models_dir = models_dir or MODELS_DIR
    try:
        z = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("That is not a zip file") from exc
    with z:
        members = [i for i in z.infolist() if not i.is_dir()]
        tops = {i.filename.split("/", 1)[0] for i in members}
        if len(tops) != 1 or not all("/" in i.filename for i in members):
            raise ValueError("Expected the result zip the GPU trainer wrote (one kronos-ft-... folder)")
        run_id = tops.pop()
        if not RUN_NAME.match(run_id):
            raise ValueError("Expected the result zip the GPU trainer wrote (one kronos-ft-... folder)")
        inner = {i.filename.split("/", 1)[1]: i for i in members}
        extra = set(inner) - RUN_FILES
        if extra:
            raise ValueError(f"Unexpected file in the zip: {sorted(extra)[0]}")
        if not {"report.json", "config.json"} <= set(inner):
            raise ValueError("The zip has no report.json / config.json - not a trainer result")
        if sum(i.file_size for i in members) > MAX_IMPORT_BYTES:
            raise ValueError("The zip is larger than a Kronos model can be")
        if max(inner["report.json"].file_size, inner["config.json"].file_size) > 1024 ** 2:
            raise ValueError("report.json / config.json are not what the trainer writes")
        try:
            report = json.loads(z.read(inner["report.json"]))
            model_cfg = json.loads(z.read(inner["config.json"]))
        except ValueError as exc:
            raise ValueError("report.json / config.json could not be read") from exc
        if (not isinstance(report, dict) or not isinstance(model_cfg, dict)
                or report.get("verdict") not in ("better", "no_gain", "worse")
                or not isinstance(report.get("original"), dict) or not isinstance(report.get("fine_tuned"), dict)
                or not report.get("base_model") or not report.get("base_tokenizer")):
            raise ValueError("report.json is not a trainer report")
        dest = models_dir / run_id
        if dest.exists():
            raise ValueError(f"{run_id} is already here")
        tmp = models_dir / f".import-{run_id}"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            for rel, info in inner.items():
                target = tmp / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
            tuned_tok = bool(report.get("tokenizer_tuned")) and (tmp / "tokenizer" / "model.safetensors").exists()
            report.update(run_id=run_id, imported=time.time(), tokenizer_tuned=tuned_tok,
                          tokenizer=str(dest / "tokenizer") if tuned_tok else report["base_tokenizer"])
            (tmp / "report.json").write_text(json.dumps(report, indent=2))
            tmp.rename(dest)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    report["has_weights"] = (dest / "model.safetensors").exists()
    return report
