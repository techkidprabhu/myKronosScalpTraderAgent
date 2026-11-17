"""Fine-tune Kronos on this computer's GPU from an exported training kit.

Run it with train.bat (Windows) or ./train.sh (Linux); they set Python up
the first time. Options, passed after the script name:

    --steps N       training steps (default from settings.json: 4000)
    --minutes N     stop after this long whatever happens (default 180)
    --batch N       windows per step (default 32; lower it if the GPU runs out of memory)
    --keep          put the model in the result zip even when it is not better
    --cpu           train without a GPU (slow; for testing)
    --model NAME    start from another model, --tokenizer NAME likewise
"""
import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parent
HOME = Path(os.environ.get("KRONOS_TRAINER_HOME") or Path.home() / ".kronos-trainer")
KRONOS_ZIP = "https://github.com/shiyu-coder/Kronos/archive/refs/heads/master.zip"


def kronos_source() -> Path:
    """The Kronos model code: in the kit, else downloaded once from GitHub
    and shared by every kit."""
    env = os.environ.get("KRONOS_PATH")
    if env and (Path(env) / "model").is_dir():
        return Path(env)
    if (KIT / "Kronos" / "model").is_dir():
        return KIT / "Kronos"
    dest = HOME / "Kronos"
    if (dest / "model").is_dir():
        return dest
    print("Downloading the Kronos model code from GitHub (first run only)...")
    HOME.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(KRONOS_ZIP, timeout=120) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))
    tmp = HOME / "Kronos.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    archive.extractall(tmp)
    inner = next(p for p in tmp.iterdir() if (p / "model").is_dir())
    inner.rename(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def local_model(name: str, sub: str) -> Path | None:
    """The original model or tokenizer without Hugging Face: from this kit
    (exported with "Include the Kronos model"), kept in ~/.kronos-trainer so
    later kits without it work too."""
    kept = HOME / "models" / name.replace("/", "--")
    bundled = KIT / "base" / sub
    if (bundled / "config.json").exists() and not (kept / "config.json").exists():
        tmp = kept.with_name(kept.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(bundled, tmp)
        tmp.rename(kept)
    return kept if (kept / "config.json").exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--tokenizer")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Start with train.bat (Windows) or ./train.sh (Linux).")
        return 2
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gb:.0f} GB)")
    elif not args.cpu:
        print("PyTorch cannot use the NVIDIA GPU on this computer.\n"
              "  1. Update the NVIDIA driver (the launcher installs the CUDA 13 build of PyTorch,\n"
              "     which needs a recent driver).\n"
              "  2. Or delete the folder " + str(HOME / "venv") + " and start again with\n"
              "     TORCH_INDEX set to https://download.pytorch.org/whl/cu126 for an older driver.\n"
              "  To train on the CPU anyway (slow), add --cpu.")
        return 2

    os.environ["KRONOS_PATH"] = str(kronos_source())
    sys.path.insert(0, str(KIT))
    from finetune.trainer import run_training

    config = json.loads((KIT / "settings.json").read_text())["config"]
    signals = json.loads((KIT / "signals.json").read_text())
    ft = config["finetune"]
    if args.steps:
        ft["max_steps"] = args.steps
    if args.minutes:
        ft["max_minutes"] = args.minutes
    if args.batch:
        ft["batch_size"] = args.batch
    if args.cpu:
        ft["device"] = "cpu"
    if args.model:
        ft["base_model"] = config["kronos"]["model_name"] = args.model
    if args.tokenizer:
        ft["base_tokenizer"] = config["kronos"]["tokenizer_name"] = args.tokenizer

    # the names the agent knows; the report keeps these, whatever folder we load from
    names = {"base_model": ft["base_model"], "base_tokenizer": ft["base_tokenizer"]}
    have = {"model": None if args.model else local_model(names["base_model"], "model"),
            "tokenizer": None if args.tokenizer else local_model(names["base_tokenizer"], "tokenizer")}
    if have["model"] and have["tokenizer"]:
        os.environ["HF_HUB_OFFLINE"] = "1"                 # never reach for Hugging Face
        ft["base_model"] = config["kronos"]["model_name"] = str(have["model"])
        ft["base_tokenizer"] = config["kronos"]["tokenizer_name"] = str(have["tokenizer"])
        config["kronos"]["local_files_only"] = True
        where = "in this kit" if (KIT / "base" / "model" / "config.json").exists() else "kept from an earlier kit"
        print(f"Starting from {names['base_model']} (model files {where}; nothing is downloaded from Hugging Face)")
    else:
        print(f"Starting from {names['base_model']} (downloaded from Hugging Face the first time, about 400 MB).\n"
              "  If Hugging Face is blocked on this network, export again on the dashboard with\n"
              "  'Include the Kronos model' ticked.")

    out = KIT / "output"
    try:
        report = run_training(config, models_dir=out, status_path=out / "status.json", training_root=KIT / "training",
                              log=lambda m: print(m, flush=True), signals_by_broker=signals, keep_always=True)
    except (RuntimeError, OSError) as exc:
        print(f"\nStopped: {exc}")
        if "huggingface" in str(exc).lower() or "hf.co" in str(exc).lower() or "connection" in str(exc).lower():
            print("Could not download the Kronos model. Export again with 'Include the Kronos model' ticked.")
        if "out of memory" in str(exc).lower():
            print("The GPU ran out of memory - try again with a smaller batch, e.g. --batch 16")
        return 1

    run_dir = out / report["run_id"]
    if have["model"] and have["tokenizer"]:     # name the original as the agent knows it, not this PC's folder
        report.update(names)
        if not report.get("tokenizer_tuned"):
            report["tokenizer"] = names["base_tokenizer"]
        (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    better = report["verdict"] == "better"
    result = KIT / f"{report['run_id']}-result.zip"
    with zipfile.ZipFile(result, "w", zipfile.ZIP_STORED) as z:
        names = ["report.json", "config.json"]
        if better or args.keep:
            names.append("model.safetensors")
            if report.get("tokenizer_tuned"):
                names += ["tokenizer/config.json", "tokenizer/model.safetensors"]
        for name in names:
            if (run_dir / name).exists():
                z.write(run_dir / name, f"{report['run_id']}/{name}")
    o, t = report["original"], report["fine_tuned"]
    print("\n" + "=" * 68)
    print(f"  {report['run_id']}   {report['steps']} steps in {report['minutes']} min on {report['device']}")
    print(f"  Held-out loss      original {o['val_loss']}  ->  fine-tuned {t['val_loss']}")
    print(f"  Direction right    original {o['direction_hit_pct']}%  ->  fine-tuned {t['direction_hit_pct']}%")
    print(f"  Result: {report['verdict'].replace('_', ' ').upper()} - {report['reason']}")
    print("=" * 68)
    if better or args.keep:
        print(f"\nBring this file back to the agent's dashboard (Self-learning -> Import trained model):\n  {result}")
    else:
        print("\nThe fine-tuned model is not better than the original, so there is no model to bring back.\n"
              f"{result.name} holds only the report (import it if you want the run listed on the dashboard).\n"
              "More trading days of collected candles usually help.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
