"""python -m src.finetune check | status | train | export [file] | import <file>"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from . import ROOT, settings
from .data import inventory
from .trainer import hardware_check, list_runs, read_status, run_training, window_sizes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.finetune", description=__doc__)
    ap.add_argument("command", choices=["check", "status", "train", "export", "import"])
    ap.add_argument("file", nargs="?", help="export: where to write the kit; import: the trainer's result zip")
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-minutes", type=float)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = yaml.safe_load(open(args.config)) or {}
    ft = config.setdefault("finetune", {})
    if args.max_steps:
        ft["max_steps"] = args.max_steps
    if args.max_minutes:
        ft["max_minutes"] = args.max_minutes

    if args.command == "export":
        from . import kit
        out = args.file or str(ROOT / "data" / "exports" / kit.export_name())
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        try:
            info = kit.build_export(config, out)
        except RuntimeError as exc:
            print(exc)
            return 1
        print(f"Wrote {out}\n  {info['files']} candle files ({', '.join(info['brokers'])}), {info['signals']} signal times.\n"
              "  Unzip it on the computer with the GPU and start train.bat / train.sh.")
        return 0

    if args.command == "import":
        from . import kit
        if not args.file:
            print("Usage: python -m src.finetune import <kronos-ft-...-result.zip>")
            return 2
        try:
            rep = kit.import_run(args.file)
        except ValueError as exc:
            print(f"Not imported: {exc}")
            return 1
        print(f"Imported {rep['run_id']}: {rep['verdict']} - {rep['reason']}")
        print("  It is listed on the dashboard (Kronos Model Lab -> Self-learning): Shadow test, or Use this model."
              if rep["has_weights"] else "  The zip held no model (it was not better than the original).")
        return 0

    if args.command == "check":
        print("Timing a few training steps on this computer (about a minute)...")
        r = hardware_check(config)
        print(f"\n  Model            {r['model']} ({r['params_m']}M parameters)")
        print(f"  CPU threads      {r['cpu_threads']}    RAM {r['ram_gb']} GB    usable GPU: {'yes' if r['cuda'] else 'no (CPU training)'}")
        print(f"  Training window  {r['window']} candles, batch {r['batch_size']}")
        print(f"  Speed            {r['seconds_per_step']} s per step")
        print(f"  A full run       about {r['est_minutes']} min for {r['max_steps']} steps (stops at {r['max_minutes']} min regardless)")
        print("\n  " + ("OK: this computer can fine-tune Kronos overnight." if r["ok"] else
                        "Too slow or too little memory for the current settings - lower finetune.max_steps / lookback, "
                        "or use Kronos-small."))
        return 0 if r["ok"] else 1

    if args.command == "status":
        lb, pr = window_sizes(config)
        print(f"Automatic collection: {'on' if settings(config)['enabled'] else 'off'}   window {lb}+{pr} candles")
        inv = inventory()
        for row in inv:
            print(f"  {row['broker']:<10} {row['timeframe']:<7} {row['symbols']:>4} symbols  {row['bars']:>9} candles  "
                  f"updated {datetime.fromtimestamp(row['updated']):%d %b %H:%M}")
        if not inv:
            print("  No candles collected yet.")
        print("Status:", json.dumps(read_status()))
        for r in list_runs():
            print(f"  {r['run_id']}: {r['verdict']} - {r['reason']}")
        return 0

    report = run_training(config)
    print(json.dumps({k: report[k] for k in ("run_id", "verdict", "reason", "original", "fine_tuned")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
