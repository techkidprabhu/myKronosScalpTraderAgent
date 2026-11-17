"""python -m src.finetune check | status | train | export [file] | import <file> | versions | save <run> [note] | restore <version>"""
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
    ap.add_argument("command", choices=["check", "status", "train", "export", "import", "versions", "save", "restore"])
    ap.add_argument("file", nargs="?", help="export: where to write the kit; import: the trainer's result zip; "
                                            "save: the run (kronos-ft-...); restore: the version number")
    ap.add_argument("note", nargs="*", help="save: a short note kept with the version")
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--with-model", action="store_true",
                    help="export: put the original Kronos model in the kit (for networks that block Hugging Face)")
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
            info = kit.build_export(config, out, include_model=args.with_model)
        except RuntimeError as exc:
            print(exc)
            return 1
        model = f", Kronos model {info['model_mb']} MB" if info["model_mb"] else ""
        print(f"Wrote {out}\n  {info['files']} candle files ({', '.join(info['brokers'])}), "
              f"{info['signals']} signal times{model}.\n"
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

    if args.command in ("versions", "save", "restore"):
        return _versions(args, config)

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


def _versions(args, config) -> int:
    from . import releases
    from .trainer import list_runs
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    repo, token = releases.target(config)
    if not token:
        print("Saved versions live on GitHub Releases. Create a fine-grained token for your repository with\n"
              "'Contents: Read and write', add it to .env as GITHUB_TOKEN=..., and run this again.")
        return 2
    if not repo:
        print("Set finetune.releases_repo: owner/name in config/config.yaml (no GitHub remote found).")
        return 2

    def bar(total):
        def show(done):
            pct = f"{done / total * 100:5.1f}%" if total else ""
            print(f"\r  {done / 1048576:7.1f} MB {pct}", end="", flush=True)
        return show

    try:
        gh = releases.GitHub(repo, token)
        if args.command == "versions":
            vs = releases.list_versions(gh)
            local = {r["run_id"] for r in list_runs() if r["has_weights"]}
            print(f"Saved Kronos models on github.com/{repo}:")
            for v in vs:
                o, t = v.get("original") or {}, v.get("fine_tuned") or {}
                print(f"  v{v['version']:<4} {v.get('run_id', '?'):<26} loss {o.get('val_loss')} -> {t.get('val_loss')}"
                      f"  {'(on this computer) ' if v.get('run_id') in local else ''}{v.get('note') or ''}")
            if not vs:
                print("  none yet - ./start.sh finetune save <run> [note]")
            return 0
        if args.command == "save":
            run = next((r for r in list_runs() if r["run_id"] == args.file), None)
            if run is None or not run["has_weights"]:
                print(f"No training run {args.file!r} with a saved model. Runs: " +
                      ", ".join(r["run_id"] for r in list_runs() if r["has_weights"]))
                return 1
            size = sum((Path(run["path"]) / f).stat().st_size for f in releases.RUN_FILES if (Path(run["path"]) / f).exists())
            print(f"Uploading {run['run_id']} to github.com/{repo} ...")
            res = releases.save_version(gh, Path(run["path"]), " ".join(args.note), bar(size))
            print(f"\nSaved as v{res['version']}: {res['url']}")
            return 0
        if not (args.file or "").lstrip("v").isdigit():
            print("Usage: ./start.sh finetune restore <version number>, e.g. restore 3")
            return 2
        version = int(args.file.lstrip("v"))
        print(f"Downloading v{version} from github.com/{repo} ...")
        v = next((x for x in releases.list_versions(gh) if x["version"] == version), {})
        res = releases.restore_version(gh, version, progress=bar(int(v.get("size") or 0)))
        print(f"\nRestored v{version} as models/{res['run_id']}. Use it from the dashboard (Self-learning).")
        return 0
    except releases.ReleaseError as exc:
        print(f"\n{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
