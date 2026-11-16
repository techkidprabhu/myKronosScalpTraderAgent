"""Fine-tune the Kronos predictor on collected candles and compare the
result with the original on held-out candles. CPU-friendly: no GPU needed."""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import MODELS_DIR, STATUS_PATH, ensure_kronos, settings
from .data import load_window_set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- status file

def read_status(path: Path | None = None) -> dict:
    try:
        st = json.loads((path or STATUS_PATH).read_text())
    except (OSError, ValueError):
        return {"state": "idle"}
    if st.get("state") == "running":
        pid = st.get("pid")
        alive = False
        if pid:
            try:
                os.kill(int(pid), 0)
                alive = True
            except (OSError, ValueError):
                alive = False
        if not alive:
            st.update(state="failed", message="The training process stopped unexpectedly (see logs/finetune.log)")
    return st


def write_status(path: Path | None = None, **fields) -> None:
    path = path or STATUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**fields, "updated": time.time()}))
    tmp.replace(path)


def list_runs(models_dir: Path | None = None) -> list[dict]:
    out = []
    for rep in sorted((models_dir or MODELS_DIR).glob("*/report.json"), reverse=True):
        try:
            r = json.loads(rep.read_text())
        except (OSError, ValueError):
            continue
        r["path"] = str(rep.parent)
        r["has_weights"] = (rep.parent / "model.safetensors").exists()
        out.append(r)
    return out


# ---------------------------------------------------------------- models

def _kronos_classes():
    if not ensure_kronos():
        raise RuntimeError("Kronos source not found (vendor/Kronos, a Kronos folder next to the trainer, or KRONOS_PATH)")
    from model import Kronos, KronosPredictor, KronosTokenizer
    return Kronos, KronosTokenizer, KronosPredictor


def original_names(config: dict) -> tuple[str, str]:
    """The model to start from: finetune.base_model, else kronos.model_name -
    and if that already points at a fine-tuned folder, the original it was
    made from, so runs never stack on top of each other."""
    ft, k = settings(config), (config.get("kronos") or {})
    model = ft.get("base_model") or k.get("model_name") or "NeoQuasar/Kronos-base"
    tok = ft.get("base_tokenizer") or k.get("tokenizer_name") or "NeoQuasar/Kronos-Tokenizer-base"
    rep = Path(str(model)) / "report.json"
    if not ft.get("base_model") and rep.exists():
        try:
            r = json.loads(rep.read_text())
            model, tok = r.get("base_model") or model, r.get("base_tokenizer") or tok
        except (OSError, ValueError):
            pass
    return str(model), str(tok)


def load_models(model_name: str, tokenizer_name: str, local_only: bool = True):
    Kronos, KronosTokenizer, _ = _kronos_classes()
    if local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    kw = {"local_files_only": True} if local_only else {}
    return Kronos.from_pretrained(model_name, **kw), KronosTokenizer.from_pretrained(tokenizer_name, **kw)


def window_sizes(config: dict) -> tuple[int, int]:
    ft = settings(config)
    max_ctx = int((config.get("kronos") or {}).get("max_context") or 512)
    predict = int(ft["predict"] or max(int(config.get("forecast_horizon") or 5), 10))
    lookback = int(ft["lookback"] or min(int(config.get("lookback_bars") or 200), 400))
    return max(20, min(lookback, max_ctx - predict)), predict


def audit_dbs(config: dict) -> dict[str, Path]:
    try:
        from src.broker_config import broker_specs
        return {s.name: s.db_path for s in broker_specs(config)}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------- loss / steps

def pick_device(want, torch) -> str:
    """'auto' -> the NVIDIA GPU when PyTorch can use it, else the CPU."""
    want = str(want or "auto").lower()
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A GPU was asked for, but this PyTorch can't use one (install the CUDA build of PyTorch)")
    return want


def _dev(model):
    return next(model.parameters()).device


def _loss(model, tokenizer, x, stamp, torch):
    """Next-token loss, exactly as finetune/train_predictor.py in the Kronos repository."""
    x, stamp = x.to(_dev(model)), stamp.to(_dev(model))
    with torch.no_grad():
        s1, s2 = tokenizer.encode(x, half=True)
    logits = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1, :])
    loss, _, _ = model.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])
    return loss


def val_loss(model, tokenizer, ws, items, batch_size, torch) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            x, st = ws.batch(items[i:i + batch_size])
            total += float(_loss(model, tokenizer, torch.from_numpy(x), torch.from_numpy(st), torch)) * len(x)
            n += len(x)
    return total / max(n, 1)


def _train_tokenizer(tokenizer, ws, ft, torch, log):
    """Official step 1 (finetune/train_tokenizer.py): reconstruction + quantiser loss."""
    import torch.nn.functional as F
    opt = torch.optim.AdamW(tokenizer.parameters(), lr=2e-4, weight_decay=0.01)
    tokenizer.train()
    for step in range(int(ft["tokenizer_steps"])):
        x, _ = ws.sample_train(int(ft["batch_size"]), float(ft["signal_fraction"]))
        x = torch.from_numpy(x).to(_dev(tokenizer))
        (z_pre, z), bsq_loss, _, _ = tokenizer(x)
        loss = (F.mse_loss(z_pre, x) + F.mse_loss(z, x) + bsq_loss) / 2
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 2.0)
        opt.step()
        if step % 50 == 0:
            log(f"tokenizer step {step}: loss {float(loss):.4f}")
    tokenizer.eval()


# ---------------------------------------------------------------- forecasts on held-out candles

def forecast_quality(model, tokenizer, ws, items, config: dict, torch) -> dict:
    """Run real forecasts (same sampling settings as the agent) from held-out
    windows and score them against what happened next."""
    _, _, KronosPredictor = _kronos_classes()
    k = config.get("kronos") or {}
    horizon = int(config.get("forecast_horizon") or 5)
    horizon = min(horizon, ws.predict)
    pred = KronosPredictor(model, tokenizer, device=str(_dev(model)), max_context=int(k.get("max_context") or 512))
    hits, errs, n = 0, [], 0
    model.eval()
    for j, (si, start) in enumerate(items):
        s = ws.series[si]
        a, b = start, start + ws.lookback
        df = pd.DataFrame(s.x[a:b, :5], columns=["open", "high", "low", "close", "volume"])
        last, actual = float(s.x[b - 1, 3]), float(s.x[b + horizon - 1, 3])
        if actual == last:
            continue
        torch.manual_seed(1000 + j)                      # same random draws for both models
        try:
            out = pred.predict(df=df, x_timestamp=s.timestamps[a:b].reset_index(drop=True),
                               y_timestamp=s.timestamps[b:b + horizon].reset_index(drop=True), pred_len=horizon,
                               T=float(k.get("temperature") or 0.5), top_p=float(k.get("top_p") or 0.9),
                               sample_count=min(int(k.get("sample_count") or 3), 3), verbose=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Held-out forecast failed: %s", exc)
            continue
        fc = float(out["close"].iloc[-1])
        hits += int((fc > last) == (actual > last))
        errs.append(abs(fc - actual) / last * 100)
        n += 1
    return {"forecasts": n, "direction_hit_pct": round(hits / n * 100, 1) if n else None,
            "close_error_pct": round(float(np.mean(errs)), 3) if errs else None, "horizon": horizon}


# ---------------------------------------------------------------- the run

def verdict(base: dict, tuned: dict, same_tokenizer: bool = True) -> tuple[str, str]:
    """('better' | 'no_gain' | 'worse', plain-English reason)."""
    bh, th = base.get("direction_hit_pct"), tuned.get("direction_hit_pct")
    hit_delta = (th - bh) if bh is not None and th is not None else None
    loss_gain = None
    if same_tokenizer and base.get("val_loss"):
        loss_gain = (base["val_loss"] - tuned["val_loss"]) / base["val_loss"] * 100
    if hit_delta is not None and hit_delta <= -5:
        return "worse", f"direction hit rate fell {abs(hit_delta):.1f} points on held-out candles"
    if loss_gain is not None and loss_gain < 0.5:
        return "no_gain", f"prediction loss changed by only {loss_gain:+.1f}% on held-out candles"
    if hit_delta is not None and hit_delta < 0 and (loss_gain is None or loss_gain < 2):
        return "no_gain", "no clear improvement on held-out candles"
    parts = []
    if loss_gain is not None:
        parts.append(f"prediction loss {loss_gain:.1f}% lower")
    if hit_delta is not None:
        parts.append(f"direction hit rate {hit_delta:+.1f} points")
    return "better", ", ".join(parts) + " on held-out candles"


def prune_runs(config: dict, models_dir: Path | None = None) -> None:
    keep = int(settings(config)["keep_runs"])
    in_use = str((config.get("kronos") or {}).get("model_name") or "")
    shadowed = str(settings(config).get("shadow_model") or "")
    with_weights = [r for r in list_runs(models_dir) if r["has_weights"]]
    for r in with_weights[keep:]:
        if os.path.abspath(r["path"]) in (os.path.abspath(in_use), os.path.abspath(shadowed) if shadowed else None):
            continue
        for name in ("model.safetensors", "tokenizer"):
            p = Path(r["path"]) / name
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()


def run_training(config: dict, models_dir: Path | None = None, status_path: Path | None = None,
                 training_root: Path | None = None, log=None, signals_by_broker: dict | None = None,
                 keep_always: bool = False) -> dict:
    """One full run: load data, fine-tune, evaluate against the original,
    save the model folder and report. Returns the report."""
    import torch

    ft = settings(config)
    log = log or logger.info
    models_dir = models_dir or MODELS_DIR
    run_id = "kronos-ft-" + datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = models_dir / run_id
    started = time.time()

    def status(**kw):
        write_status(status_path, run_id=run_id, pid=os.getpid(), started=started,
                     max_steps=int(ft["max_steps"]), **kw)

    status(state="running", step=0, message="Loading candles")
    try:
        if ft["threads"]:
            torch.set_num_threads(int(ft["threads"]))
        try:
            os.nice(10)                               # stay out of the trading agent's way
        except (OSError, AttributeError):
            pass
        torch.manual_seed(int(ft["seed"]))
        lookback, predict = window_sizes(config)
        ws = load_window_set(lookback, predict, float(ft["val_fraction"]), audit_dbs(config), training_root,
                             int(ft["seed"]), signals_by_broker)
        if len(ws.train) < int(ft["min_train_windows"]) or not ws.val:
            raise RuntimeError(
                f"Not enough candles yet: {len(ws.train)} training windows from {len(ws.series)} symbols "
                f"(need {ft['min_train_windows']}, plus held-out candles). Let the agent collect for more sessions.")
        log(f"{len(ws.series)} symbols, {len(ws.train)} training windows ({len(ws.train_signal)} at the agent's "
            f"signals), {len(ws.val)} held-out; window {lookback}+{predict} candles")

        base_name, tok_name = original_names(config)
        local = bool((config.get("kronos") or {}).get("local_files_only", True))
        status(state="running", step=0, message=f"Loading {base_name}")
        model, tokenizer = load_models(base_name, tok_name, local)
        device = pick_device(ft.get("device"), torch)
        device_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU"
        log(f"Training on {device_name}")
        model, tokenizer = model.to(device), tokenizer.to(device)
        tokenizer.eval()
        for p in tokenizer.parameters():
            p.requires_grad_(False)

        bs = int(ft["batch_size"])
        val_items = ws.val_items(int(ft["val_windows"]))
        fc_items = ws.val_items(int(ft["eval_forecasts"]))
        status(state="running", step=0, message="Scoring the original model on held-out candles")
        base = {"val_loss": round(val_loss(model, tokenizer, ws, val_items, bs, torch), 5),
                **forecast_quality(model, tokenizer, ws, fc_items, config, torch)}
        log(f"original: held-out loss {base['val_loss']}, direction hit {base['direction_hit_pct']}%")

        tuned_tokenizer = bool(ft["train_tokenizer"])
        if tuned_tokenizer:
            status(state="running", step=0, message="Fine-tuning the tokenizer")
            for p in tokenizer.parameters():
                p.requires_grad_(True)
            _train_tokenizer(tokenizer, ws, ft, torch, log)
            for p in tokenizer.parameters():
                p.requires_grad_(False)

        max_steps, lr = int(ft["max_steps"]), float(ft["learning_rate"])
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max_steps, pct_start=0.03,
                                                    div_factor=10)
        deadline = started + float(ft["max_minutes"]) * 60
        best, best_step, bad, step, losses, stopped = float("inf"), 0, 0, 0, [], "max_steps"
        run_dir.mkdir(parents=True, exist_ok=True)
        t_train = time.time()
        while step < max_steps:
            model.train()
            x, st = ws.sample_train(bs, float(ft["signal_fraction"]))
            loss = _loss(model, tokenizer, torch.from_numpy(x), torch.from_numpy(st), torch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            if step % int(ft["eval_every"]) == 0 or step == max_steps:
                v = val_loss(model, tokenizer, ws, val_items, bs, torch)
                per = (time.time() - t_train) / step
                eta = min((max_steps - step) * per, max(0.0, deadline - time.time()))
                log(f"step {step}/{max_steps}: train {np.mean(losses[-50:]):.4f}, held-out {v:.4f} "
                    f"(original {base['val_loss']:.4f})")
                if v < best - 1e-5:
                    best, best_step, bad = v, step, 0
                    model.save_pretrained(run_dir)
                else:
                    bad += 1
                status(state="running", step=step, train_loss=round(float(np.mean(losses[-50:])), 5),
                       val_loss=round(v, 5), best_val_loss=round(best, 5), base_val_loss=base["val_loss"],
                       eta_s=round(eta), message="Training")
                if bad >= int(ft["patience"]):
                    stopped = "no further improvement"
                    break
            if time.time() > deadline:
                stopped = "time limit"
                if best == float("inf"):
                    best, best_step = val_loss(model, tokenizer, ws, val_items, bs, torch), step
                    model.save_pretrained(run_dir)
                break

        status(state="running", step=step, message="Scoring the fine-tuned model on held-out candles")
        Kronos, _, _ = _kronos_classes()
        best_model = Kronos.from_pretrained(str(run_dir), local_files_only=True).to(device)
        tuned = {"val_loss": round(val_loss(best_model, tokenizer, ws, val_items, bs, torch), 5),
                 **forecast_quality(best_model, tokenizer, ws, fc_items, config, torch)}
        kind, why = verdict(base, tuned, same_tokenizer=not tuned_tokenizer)
        tok_path = tok_name
        if tuned_tokenizer:
            tokenizer.save_pretrained(run_dir / "tokenizer")
            tok_path = str(run_dir / "tokenizer")
        by_broker = {}
        for s in ws.series:
            by_broker[s.broker] = by_broker.get(s.broker, 0) + 1
        report = {
            "run_id": run_id, "created": time.time(), "minutes": round((time.time() - started) / 60, 1),
            "device": device_name,
            "base_model": base_name, "base_tokenizer": tok_name, "tokenizer": tok_path,
            "tokenizer_tuned": tuned_tokenizer, "steps": step, "best_step": best_step, "stopped": stopped,
            "lookback": lookback, "predict": predict, "symbols": len(ws.series), "symbols_by_broker": by_broker,
            "train_windows": len(ws.train), "signal_windows": len(ws.train_signal), "val_windows": len(ws.val),
            "held_out_from": ws.cutoff_ms / 1000 if ws.cutoff_ms else None,
            "original": base, "fine_tuned": tuned, "verdict": kind, "reason": why,
        }
        (run_dir / "report.json").write_text(json.dumps(report, indent=2))
        if kind != "better" and not keep_always:       # nothing worth keeping: free the disk space
            (run_dir / "model.safetensors").unlink(missing_ok=True)
        prune_runs(config, models_dir)
        log(f"{run_id}: {kind} - {why}")
        write_status(status_path, state="done", run_id=run_id, started=started, step=step, max_steps=max_steps,
                     verdict=kind, message=why)
        return report
    except BaseException as exc:
        write_status(status_path, state="failed", run_id=run_id, started=started,
                     message=str(exc)[:400] or type(exc).__name__)
        if run_dir.exists() and not (run_dir / "report.json").exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------- can this computer do it?

def hardware_check(config: dict, steps: int = 3) -> dict:
    """Time a few real training steps on this computer and estimate a run."""
    import torch

    ft = settings(config)
    lookback, predict = window_sizes(config)
    out = {"cpu_threads": torch.get_num_threads(), "cuda": bool(torch.cuda.is_available()),
           "torch": torch.__version__, "window": lookback + predict, "batch_size": int(ft["batch_size"])}
    try:
        import psutil
        out["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # noqa: BLE001
        try:
            out["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
        except (ValueError, OSError, AttributeError):
            out["ram_gb"] = None
    base_name, tok_name = original_names(config)
    out["model"] = base_name
    model, tokenizer = load_models(base_name, tok_name, bool((config.get("kronos") or {}).get("local_files_only", True)))
    out["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-9)       # timing only: effectively no update
    bs, w = out["batch_size"], out["window"]
    x = torch.randn(bs, w, 6).clamp(-5, 5)
    st = torch.zeros(bs, w, 5)
    times = []
    for _ in range(steps + 1):
        t0 = time.time()
        loss = _loss(model, tokenizer, x, st, torch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        times.append(time.time() - t0)
    per = float(np.median(times[1:]))
    out["seconds_per_step"] = round(per, 2)
    out["est_minutes"] = round(per * int(ft["max_steps"]) / 60 * 1.15 + 5)     # + evaluations
    out["max_steps"], out["max_minutes"] = int(ft["max_steps"]), int(ft["max_minutes"])
    out["ok"] = out["est_minutes"] <= 12 * 60 and (out["ram_gb"] is None or out["ram_gb"] >= 8)
    return out
