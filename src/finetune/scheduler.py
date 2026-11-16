"""Starts the scheduled training run (finetune.train_time) as a separate,
low-priority process, and only while every market the agent trades is closed."""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from datetime import datetime

from . import ROOT, settings
from .trainer import read_status

logger = logging.getLogger(__name__)
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def start_training_process(extra_args: list[str] | None = None) -> int:
    """Launch `python -m src.finetune train` in the background; its output
    goes to logs/finetune.log. Raises if a run is already going."""
    if read_status().get("state") == "running":
        raise RuntimeError("A training run is already in progress")
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "finetune.log", "ab") as log:
        log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} starting fine-tuning =====\n".encode())
        proc = subprocess.Popen([sys.executable, "-u", "-m", "src.finetune", "train", *(extra_args or [])],
                                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return proc.pid


def due(ft: dict, now: datetime, last_started: float | None) -> bool:
    """True when the scheduled time has passed today, today is a training
    day, and no run has been started today."""
    spec = str(ft.get("train_time") or "").strip()
    if not ft.get("enabled") or not spec:
        return False
    try:
        hh, mm = (int(x) for x in spec.split(":")[:2])
    except ValueError:
        return False
    days = [str(d).strip().lower()[:3] for d in (ft.get("train_days") or DAYS)]
    if DAYS[now.weekday()] not in days:
        return False
    slot = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if now < slot or (now - slot).total_seconds() > 6 * 3600:      # don't start very late
        return False
    return not (last_started and last_started >= slot.timestamp())


def start(config: dict, phases) -> threading.Thread | None:
    """`phases()` returns the current market phase of every broker loop."""
    ft = settings(config)
    if not ft["enabled"] or not ft["train_time"]:
        return None

    def loop():
        while True:
            time.sleep(60)
            try:
                st = read_status()
                if st.get("state") == "running" or not due(ft, datetime.now(), st.get("started")):
                    continue
                if any(p in ("active", "always", "warmup") for p in phases()):
                    continue                               # a market is open: wait
                pid = start_training_process()
                logger.info("Scheduled Kronos fine-tuning started (pid %s) - see logs/finetune.log", pid)
            except Exception:
                logger.exception("Fine-tuning scheduler error")

    t = threading.Thread(target=loop, name="finetune-scheduler", daemon=True)
    t.start()
    logger.info("Kronos fine-tuning scheduled at %s on %s (only while markets are closed)",
                ft["train_time"], ", ".join(ft["train_days"] or DAYS))
    return t
