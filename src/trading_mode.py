"""
Who makes the trading decisions: Kronos (the default) or a skill file.

One switch for every broker, kept in data/trading_mode.json so it can be
flipped from the dashboard without a restart. The agent re-reads it each
cycle (only when the file changes). A missing or unreadable file means
Kronos - the switch can only ever be turned on on purpose.

    {"mode": "skill", "file": "config/PragaPriceActionSkill.md",
     "changed_ts": 1795100000.0}

In "skill" mode Kronos is not called at all. The skill file's settings
block decides entries, stops, targets and exits; your risk limits (max
loss per trade, daily limits, per-stock limits, flat-by-close) still apply.
Positions keep the manager they were opened with when the switch changes.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SWITCH_PATH = ROOT / "data" / "trading_mode.json"
MODES = ("kronos", "skill")
DEFAULT_FILE = "config/PragaPriceActionSkill.md"

_cache: dict = {"mtime": None, "path": None, "value": None}


def _default() -> dict:
    return {"mode": "kronos", "file": DEFAULT_FILE, "changed_ts": None}


def read(path: Path | None = None) -> dict:
    """The current switch: {"mode", "file", "changed_ts"}."""
    path = Path(path or SWITCH_PATH)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return _default()
    if _cache["path"] == str(path) and _cache["mtime"] == mtime and _cache["value"] is not None:
        return dict(_cache["value"])
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        mode = str(raw.get("mode") or "kronos").lower()
        value = {"mode": mode if mode in MODES else "kronos",
                 "file": str(raw.get("file") or DEFAULT_FILE),
                 "changed_ts": raw.get("changed_ts")}
    except Exception:  # noqa: BLE001
        logger.warning("Unreadable %s - using Kronos", path)
        value = _default()
    _cache.update(mtime=mtime, path=str(path), value=value)
    return dict(value)


def write(mode: str, file: str | None = None, path: Path | None = None) -> dict:
    mode = str(mode).lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    path = Path(path or SWITCH_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"mode": mode, "file": file or read(path).get("file") or DEFAULT_FILE, "changed_ts": time.time()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)
    _cache.update(mtime=None, value=None)
    return value


def resolve_file(rel: str) -> Path:
    """The skill file as a path inside the project's config folder."""
    p = (ROOT / rel).resolve()
    cfg = (ROOT / "config").resolve()
    if p.suffix.lower() != ".md" or cfg not in p.parents:
        raise ValueError("The skill file must be a .md file inside the config folder")
    return p
