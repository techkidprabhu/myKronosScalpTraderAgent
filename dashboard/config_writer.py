"""
Write dashboard settings changes into config/config.yaml without losing
its comments or layout (ruamel.yaml round-trip), at the shared top level
or as a per-broker override under `brokers.<name>`.

Every write is checked end to end before it lands: the new file must load,
every broker must resolve, at least one must stay enabled, cross-field
rules must hold for each broker's effective settings, and each changed
value must read back exactly as requested. The previous file is kept as
config.yaml.bak plus a timestamped copy in config/backups/ (last 20).
"""
from __future__ import annotations

import io
import os
import shutil
import time
from pathlib import Path

import yaml

from dashboard.schema import FIELDS, coerce, cross_check
from src.broker_config import BROKER_ONLY_KEYS, broker_specs, enabled_specs

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
except ImportError:  # pragma: no cover - checked at request time
    YAML = None
    CommentedMap = dict

BACKUPS_KEPT = 20


class SettingsError(ValueError):
    pass


def _yaml():
    if YAML is None:
        raise SettingsError("Saving settings needs ruamel.yaml: pip install ruamel.yaml")
    y = YAML(typ="rt")
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _get(node, parts):
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return None, False
        node = node[p]
    return node, True


def _set(node, parts, value):
    for p in parts[:-1]:
        if not isinstance(node.get(p), dict):
            node[p] = CommentedMap()
        node = node[p]
    node[parts[-1]] = value


def _delete(node, parts):
    chain = []
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            return
        chain.append((node, p))
        node = node[p]
    if isinstance(node, dict):
        node.pop(parts[-1], None)
    for parent, key in reversed(chain):        # prune maps left empty
        if isinstance(parent[key], dict) and not parent[key]:
            del parent[key]
        else:
            break


def apply_changes(text: str, scope: str, changes: dict, resets: list | None = None) -> tuple[str, dict]:
    """Return (new_text, applied) for `changes` {dotted key: value} at
    `scope` ("shared" or a broker name). `resets` removes per-broker
    overrides so the shared value applies again."""
    resets = list(resets or [])
    y = _yaml()
    doc = y.load(text) or CommentedMap()
    plain = yaml.safe_load(text) or {}
    names = [s.name for s in broker_specs(plain)]

    if scope != "shared" and scope not in names:
        raise SettingsError(f"Unknown broker {scope!r}")
    applied = {}
    for key, raw in (changes or {}).items():
        f = FIELDS.get(key)
        if f is None:
            raise SettingsError(f"{key}: not editable from the dashboard")
        fscope = f.get("scope", "any")
        if scope == "shared" and fscope == "broker":
            raise SettingsError(f"{f['label']}: set it per broker")
        if scope != "shared" and fscope == "shared":
            raise SettingsError(f"{f['label']}: shared by all brokers - change it under Shared settings")
        try:
            applied[key] = coerce(key, raw)
        except ValueError as exc:
            raise SettingsError(str(exc)) from None

    if scope != "shared" and (applied or resets):
        if not isinstance(doc.get("brokers"), dict):
            # First per-broker setting: make the implicit default explicit.
            doc["brokers"] = CommentedMap()
            doc["brokers"]["moomoo"] = CommentedMap([("enabled", True)])
        if not isinstance(doc["brokers"].get(scope), dict):
            doc["brokers"][scope] = CommentedMap()
        base = doc["brokers"][scope]
    else:
        base = doc

    for key, value in applied.items():
        _set(base, key.split("."), value)
    for key in resets:
        if scope == "shared":
            raise SettingsError("Only per-broker overrides can be reset")
        if key in BROKER_ONLY_KEYS:
            continue
        _delete(base, key.split("."))

    out = io.StringIO()
    y.dump(doc, out)
    new_text = out.getvalue()
    validate_text(new_text, scope, applied)
    return new_text, applied


def validate_text(text: str, scope: str, applied: dict):
    try:
        cfg = yaml.safe_load(text) or {}
        specs = broker_specs(cfg)
        enabled_specs(cfg)
    except Exception as exc:  # noqa: BLE001
        raise SettingsError(f"Config would be invalid: {exc}") from None
    by_msg: dict[str, list[str]] = {}
    for s in specs:
        for msg in cross_check(s.config):
            by_msg.setdefault(msg, []).append(s.name)
    if by_msg:
        # Say it once when every broker has the same problem (a shared setting).
        parts = [msg if len(names) == len(specs) else f"{', '.join(names)}: {msg}"
                 for msg, names in by_msg.items()]
        raise SettingsError("; ".join(parts))
    # read back what we wrote
    for key, value in applied.items():
        if scope == "shared":
            got, ok = _get(cfg, key.split("."))
        else:
            got, ok = _get((cfg.get("brokers") or {}).get(scope, {}), key.split("."))
        if not ok or got != value:
            raise SettingsError(f"{key}: config rewrite did not round-trip ({got!r} != {value!r})")


def write_config(path: Path, new_text: str):
    path = Path(path)
    if path.exists():
        backups = path.parent / "backups"
        backups.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, backups / f"config.{stamp}.yaml")
        shutil.copy2(path, path.with_suffix(".yaml.bak"))
        for old in sorted(backups.glob("config.*.yaml"))[:-BACKUPS_KEPT]:
            old.unlink(missing_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(new_text)
    os.replace(tmp, path)
