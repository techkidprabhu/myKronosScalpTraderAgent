"""
Multi-broker configuration.

config.yaml keeps every shared setting at the top level, exactly as
before. An optional `brokers:` section lists the brokers to run; each
entry can override ANY top-level setting for that broker only - the
override is deep-merged over the shared value, so you write just what
differs:

    brokers:
      moomoo:
        enabled: true                 # trades the shared settings as-is
      tiger:
        enabled: true
        market: SG
        timeframe: HOUR_1             # overrides the shared timeframe
        default_quantity: 100
        risk: {lot_size: 100, max_daily_loss: 400}   # only these two risk keys change
        screener: {candidate_symbols: [SG.D05, SG.O39, SG.U11]}

Per-broker-only keys (not settings overrides):
    enabled        run this broker (default true for moomoo, false otherwise)
    adapter        which adapter to use (default: the entry's own name)
    label          display name (default: the adapter's label)
    market         market the broker trades (default: the adapter's .env setting)
    db_path        audit database (default data/audit.db for moomoo,
                   data/audit_<name>.db for others) - one per broker, so
                   positions, cooldowns and the daily loss cap never mix

Credentials always stay in .env. With no `brokers:` section the agent
runs moomoo alone on the top-level settings - identical to before.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from src.brokers import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
BROKER_ONLY_KEYS = {"enabled", "adapter", "label", "market", "db_path"}
DEFAULT_BROKERS = {"moomoo": {"enabled": True}}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class BrokerSpec:
    name: str                       # key under brokers:
    adapter: str
    enabled: bool
    label: str
    market: str | None
    db_path: Path
    overrides: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)   # shared settings with overrides applied

    def to_dict(self) -> dict:
        return {"name": self.name, "adapter": self.adapter, "enabled": self.enabled, "label": self.label,
                "market": self.market, "db_path": str(self.db_path), "overrides": sorted(self.overrides)}


def _default_db(name: str) -> Path:
    return ROOT / "data" / ("audit.db" if name == "moomoo" else f"audit_{name}.db")


def broker_specs(config: dict) -> list[BrokerSpec]:
    """Every broker entry (enabled or not), in config order."""
    shared = {k: v for k, v in (config or {}).items() if k != "brokers"}
    entries = (config or {}).get("brokers") or DEFAULT_BROKERS
    if not isinstance(entries, dict):
        raise ValueError("brokers: must be a mapping of broker name -> settings")
    specs = []
    for name, raw in entries.items():
        raw = dict(raw or {})
        adapter = str(raw.get("adapter", name)).lower()
        if adapter not in REGISTRY:
            raise ValueError(f"brokers.{name}.adapter: unknown adapter {adapter!r} "
                             f"(known: {', '.join(REGISTRY)})")
        overrides = {k: v for k, v in raw.items() if k not in BROKER_ONLY_KEYS}
        db = Path(raw["db_path"]) if raw.get("db_path") else _default_db(name)
        if not db.is_absolute():
            db = ROOT / db
        try:
            from src.brokers import broker_class
            default_label = broker_class(adapter).label
        except Exception:  # noqa: BLE001 - SDK missing; label is cosmetic
            default_label = adapter
        specs.append(BrokerSpec(
            name=name, adapter=adapter,
            enabled=bool(raw.get("enabled", name == "moomoo")),
            label=str(raw.get("label") or default_label),
            market=(str(raw["market"]).upper() if raw.get("market") else None),
            db_path=db, overrides=overrides,
            config=deep_merge(shared, overrides),
        ))
    dbs = [s.db_path for s in specs]
    if len(set(dbs)) != len(dbs):
        raise ValueError("brokers: each broker needs its own db_path")
    return specs


def enabled_specs(config: dict) -> list[BrokerSpec]:
    specs = [s for s in broker_specs(config) if s.enabled]
    if not specs:
        raise ValueError("brokers: no broker is enabled")
    return specs


def spec_for(config: dict, name: str | None) -> BrokerSpec:
    """The named broker, or the first enabled one when name is None."""
    specs = broker_specs(config)
    if name is None:
        enabled = [s for s in specs if s.enabled]
        return (enabled or specs)[0]
    for s in specs:
        if s.name == name:
            return s
    raise KeyError(name)
