"""
Broker registry.

To add a broker: write an adapter implementing src.brokers.base.BrokerClient
and add one line to REGISTRY. The orchestrator, screener, dashboard and the
Kronos pipeline need no changes.

    adapter     status        notes
    moomoo      ready         OpenD gateway (src/moomoo_client.py)
    tiger       ready         tigeropen SDK, ported from tiger-swing-agent
    ig          experimental  IG REST API - not yet tested on a real account
    zerodha     experimental  Kite Connect (NSE/BSE), GTT exits - daily login via kite_login.py
    angel       experimental  SmartAPI (NSE/BSE), automatic TOTP login
"""
from __future__ import annotations

import importlib
import threading

from src.brokers.base import BrokerClient, BrokerError, Capabilities, base_symbol  # noqa: F401

REGISTRY = {
    "moomoo": ("src.moomoo_client", "MoomooClient"),
    "tiger": ("src.brokers.tiger", "TigerClient"),
    "ig": ("src.brokers.ig", "IGClient"),
    "zerodha": ("src.brokers.zerodha", "ZerodhaClient"),
    "angel": ("src.brokers.angel", "AngelClient"),
}


def broker_class(adapter: str) -> type[BrokerClient]:
    if adapter not in REGISTRY:
        raise BrokerError(f"Unknown broker adapter {adapter!r}. Known: {', '.join(REGISTRY)}")
    module, cls = REGISTRY[adapter]
    return getattr(importlib.import_module(module), cls)


def capabilities() -> dict:
    """Capability matrix for the dashboard - works without the SDKs installed."""
    out = {}
    for adapter in REGISTRY:
        try:
            cls = broker_class(adapter)
            out[adapter] = {"label": cls.label, **cls.capabilities.to_dict()}
        except Exception as exc:  # noqa: BLE001 - e.g. SDK not installed
            out[adapter] = {"label": adapter, **Capabilities(status="unavailable", notes=str(exc)).to_dict()}
    return out


def connect(adapter: str, timeout: float = 35.0, data_source: str | None = None, **kwargs):
    """Connect the trading broker and, when `data_source` names a different
    adapter, a second broker that supplies all price data (see
    src/brokers/data_source.py). Both connects are bounded."""
    trade = connect_with_timeout(adapter, timeout, **kwargs)
    if not data_source or data_source == adapter:
        return trade
    from src.brokers.data_source import DataRoutedBroker
    data_kwargs = dict(kwargs)
    data_kwargs.setdefault("market", getattr(trade, "market", None))
    try:
        data = connect_with_timeout(data_source, timeout, **data_kwargs)
    except BaseException as exc:
        trade.close()
        label = REGISTRY.get(data_source, (None, data_source))[1]
        raise BrokerError(f"Price data source {data_source} ({label}) failed to connect: {exc}") from exc
    return DataRoutedBroker(trade, data)


def connect_with_timeout(adapter: str, timeout: float = 35.0, **kwargs) -> BrokerClient:
    """Build the adapter with a hard wall-clock limit, so a broker that
    accepts the connection but never answers can't freeze the agent. Runs the
    connect in a daemon thread; a client that arrives after we gave up is
    closed."""
    if adapter == "moomoo":   # moomoo already has its own bounded connect
        from src.moomoo_client import connect_with_timeout as moomoo_connect
        return moomoo_connect(timeout, **kwargs)

    cls = broker_class(adapter)
    result: dict = {}
    lock = threading.Lock()
    gave_up = False

    def target():
        try:
            client = cls(**kwargs)
        except BaseException as exc:  # reported to the caller
            result["error"] = exc
            return
        with lock:
            late = gave_up
            if not late:
                result["client"] = client
        if late:
            client.close()

    worker = threading.Thread(target=target, name=f"{adapter}-connect", daemon=True)
    worker.start()
    worker.join(timeout)
    with lock:
        if "client" not in result and "error" not in result:
            gave_up = True
    if gave_up:
        raise ConnectionError(f"{cls.label} gave no response within {timeout:.0f}s")
    if "error" in result:
        raise result["error"]
    return result["client"]
