"""
Run every enabled broker from one process.

Each broker gets its own Orchestrator - its own connection, audit DB,
risk state (daily loss cap, cooldowns), screener and market calendar - in
its own thread. They share ONE offline Kronos model: it is loaded once and
calls into it are serialised with a lock, so memory use doesn't multiply
and the model never runs two forecasts at the same time.

With a single broker the orchestrator runs on the main thread, exactly as
it always has.
"""
from __future__ import annotations

import logging
import threading
import time
from html import escape as esc

from src.audit_log import AuditLog
from src.broker_config import BrokerSpec, enabled_specs
from src.brokers import capabilities as broker_capabilities
from src.kronos_agent import KronosAgent
from src.orchestrator import Orchestrator
from src.telegram_bot import TelegramReporter

logger = logging.getLogger(__name__)

RESTART_DELAY_SECONDS = 60


class SharedKronos:
    """Thread-safe front for one KronosAgent. The model and its use are
    unchanged - this only stops two broker threads calling it at once."""

    def __init__(self, agent, shadow=None):
        self._agent = agent
        self._lock = threading.Lock()
        self.shadow = shadow          # finetune.shadow.ShadowRunner or None

    def shadow_submit(self, *args, **kwargs) -> bool:
        """Hand the same candles to the shadow model (background, never
        blocks, never affects the verdict). False when shadow mode is off."""
        return bool(self.shadow and self.shadow.submit(*args, **kwargs))

    def confirm_direction(self, *args, **kwargs):
        with self._lock:
            return self._agent.confirm_direction(*args, **kwargs)

    def forecast(self, *args, **kwargs):
        with self._lock:
            return self._agent.forecast(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._agent, name)


def build_orchestrator(spec: BrokerSpec, kronos, multi: bool) -> Orchestrator:
    reporter = TelegramReporter(prefix=f"[{esc(spec.label)}] " if multi else "")
    return Orchestrator(
        spec.config, broker_name=spec.name, adapter=spec.adapter, market=spec.market,
        data_source=spec.data_source,
        label=spec.label, audit_log=AuditLog(str(spec.db_path)), kronos_agent=kronos, reporter=reporter,
    )


class MultiBrokerRunner:
    def __init__(self, config: dict):
        self.config = config
        self.specs = enabled_specs(config)
        self.orchestrators: dict[str, Orchestrator] = {}
        self._lock = threading.Lock()
        kronos_cfg = config.get("kronos", {})
        caps = broker_capabilities()
        for spec in self.specs:
            risk = spec.config.get("risk") or {}
            if risk.get("allow_short") and not caps.get(spec.adapter, {}).get("short_selling", True):
                # e.g. Zerodha CNC / Angel DELIVERY: a SELL entry would only be rejected.
                logger.warning("%s can't short sell - allow_short turned off for this broker", spec.label)
                spec.config["risk"] = {**risk, "allow_short": False}
            if "kronos" in spec.overrides:
                logger.warning("brokers.%s.kronos is ignored: the Kronos model is shared by all brokers "
                               "(set kronos: at the top level)", spec.name)
        logger.info("Loading the shared offline Kronos model for %d broker(s): %s",
                    len(self.specs), ", ".join(s.name for s in self.specs))
        shadow = None
        try:
            from src.finetune import shadow as ft_shadow
            shadow = ft_shadow.build(config)
        except Exception:  # noqa: BLE001
            logger.exception("Shadow mode could not start")
        self.kronos = SharedKronos(KronosAgent(**kronos_cfg), shadow)

    def _run_one(self, spec: BrokerSpec, multi: bool):
        """Build and run one broker's loop; if it dies, rebuild it after a
        pause instead of taking the other brokers down."""
        while True:
            try:
                orch = build_orchestrator(spec, self.kronos, multi)
                with self._lock:
                    self.orchestrators[spec.name] = orch
                orch.run_forever()
            except Exception:  # noqa: BLE001
                logger.exception("Broker %s stopped unexpectedly - restarting in %ss",
                                 spec.name, RESTART_DELAY_SECONDS)
                time.sleep(RESTART_DELAY_SECONDS)

    def _phases(self) -> list:
        with self._lock:
            return [getattr(o, "phase", None) for o in self.orchestrators.values()]

    def run_forever(self):
        try:
            from src.finetune import scheduler as ft_scheduler
            ft_scheduler.start(self.config, self._phases)
        except Exception:  # noqa: BLE001
            logger.exception("Could not start the fine-tuning scheduler")
        if len(self.specs) == 1:
            spec = self.specs[0]
            orch = build_orchestrator(spec, self.kronos, multi=False)
            self.orchestrators[spec.name] = orch
            orch.run_forever()
            return
        threads = []
        for spec in self.specs:
            t = threading.Thread(target=self._run_one, args=(spec, True), name=f"broker-{spec.name}",
                                 daemon=True)
            t.start()
            threads.append(t)
            logger.info("Started %s (%s, market %s, db %s)", spec.name, spec.adapter,
                        spec.market or "from .env", spec.db_path.name)
        while True:   # daemon threads: Ctrl-C / SIGTERM in the main thread ends everything
            time.sleep(1)

    def close(self):
        with self._lock:
            orchestrators = list(self.orchestrators.values())
        for orch in orchestrators:
            try:
                orch.broker.close()
            except Exception:  # noqa: BLE001
                logger.debug("Closing %s failed", getattr(orch, "broker_name", "?"), exc_info=True)
