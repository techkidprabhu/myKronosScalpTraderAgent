import argparse
import logging
import signal

import yaml
from dotenv import load_dotenv

from src.multi_broker import MultiBrokerRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s",
)


def _stop(signum, frame):
    # systemd stops services with SIGTERM: leave through the same path as
    # Ctrl-C so broker connections are released.
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description="Kronos scalp trading agent (multi-broker)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    load_dotenv()
    signal.signal(signal.SIGTERM, _stop)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    runner = None
    try:
        # One orchestrator per enabled broker (config.yaml -> brokers:). Each
        # waits, bounded and retrying, until its broker answers - so the agent
        # can be started before OpenD / the broker gateway.
        runner = MultiBrokerRunner(config)
        runner.run_forever()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopping agent")
    finally:
        if runner is not None:
            runner.close()  # release broker connections


if __name__ == "__main__":
    main()
