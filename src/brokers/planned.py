"""
Base for a broker that is listed in config but not wired in yet: enabling
it fails at start-up with a clear message instead of half-trading.

Not used by any registered broker today (Zerodha and Angel have adapters
in zerodha.py / angel.py). Subclass it to reserve a name in REGISTRY.
"""
from src.brokers.base import BrokerClient, BrokerError


class PlannedClient(BrokerClient):
    def __init__(self, *args, **kwargs):
        raise BrokerError(f"The {self.label} adapter is not implemented yet - "
                          f"set brokers.{self.name}.enabled: false")

    # Never reached - __init__ raises. Defined so the class isn't abstract.
    def get_historical_bars(self, *a, **k): raise NotImplementedError
    def get_current_price(self, *a, **k): raise NotImplementedError
    def get_open_positions(self): raise NotImplementedError
    def get_open_orders(self): raise NotImplementedError
    def get_account_summary(self): raise NotImplementedError
    def get_order_state(self, *a, **k): raise NotImplementedError
    def create_entry(self, *a, **k): raise NotImplementedError
    def place_stop_order(self, *a, **k): raise NotImplementedError
    def place_target_order(self, *a, **k): raise NotImplementedError
    def modify_stop_order(self, *a, **k): raise NotImplementedError
    def cancel_order(self, *a, **k): raise NotImplementedError
    def close_position(self, *a, **k): raise NotImplementedError
