"""
Live quote / K-line push updates via moomoo's subscribe + handler model.
OpenD pushes updates to any handler registered on the quote context
after a `subscribe()` call.

The scalp agent polls `get_historical_bars` each cycle (see
orchestrator.py), which is simpler and sufficient for 1-3 minute candles - this module is for anyone who wants real-time updates
instead. Note each subscription consumes moomoo subscription quota and
must stay subscribed for at least one minute before unsubscribing.
"""
import logging

from moomoo import RET_OK, StockQuoteHandlerBase, SubType

logger = logging.getLogger(__name__)


class QuoteStream(StockQuoteHandlerBase):
    def __init__(self, quote_ctx, on_quote_update):
        """on_quote_update: callback(df: pandas.DataFrame) on every push."""
        super().__init__()
        self.quote_ctx = quote_ctx
        self.on_quote_update = on_quote_update
        quote_ctx.set_handler(self)

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret != RET_OK:
            logger.warning("Quote push error: %s", data)
            return ret, data
        self.on_quote_update(data)
        return RET_OK, data

    def subscribe(self, codes: list[str]):
        ret, data = self.quote_ctx.subscribe(codes, [SubType.QUOTE], subscribe_push=True)
        if ret != RET_OK:
            raise RuntimeError(f"subscribe failed: {data}")
        logger.info("Subscribed to live quotes for %s", codes)

    def disconnect(self):
        self.quote_ctx.unsubscribe_all()
