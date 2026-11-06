"""
OpenD execution smoke test for the moomoo swing agent.

Uses the agent's own MoomooClient, so it exercises the exact code path
the orchestrator uses. Run from the project folder with OpenD running:

    source ../venv/bin/activate
    python verify_opend.py                      # read-only checks
    python verify_opend.py --place-test-order   # + place & cancel a paper order
    python verify_opend.py --order-id 3199841   # inspect a specific order

--place-test-order refuses to run unless MOOMOO_TRADING_ENV=SIMULATE.
It places a 1-share BUY LIMIT ~15% below the last price (it should not
fill), confirms OpenD lists it as working, cancels it, and confirms the
cancel.
"""
import argparse
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from moomoo import TrdEnv  # noqa: E402

from src.moomoo_client import MoomooClient  # noqa: E402

OK, FAIL = "PASS", "FAIL"
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((OK, name, detail))
        print(f"[{OK}] {name}: {detail}")
        return detail
    except Exception as exc:
        results.append((FAIL, name, str(exc)))
        print(f"[{FAIL}] {name}: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="US.AAPL")
    ap.add_argument("--place-test-order", action="store_true")
    ap.add_argument("--order-id", action="append", default=[])
    args = ap.parse_args()

    broker = check("connect to OpenD", lambda: MoomooClient())
    if broker is None:
        sys.exit(1)
    results[-1] = (OK, "connect to OpenD",
                   f"{broker.host}:{broker.port} env={broker.trd_env} market={broker.market} acc_id={broker.acc_id}")
    print(f"        -> {results[-1][2]}")

    try:
        check("account summary", broker.get_account_summary)
        price = check(f"last price {args.symbol}", lambda: broker.get_current_price(args.symbol))
        check(f"history K-lines {args.symbol} (HOUR_1)",
              lambda: f"{len(broker.get_historical_bars(args.symbol, 'HOUR_1', 50))} bars")
        check("positions", lambda: broker.get_open_positions() or "none")
        check("working orders", lambda: broker.get_open_orders() or "none")
        check(f"net position {args.symbol}", lambda: broker.get_position_quantity(args.symbol))
        for oid in args.order_id:
            check(f"order {oid}", lambda oid=oid: broker._find_order(oid))

        if args.place_test_order:
            if broker.trd_env != TrdEnv.SIMULATE:
                print(f"[{FAIL}] refusing to place a test order: MOOMOO_TRADING_ENV is not SIMULATE")
                results.append((FAIL, "test order", "not SIMULATE"))
            elif price:
                limit = broker.round_price(price * 0.85)
                oid = check(f"place BUY 1 {args.symbol} LIMIT {limit} (DAY)",
                            lambda: broker.create_entry(args.symbol, "BUY", 1, limit))
                if oid:
                    time.sleep(2)
                    check("order visible and working", lambda: _expect(broker.get_order_state(oid), active=True))
                    check("cancel order", lambda: broker.cancel_order(oid) is not None)
                    time.sleep(2)
                    check("order no longer working", lambda: _expect(broker.get_order_state(oid), active=False))
    finally:
        broker.close()

    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


def _expect(state, active):
    if state["active"] != active:
        raise RuntimeError(f"expected active={active}, got {state}")
    return state


if __name__ == "__main__":
    main()
