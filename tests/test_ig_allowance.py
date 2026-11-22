"""Staying inside IG's request allowances (shared by the agent and the dashboard)."""
import pytest

import src.brokers.ig as igmod
from src.brokers.base import BrokerError
from src.brokers.ig import IGClient
from tests.test_ig24 import SNAPS, FakeHttp, FakeResp

LIMIT = FakeResp(403, {"errorCode": "error.public-api.exceeded-api-key-allowance"})


class CountingHttp(FakeHttp):
    def __init__(self, login_needs_encryption=False):
        super().__init__()
        self.posts, self.enc = [], login_needs_encryption
        self.positions = {"positions": []}
        self.orders = {"workingOrders": []}
        self.fail_next = 0

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append(json)
        if self.fail_next:
            self.fail_next -= 1
            return LIMIT
        if self.enc and not json.get("encryptedPassword"):
            return FakeResp(400, {"errorCode": "error.public-api.failure.encryption.required"})
        return super().post(url, headers, json, timeout)

    def get(self, url, headers=None, timeout=None):
        return FakeResp(200, {"encryptionKey": "x", "timeStamp": 1})

    def request(self, method, url, headers=None, json=None, params=None, timeout=None):
        path = url.split("/gateway/deal", 1)[1]
        if self.fail_next:
            self.fail_next -= 1
            self.calls.append((path, params))
            return LIMIT
        if path == "/positions":
            self.calls.append((path, params))
            return FakeResp(200, self.positions)
        if path == "/workingorders":
            self.calls.append((path, params))
            return FakeResp(200, self.orders)
        if path.startswith("/positions/otc") or path.startswith("/workingorders/otc"):
            self.calls.append((path, params))
            return FakeResp(200, {"dealReference": "REF"})
        if path.startswith("/confirms/"):
            return FakeResp(200, {"dealStatus": "ACCEPTED", "dealId": "D1"})
        if path == "/session/encryptionKey":
            return FakeResp(200, {"encryptionKey": "x", "timeStamp": 1})
        return super().request(method, url, headers, json, params, timeout)


@pytest.fixture
def creds(monkeypatch):
    for k, v in {"IG_API_KEY": "key", "IG_USERNAME": "u", "IG_PASSWORD": "p"}.items():
        monkeypatch.setenv(k, v)


@pytest.fixture
def clock(monkeypatch):
    c = {"t": 1_000_000.0}
    monkeypatch.setattr(igmod.time, "time", lambda: c["t"])
    monkeypatch.setattr(igmod.time, "sleep", lambda s: c.__setitem__("t", c["t"] + s))
    return c


def calls(http, path):
    return sum(1 for p, _ in http.calls if p == path)


def test_order_checks_in_one_cycle_share_one_positions_and_orders_request(creds, clock):
    http = CountingHttp()
    ig = IGClient(market="US", session=http)
    http.orders = {"workingOrders": [{"workingOrderData": {"dealId": "W1", "epic": "E", "direction": "SELL",
                                                           "orderSize": 1, "orderLevel": 1.0}}]}
    for oid in ("W1", "W1", "W1"):                 # entry, stop and target of open trades
        ig.get_order_state(oid)
    assert calls(http, "/workingorders") == 1 and calls(http, "/positions") <= 1
    clock["t"] += 5                                 # next cycle: fresh figures
    ig.get_order_state("W1")
    assert calls(http, "/workingorders") == 2


def test_placing_an_order_drops_the_cached_positions(creds, clock):
    http = CountingHttp()
    ig = IGClient(market="US", session=http)
    ig.get_open_positions()
    ig.cancel_order("W1")
    ig.get_open_positions()
    assert calls(http, "/positions") == 2


def test_budget_waits_for_room_and_orders_are_never_held_back(creds, clock, monkeypatch):
    monkeypatch.setenv("IG_REQUESTS_PER_MIN", "3")
    monkeypatch.setenv("IG_REQUEST_MAX_WAIT", "120")
    http = CountingHttp()
    ig = IGClient(market="US", session=http)       # login: 1 of 3
    t0 = clock["t"]
    ig._request("GET", "/accounts", 1)              # 2
    ig._request("GET", "/session", 1)               # 3
    ig.cancel_order("W1")                           # an order: not counted, no wait
    assert clock["t"] == t0
    ig._request("GET", "/session", 1)               # 4th in a minute: waits for room
    assert clock["t"] - t0 >= 59


def test_dashboard_fails_fast_when_its_share_is_used(creds, clock, monkeypatch):
    monkeypatch.setenv("IG_REQUESTS_PER_MIN", "2")
    monkeypatch.setenv("IG_REQUEST_MAX_WAIT", "2")
    ig = IGClient(market="US", session=CountingHttp())
    ig._request("GET", "/session", 1)
    with pytest.raises(BrokerError, match="budget"):
        ig._request("GET", "/session", 1)


def test_allowance_error_pauses_requests_instead_of_hammering(creds, clock):
    http = CountingHttp()
    ig = IGClient(market="US", session=http)
    http.fail_next = 1
    with pytest.raises(BrokerError, match="exceeded-api-key-allowance"):
        ig._request("GET", "/session", 1)
    sent = len(http.calls)
    for _ in range(5):                                # the agent's and dashboard's next polls
        with pytest.raises(BrokerError, match="pausing IG requests until"):
            ig._request("GET", "/session", 1)
    with pytest.raises(BrokerError, match="pausing"):
        IGClient(market="US", session=http)           # reconnect attempts don't log in either
    assert len(http.calls) == sent and len(http.posts) == 1
    ig.cancel_order("W1")                             # orders still go out (their own allowance)
    clock["t"] += 61
    ig._request("GET", "/session", 1)                 # resumes after the pause


def test_repeated_limits_pause_longer_up_to_15_minutes(creds, clock):
    http = CountingHttp()
    IGClient(market="US", session=http)
    pauses = []
    for _ in range(6):
        http.fail_next = 1
        with pytest.raises(BrokerError):
            IGClient(market="US", session=http)
        pauses.append(round(igmod.GUARD.paused_until - clock["t"]))
        clock["t"] = igmod.GUARD.paused_until + 1
    assert pauses == [60, 120, 240, 480, 900, 900]


def test_login_remembers_the_account_needs_an_encrypted_password(creds, clock, monkeypatch):
    monkeypatch.setattr(igmod, "encrypt_password", lambda pw, key, ts: "ENC")
    http = CountingHttp(login_needs_encryption=True)
    IGClient(market="US", session=http)
    IGClient(market="US", session=http)
    assert [bool(p.get("encryptedPassword")) for p in http.posts] == [False, True, True]


def test_quotes_for_many_markets_come_in_one_request(creds, clock):
    http = CountingHttp()
    ig = IGClient(market="IG24", session=http)
    epics = list(SNAPS)[:3]
    df = ig.get_snapshots(epics)
    assert sorted(df["code"]) == sorted(epics) and calls(http, "/markets") == 1
