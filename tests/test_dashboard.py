import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import dashboard.app as dash
from src.audit_log import AuditLog

ROOT = Path(__file__).resolve().parent.parent
TOKEN = dash.DASHBOARD_TOKEN
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    shutil.copy(ROOT / "config" / "config.example.yaml", path)
    monkeypatch.setattr(dash, "CONFIG_PATH", path)
    return path


@pytest.fixture
def client(tmp_path, config_path, monkeypatch):
    db = tmp_path / "audit.db"
    AuditLog(str(db)).close()
    monkeypatch.setattr(dash, "DB_PATH", db)
    # Never touch the real broker or load the model in tests.
    monkeypatch.setattr(dash, "get_broker", lambda: None)
    monkeypatch.setattr(dash, "get_kronos_agent", lambda: None)
    dash.app.config["TESTING"] = True
    with dash.app.test_client() as c:
        yield c


# ---- authentication ----

def test_requests_without_token_are_rejected(client):
    assert client.get("/").status_code == 401
    assert client.get("/api/status").status_code == 401
    assert client.post("/api/config", json={}).status_code == 401


def test_wrong_token_rejected(client):
    assert client.get("/api/status", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/?token=nope").status_code == 401


def test_login_link_sets_strict_httponly_cookie(client):
    rv = client.get(f"/?token={TOKEN}")
    assert rv.status_code == 302
    assert rv.headers["Location"] == "/"
    cookie = rv.headers["Set-Cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie

    # Cookie now authenticates subsequent requests.
    assert client.get("/api/status").status_code == 200


def test_static_files_are_public(client):
    assert client.get("/static/css/style.css").status_code == 200


def test_cross_origin_post_blocked(client):
    rv = client.post("/api/config", json={}, headers={**AUTH, "Origin": "http://evil.example"})
    assert rv.status_code == 403


def test_no_cors_headers(client):
    rv = client.get("/api/status", headers={**AUTH, "Origin": "http://evil.example"})
    assert "Access-Control-Allow-Origin" not in rv.headers


# ---- config lockdown ----

def test_risk_section_cannot_be_changed(client, config_path):
    before = config_path.read_text()
    cfg = yaml.safe_load(before)
    cfg["risk"]["max_daily_loss"] = 1_000_000

    rv = client.post("/api/config", json=cfg, headers=AUTH)

    assert rv.status_code == 403
    assert "risk" in rv.get_json()["error"]
    assert config_path.read_text() == before


def test_editable_setting_saved_with_comments_preserved(client, config_path):
    cfg = yaml.safe_load(config_path.read_text())
    cfg["kronos_confidence_threshold"] = 0.7
    cfg["timeframe"] = "DAY"

    rv = client.post("/api/config", json=cfg, headers=AUTH)

    assert rv.status_code == 200, rv.get_json()
    saved = config_path.read_text()
    assert yaml.safe_load(saved)["kronos_confidence_threshold"] == 0.7
    assert yaml.safe_load(saved)["timeframe"] == "DAY"
    assert "# Trailing stop" in saved  # comments survive
    assert config_path.with_suffix(".yaml.bak").exists()


def test_invalid_setting_value_rejected(client, config_path):
    cfg = yaml.safe_load(config_path.read_text())
    cfg["kronos_confidence_threshold"] = 0.1
    rv = client.post("/api/config", json=cfg, headers=AUTH)
    assert rv.status_code == 400


def test_unknown_key_rejected(client):
    rv = client.post("/api/config", json={"evil": 1}, headers=AUTH)
    assert rv.status_code == 403


# ---- input validation / resource limits ----

@pytest.mark.parametrize("body", [
    {"symbol": "D05.SI'); alert(1)//"},
    {"horizon": 10_000},
    {"sample_count": 500},
    {"temperature": "hot"},
    {"timeframe": "YEAR"},
])
def test_forecast_rejects_bad_input(client, body):
    assert client.post("/api/forecast", json=body, headers=AUTH).status_code == 400


def test_forecast_rejects_concurrent_runs(client, monkeypatch):
    monkeypatch.setattr(dash, "get_broker", lambda: MagicMock())
    monkeypatch.setattr(dash, "get_kronos_agent", lambda: MagicMock())
    assert dash._kronos_lock.acquire(blocking=False)
    try:
        rv = client.post("/api/forecast", json={"symbol": "US.AAPL"}, headers=AUTH)
        assert rv.status_code == 429
    finally:
        dash._kronos_lock.release()


def test_limit_parameter_bounded(client):
    assert client.get("/api/decisions?limit=abc", headers=AUTH).status_code == 400
    assert client.get("/api/decisions?limit=100000", headers=AUTH).status_code == 400


# ---- existing endpoints still work ----

def test_index_route(client):
    rv = client.get("/", headers=AUTH)
    assert rv.status_code == 200
    assert b"Kronos Scalp Agent" in rv.data
    assert b"Strategies & Tactics" in rv.data
    assert b"tab-strategies" in rv.data
    # the playbook covers the whole current pipeline, self-learning included
    for card in (b'id="st-markets"', b'id="st-indicators"', b'id="st-kronos"', b'id="st-risk"', b'id="st-manage"',
                 b'id="st-learn"', b'id="st-shadow"', b'id="st-versions"', b"Project map"):
        assert card in rv.data


def test_api_status(client):
    data = client.get("/api/status", headers=AUTH).get_json()
    assert data["status"] == "ok"
    assert "kronos" in data
    assert data["broker"]["name"] == "moomoo"
    assert data["broker"]["connected"] is False
    assert data["kronos"]["offline"] is True


def test_api_decisions(client):
    data = client.get("/api/decisions?limit=5", headers=AUTH).get_json()
    assert isinstance(data["decisions"], list)


def test_api_snapshots(client):
    data = client.get("/api/snapshots?limit=5", headers=AUTH).get_json()
    assert isinstance(data["snapshots"], list)


def test_api_positions(client):
    data = client.get("/api/positions", headers=AUTH).get_json()
    assert isinstance(data["positions"], list)


def test_api_config(client):
    data = client.get("/api/config", headers=AUTH).get_json()
    assert "timeframe" in data
    assert "kronos" in data


# ---- active symbol list ----

def _status(client):
    return client.get("/api/status", headers=AUTH).get_json()


def test_symbols_fall_back_to_config_list(client):
    data = _status(client)
    assert data["symbols_source"] == "config"
    assert data["symbols"] == ["US.AAPL", "US.MSFT"]


def test_symbols_fall_back_to_screener_candidates(client, config_path):
    cfg = yaml.safe_load(config_path.read_text())
    cfg["screener"]["enabled"] = True
    config_path.write_text(yaml.safe_dump(cfg))

    data = _status(client)
    assert data["symbols_source"] == "screener_candidates"
    assert data["symbols"] == cfg["screener"]["candidate_symbols"]


def test_symbols_published_by_agent_win(client):
    audit = AuditLog(str(dash.DB_PATH))
    audit.set_kv("active_symbols", {"source": "screener", "updated": 1.0, "symbols": [
        {"symbol": "US.NVDA", "name": "US.NVDA"}, {"symbol": "US.AMD", "name": "US.AMD"}]})
    audit.record_pending_entry("US.TSLA", "BUY", 100, 30.0, 29.5, 31.0, entry_order_id="1")

    data = _status(client)

    assert data["symbols_source"] == "screener"
    # Symbol with a working position stays visible even if not screened in.
    assert data["symbols"] == ["US.NVDA", "US.AMD", "US.TSLA"]
    assert data["symbol_details"][2]["has_position"] is True


def test_config_names_used_for_published_symbols(client):
    AuditLog(str(dash.DB_PATH)).set_kv("active_symbols", {"source": "screener", "symbols": [
        {"symbol": "US.AAPL", "name": "US.AAPL"}]})
    assert _status(client)["symbol_details"][0]["name"] == "Apple"


def test_forecast_requires_symbol(client):
    assert client.post("/api/forecast", json={}, headers=AUTH).status_code == 400


# ---- positions / working orders ----

def _fake_broker(positions, orders):
    broker = MagicMock()
    broker.market, broker.account = "SG", "acc"
    broker.get_open_positions.return_value = positions
    broker.get_open_orders.return_value = orders
    return broker


def _pos(symbol, qty, cost, price, pnl):
    # Shape returned by MoomooClient.get_open_positions
    return {"symbol": symbol, "name": symbol, "quantity": qty, "average_cost": cost,
            "market_price": price, "unrealized_pnl": pnl, "currency": "SGD"}


def _order(oid, symbol, action, otype, qty, limit=None, stop=None):
    # Shape returned by MoomooClient.get_open_orders
    return {"id": str(oid), "symbol": symbol, "action": action, "order_type": otype,
            "quantity": qty, "filled": 0, "limit_price": limit, "stop_price": stop,
            "status": "SUBMITTED", "time_in_force": "GTC", "order_time": "2026-11-02 10:15:40.000"}


@pytest.fixture
def broker(client, monkeypatch):
    monkeypatch.setattr(dash, "_broker_cache", {"ts": 0.0, "data": None})
    holder = {}
    monkeypatch.setattr(dash, "get_broker", lambda: holder["broker"])
    return holder


def test_positions_merge_broker_and_agent(client, broker):
    audit = AuditLog(str(dash.DB_PATH))
    audit.record_pending_entry("SG.O39", "BUY", 100, 28.32, 28.17, 28.62, entry_order_id="1")
    audit.mark_position_open("SG.O39", 100, 28.30, "2", "3")
    audit.record_pending_entry("SG.U11", "BUY", 100, 30.0, 29.5, 31.0, entry_order_id="4")
    broker["broker"] = _fake_broker(
        positions=[_pos("SG.O39", 100, 28.30, 28.50, 20.0), _pos("SG.D05", 30, 71.71, 71.0, -21.3)],
        orders=[_order(2, "SG.O39", "SELL", "STOP", 100, stop=28.17),
                _order(9, "SG.S68", "BUY", "STOP", 100, stop=20.47)],
    )

    data = client.get("/api/positions", headers=AUTH).get_json()

    assert data["broker_connected"] is True
    by_symbol = {p["symbol"]: p for p in data["positions"]}
    assert by_symbol["SG.O39"]["state"] == "managed"
    assert by_symbol["SG.O39"]["stop"] == 28.17
    assert by_symbol["SG.O39"]["current_price"] == 28.50
    assert by_symbol["SG.D05"]["state"] == "unmanaged"
    assert by_symbol["SG.U11"]["state"] == "pending_entry"
    managed = {o["id"]: o["managed"] for o in data["orders"]}
    assert managed == {"2": True, "9": False}


def test_positions_without_broker_shows_agent_view(client, broker):
    broker["broker"] = None
    AuditLog(str(dash.DB_PATH)).record_pending_entry("SG.U11", "BUY", 100, 30.0, 29.5, 31.0, entry_order_id="4")

    data = client.get("/api/positions", headers=AUTH).get_json()

    assert data["broker_connected"] is False
    assert [p["state"] for p in data["positions"]] == ["pending_entry"]


def test_broker_queries_are_cached(client, broker):
    broker["broker"] = _fake_broker([], [])
    client.get("/api/positions", headers=AUTH)
    client.get("/api/positions", headers=AUTH)
    assert broker["broker"].get_open_positions.call_count == 1


# ---- market bar / agent health ----

def test_markets_endpoint_lists_nasdaq_and_nse(client):
    data = client.get("/api/markets", headers=AUTH).get_json()
    codes = [s["code"] for s in data["sessions"]]
    assert codes[:2] == ["NASDAQ", "NSE"]
    nasdaq = data["sessions"][0]
    assert nasdaq["agent"] is True
    assert nasdaq["state"] in ("open", "pre", "post", "closed", "holiday", "weekend")
    assert nasdaq["next_open"] > 0 and nasdaq["upcoming_holidays"]


def test_status_reports_phase_broker_health_and_unmanaged(client, tmp_path):
    import time as _t
    audit = AuditLog(str(dash.DB_PATH))
    audit.set_kv("agent_phase", {"phase": "sleeping", "exchange": "NASDAQ", "market": "US",
                                 "next_open": _t.time() + 3600, "ts": _t.time()})
    audit.set_kv("unmanaged", {"ts": _t.time(), "orders": [],
                               "positions": [{"symbol": "US.TSLA", "quantity": -25}]})
    data = client.get("/api/status", headers=AUTH).get_json()
    assert data["agent_status"] == "sleeping"
    assert data["unmanaged"]["positions"][0]["symbol"] == "US.TSLA"

    audit.set_kv("broker_health", {"ok": False, "detail": "not logged in", "ts": _t.time(), "since": _t.time()})
    data = client.get("/api/status", headers=AUTH).get_json()
    assert data["agent_status"] == "broker_down"
    assert data["agent_broker"]["detail"] == "not logged in"
