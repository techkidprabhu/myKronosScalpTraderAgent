"""Dashboard with several brokers: per-broker config/DB/client, the broker
list, and the schema-driven settings writer (comments kept, overrides)."""
import shutil
from pathlib import Path

import pytest
import yaml

import dashboard.app as dash
from src.audit_log import AuditLog

ROOT = Path(__file__).resolve().parent.parent
AUTH = {"Authorization": f"Bearer {dash.DASHBOARD_TOKEN}"}

CONFIG_TEXT = """\
# Shared settings - every broker uses these unless it overrides them.
timeframe: "MIN_5"            # 5-minute candles
lookback_bars: 200
kronos_confidence_threshold: 0.6
forecast_horizon: 5
kronos:
  model_name: "NeoQuasar/Kronos-base"   # offline model
  sample_count: 5
indicators:
  enabled: true
  ma_fast: 20
  ma_slow: 50
  min_bars: 60
screener:
  enabled: true
  candidate_symbols: ["US.AAPL", "US.MSFT"]
  min_price: 5.0
  max_price: 1500.0
  min_atr_pct: 0.15
  max_atr_pct: 1.5
risk:
  max_open_positions: 5   # across all symbols
  max_daily_loss: 600.0
  lot_size: 1
brokers:
  moomoo:
    enabled: true
  tiger:
    enabled: true
    market: SG
    timeframe: HOUR_1     # Tiger trades hourly candles
    risk:
      lot_size: 100
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text(CONFIG_TEXT)
    monkeypatch.setattr(dash, "CONFIG_PATH", cfg_path)
    db = tmp_path / "audit.db"
    AuditLog(str(db)).close()
    monkeypatch.setattr(dash, "DB_PATH", db)
    # every non-default broker DB goes to tmp
    import src.broker_config as bc
    monkeypatch.setattr(bc, "_default_db", lambda name: (bc.ROOT / "data" / "audit.db") if name == "moomoo"
                        else tmp_path / f"audit_{name}.db")
    monkeypatch.setattr(dash, "get_kronos_agent", lambda: None)
    monkeypatch.setattr(dash, "_clients", {})
    monkeypatch.setattr(dash, "_broker_cache", {})
    monkeypatch.setattr(dash, "_funds_cache", {})
    dash.app.config["TESTING"] = True
    with dash.app.test_client() as c:
        c.tmp, c.cfg_path = tmp_path, cfg_path
        yield c


class FakeClient:
    def __init__(self, adapter, market):
        self.adapter, self.market, self.account, self.sandbox = adapter, market, f"{adapter}-acct-123", True

    def ping(self):
        return True, "ok"

    def get_open_positions(self):
        return [{"symbol": f"{self.market}.X", "quantity": 1, "average_cost": 1.0, "market_price": 1.0,
                 "unrealized_pnl": 0.0, "currency": "SGD", "name": "X"}]

    def get_open_orders(self):
        return []

    def get_account_summary(self):
        return {"currency": "SGD" if self.market == "SG" else "USD", "total_assets": 1.0}

    def close(self):
        pass


@pytest.fixture
def fake_connect(monkeypatch):
    calls = []

    def connect(adapter, timeout, **kw):
        calls.append((adapter, kw.get("market")))
        return FakeClient(adapter, kw.get("market") or "US")
    monkeypatch.setattr(dash.broker_registry, "connect_with_timeout", connect)
    return calls


def test_broker_list(client):
    data = client.get("/api/brokers", headers=AUTH).get_json()
    assert [b["name"] for b in data["brokers"]] == ["moomoo", "tiger"]
    assert data["multi"] is True and data["current"] == "moomoo"
    tiger = data["brokers"][1]
    assert tiger["market"] == "SG" and tiger["overrides"] == ["risk", "timeframe"]
    assert tiger["capabilities"]["status"] == "ready"


def test_each_broker_gets_its_own_settings_db_and_client(client, fake_connect):
    m = client.get("/api/status", headers=AUTH).get_json()
    t = client.get("/api/status?broker=tiger", headers=AUTH).get_json()
    assert m["timeframe"] == "MIN_5" and t["timeframe"] == "HOUR_1"
    assert m["broker"]["name"] == "moomoo" and t["broker"]["label"] == "Tiger"
    assert t["broker"]["market_code"] == "SG" and t["broker"]["opend"] is None
    assert ("tiger", "SG") in fake_connect and ("moomoo", None) in fake_connect

    # decisions land in each broker's own DB
    AuditLog(str(client.tmp / "audit_tiger.db")).log_decision(
        "SG.D05", "hammer", "BUY", 0.6, True, 0.8, True, "ok", "opened")
    assert client.get("/api/decisions?broker=tiger", headers=AUTH).get_json()["decisions"]
    assert client.get("/api/decisions", headers=AUTH).get_json()["decisions"] == []

    pos = client.get("/api/positions?broker=tiger", headers=AUTH).get_json()
    assert {p["symbol"] for p in pos["positions"]} == {"SG.X"}


def test_unknown_broker_is_a_400(client):
    assert client.get("/api/status?broker=nope", headers=AUTH).status_code == 400


def test_settings_get_reports_values_and_overrides(client):
    data = client.get("/api/settings", headers=AUTH).get_json()
    assert {s["id"] for s in data["schema"]} >= {"broker", "schedule", "kronos", "indicators", "risk"}
    assert data["values"]["shared"]["timeframe"] == "MIN_5"
    assert data["values"]["tiger"]["timeframe"] == "HOUR_1"
    assert data["values"]["tiger"]["risk.lot_size"] == 100 and data["values"]["moomoo"]["risk.lot_size"] == 1
    assert data["overrides"]["tiger"] == ["risk.lot_size", "timeframe"]
    assert data["brokers"]["tiger"]["markets"] == ["SG", "US", "HK"]


def _post(client, body):
    return client.post("/api/settings", json=body, headers=AUTH)


def test_shared_change_keeps_comments_and_makes_backup(client):
    r = _post(client, {"scope": "shared", "changes": {"risk.max_open_positions": 4, "indicators.ma_fast": 9}})
    assert r.status_code == 200, r.get_json()
    text = client.cfg_path.read_text()
    assert "max_open_positions: 4   # across all symbols" in text
    assert "# Shared settings - every broker uses these" in text and "# Tiger trades hourly candles" in text
    cfg = yaml.safe_load(text)
    assert cfg["indicators"]["ma_fast"] == 9 and cfg["brokers"]["tiger"]["risk"]["lot_size"] == 100
    assert list((client.cfg_path.parent / "backups").glob("config.*.yaml"))
    assert client.cfg_path.with_suffix(".yaml.bak").read_text() == CONFIG_TEXT


def test_broker_override_and_reset(client):
    r = _post(client, {"scope": "tiger", "changes": {"risk.max_daily_loss": 250, "enabled": False,
                                                       "screener.candidate_symbols": "sg.d05, SG.O39"}})
    assert r.status_code == 200, r.get_json()
    cfg = yaml.safe_load(client.cfg_path.read_text())
    tiger = cfg["brokers"]["tiger"]
    assert tiger["risk"] == {"lot_size": 100, "max_daily_loss": 250} and tiger["enabled"] is False
    assert tiger["screener"]["candidate_symbols"] == ["SG.D05", "SG.O39"]
    assert cfg["risk"]["max_daily_loss"] == 600.0          # shared untouched

    r = _post(client, {"scope": "tiger", "reset": ["risk.max_daily_loss", "risk.lot_size", "timeframe"]})
    assert r.status_code == 200
    tiger = yaml.safe_load(client.cfg_path.read_text())["brokers"]["tiger"]
    assert "risk" not in tiger and "timeframe" not in tiger and tiger["market"] == "SG"


def test_first_broker_override_creates_brokers_section(client):
    client.cfg_path.write_text("timeframe: MIN_5  # shared\nrisk:\n  lot_size: 1\n")
    r = _post(client, {"scope": "moomoo", "changes": {"risk.lot_size": 10}})
    assert r.status_code == 200, r.get_json()
    cfg = yaml.safe_load(client.cfg_path.read_text())
    assert cfg["brokers"]["moomoo"] == {"enabled": True, "risk": {"lot_size": 10}}
    assert "# shared" in client.cfg_path.read_text()


@pytest.mark.parametrize("body, msg", [
    ({"scope": "tiger", "changes": {"kronos.sample_count": 3}}, "shared by all brokers"),
    ({"scope": "shared", "changes": {"enabled": False}}, "set it per broker"),
    ({"scope": "shared", "changes": {"risk.max_open_positions": 0}}, "at least 1"),
    ({"scope": "shared", "changes": {"timeframe": "WEEK"}}, "one of"),
    ({"scope": "shared", "changes": {"kronos.model_name": "x"}}, "not editable"),
    ({"scope": "shared", "changes": {"indicators.ma_fast": 80}}, "Fast MA period must be smaller"),
    ({"scope": "moomoo", "changes": {"enabled": False}}, None),   # still valid: tiger stays enabled
    ({"scope": "nobody", "changes": {"timeframe": "DAY"}}, "Unknown broker"),
])
def test_settings_validation(client, body, msg):
    r = _post(client, body)
    if msg is None:
        assert r.status_code == 200
    else:
        assert r.status_code == 400 and msg in r.get_json()["error"]
        assert client.cfg_path.read_text() == CONFIG_TEXT     # nothing written


def test_cannot_disable_every_broker(client):
    assert _post(client, {"scope": "tiger", "changes": {"enabled": False}}).status_code == 200
    r = _post(client, {"scope": "moomoo", "changes": {"enabled": False}})
    assert r.status_code == 400 and "no broker is enabled" in r.get_json()["error"]


def test_nullable_rsi_limit_can_be_turned_off(client):
    r = _post(client, {"scope": "shared", "changes": {"indicators.max_rsi_for_buy": ""}})
    assert r.status_code == 200
    assert yaml.safe_load(client.cfg_path.read_text())["indicators"]["max_rsi_for_buy"] is None


def test_settings_daily_limits_and_selflearning_sections(client):
    body = client.get("/api/settings", headers=AUTH).get_json()
    secs = {s["id"]: [f["key"] for f in s["fields"]] for s in body["schema"]}
    assert {"risk.daily_profit_target_pct", "risk.daily_profit_target", "risk.max_daily_loss",
            "risk.max_symbol_loss_per_day", "risk.max_losses_per_symbol_per_day"} <= set(secs["daily"])
    assert "risk.daily_profit_target_pct" not in secs["daytrader"]
    assert {"finetune.enabled", "finetune.batch_size", "finetune.device", "finetune.releases_repo"} <= set(secs["selflearning"])
    assert "skills.file" in secs["skills"]
    # Not in this test config.yaml: the code's default is shown, not a blank
    assert body["values"]["shared"]["finetune.batch_size"] == 8
    assert body["values"]["shared"]["finetune.device"] == "auto"
    assert body["values"]["shared"]["skills.file"] == "config/skills.md"


def test_settings_rules_file_must_be_markdown(client):
    r = _post(client, {"scope": "shared", "changes": {"skills.file": "config/rules.txt"}})
    assert r.status_code == 400
    r = _post(client, {"scope": "shared", "changes": {"finetune.device": "cuda", "skills.file": "config/my-rules.md"}})
    assert r.status_code == 200, r.get_json()


def test_trading_mode_switch(client, monkeypatch):
    from src import trading_mode
    monkeypatch.setattr(trading_mode, "ROOT", client.tmp)
    monkeypatch.setattr(trading_mode, "SWITCH_PATH", client.tmp / "data" / "trading_mode.json")
    trading_mode._cache.update(mtime=None, value=None)
    (client.tmp / "config" / "Praga.md").write_text("# m\n```yaml\nstrategy: praga_price_action\n```\n")
    (client.tmp / "config" / "notes.md").write_text("# just notes\n")

    body = client.get("/api/trading_mode", headers=AUTH).get_json()
    assert body["mode"] == "kronos"
    assert [f["file"] for f in body["files"]] == ["config/Praga.md"]

    r = client.post("/api/trading_mode", json={"mode": "skill", "file": "config/notes.md"}, headers=AUTH)
    assert r.status_code == 400 and trading_mode.read()["mode"] == "kronos"
    r = client.post("/api/trading_mode", json={"mode": "skill", "file": "../x.md"}, headers=AUTH)
    assert r.status_code == 400

    r = client.post("/api/trading_mode", json={"mode": "skill", "file": "config/Praga.md"}, headers=AUTH)
    assert r.status_code == 200 and r.get_json()["mode"] == "skill"
    assert trading_mode.read() ["file"] == "config/Praga.md"
    assert r.get_json()["settings"]["swing_candles"] == 2

    r = client.post("/api/trading_mode", json={"mode": "kronos"}, headers=AUTH)
    assert r.get_json()["mode"] == "kronos"
    trading_mode._cache.update(mtime=None, value=None)
