"""Which code is running: src/version.py, the agent's record and /api/version."""
import json
import subprocess
from unittest.mock import MagicMock

import pytest

from src import version
from tests.test_dashboard_multibroker import AUTH, client  # noqa: F401  (temp config + DBs)


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def test_reads_tag_pull_request_and_commit(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("1")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-qm", "first")
    git(tmp_path, "tag", "v1.0.0")
    for msg in ("Merge pull request #7 from me/feature/x", "Fix typo"):
        (tmp_path / "a.txt").write_text(msg)
        git(tmp_path, "commit", "-qam", msg)
    git(tmp_path, "remote", "add", "origin", "https://github.com/me/agent.git")
    v = version.read(tmp_path)
    assert v["tag"] == "v1.0.0" and v["ahead"] == 2 and v["pr"] == 7 and len(v["short"]) == 7
    assert v["repo"] == "me/agent" and not v["modified"]
    assert v["label"] == f"v1.0.0+2 · PR #7 · {v['short']}"
    (tmp_path / "a.txt").write_text("edited")
    assert version.read(tmp_path)["modified"] and "(local changes)" in version.read(tmp_path)["label"]
    assert version.read(tmp_path / "nowhere")["label"] == "unknown (no git)"


def test_agent_records_its_version_when_it_starts(tmp_path):
    import yaml
    from pathlib import Path
    from src.audit_log import AuditLog
    from src.orchestrator import Orchestrator
    from tests.test_orchestrator import FakeBroker

    class Broker(FakeBroker):
        def get_open_positions(self):
            return []

        def get_open_orders(self):
            return []

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "config.example.yaml").read_text())
    audit = AuditLog(str(tmp_path / "audit.db"))
    try:
        Orchestrator(cfg, broker=Broker(), audit_log=audit, kronos_agent=MagicMock(), reporter=MagicMock())
    except Exception:  # noqa: BLE001 - only the record made at start-up matters here
        pass
    rec = audit.get_kv(version.AGENT_VERSION_KEY)
    assert rec["commit"] == version.RUNNING["commit"] and rec["started"] and rec["pid"]


def test_dashboard_shows_versions_and_asks_for_a_restart(client, tmp_path, monkeypatch):  # noqa: F811
    import dashboard.app as app_mod
    monkeypatch.setattr(app_mod.code_version, "RUNNING", {"commit": "a" * 40, "short": "aaaaaaa", "pr": 31, "label": "PR #31 · aaaaaaa"})
    disk = {"commit": "b" * 40, "short": "bbbbbbb", "pr": 32, "label": "PR #32 · bbbbbbb", "modified": False, "repo": "me/agent"}
    monkeypatch.setattr(app_mod.code_version, "read", lambda: disk)
    app_mod._disk_version.update(ts=0.0)
    c = client
    assert c.get("/api/version").status_code == 401
    db = app_mod._db_path()
    assert str(db).startswith(str(tmp_path))                 # never the real data/audit.db
    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                 ("agent_version", json.dumps({"commit": "b" * 40, "short": "bbbbbbb", "pr": 32, "label": "PR #32 · bbbbbbb", "started": 1.0})))
    conn.commit()
    conn.close()
    d = c.get("/api/version", headers=AUTH).get_json()
    assert d["dashboard"]["pr"] == 31 and d["agent"]["pr"] == 32 and d["on_disk"]["pr"] == 32
    assert d["restart_needed"] and "Restart the dashboard to run it" in d["notes"][0]
    monkeypatch.setattr(app_mod.code_version, "RUNNING", {**disk})
    app_mod._disk_version.update(ts=0.0)
    d = c.get("/api/version", headers=AUTH).get_json()
    assert not d["restart_needed"] and d["notes"] == []
