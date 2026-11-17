"""Saved model versions on GitHub Releases, against a stand-in GitHub."""
import json
import threading
import time
import zipfile

import pytest
from flask import Flask, Response, redirect, request
from werkzeug.serving import make_server

from src.finetune import releases
from src.finetune.trainer import list_runs

TOKEN = "good-token"


class FakeGitHub:
    def __init__(self):
        self.releases, self.blobs, self.deleted, self.fail_upload = [], {}, [], False
        app = Flask("fakegh")
        self.app = app

        def auth():
            return request.headers.get("Authorization") == f"Bearer {TOKEN}"

        @app.get("/repos/<o>/<r>/releases")
        def list_(o, r):
            if not auth():
                return Response('{"message":"Bad credentials"}', 401)
            page = int(request.args.get("page", 1))
            return json.dumps(self.releases[::-1][(page - 1) * 100: page * 100])

        @app.post("/repos/<o>/<r>/releases")
        def create(o, r):
            d = request.get_json()
            assert d["make_latest"] == "false"
            rid = len(self.releases) + 100
            rel = {"id": rid, "tag_name": d["tag_name"], "name": d["name"], "body": d["body"], "assets": [],
                   "html_url": f"https://github.com/{o}/{r}/releases/tag/{d['tag_name']}",
                   "upload_url": f"{self.base}/uploads/{rid}/assets{{?name,label}}", "published_at": "2026-11-17T10:00:00Z"}
            self.releases.append(rel)
            return json.dumps(rel), 201

        @app.post("/uploads/<int:rid>/assets")
        def upload(rid):
            if self.fail_upload:
                return Response('{"message":"boom"}', 500)
            assert request.headers["Content-Type"] == "application/zip"
            data = request.get_data()
            assert len(data) == int(request.headers["Content-Length"])
            aid = 9000 + len(self.blobs)
            self.blobs[aid] = data
            asset = {"id": aid, "name": request.args["name"], "size": len(data)}
            next(x for x in self.releases if x["id"] == rid)["assets"].append(asset)
            return json.dumps(asset), 201

        @app.delete("/repos/<o>/<r>/releases/<int:rid>")
        def delete(o, r, rid):
            self.releases = [x for x in self.releases if x["id"] != rid]
            self.deleted.append(rid)
            return "", 204

        @app.delete("/repos/<o>/<r>/git/refs/tags/<tag>")
        def delete_tag(o, r, tag):
            return "", 204

        @app.get("/repos/<o>/<r>/releases/assets/<int:aid>")
        def asset(o, r, aid):
            assert request.headers["Accept"] == "application/octet-stream"
            return redirect(f"{self.base}/blob/{aid}")          # GitHub sends you to storage

        @app.get("/blob/<int:aid>")
        def blob(aid):
            return Response(self.blobs[aid], mimetype="application/octet-stream")

    def __enter__(self):
        self.srv = make_server("127.0.0.1", 0, self.app)
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()


@pytest.fixture
def gh(monkeypatch):
    with FakeGitHub() as fake:
        monkeypatch.setattr(releases, "API", fake.base)
        yield fake


def make_run(models, run_id="kronos-ft-20261116-2145", weights=b"w" * 5000):
    d = models / run_id
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(weights)
    (d / "config.json").write_text("{}")
    (d / "report.json").write_text(json.dumps({
        "run_id": run_id, "created": time.time(), "verdict": "better", "reason": "loss 2% lower", "steps": 4000,
        "symbols": 40, "device": "RTX 2000 Ada", "base_model": "NeoQuasar/Kronos-base",
        "base_tokenizer": "NeoQuasar/Kronos-Tokenizer-base", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "original": {"val_loss": 5.2, "direction_hit_pct": 52.0}, "fine_tuned": {"val_loss": 5.1, "direction_hit_pct": 55.5}}))
    return d


def test_repo_names_from_git_remotes():
    assert releases.parse_repo("https://github.com/techkidprabhu/myKronosSwingTraderAgent.git") == \
        "techkidprabhu/myKronosSwingTraderAgent"
    assert releases.parse_repo("git@github.com:me/agent") == "me/agent"
    assert releases.parse_repo("https://gitlab.com/me/agent") is None
    with pytest.raises(releases.ReleaseError):
        releases.GitHub("not a repo", TOKEN)


def test_save_list_and_restore_round_trip(gh, tmp_path):
    models = tmp_path / "models"
    run = make_run(models)
    api = releases.GitHub("me/agent", TOKEN)
    seen = []
    v1 = releases.save_version(api, run, "  first   GPU run ", seen.append)
    assert v1["version"] == 1 and v1["note"] == "first GPU run" and seen[-1] == v1["size"]
    rel = gh.releases[0]
    assert rel["tag_name"] == "model-v1" and "first GPU run" in rel["body"] and "| 5.2 | 5.1 |" in rel["body"]
    with zipfile.ZipFile(__import__("io").BytesIO(gh.blobs[9000])) as z:
        assert sorted(z.namelist()) == [f"{run.name}/config.json", f"{run.name}/model.safetensors", f"{run.name}/report.json"]
    with pytest.raises(releases.ReleaseError, match="already saved as v1"):
        releases.save_version(api, run, "")

    run2 = make_run(models, "kronos-ft-20261118-2145")
    assert releases.save_version(api, run2, "")["version"] == 2
    vs = releases.list_versions(api)
    assert [v["version"] for v in vs] == [2, 1] and vs[1]["run_id"] == run.name and vs[1]["asset_id"]
    assert vs[1]["fine_tuned"]["direction_hit_pct"] == 55.5

    (run / "model.safetensors").unlink()                       # cleaned up locally, as old runs are
    got = []
    rep = releases.restore_version(api, 1, models, got.append)
    assert rep["run_id"] == run.name and (run / "model.safetensors").read_bytes() == b"w" * 5000 and got
    assert next(r for r in list_runs(models) if r["run_id"] == run.name)["has_weights"]
    with pytest.raises(releases.ReleaseError, match="already here"):
        releases.restore_version(api, 1, models)
    with pytest.raises(releases.ReleaseError, match="no saved version v7"):
        releases.restore_version(api, 7, models)


def test_failures_are_explained_and_leave_nothing_half_done(gh, tmp_path):
    run = make_run(tmp_path / "models")
    with pytest.raises(releases.ReleaseError, match="GITHUB_TOKEN"):
        releases.list_versions(releases.GitHub("me/agent", "wrong"))
    gh.fail_upload = True
    with pytest.raises(releases.ReleaseError, match="upload"):
        releases.save_version(releases.GitHub("me/agent", TOKEN), run, "")
    assert gh.releases == [] and gh.deleted                      # the empty release was removed
    gh.fail_upload = False
    api = releases.GitHub("me/agent", TOKEN)
    releases.save_version(api, run, "")
    aid = next(iter(gh.blobs))
    gh.blobs[aid] = gh.blobs[aid][:-10] + b"0123456789"           # corrupted on the way back
    (run / "model.safetensors").unlink()
    with pytest.raises(releases.ReleaseError, match="checksum"):
        releases.restore_version(api, 1, tmp_path / "models")
    assert not (run / "model.safetensors").exists()


def test_dashboard_save_and_download(gh, tmp_path, monkeypatch):
    import dashboard.app as app_mod
    from src.finetune import trainer
    models = tmp_path / "models"
    make_run(models)
    monkeypatch.setattr(trainer, "MODELS_DIR", models)
    monkeypatch.setattr(releases, "MODELS_DIR", models)
    monkeypatch.setattr(releases, "STATUS_PATH", tmp_path / "job.json")
    monkeypatch.setattr(app_mod, "load_config", lambda: {"finetune": {"releases_repo": "me/agent"}})
    app_mod._versions_cache.update(ts=0.0, key=None, data=None)
    app_mod.app.config["TESTING"] = True
    c = app_mod.app.test_client()
    hdr = {"Authorization": f"Bearer {app_mod.DASHBOARD_TOKEN}"}

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    d = c.get("/api/finetune/versions", headers=hdr).get_json()
    assert d["configured"] is False and d["has_token"] is False
    assert c.post("/api/finetune/release", json={"run_id": "x"}, headers=hdr).status_code == 400

    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert c.post("/api/finetune/release", json={"run_id": "nope"}, headers=hdr).status_code == 404
    r = c.post("/api/finetune/release", json={"run_id": "kronos-ft-20261116-2145", "note": "keep"}, headers=hdr)
    assert r.status_code == 200, r.get_json()
    for _ in range(100):
        if releases.read_job(tmp_path / "job.json")["state"] != "running":
            break
        time.sleep(0.05)
    assert releases.read_job(tmp_path / "job.json")["state"] == "done"
    v = c.get("/api/finetune/versions?refresh=1", headers=hdr).get_json()
    assert v["configured"] and v["versions"][0]["version"] == 1 and v["versions"][0]["local"] is True
    assert "GITHUB_TOKEN" not in json.dumps(v) and TOKEN not in json.dumps(v)

    (models / "kronos-ft-20261116-2145" / "model.safetensors").unlink()
    assert c.post("/api/finetune/restore", json={"version": "x"}, headers=hdr).status_code == 400
    assert c.post("/api/finetune/restore", json={"version": 1}, headers=hdr).status_code == 200
    for _ in range(100):
        if releases.read_job(tmp_path / "job.json")["state"] != "running":
            break
        time.sleep(0.05)
    assert releases.read_job(tmp_path / "job.json")["state"] == "done"
    assert (models / "kronos-ft-20261116-2145" / "model.safetensors").exists()
