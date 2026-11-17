"""Saved model versions, kept as GitHub Releases on your own repository.

Each version is one release tagged ``model-v<N>`` whose asset is the run's
zip (model.safetensors, config.json, report.json, a tuned tokenizer if any),
with the scores and your note in the release text. The release list is the
version history; any version can be downloaded back into models/ later and
then shadow-tested or used like a run trained here.

Needs a GitHub token in .env:

    GITHUB_TOKEN=...   fine-grained token, only this repository,
                       permission "Contents: Read and write"

The repository is finetune.releases_repo ("owner/name"), or the one this
folder was cloned from. Saving or downloading never changes which model the
agent trades with.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path

from . import MODELS_DIR, ROOT, settings

logger = logging.getLogger(__name__)

API = "https://api.github.com"
TAG_PREFIX = "model-v"
META = re.compile(r"<!--\s*kronos-model\s+(\{.*?\})\s*-->", re.S)
STATUS_PATH = ROOT / "data" / "model_versions_job.json"
RUN_FILES = ("report.json", "config.json", "model.safetensors", "tokenizer/config.json", "tokenizer/model.safetensors")


class ReleaseError(RuntimeError):
    """Something the person can act on (missing token, no such version...)."""


# ---------------------------------------------------------------- where

def repo_from_git(root: Path = ROOT) -> str | None:
    try:
        url = subprocess.run(["git", "-C", str(root), "remote", "get-url", "origin"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_repo(url)


def parse_repo(url: str) -> str | None:
    """owner/name from a GitHub remote (https or ssh), else None."""
    m = re.search(r"github\.com[:/]+([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", url or "")
    return f"{m.group(1)}/{m.group(2)}" if m else None


def target(config: dict) -> tuple[str | None, str | None]:
    """(repository, token); either may be None when not set up."""
    repo = settings(config).get("releases_repo") or repo_from_git()
    token = os.environ.get("GITHUB_TOKEN", "").strip() or None
    return repo, token


class GitHub:
    def __init__(self, repo: str, token: str, api: str | None = None, timeout: float = 30):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo or ""):
            raise ReleaseError(f"finetune.releases_repo should look like owner/name, not {repo!r}")
        import requests
        self.repo, self.api, self.timeout = repo, (api or API).rstrip("/"), timeout
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "kronos-swing-agent"})

    def _check(self, r, what: str):
        if r.status_code in (401, 403):
            raise ReleaseError(f"GitHub refused to {what} (HTTP {r.status_code}). Check GITHUB_TOKEN: it needs "
                               f"access to {self.repo} with Contents: Read and write.")
        if r.status_code == 404:
            raise ReleaseError(f"GitHub could not find {self.repo} (or the token can't see it).")
        if r.status_code >= 400:
            msg = ""
            try:
                msg = r.json().get("message", "")
            except ValueError:
                pass
            raise ReleaseError(f"GitHub could not {what}: HTTP {r.status_code} {msg}".strip())
        return r

    def releases(self) -> list[dict]:
        out, page = [], 1
        while page <= 10:
            r = self._check(self.s.get(f"{self.api}/repos/{self.repo}/releases", params={"per_page": 100, "page": page},
                                       timeout=self.timeout), "list releases")
            batch = r.json()
            out += batch
            if len(batch) < 100:
                break
            page += 1
        return out

    def create_release(self, payload: dict) -> dict:
        return self._check(self.s.post(f"{self.api}/repos/{self.repo}/releases", json=payload, timeout=self.timeout),
                           "create the release").json()

    def delete_release(self, release_id: int, tag: str) -> None:
        self.s.delete(f"{self.api}/repos/{self.repo}/releases/{release_id}", timeout=self.timeout)
        self.s.delete(f"{self.api}/repos/{self.repo}/git/refs/tags/{tag}", timeout=self.timeout)

    def upload(self, release: dict, name: str, fh, size: int) -> dict:
        url = release["upload_url"].split("{", 1)[0]
        r = self.s.post(url, params={"name": name}, data=fh, timeout=max(self.timeout, 600),
                        headers={"Content-Type": "application/zip", "Content-Length": str(size)})
        return self._check(r, "upload the model").json()

    def download(self, asset_id: int, dest, progress=None) -> None:
        r = self._check(self.s.get(f"{self.api}/repos/{self.repo}/releases/assets/{asset_id}", stream=True,
                                   headers={"Accept": "application/octet-stream"}, timeout=max(self.timeout, 600)),
                        "download the model")
        done = 0
        for chunk in r.iter_content(1 << 20):
            dest.write(chunk)
            done += len(chunk)
            if progress:
                progress(done)


# ---------------------------------------------------------------- versions

def _meta(release: dict) -> dict | None:
    if not str(release.get("tag_name", "")).startswith(TAG_PREFIX):
        return None
    m = META.search(release.get("body") or "")
    try:
        meta = json.loads(m.group(1)) if m else {}
    except ValueError:
        meta = {}
    try:
        version = int(release["tag_name"][len(TAG_PREFIX):])
    except ValueError:
        return None
    asset = next((a for a in release.get("assets") or [] if a.get("name", "").endswith(".zip")), None)
    return {**meta, "version": version, "tag": release["tag_name"], "release_id": release.get("id"),
            "url": release.get("html_url"), "published": release.get("published_at") or release.get("created_at"),
            "asset_id": asset.get("id") if asset else None, "size": asset.get("size") if asset else meta.get("size")}


def list_versions(gh: GitHub) -> list[dict]:
    return sorted((m for m in map(_meta, gh.releases()) if m), key=lambda v: -v["version"])


def _fmt(v, suffix=""):
    return "–" if v is None else f"{v}{suffix}"


def release_text(report: dict, note: str, meta: dict) -> str:
    o, t = report.get("original") or {}, report.get("fine_tuned") or {}
    lines = [
        f"Fine-tuned Kronos model **{report['run_id']}**, saved from the swing agent's dashboard.",
        "",
        *([f"> {note}", ""] if note else []),
        "| | Original | Fine-tuned |",
        "|---|---|---|",
        f"| Held-out loss (lower is better) | {_fmt(o.get('val_loss'))} | {_fmt(t.get('val_loss'))} |",
        f"| Direction right | {_fmt(o.get('direction_hit_pct'), '%')} | {_fmt(t.get('direction_hit_pct'), '%')} |",
        "",
        f"Result: **{str(report.get('verdict', '')).replace('_', ' ')}** - {report.get('reason', '')}",
        f"Started from {report.get('base_model')}; {report.get('symbols')} symbols, {report.get('steps')} steps"
        f"{' on ' + report['device'] if report.get('device') else ''}.",
        "",
        "Restore it on the dashboard (Kronos Model Lab -> Self-learning -> Saved versions -> Download) or with "
        f"`./start.sh finetune restore {meta['version']}`.",
        "",
        f"<!-- kronos-model {json.dumps(meta, separators=(',', ':'))} -->",
    ]
    return "\n".join(lines)


def _zip_run(run_dir: Path, out: Path) -> tuple[int, str]:
    run_id = run_dir.name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:       # weights don't compress
        for rel in RUN_FILES:
            if (run_dir / rel).exists():
                z.write(run_dir / rel, f"{run_id}/{rel}")
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return out.stat().st_size, h.hexdigest()


class _Reader:
    """A file object that reports how much has been read (upload progress)."""

    def __init__(self, fh, progress, size):
        self.fh, self.progress, self.done, self.size = fh, progress, 0, size

    def __len__(self):                 # lets requests send a Content-Length, not chunks
        return self.size

    def read(self, n=-1):
        b = self.fh.read(n if n and n > 0 else 1 << 20)
        self.done += len(b)
        if self.progress:
            self.progress(self.done)
        return b


def save_version(gh: GitHub, run_dir: Path, note: str = "", progress=None) -> dict:
    """Upload one training run as the next model-v<N> release."""
    report = json.loads((run_dir / "report.json").read_text())
    if not (run_dir / "model.safetensors").exists():
        raise ReleaseError(f"{run_dir.name} has no saved model to keep")
    versions = list_versions(gh)
    dup = next((v for v in versions if v.get("run_id") == run_dir.name), None)
    if dup:
        raise ReleaseError(f"{run_dir.name} is already saved as v{dup['version']}")
    version = max((v["version"] for v in versions), default=0) + 1
    note = " ".join(str(note or "").split())[:300]
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / f"{run_dir.name}.zip"
        size, sha = _zip_run(run_dir, zpath)
        o, t = report.get("original") or {}, report.get("fine_tuned") or {}
        meta = {"run_id": run_dir.name, "version": version, "note": note, "saved_at": time.time(),
                "created": report.get("created"), "verdict": report.get("verdict"), "reason": report.get("reason"),
                "base_model": report.get("base_model"), "base_tokenizer": report.get("base_tokenizer"),
                "tokenizer_tuned": bool(report.get("tokenizer_tuned")), "device": report.get("device"),
                "symbols": report.get("symbols"), "steps": report.get("steps"),
                "original": {k: o.get(k) for k in ("val_loss", "direction_hit_pct")},
                "fine_tuned": {k: t.get(k) for k in ("val_loss", "direction_hit_pct")},
                "sha256": sha, "size": size}
        rel = gh.create_release({"tag_name": f"{TAG_PREFIX}{version}", "name": f"Kronos model v{version}",
                                 "body": release_text(report, note, meta), "make_latest": "false"})
        try:
            with open(zpath, "rb") as f:
                gh.upload(rel, zpath.name, _Reader(f, progress, size), size)
        except BaseException:
            gh.delete_release(rel["id"], f"{TAG_PREFIX}{version}")      # no empty versions left behind
            raise
    logger.info("Saved %s as Kronos model v%s (%s)", run_dir.name, version, rel.get("html_url"))
    return {**meta, "url": rel.get("html_url"), "tag": f"{TAG_PREFIX}{version}"}


def restore_version(gh: GitHub, version: int, models_dir: Path | None = None, progress=None) -> dict:
    """Download model-v<version> into models/ (checked like any import)."""
    from .kit import import_run
    v = next((x for x in list_versions(gh) if x["version"] == int(version)), None)
    if v is None:
        raise ReleaseError(f"There is no saved version v{version}")
    if not v.get("asset_id"):
        raise ReleaseError(f"v{version} has no model file attached")
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        gh.download(v["asset_id"], tmp, progress)
        tmp.flush()
        if v.get("sha256"):
            h = hashlib.sha256()
            with open(tmp.name, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != v["sha256"]:
                raise ReleaseError(f"v{version} did not download completely (checksum differs) - try again")
        try:
            rep = import_run(tmp.name, models_dir or MODELS_DIR, replace_without_weights=True)
        except ValueError as exc:
            raise ReleaseError(str(exc)) from exc
    logger.info("Restored Kronos model v%s (%s)", version, rep["run_id"])
    return {**rep, "version": int(version)}


# ---------------------------------------------------------------- background jobs (dashboard)

_job_lock = threading.Lock()


def read_job(path: Path | None = None) -> dict:
    try:
        return json.loads((path or STATUS_PATH).read_text())
    except (OSError, ValueError):
        return {"state": "idle"}


def _write_job(path: Path | None = None, **fields) -> None:
    path = path or STATUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**fields, "updated": time.time()}))
    tmp.replace(path)


def start_job(action: str, gh: GitHub, *, run_dir: Path | None = None, note: str = "", version: int | None = None,
              models_dir: Path | None = None, status_path: Path | None = None, on_done=None) -> threading.Thread:
    """Run save / restore on a thread (uploads take minutes). One at a time."""
    if not _job_lock.acquire(blocking=False):
        raise ReleaseError("Another save or download is still running")
    total = {"n": 0}
    label = run_dir.name if run_dir else f"v{version}"

    def progress(done):
        _write_job(status_path, state="running", action=action, subject=label, done_bytes=done,
                   total_bytes=total["n"] or None)

    def work():
        try:
            if action == "save":
                total["n"] = sum((run_dir / r).stat().st_size for r in RUN_FILES if (run_dir / r).exists())
                progress(0)
                res = save_version(gh, run_dir, note, progress)
                msg = f"Saved {label} as v{res['version']}"
            else:
                v = next((x for x in list_versions(gh) if x["version"] == int(version)), {})
                total["n"] = int(v.get("size") or 0)
                progress(0)
                res = restore_version(gh, version, models_dir, progress)
                msg = f"Downloaded v{version} ({res['run_id']}). It is listed under Training runs."
            _write_job(status_path, state="done", action=action, subject=label, message=msg)
            if on_done:
                on_done()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model %s failed", action)
            _write_job(status_path, state="failed", action=action, subject=label, message=str(exc)[:300])
        finally:
            _job_lock.release()

    _write_job(status_path, state="running", action=action, subject=label, done_bytes=0)
    th = threading.Thread(target=work, name=f"model-{action}", daemon=True)
    th.start()
    return th
