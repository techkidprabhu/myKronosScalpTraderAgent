"""Which code is running: the release tag, the last merged pull request and
the commit, read from git.

The agent records the version it started with in each broker's audit DB
(kv "agent_version"); the dashboard shows that, its own, and what is on
disk now, so a `git pull` without a restart is visible.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT_VERSION_KEY = "agent_version"
_PR = re.compile(r"(?:Merge pull request #|\(#)(\d+)")


def _git(*args: str, root: Path = ROOT) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def read(root: Path = ROOT) -> dict:
    """{tag, pr, commit, short, date, branch, modified, label}. Fields are
    None when git isn't available (e.g. a copy without the .git folder)."""
    commit = _git("rev-parse", "HEAD", root=root)
    if not commit:
        return {"tag": None, "pr": None, "commit": None, "short": None, "date": None, "branch": None,
                "modified": False, "label": "unknown (no git)", "read_at": time.time()}
    describe = _git("describe", "--tags", "--long", "--always", root=root) or ""
    tag, ahead = None, 0
    m = re.match(r"^(.*)-(\d+)-g[0-9a-f]+$", describe)
    if m:
        tag, ahead = m.group(1), int(m.group(2))
    pr = None
    for subject in (_git("log", "-40", "--format=%s", root=root) or "").splitlines():
        hit = _PR.search(subject)
        if hit:
            pr = int(hit.group(1))
            break
    modified = bool(_git("status", "--porcelain", "--untracked-files=no", root=root))
    short = commit[:7]
    parts = [f"{tag}+{ahead}" if tag and ahead else tag, f"PR #{pr}" if pr else None, short]
    label = " · ".join(p for p in parts if p) + (" (local changes)" if modified else "")
    remote = _git("remote", "get-url", "origin", root=root) or ""
    rm = re.search(r"github\.com[:/]+([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", remote)
    return {"tag": tag, "ahead": ahead, "pr": pr, "commit": commit, "short": short,
            "repo": f"{rm.group(1)}/{rm.group(2)}" if rm else None,
            "date": _git("log", "-1", "--format=%cI", root=root), "branch": _git("rev-parse", "--abbrev-ref", "HEAD", root=root),
            "modified": modified, "label": label, "read_at": time.time()}


# the code this process loaded
RUNNING = read()
