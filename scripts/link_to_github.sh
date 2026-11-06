#!/usr/bin/env bash
# One-time: turn this project folder into a working copy of the GitHub repo,
# keeping every local file (including the ignored .env, data/ and your live
# config/config.yaml, which never leave this machine).
#
#   cd ~/mySwingTraders/moomoo-swing-agent
#   bash scripts/link_to_github.sh https://github.com/<you>/moomoo-swing-agent.git
set -euo pipefail

REMOTE="${1:?usage: bash scripts/link_to_github.sh <repo-url>}"
cd "$(dirname "$0")/.."

command -v git >/dev/null || { echo "Install git first:  sudo apt install git"; exit 1; }

if [ -d .git ]; then
  echo "This folder is already a git repo:"; git remote -v; exit 0
fi

# GitHub sign-in for pushes from this machine (opens the browser once).
if command -v gh >/dev/null; then
  gh auth status >/dev/null 2>&1 || gh auth login --hostname github.com --git-protocol https --web
  gh auth setup-git
else
  echo "Tip: 'sudo apt install gh' then 'gh auth login' makes git push sign in through your browser."
fi

git init -q -b main
if ! git config user.email >/dev/null; then
  echo "Set your commit identity once (use your GitHub email or its noreply address):"
  echo '  git config --global user.name  "Your Name"'
  echo '  git config --global user.email "you@example.com"'
fi
git remote add origin "$REMOTE"
git fetch -q origin
# Adopt the GitHub history without touching files on disk.
git reset -q origin/main
git branch -q --set-upstream-to=origin/main main

# Files that exist only in the repo (CHANGELOG.md, docs/, ...) - bring them in.
git ls-files -d -z | xargs -0 -r git checkout -q --

echo
echo "Linked to $REMOTE"
echo "Local differences from GitHub (empty = identical):"
git status --short
echo
echo "Never-committed files check (should list nothing):"
git ls-files | grep -E '(^|/)\.env$|\.pem$|^data/|^config/config\.yaml$' || echo "  OK - no secrets tracked"
