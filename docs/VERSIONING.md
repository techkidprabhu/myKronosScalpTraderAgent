# Versioning & change history

The GitHub repo is the history of this project. Every feature, bug fix
and release is a commit there, and releases are tags you can return to.

## Branches

| Branch | Use |
|---|---|
| `main` | Always the version you are running live. Only merge tested work here. |
| `feature/<name>` | A new feature, e.g. `feature/manual-override`. |
| `fix/<name>` | A bug fix, e.g. `fix/positions-stale-price`. |

## Day-to-day workflow

```bash
cd ~/mySwingTraders/moomoo-swing-agent
git checkout main && git pull                 # start from the latest
git checkout -b feature/my-new-thing          # or fix/what-broke

# ... edit, then run the tests ...
python -m pytest -q -p no:warnings --deselect tests/test_kronos_agent.py

git add -A
git commit -m "feat: short description of the change"
git push -u origin feature/my-new-thing
```

Then merge it into `main` with a pull request on GitHub (it shows the
full diff and keeps a record), or locally:

```bash
git checkout main && git merge --no-ff feature/my-new-thing && git push
```

## Commit messages

Start each message with a type so the history reads like a changelog:

| Prefix | For |
|---|---|
| `feat:` | a new feature |
| `fix:` | a bug fix |
| `refactor:` | code restructuring, no behaviour change |
| `test:` | tests only |
| `docs:` | README / docs / changelog |
| `chore:` | dependencies, config templates, tooling |

Example: `fix: anchor bracket stop to fill price, not signal close`

## Releasing a version

1. Move the notes under **Unreleased** in `CHANGELOG.md` to a new heading,
   e.g. `## [1.1.0] - 2026-11-20`.
2. Put the same number in `VERSION`.
3. Commit, tag and push:

```bash
git commit -am "chore: release 1.1.0"
git tag -a v1.1.0 -m "v1.1.0"
git push && git push --tags
```

On GitHub, **Releases -> Draft a new release** on that tag lets you paste
the changelog section as release notes.

## Going back

```bash
git log --oneline --graph          # see history
git checkout v1.0.0                # look at an old version (read-only)
git checkout main                  # back to the latest
git revert <commit>                # undo one commit safely on main
```

## What is never committed

`.gitignore` keeps these out of GitHub: `.env` (API keys, passwords, TOTP
secrets), `*.pem` keys, everything in `data/` (trade audit databases,
dashboard token, Kite access token), and your live `config/config.yaml`
(and its backups). `config/config.example.yaml` is the tracked template -
when you add a new setting, add it there too.

Check before committing if unsure: `git status` should never list any of
those files.
