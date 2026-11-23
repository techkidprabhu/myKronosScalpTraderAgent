import pytest


@pytest.fixture(autouse=True)
def fresh_ig_guard(monkeypatch):
    """Each test gets its own IG request budget (high, unless the test sets
    one) and forgets what an earlier test's login learnt."""
    from src.brokers import ig
    monkeypatch.setattr(ig, "GUARD", ig.RequestGuard())
    monkeypatch.setattr(ig.IGClient, "_needs_encryption", False)
    monkeypatch.setenv("IG_REQUESTS_PER_MIN", "100000")
    monkeypatch.setenv("IG_REQUEST_MAX_WAIT", "0")
    yield


@pytest.fixture(autouse=True)
def focus_switch_off(monkeypatch, tmp_path):
    """Single-stock focus is off unless a test switches it on (never reads
    the real data/focus_mode.json)."""
    from src import focus
    monkeypatch.setattr(focus, "SWITCH_PATH", tmp_path / "focus_mode.json")
    yield
