import pytest

from agents.common.config import settings


@pytest.fixture(autouse=True)
def isolated_run_store(tmp_path, monkeypatch):
    """Every test gets its own run-state directory so tests never see each
    other's runs and never touch the real data/runs/ on disk."""
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path / "runs"))
    yield
