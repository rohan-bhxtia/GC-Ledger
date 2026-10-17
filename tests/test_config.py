import os

import pytest
from pydantic import ValidationError

from gcl.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isolate tests from the developer's real environment."""
    for key in list(os.environ):
        if key.startswith("GCL_"):
            monkeypatch.delenv(key)


def make(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)  # never read the real .env in tests


def test_defaults():
    s = make()
    assert s.default_timezone == "Asia/Kolkata"
    assert s.max_import_bytes == 10 * 1024 * 1024
    assert s.gemini_api_key is None


def test_reads_env_vars(monkeypatch):
    monkeypatch.setenv("GCL_MAX_IMPORT_MB", "5")
    monkeypatch.setenv("GCL_LOG_LEVEL", "debug")
    s = make()
    assert s.max_import_mb == 5
    assert s.log_level == "DEBUG"


def test_invalid_timezone_rejected():
    with pytest.raises(ValidationError):
        make(default_timezone="Mars/Olympus")


def test_import_limit_bounds():
    with pytest.raises(ValidationError):
        make(max_import_mb=0)


def test_missing_api_key_fails_clearly():
    with pytest.raises(RuntimeError, match="GCL_GEMINI_API_KEY"):
        make().require_api_key()


def test_api_key_not_leaked_in_repr(monkeypatch):
    monkeypatch.setenv("GCL_GEMINI_API_KEY", "super-secret-value")
    s = make()
    assert s.require_api_key() == "super-secret-value"
    assert "super-secret-value" not in repr(s)