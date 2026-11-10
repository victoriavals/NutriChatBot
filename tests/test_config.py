"""Unit tests for config module."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from config import ConfigurationError, SensitiveDataFilter, load_settings


def test_load_settings_missing_key_raises_when_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    empty_env = tmp_path / ".env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError):
        load_settings(env_file=empty_env, require_api_key=True)


def test_load_settings_optional_key_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    empty_env = tmp_path / ".env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    settings = load_settings(env_file=empty_env, require_api_key=False)
    assert settings.gemini_api_key is None
    assert settings.gemini_model == "gemini-2.0-flash"


def test_load_settings_reads_env_file_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key-123\nGEMINI_MODEL=gemini-2.5-flash\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    settings = load_settings(env_file=env_file, require_api_key=True)
    assert settings.gemini_api_key == "test-key-123"
    assert settings.gemini_model == "gemini-2.5-flash"


def test_sensitive_data_filter_redacts_google_api_key() -> None:
    log_filter = SensitiveDataFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Failed using key %s",
        args=("AIzaSyDummyKeyForTestingPurposes12345",),
        exc_info=None,
    )
    assert log_filter.filter(record) is True
    assert "AIzaSy" not in record.getMessage()
    assert "[REDACTED_API_KEY]" in record.getMessage()

