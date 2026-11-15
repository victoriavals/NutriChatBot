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
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SUMOPOD_API_KEY", raising=False)

    with pytest.raises(ConfigurationError):
        load_settings(env_file=empty_env, require_api_key=True)


def test_load_settings_optional_key_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    empty_env = tmp_path / ".env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_LLM", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    settings = load_settings(env_file=empty_env, require_api_key=False)
    assert settings.api_key is None
    assert settings.model_llm == "glm-5.3-flash"
    assert settings.base_url == "https://ai.sumopod.com/v1"
    assert settings.model_embedding == "text-embedding-3-small"


def test_load_settings_reads_env_file_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "API_KEY=sk-test-key-1234567890\n"
        "BASE_URL=https://ai.sumopod.com/v1\n"
        "MODEL_LLM=glm-5.3-flash\n"
        "MODEL_EMBEDDING=text-embedding-3-small\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_LLM", raising=False)
    monkeypatch.delenv("MODEL_EMBEDDING", raising=False)

    settings = load_settings(env_file=env_file, require_api_key=True)
    assert settings.api_key == "sk-test-key-1234567890"
    assert settings.model_llm == "glm-5.3-flash"
    assert settings.model_embedding == "text-embedding-3-small"
    assert settings.base_url == "https://ai.sumopod.com/v1"


def test_sensitive_data_filter_redacts_keys() -> None:
    log_filter = SensitiveDataFilter()
    record_google = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Failed using google key %s",
        args=("AIzaSyDummyKeyForTestingPurposes12345",),
        exc_info=None,
    )
    assert log_filter.filter(record_google) is True
    assert "AIzaSy" not in record_google.getMessage()
    assert "[REDACTED_API_KEY]" in record_google.getMessage()

    record_sumopod = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Failed using sumopod key %s",
        args=("sk-1234567890abcdefghijklmnopqrstuvwxyz",),
        exc_info=None,
    )
    assert log_filter.filter(record_sumopod) is True
    assert "sk-" not in record_sumopod.getMessage()
    assert "[REDACTED_API_KEY]" in record_sumopod.getMessage()
