"""Centralized configuration, constants, and logging setup for NutriChatBot."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR: Path = Path(__file__).resolve().parent
DEFAULT_DB_PATH: Path = BASE_DIR / "nutrition.db"
DEFAULT_CSV_PATH: Path = BASE_DIR / "data" / "nutrition.csv"
DEFAULT_CHROMA_PATH: Path = BASE_DIR / "chroma"

CHROMA_COLLECTION_NAME: str = "nutrition"
DEFAULT_GEMINI_MODEL: str = "gemini-2.0-flash"

EMBEDDING_BATCH_SIZE: int = 100
DEFAULT_RAG_RESULTS: int = 3
MAX_USER_INPUT_LENGTH: int = 1000
MIN_DAILY_CALORIES: int = 1000
MAX_DAILY_CALORIES: int = 5000
DEFAULT_DAILY_CALORIES: int = 2000

RAG_TEMPERATURE: float = 0.2
RAG_MAX_TOKENS: int = 512

RECIPE_TEMPERATURE: float = 0.3
RECIPE_MAX_TOKENS: int = 1024

MENU_TEMPERATURE: float = 0.2
MENU_MAX_TOKENS: int = 2048

ALTERNATIVE_TEMPERATURE: float = 0.5
ALTERNATIVE_MAX_TOKENS: int = 512

_API_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing or invalid."""


class SensitiveDataFilter(logging.Filter):
    """Redacts Google API key patterns from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize message and arguments before emission."""
        if isinstance(record.msg, str):
            record.msg = _API_KEY_PATTERN.sub("[REDACTED_API_KEY]", record.msg)
        if record.args:
            sanitized_args = tuple(
                _API_KEY_PATTERN.sub("[REDACTED_API_KEY]", arg) if isinstance(arg, str) else arg
                for arg in (record.args if isinstance(record.args, tuple) else (record.args,))
            )
            record.args = sanitized_args
        return True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger configured with consistent formatting and secret redaction."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)
    return logger


@dataclass(frozen=True)
class AppSettings:
    """Immutable application settings loaded from environment variables."""

    gemini_api_key: str | None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    db_path: Path = DEFAULT_DB_PATH
    csv_path: Path = DEFAULT_CSV_PATH
    chroma_path: Path = DEFAULT_CHROMA_PATH


def load_settings(
    env_file: str | Path | None = None,
    require_api_key: bool = False,
) -> AppSettings:
    """Load environment variables once and return structured application settings.

    Args:
        env_file: Optional path to a `.env` file. Defaults to automatic discovery.
        require_api_key: When True, raises ConfigurationError if GEMINI_API_KEY is missing.

    Returns:
        Populated AppSettings instance.

    Raises:
        ConfigurationError: If require_api_key is True and GEMINI_API_KEY is missing or blank.
    """
    if env_file is not None:
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    raw_key = os.getenv("GEMINI_API_KEY")
    api_key = raw_key.strip() if raw_key and raw_key.strip() else None

    if require_api_key and not api_key:
        raise ConfigurationError(
            "GEMINI_API_KEY tidak ditemukan. Silakan set di file .env atau environment variables."
        )

    raw_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    model_name = raw_model or DEFAULT_GEMINI_MODEL

    return AppSettings(
        gemini_api_key=api_key,
        gemini_model=model_name,
    )

