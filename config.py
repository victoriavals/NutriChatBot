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

# AI Provider Configuration (SumoPod AI / OpenAI compatible)
DEFAULT_BASE_URL: str = "https://ai.sumopod.com/v1"
DEFAULT_LLM_MODEL: str = "glm-5.3-flash"
DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"

# Legacy alias for backward compatibility
DEFAULT_GEMINI_MODEL: str = DEFAULT_LLM_MODEL

EMBEDDING_BATCH_SIZE: int = 100
DEFAULT_RAG_RESULTS: int = 3
MAX_USER_INPUT_LENGTH: int = 1000
MIN_DAILY_CALORIES: int = 1000
MAX_DAILY_CALORIES: int = 5000
DEFAULT_DAILY_CALORIES: int = 2000

# Token budgets (increased for reasoning models like GLM)
RAG_TEMPERATURE: float = 0.2
RAG_MAX_TOKENS: int = 1536

RECIPE_TEMPERATURE: float = 0.3
RECIPE_MAX_TOKENS: int = 2048

MENU_TEMPERATURE: float = 0.2
MENU_MAX_TOKENS: int = 3072

ALTERNATIVE_TEMPERATURE: float = 0.5
ALTERNATIVE_MAX_TOKENS: int = 1024

_API_KEY_PATTERN = re.compile(r"(AIza[0-9A-Za-z_\-]{10,}|sk-[0-9A-Za-z_\-]{10,})")


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing or invalid."""


class SensitiveDataFilter(logging.Filter):
    """Redacts API key patterns (Google AI Studio and OpenAI/SumoPod) from log records."""

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

    api_key: str | None
    base_url: str = DEFAULT_BASE_URL
    model_llm: str = DEFAULT_LLM_MODEL
    model_embedding: str = DEFAULT_EMBEDDING_MODEL
    db_path: Path = DEFAULT_DB_PATH
    csv_path: Path = DEFAULT_CSV_PATH
    chroma_path: Path = DEFAULT_CHROMA_PATH

    @property
    def gemini_api_key(self) -> str | None:
        """Backward-compatible alias for api_key."""
        return self.api_key

    @property
    def gemini_model(self) -> str:
        """Backward-compatible alias for model_llm."""
        return self.model_llm


def load_settings(
    env_file: str | Path | None = None,
    require_api_key: bool = False,
) -> AppSettings:
    """Load environment variables once and return structured application settings.

    Args:
        env_file: Optional path to a `.env` file. Defaults to automatic discovery.
        require_api_key: When True, raises ConfigurationError if API_KEY is missing.

    Returns:
        Populated AppSettings instance.

    Raises:
        ConfigurationError: If require_api_key is True and API_KEY is missing or blank.
    """
    if env_file is not None:
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    # Prioritize API_KEY as specified by user, with fallback to other common names
    raw_key = (
        os.getenv("API_KEY")
        or os.getenv("SUMOPOD_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    api_key = raw_key.strip() if raw_key and raw_key.strip() else None

    if require_api_key and not api_key:
        raise ConfigurationError(
            "API_KEY tidak ditemukan. Silakan set di file .env atau environment variables."
        )

    raw_base_url = (
        os.getenv("BASE_URL")
        or os.getenv("SUMOPOD_BASE_URL")
        or DEFAULT_BASE_URL
    ).strip()
    base_url = raw_base_url or DEFAULT_BASE_URL

    raw_llm = (
        os.getenv("MODEL_LLM")
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_LLM_MODEL
    ).strip()
    model_llm = raw_llm or DEFAULT_LLM_MODEL

    raw_embedding = os.getenv("MODEL_EMBEDDING", DEFAULT_EMBEDDING_MODEL).strip()
    model_embedding = raw_embedding or DEFAULT_EMBEDDING_MODEL

    return AppSettings(
        api_key=api_key,
        base_url=base_url,
        model_llm=model_llm,
        model_embedding=model_embedding,
    )
